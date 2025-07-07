import gym
from gym import spaces
from gym.wrappers.frame_stack import FrameStack
import cv2
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import random
import os

from config import DEADLOCK_PENALTY, DEADLOCK_STEPS, DEATH_PENALTY, COMPLETION_REWARD, ITEM_REWARD_FACTOR, \
    RANDOM_STAGES, SCORE_REWARD_FACTOR, MOVE_REWARD


# Environment preprocessing wrappers
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        obs_shape = self.shape  # Add channel dimension for grayscale
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # Remove channel dimension for resizing
        if len(observation.shape) == 3:
            observation = observation.squeeze(0)
        # Resize
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        # Add channel dimension back
        return observation


# wrapper to reduce frame amount
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        # sum up all rewards in skipped frames and keep doing the same action on skipped frames
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ScoreRewardWrapper(gym.Wrapper):
    def __init__(self, env, score_reward_scale=0.1):
        super().__init__(env)
        self.score_reward_scale = score_reward_scale
        self.last_score = 0

    def reset(self, **kwargs):
        self.last_score = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        # Calculate score reward
        current_score = info.get('score', 0)
        score_reward = (current_score - self.last_score) * self.score_reward_scale
        self.last_score = current_score
        
        # Add score reward to the original reward
        reward += score_reward
        
        return state, reward, done, info

class DeadlockEnv(gym.Wrapper):
    def __init__(self, env, threshold, deadlock_penalty):
        super().__init__(env)
        self.threshold = threshold
        self.last_x_pos = 0
        self.count = 0
        self.deadlock_penalty = deadlock_penalty

    def reset(self, **kwargs):
        self.last_x_pos = 0
        self.count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        x_pos = info['x_pos']

        if x_pos <= self.last_x_pos:
            self.count += 1
        else:
            self.last_x_pos = x_pos
            self.count = 0

        if self.count >= self.threshold:
            done = True
            reward -= self.deadlock_penalty

        return state, reward, done, info

# limits marios lives to a single one
class LifeLimitEnv(gym.Wrapper):
    def __init__(self, env, death_penalty):
        super().__init__(env)
        self.death_penalty = death_penalty

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if info['life'] < 2:
            done = True
            reward -= self.death_penalty

        return state, reward, done, info

# limits mario to play a single level and gives a reward for completion
class LevelLimitEnv(gym.Wrapper):
    def __init__(self, env, completion_reward):
        super().__init__(env)
        self.completion_reward = completion_reward

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # if mario reached flag end episode and give reward
        if info['flag_get']:
            done = True
            reward += self.completion_reward

        return state, reward, done, info

# overwrites the whole existing reward structure and rewards moving right, getting score
class RewardShaperEnv(gym.Wrapper):
    def __init__(self, env, death_penalty, score_reward_factor):
        super().__init__(env)
        self.last_x_pos = 0
        self.last_life = 2  # Track life to detect death
        # reward = distance * factor
        self.pos_mov_factor = MOVE_REWARD
        # separate factor for moving backwards
        self.neg_mov_factor = MOVE_REWARD / 2.0

        self.last_score = 0
        self.score_reward_scale = score_reward_factor
        self.death_penalty = death_penalty

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Get Mario's actual starting position by taking a no-op step
        _, _, _, info = self.env.step(0)  # No-op action to get initial info
        self.last_x_pos = info.get('x_pos', 0)
        self.last_life = info.get('life', 2)
        self.last_score = info.get('score', 0)
        return obs

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        reward = 0

        # Calculate score reward
        current_score = info.get('score', 0)
        score_reward = (current_score - self.last_score) * self.score_reward_scale
        self.last_score = current_score
        reward += score_reward

        # Check if Mario died (life decreased)
        current_life = info.get('life', 2)
        if current_life < self.last_life:
            # Mario died - apply death penalty and skip movement calculation
            reward -= self.death_penalty
        else:
            # Normal movement - calculate movement reward
            current_x_pos = info['x_pos']
            distance_moved = current_x_pos - self.last_x_pos
            if distance_moved > 0:
                reward += distance_moved * self.pos_mov_factor
            else:
                reward += distance_moved * self.neg_mov_factor
        
        # Update tracking variables
        self.last_x_pos = info['x_pos']
        self.last_life = current_life

        return state, reward, done, info

# gives mario reward and penalty for gaining or loosing item effects
class ItemRewardEnv(gym.Wrapper):
    def __init__(self, env, item_reward_factor):
        super().__init__(env)
        self.last_state = 'small'
        self.item_reward_factor = item_reward_factor
        self.states = ['small', 'tall', 'fireball']

    def reset(self, **kwargs):
        self.last_state = 'small'
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        current_state = info['status']

        if current_state != self.last_state:
            reward += (self.states.index(current_state) - self.states.index(self.last_state)) * self.item_reward_factor
            self.last_state = current_state

        return state, reward, done, info


# All available Mario stages (World 1-8, Stage 1-4 each = 32 total stages)
ALL_MARIO_STAGES = [
    (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 1), (2, 2), (2, 3), (2, 4),
    (3, 1), (3, 2), (3, 3), (3, 4),
    (4, 1), (4, 2), (4, 3), (4, 4),
    (5, 1), (5, 2), (5, 3), (5, 4),
    (6, 1), (6, 2), (6, 3), (6, 4),
    (7, 1), (7, 2), (7, 3), (7, 4),
    (8, 1), (8, 2), (8, 3), (8, 4)
]


class RandomSaveStateWrapper(gym.Wrapper):
    """Wrapper that loads random save states on reset."""
    
    def __init__(self, env, save_states_dir="save_states"):
        super().__init__(env)
        self.save_states_dir = save_states_dir
        
        # Find the NES environment
        self.nes_env = env
        while hasattr(self.nes_env, 'env'):
            self.nes_env = self.nes_env.env
    
    def reset(self, **kwargs):
        """Reset environment and optionally load a random save state."""
        obs = self.env.reset(**kwargs)
        
        # Try to load a random save state for current stage
        if self._load_random_save_state():
            print("🎯 Loaded random save state")
        
        return obs
    
    def _get_current_stage_info(self):
        """Get current world and stage from environment."""
        try:
            # Take a step to get info
            obs, reward, done, info = self.env.step(0)
            world = info.get('world', 1)
            stage = info.get('stage', 1)
            return world, stage
        except:
            return 1, 1  # Default to world 1, stage 1
    
    def _load_random_save_state(self):
        """Load a random save state for the current stage using surgical restoration."""
        try:
            # Get current stage
            world, stage = self._get_current_stage_info()
            
            # Find save states for this stage
            stage_dir = os.path.join(self.save_states_dir, f"world_{world}_stage_{stage}")
            if not os.path.exists(stage_dir):
                return False
            
            # Get all RAM save files
            ram_files = [f for f in os.listdir(stage_dir) if f.startswith('ram_') and f.endswith('.npy')]
            if not ram_files:
                return False
            
            # Load a random save state
            random_file = random.choice(ram_files)
            ram_path = os.path.join(stage_dir, random_file)
            saved_ram = np.load(ram_path)
            
            # Improved surgical restoration - use only addresses that work well
            # Based on analysis: avoid 0x006D as it causes interference
            essential_addresses = [
                0x0086,  # Player X Position (high byte) - primary position
                0x03AD,  # Current Screen/Page Position - screen scrolling
                # Note: Excluding 0x006D as it causes position to reset to 0
            ]
            
            # Restore only the essential addresses
            for addr in essential_addresses:
                if addr < len(self.nes_env.ram) and addr < len(saved_ram):
                    self.nes_env.ram[addr] = saved_ram[addr]
            
            # Skip _frame_advance as it might be resetting values
            # Let the environment naturally sync on the next step
            
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load save state: {e}")
            return False


def create_env():
    env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0' if RANDOM_STAGES else 'SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    # Apply all the preprocessing wrappers
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 128)
    env = SkipFrame(env, skip=6)
    env = FrameStack(env, 8)
    # Apply reward-modifying wrappers first
    env = LifeLimitEnv(env, death_penalty=DEATH_PENALTY)
    env = LevelLimitEnv(env, completion_reward=COMPLETION_REWARD)
    env = ItemRewardEnv(env, item_reward_factor=ITEM_REWARD_FACTOR)
    env = DeadlockEnv(env, threshold=DEADLOCK_STEPS, deadlock_penalty=DEADLOCK_PENALTY)
    # Apply RewardShaperEnv LAST to completely overwrite all rewards
    env = RewardShaperEnv(env, death_penalty=DEATH_PENALTY, score_reward_factor=SCORE_REWARD_FACTOR)
    return env
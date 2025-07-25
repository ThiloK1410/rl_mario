from typing import Tuple

import gym
from gym import spaces
from gym.core import ActType, ObsType
from gym.wrappers.frame_stack import FrameStack
import cv2
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
# Use expanded action space to match recording environment
import random
import os
import pickle
import glob
import json
from collections import deque



from config import (
    DEADLOCK_PENALTY, DEADLOCK_STEPS, DEATH_PENALTY, COMPLETION_REWARD,
    ITEM_REWARD_FACTOR, RANDOM_STAGES, SCORE_REWARD_FACTOR,
    USE_RECORDED_GAMEPLAY, RECORDED_GAMEPLAY_DIR, get_recorded_start_probability,
    PREFER_ADVANCED_CHECKPOINTS, MIN_CHECKPOINT_X_POS, ONE_RECORDING_PER_STAGE, MOVE_REWARD,
    MIN_SAMPLING_PERCENTAGE, MAX_SAMPLING_PERCENTAGE, MOVE_REWARD_CAP, DOWNSCALE_RESOLUTION, SKIPPED_FRAMES,
    STACKED_FRAMES, USED_MOVESET, EXPANDED_COMPLEX_MOVEMENT, SPARSE_FRAME_INTERVAL
)


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
        self.last_x_pos = None  # Will be set on first step
        self.last_score = None  # Will be set on first step
        self.count = 0
        self.deadlock_penalty = deadlock_penalty
        self.first_step = True

    def reset(self, **kwargs):
        self.last_x_pos = None  # Will be set on first step
        self.last_score = None  # Will be set on first step
        self.count = 0
        self.first_step = True
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        x_pos = info['x_pos']
        current_score = info.get('score', 0)

        if self.first_step:
            # First step: establish baseline position and score
            self.last_x_pos = x_pos
            self.last_score = current_score
            self.first_step = False
            self.count = 0
        else:
            # Normal deadlock detection
            # Reset counter if either position or score has improved
            if x_pos > self.last_x_pos or current_score > self.last_score:
                # Update to best position and score
                if x_pos > self.last_x_pos:
                    self.last_x_pos = x_pos
                if current_score > self.last_score:
                    self.last_score = current_score
                self.count = 0
            else:
                self.count += 1

            if self.count >= self.threshold:
                done = True
                reward -= self.deadlock_penalty

        return state, reward, done, info


# limits marios lives to a single one
class LifeLimitEnv(gym.Wrapper):
    def __init__(self, env, death_penalty):
        super().__init__(env)
        self.death_penalty = death_penalty
        self.last_lives = None
        self.previous_state = None
        self.previous_info = None

    def reset(self, **kwargs):
        self.last_lives = None
        self.previous_state = None
        self.previous_info = None
        return self.env.reset(**kwargs)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # Take the step in the base environment
        state, reward, done, info = self.env.step(action)
        
        current_life = info['life']
        
        # Initialize on first step
        if self.last_lives is None:
            self.last_lives = current_life
            self.previous_state = state
            self.previous_info = info
            return state, reward, done, info
        
        # Check if Mario died (life decreased)
        if current_life < self.last_lives:
            # Mario died! Return the previous state (before death)
            done = True
            
            # Add death information to the info dict for RewardShaperEnv
            death_info = self.previous_info.copy()
            death_info['mario_died'] = True
            death_info['death_penalty'] = self.death_penalty
            
            # Return the state from before death occurred
            return self.previous_state, reward, done, death_info
        
        # Normal step - update tracking and return current state
        self.last_lives = current_life
        self.previous_state = state
        self.previous_info = info
        
        return state, reward, done, info


# limits mario to play a single level and gives a reward for completion
class LevelLimitEnv(gym.Wrapper):
    def __init__(self, env, completion_reward):
        super().__init__(env)
        self.completion_reward = completion_reward

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # Check for level completion using only flag_get
        if info.get('flag_get', False):
            done = True
            reward += self.completion_reward

        return state, reward, done, info

# overwrites the whole existing reward structure and rewards moving right, getting score
class RewardShaperEnv(gym.Wrapper):
    def __init__(self, env, death_penalty, score_reward_factor):
        super().__init__(env)
        self.last_x_pos = None  # Will be set on first step
        self.last_life = 2  # Track life to detect death
        # reward = distance * factor
        self.pos_mov_factor = MOVE_REWARD
        # separate factor for moving backwards
        self.neg_mov_factor = MOVE_REWARD / 2.0

        self.last_score = 0
        self.score_reward_scale = score_reward_factor
        self.death_penalty = death_penalty
        self.first_step = True  # Track if this is the first step after reset

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Reset tracking variables but don't probe for position
        # This avoids interfering with RecordedGameplayWrapper
        self.last_x_pos = None  # Will be set on first step
        self.last_life = 2
        self.last_score = 0
        self.first_step = True
        return obs

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        # REPLACE base reward with our custom reward structure
        # Start with 0 and build up our own reward system
        reward = 0.0    

        # Calculate score reward
        current_score = info.get('score', 0)
        score_reward = (current_score - self.last_score) * self.score_reward_scale
        self.last_score = current_score
        reward += score_reward

        # Check if Mario died - either from life decrease or info flag
        current_life = info.get('life', 2)
        mario_died = info.get('mario_died', False)
        
        if current_life < self.last_life or mario_died:
            # Mario died - apply death penalty and skip movement calculation
            reward -= self.death_penalty
        else:
            # Handle movement reward
            current_x_pos = info['x_pos']

            if self.first_step:
                # First step after reset: set initial position without movement reward
                self.last_x_pos = current_x_pos
                self.first_step = False
                # No movement reward on first step (establishes baseline)
            else:
                # Normal movement - calculate movement reward
                distance_moved = current_x_pos - self.last_x_pos
                if distance_moved > 0:
                    move_reward = distance_moved * self.pos_mov_factor
                else:
                    move_reward = distance_moved * self.neg_mov_factor
                
                # Cap the movement reward to prevent it from overwhelming other signals
                move_reward = max(-MOVE_REWARD_CAP, min(MOVE_REWARD_CAP, move_reward))
                
                reward += move_reward

                # Update position tracking
                self.last_x_pos = current_x_pos
        
        # Update life tracking (but only if we don't have death flag from previous wrapper)
        if not mario_died:
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
# You can modify this list to exclude stages you don't want to train on
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


class CustomRandomStageWrapper(gym.Wrapper):
    """Custom wrapper that randomly selects stages from ALL_MARIO_STAGES list."""
    
    def __init__(self, env):
        super().__init__(env)
        self.current_world = 1
        self.current_stage = 1
        self.select_random_stage()
    
    def select_random_stage(self):
        """Select a random stage from ALL_MARIO_STAGES list."""
        if not ALL_MARIO_STAGES:
            raise ValueError("ALL_MARIO_STAGES list is empty. Please add stages to train on.")
        
        # Select random stage from the list
        self.current_world, self.current_stage = random.choice(ALL_MARIO_STAGES)
        
        # Instead of creating a new environment (which causes window spawning),
        # we'll change the stage by creating a new environment but closing the old one properly
        try:
            # Close the old environment properly to avoid window spawning
            if hasattr(self.env, 'close'):
                self.env.close()
            
            # Create new environment for this stage
            env_name = f'SuperMarioBros-{self.current_world}-{self.current_stage}-v0'
            new_env = gym_super_mario_bros.make(env_name)
            new_env = JoypadSpace(new_env, USED_MOVESET)
            
            # Update the underlying environment
            self.env = new_env

        except Exception as e:
            print(f"Warning: Failed to create environment for stage {self.current_world}-{self.current_stage}: {e}")
            # Fallback to World 1-1
            self.current_world, self.current_stage = 1, 1
            try:
                if hasattr(self.env, 'close'):
                    self.env.close()
                fallback_env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
                fallback_env = JoypadSpace(fallback_env, USED_MOVESET)
                self.env = fallback_env
            except:
                pass  # Keep the existing env if fallback fails
    
    def reset(self, **kwargs):
        """Reset and select a new random stage."""
        self.select_random_stage()
        return self.env.reset(**kwargs)
    
    def get_current_stage_info(self):
        """Get the current world and stage info."""
        return self.current_world, self.current_stage


class RecordedGameplayWrapper(gym.Wrapper):
    """Wrapper that replays recorded actions to start from random positions."""
    
    def __init__(self, env):
        super().__init__(env)
        self.recorded_sessions = {}  # Cache for loaded sessions
        self.last_loaded_stage = None
        self.used_recorded_start = False  # Track if current episode used recorded start

    def get_current_stage_info(self):
        """Get current world and stage from environment."""
        try:
            # First check if we have a CustomRandomStageWrapper in the chain
            current_env = self.env
            while hasattr(current_env, 'env'):
                if hasattr(current_env, 'get_current_stage_info'):
                    # Found CustomRandomStageWrapper
                    return current_env.get_current_stage_info()
                current_env = current_env.env
            
            # Check if the current env itself has get_current_stage_info (it might be the CustomRandomStageWrapper)
            if hasattr(self.env, 'get_current_stage_info'):
                return self.env.get_current_stage_info()
            
            # Fallback: try to get stage info from the environment spec
            current_env = self.env
            while hasattr(current_env, 'env'):
                if hasattr(current_env, 'spec') and current_env.spec and 'SuperMarioBros' in current_env.spec.id:
                    # Found the base Mario environment
                    break
                current_env = current_env.env
            
            # Try to get stage info from the environment spec or take a minimal step
            if hasattr(current_env, 'spec') and current_env.spec:
                spec_id = current_env.spec.id
                if 'SuperMarioBros-' in spec_id:
                    # Extract world and stage from spec ID (e.g., "SuperMarioBros-1-1-v0")
                    parts = spec_id.split('-')
                    if len(parts) >= 3:
                        try:
                            world = int(parts[1])
                            stage = int(parts[2])
                            return world, stage
                        except ValueError:
                            pass
            
            # Final fallback: take a no-op step to get info
            obs, reward, done, info = self.env.step(0)
            world = info.get('world', 1)
            stage = info.get('stage', 1)
            return world, stage
        except:
            return 1, 1  # Default to world 1, stage 1
    
    def _load_recorded_sessions(self, world, stage):
        """Load recorded sessions for a specific world/stage."""
        from config import RECORDED_GAMEPLAY_DIR
        
        stage_key = f"world_{world}_stage_{stage}"

        # Check if already loaded
        if stage_key in self.recorded_sessions:
            return self.recorded_sessions[stage_key]

        # Find recorded action files for this stage
        stage_dir = os.path.join(RECORDED_GAMEPLAY_DIR, stage_key)
        
        if not os.path.exists(stage_dir):
            self.recorded_sessions[stage_key] = []
            return []

        action_files = glob.glob(os.path.join(stage_dir, "actions_*.json"))
        
        if not action_files:
            self.recorded_sessions[stage_key] = []
            return []

        # Load the first recording found
        sessions = []
        try:
            with open(action_files[0], 'r') as f:
                session_data = json.load(f)
                if session_data.get('actions'):  # Just check if actions exist
                    sessions.append(session_data)
        except Exception:
            pass

        self.recorded_sessions[stage_key] = sessions
        return sessions

    def _replay_to_random_position(self, world, stage):
        """Replay actions to reach a random position."""
        try:
            # Import config values dynamically to allow runtime changes
            from config import PREFER_ADVANCED_CHECKPOINTS, MIN_SAMPLING_PERCENTAGE, MAX_SAMPLING_PERCENTAGE
            
            # Load recorded sessions for this stage
            sessions = self._load_recorded_sessions(world, stage)
            if not sessions:
                return False
            
            # Choose a random session
            session = random.choice(sessions)
            actions = session.get('actions', [])
            if not actions:
                return False
            
            # Choose a random position in the action sequence
            if PREFER_ADVANCED_CHECKPOINTS:
                # Focus on the productive middle-to-late range (configurable)
                min_position = int(len(actions) * MIN_SAMPLING_PERCENTAGE)
                max_position = int(len(actions) * MAX_SAMPLING_PERCENTAGE)
                
                # Use power distribution within this range to favor later positions
                relative_position = random.random() ** 0.7
                position = min_position + int((max_position - min_position) * relative_position)
            else:
                # Improved sampling: use the configured range instead of full range
                min_position = int(len(actions) * MIN_SAMPLING_PERCENTAGE)
                max_position = int(len(actions) * MAX_SAMPLING_PERCENTAGE)
                
                # Uniform distribution within the configured range
                position = random.randint(min_position, max_position)
            
            # Reduce position by 1 to avoid reaching terminal states
            position = max(1, position - 1)
            
            # Replay actions up to the chosen position
            for i, action in enumerate(actions[:position]):
                if i >= position:
                    break
                obs, reward, done, info = self.env.step(action)
                if done:
                    # If episode ends during replay, reset and return False
                    self.env.reset()
                    return False
            
            return True
            
        except Exception:
            return False

    def step(self, action):
        """Step the environment."""
        return self.env.step(action)

    def reset(self, **kwargs):
        """Reset environment."""
        obs = self.env.reset(**kwargs)
        
        # Reset the flag
        self.used_recorded_start = False
        
        # Import config values dynamically to allow runtime changes
        from config import USE_RECORDED_GAMEPLAY, get_recorded_start_probability
        
        # Get current recorded start probability based on training progress
        current_recorded_probability = get_recorded_start_probability()
        
        # Check if we should use recorded gameplay
        if USE_RECORDED_GAMEPLAY and random.random() < current_recorded_probability:
            # Get current stage info
            world, stage = self.get_current_stage_info()
            
            # Try to replay actions to a random position
            if self._replay_to_random_position(world, stage):
                self.used_recorded_start = True
        
        return obs

# by thilo
class SparseFrameStack(gym.Wrapper):
    def __init__(self, env, num_stack=4, sparse_interval=2):
        super().__init__(env)
        self.num_stack = num_stack
        self.sparse_interval = sparse_interval
        
        # Calculate buffer size needed to hold enough frames for sparse stacking
        # We need at least (num_stack - 1) * sparse_interval + 1 frames
        self.buffer_size = (num_stack - 1) * sparse_interval + 1
        
        # Initialize frame buffer as a deque for efficient operations
        self.frame_buffer = deque(maxlen=self.buffer_size)
        
        # Get the shape of individual frames from the environment
        obs_shape = env.observation_space.shape
        
        # Create new observation space with stacked frames in (C, H, W) format
        # If the original observation has channels, multiply by num_stack
        if len(obs_shape) == 3:  # (height, width, channels)
            new_shape = (obs_shape[2] * num_stack, obs_shape[0], obs_shape[1])
        else:  # (height, width) - grayscale
            new_shape = (num_stack, obs_shape[0], obs_shape[1])
        
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8
        )
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        # Clear the buffer
        self.frame_buffer.clear()
        
        # Fill the buffer with the initial frame
        for _ in range(self.buffer_size):
            self.frame_buffer.append(obs)
        
        # Return the stacked observation
        return self._get_stacked_observation()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Add new frame to buffer (automatically removes oldest if full)
        self.frame_buffer.append(obs)
        
        # Get stacked observation
        stacked_obs = self._get_stacked_observation()
        
        return stacked_obs, reward, done, info
    
    def _get_stacked_observation(self):
        if len(self.frame_buffer) < self.buffer_size:
            # Not enough frames yet, duplicate the available frames
            available_frames = list(self.frame_buffer)
            while len(available_frames) < self.num_stack:
                available_frames.insert(0, available_frames[0])
            selected_frames = available_frames[-self.num_stack:]
        else:
            # Select frames at sparse intervals, working backwards from most recent
            selected_frames = []
            for i in range(self.num_stack):
                frame_index = -(1 + i * self.sparse_interval)
                selected_frames.insert(0, self.frame_buffer[frame_index])
        
        # Stack the selected frames
        return self._stack_frames(selected_frames)
    
    def _stack_frames(self, frames):
        if len(frames[0].shape) == 3:  # (channels, height, width)
            # Stack along the channel dimension
            stacked = np.concatenate(frames, axis=2)
        else:  # (height, width) - grayscale
            # Stack along a new channel dimension
            stacked = np.stack(frames, axis=2)
        
        # Convert from (H, W, C) to (C, H, W) format for compatibility with DQN
        return np.transpose(stacked, (2, 0, 1))


def create_env(use_level_start=False):
    # Import config values dynamically to allow runtime changes
    from config import RANDOM_STAGES, USE_RECORDED_GAMEPLAY, DEADLOCK_STEPS, DEATH_PENALTY, SCORE_REWARD_FACTOR, COMPLETION_REWARD, ITEM_REWARD_FACTOR, DEADLOCK_PENALTY
    
    # Create base environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')  # Default to World 1-1
    env = JoypadSpace(env, USED_MOVESET)

    # Apply the custom random stage wrapper ONLY if RANDOM_STAGES is True
    if RANDOM_STAGES:
        env = CustomRandomStageWrapper(env)
        print(f"[ENV] Using custom random stage selection from {len(ALL_MARIO_STAGES)} stages")
    else:
        print(f"[ENV] Using fixed stage: World 1-1")

    # Apply all the preprocessing wrappers
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, DOWNSCALE_RESOLUTION)
    env = SkipFrame(env, skip=SKIPPED_FRAMES)
    
    # Apply recorded gameplay wrapper AFTER SkipFrame to match recording level
    # This ensures recorded actions are replayed at the same abstraction level they were recorded
    if USE_RECORDED_GAMEPLAY and not use_level_start:
        env = RecordedGameplayWrapper(env)
    
    env = FrameStack(env, STACKED_FRAMES)
    
    # Apply RewardShaperEnv FIRST (it handles death penalty and movement rewards)
    env = RewardShaperEnv(env, death_penalty=DEATH_PENALTY, score_reward_factor=SCORE_REWARD_FACTOR)
    
    # Apply other reward-modifying wrappers that ADD to the shaped reward
    env = LifeLimitEnv(env, death_penalty=0)  # No additional death penalty - RewardShaperEnv handles it
    env = LevelLimitEnv(env, completion_reward=COMPLETION_REWARD)
    env = ItemRewardEnv(env, item_reward_factor=ITEM_REWARD_FACTOR)
    env = DeadlockEnv(env, threshold=DEADLOCK_STEPS, deadlock_penalty=DEADLOCK_PENALTY)

    # Add helper method to access recorded start flag through wrapper chain
    def get_used_recorded_start():
        """Helper method to find RecordedGameplayWrapper in the wrapper chain."""
        current_env = env
        while hasattr(current_env, 'env'):
            if hasattr(current_env, 'used_recorded_start'):
                return current_env.used_recorded_start
            current_env = current_env.env
        return False  # Default to False if no RecordedGameplayWrapper found
    
    env.get_used_recorded_start = get_used_recorded_start

    return env

def create_env_new():
    env = gym_super_mario_bros.make('SuperMarioBros-v1')
    env = JoypadSpace(env, USED_MOVESET)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, DOWNSCALE_RESOLUTION)
    env = DeadlockEnv(env, deadlock_penalty=DEADLOCK_PENALTY, threshold=DEADLOCK_STEPS)

    env = SkipFrame(env, skip=8)
    
    env = SparseFrameStack(env, num_stack=STACKED_FRAMES, sparse_interval=SPARSE_FRAME_INTERVAL)
       

    return env
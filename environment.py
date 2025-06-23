import gym
import cv2
import numpy as np
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from config import DEADLOCK_PENALTY, DEADLOCK_STEPS, DEATH_PENALTY, COMPLETION_REWARD, ITEM_REWARD_FACTOR, RANDOM_STAGES


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

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # if mario reached flag end episode and give reward
        if info['flag_get']:
            done = True
            reward += self.completion_reward

        return state, reward, done, info

# overwrites the whole existing reward structure and rewards moving right, getting score
class RewardShaperEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_x_pos = 0
        # reward = distance * factor
        self.pos_mov_factor = 1.0 / 12.0
        # separate factor for moving backwards
        self.neg_mov_factor = self.pos_mov_factor / 2.0

        self.last_score = 0
        self.score_reward_scale = 0.01

    def reset(self, **kwargs):
        self.last_x_pos = 0
        self.last_score = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        reward = 0

        # Calculate score reward
        current_score = info.get('score', 0)
        score_reward = (current_score - self.last_score) * self.score_reward_scale
        self.last_score = current_score
        # Add score reward to the original reward
        reward += score_reward

        # Calculate moving score reward
        current_x_pos = info['x_pos']
        distance_moved = current_x_pos - self.last_x_pos
        if distance_moved > 0:
            reward += distance_moved * self.pos_mov_factor
        else:
            reward += distance_moved * self.neg_mov_factor
        self.last_x_pos = current_x_pos

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






def create_env():
    """Create and wrap a Mario environment with all necessary wrappers."""
    env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0' if RANDOM_STAGES else 'SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 128)
    env = SkipFrame(env, skip=6)
    env = FrameStack(env, 8)
    env = RewardShaperEnv(env)
    env = LifeLimitEnv(env, death_penalty=DEATH_PENALTY)
    env = LevelLimitEnv(env, completion_reward=COMPLETION_REWARD)
    env = ItemRewardEnv(env, item_reward_factor=ITEM_REWARD_FACTOR)
    env = DeadlockEnv(env, threshold=DEADLOCK_STEPS, deadlock_penalty=DEADLOCK_PENALTY)
    return env
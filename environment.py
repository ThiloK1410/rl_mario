import gym
import cv2
import numpy as np
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

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
    def __init__(self, env, threshold=10):
        super().__init__(env)
        self.last_x_pos = 0
        self.count = 0
        self.threshold = threshold
        self.lifes = 3
        self.stage = 1
        self.world = 1

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
            self.count = 0
            self.last_x_pos = x_pos

        # Reset counter when life/stage/world changes
        if info['life'] != self.lifes or info["stage"] != self.stage or info["world"] != self.world:
            self.last_x_pos = x_pos
            self.count = 0
            self.lifes = info['life']
            self.stage = info["stage"]
            self.world = info["world"]

        if self.count >= self.threshold:
            done = True

        if info["life"] < 2:
            #reward -= 30  # Penalty for dying
            done = True

        return state, reward, done, info


def create_env(deadlock_steps=10, random_stages=True):
    """Create and wrap a Mario environment with all necessary wrappers."""
    env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0' if random_stages else 'SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 128)
    env = SkipFrame(env, skip=4)
    env = FrameStack(env, 4)
    env = ScoreRewardWrapper(env)  # Add score reward wrapper
    env = DeadlockEnv(env, threshold=deadlock_steps)
    return env
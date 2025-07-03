from time import sleep

import torch

from dqn_agent import MarioAgent
from environment import create_env
from mario_rl_simple import find_latest_checkpoint, load_checkpoint
from config import BUFFER_SIZE, DEADLOCK_STEPS, AGENT_FOLDER, RANDOM_STAGES, RANDOM_SAVES

import numpy as np


def main():
    env = create_env(sample_random_stages=RANDOM_STAGES, use_random_saves=RANDOM_SAVES)
    agent = MarioAgent((128, 128), env.action_space.n, experience_queue=None)  # type: ignore
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir=AGENT_FOLDER)
    if latest_checkpoint:
        start_epoch = load_checkpoint(agent, latest_checkpoint)
        print(f"loaded agent from epoch {start_epoch}")
    else:
        print("No checkpoint found")
        exit(1)

    for i in range(20):
        print(f"Starting episode {i}")
        state = env.reset()
        done = False
        total_reward = 0
        current_step = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            env.render()
            total_reward += reward
            state = next_state
            print(f"Step {current_step}\treward: {reward}\tx_position: {info['x_pos']}\ty_position: {info['y_pos']}\tstate: {info['status']}")
            sleep(1.0/5.0)

        print(f"total reward in episode {i}: {total_reward}")


if __name__ == "__main__":
    main()
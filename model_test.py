from time import sleep

import torch

from dqn_agent import MarioAgent
from environment import create_env
from mario_rl import find_latest_checkpoint, load_checkpoint, BUFFER_SIZE
from multiprocessing import Queue

import numpy as np


def main():
    env = create_env(random_stages=True, deadlock_steps=200)
    experience_queue = Queue()
    agent = MarioAgent((128, 128), env.action_space.n, experience_queue)
    latest_checkpoint = find_latest_checkpoint()
    if latest_checkpoint:
        start_epoch = load_checkpoint(agent, latest_checkpoint)
        print(f"loaded agent from epoch {start_epoch}")
    else:
        print("No checkpoint found")
        exit(1)

    for i in range(10):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            env.render()
            total_reward += reward
            state = next_state
            sleep(1.0/15.0)

        print(f"total reward in game {i}: {total_reward}")


if __name__ == "__main__":
    main()
from time import sleep

import torch

from dqn_agent import MarioAgent
from environment import create_env
from mario_rl_simple import find_latest_checkpoint, load_checkpoint, list_available_experiments
from config import AGENT_FOLDER

import numpy as np


def main():
    env = create_env()
    agent = MarioAgent((128, 128), env.action_space.n, experience_queue=None)  # type: ignore
    
    # Show available experiments
    available_experiments = list_available_experiments(checkpoint_dir=AGENT_FOLDER)
    if available_experiments:
        print(f"Available experiments: {', '.join(available_experiments)}")
        experiment_name = input("Enter experiment name to test (or press Enter for latest): ").strip()
        if not experiment_name:
            experiment_name = None
    else:
        experiment_name = None
    
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir=AGENT_FOLDER, experiment_name=experiment_name)
    if latest_checkpoint:
        start_epoch, _ = load_checkpoint(agent, latest_checkpoint)
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
            sleep(1.0/20.0)

        print(f"total reward in episode {i}: {total_reward}")


if __name__ == "__main__":
    main()
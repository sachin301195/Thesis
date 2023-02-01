import gym
import numpy as np
import matplotlib.pyplot as plt

from jsp_env.src.graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from pathlib import Path
# from main import instance_creator
# from main import JspEnv_v1
import random
import logging

from evaluate import evaluate_instance

import json
import time
import networkx as nx


def configure_logger():
    Path(f'./agents_runs/ConveyorEnv_D/random/').mkdir(parents=True, exist_ok=True)
    agent_save_path = './agents_runs/' + 'ConveyorEnv_D' + '/' + 'random'
    # best_agent_save_path = './agents_runs/' + 'ConveyorEnv_v3' + '/' + 'random' + '_best_agents'
    # Path(best_agent_save_path).mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d")
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    # Path("./logs_new").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(agent_save_path + timestamp + '.log')
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    return _logger


logger = configure_logger()
agent_save_path = "./agents_runs/random/"

NUM_EPISODES = 10
REWARDS = []
MAKESPAN = []
OPTIMUM_VALUE = []
TRIAL = False


if __name__ == "__main__":
    size = "3x3"
    c_episode = 0
    while c_episode <= NUM_EPISODES:
        jsp = evaluate_instance(size, c_episode)
        env_config = {
            "jsp": jsp,
            "reward_version": "A",
            # "reward_version": "A",
            "scaling_divisor": 100,
            "scale_reward": True,
            "perform_left_shift_if_possible": True,
            "normalize_observation_space": False,
            "flat_observation_space": False,
            "action_mode": "task",
            "env_transform": "mask",
            "verbose": 2,
            "dtype": "float32",
        }
        env = DisjunctiveGraphJspEnv(env_config)
        c_episode += 1
        obs = env.reset(jsp)
        print(obs["observations"])
        # nx.draw(env.G, with_labels=True)
        done = False
        # print(nx.to_numpy_array(env.G)[1:-1, 2:].astype(dtype=int))
        step = 0
        score = 0
        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            print(state["observations"])
            score += reward

        REWARDS.append(score)
        MAKESPAN.append(info["makespan"])
        OPTIMUM_VALUE.append(info["optimal_value"])

    # plt.scatter(OPTIMUM_VALUE.keys(), MAKESPAN, marker='^')
    # plt.scatter(OPTIMUM_VALUE.keys(), OPTIMUM_VALUE.values(), marker='o')
    # plt.savefig(r"./plots/testPlot.png")





import gym
import numpy as np
import matplotlib.pyplot as plt

from jsp_env.src.graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from pathlib import Path
from main import instance_creator
from main import JspEnv_v1
import random
import logging

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

SIZE = "3x3"
NUM_EPISODES = 10
REWARDS = []
MAKESPAN = []
OPTIMUM_VALUE = {}
TRIAL = False


if __name__ == "__main__":
    size = "3x3"
    c_episode = 1
    while c_episode <= NUM_EPISODES:
        c_episode += 1
        jsp = instance_creator(size, "evaluate")

        env = DisjunctiveGraphJspEnv(jps_instance=jsp[0],
                                     scaling_divisor=40,
                                     action_mode="job",
                                     env_transform="mask",
                                     perform_left_shift_if_possible=True,
                                     normalize_observation_space=True,
                                     flat_observation_space=True,
                                     verbose=2)
        OPTIMUM_VALUE[jsp[1]] = jsp[2]
        obs = env.reset()
        print(nx.to_numpy_array(env.G).astype(dtype=int))
        # nx.draw(env.G, with_labels=True)
        done = False
        # print(nx.to_numpy_array(env.G)[1:-1, 2:].astype(dtype=int))
        step = 0
        score = 0
        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            score += reward

        REWARDS.append(score)
        MAKESPAN.append(info["makespan"])

    plt.scatter(OPTIMUM_VALUE.keys(), MAKESPAN, marker='^')
    plt.scatter(OPTIMUM_VALUE.keys(), OPTIMUM_VALUE.values(), marker='o')
    plt.savefig(r"./plots/testPlot.png")





import argparse
import platform
import sys
import logging
from abc import ABC

import gym

from util import *

import numpy as np
import pandas as pd

from pathlib import Path
import os
import matplotlib.pyplot as plt
import time
import random
import json
from typing import Dict

import ray
from ray import tune
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms import a3c
from ray.rllib.algorithms import dqn
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from jsp_env.src.graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from main import JspEnv_v1
# from main import instance_creator
from jsp_env.src.graph_jsp_env.disjunctive_graph_logger import log
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

EVALUATE = [120,  12, 278, 797, 230, 692, 917, 565, 301, 844,  68, 930, 851,
            464, 621, 307, 674, 206, 106, 840, 730, 489, 680, 742, 280, 667,
            998, 810, 621, 162, 962, 709, 467, 760, 426, 442, 716,  11, 262,
            665, 808, 661, 373, 956, 223, 985, 449, 293, 856, 500, 725, 641,
            167, 339, 898, 534, 450, 150, 906, 303, 319, 755, 240, 953,  28,
            661, 558, 632, 356, 199, 891, 747, 352, 560, 830, 319, 231,  91,
            752, 271, 244, 466, 920, 189, 818, 370, 873, 809, 209, 128, 972,
            477, 827, 375, 794, 454, 284, 256, 757, 678]


def configure_logger():
    timestamp = time.strftime("%Y-%m-%d")
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)
    Path(f"./logs/{args.algo}").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(f'./logs/application-{args.algo}-{str(args.file_no)}' + timestamp + '.log')
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    formatter = logging.Formatter('%(asctime)s  %(name)s  %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    return _logger, timestamp


torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--algo",
    type=str,
    default="PPO",
    choices=["PPO", "A2C", "A3C", "DQN"],
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Weather this script should be run as a test: --stop-reward must "
         "be achieved within --stop-timesteps AND --stop-iters")
parser.add_argument(
    "--run-type",
    default="evaluate",
    type=str,
    choices=["train", "test", "trial", "evaluate"],
    help="Algo running as"
)
parser.add_argument(
    "--instance-size",
    type=str,
    default="6x6",
    choices=["3x3", "6x6", "8x8", "10x10", "15x15", "any"],
    help="Jsp instance size")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=500,
    help="Number of iterations to train")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--eval-tune",
    default=True,
    type=bool,
    help="Evaluation while training TRUE/FALSE"
)
parser.add_argument(
    "--eval-interval",
    default=50,
    type=int,
    help="Evaluation after n training iterations"
)
parser.add_argument(
    "--no-tune",
    default=False,
    type=bool,
    help="Run without Tune using a manual train loop instead. In this case,"
         "use ALGO without TensorBoard.")
parser.add_argument(
    "--lstm",
    default=False,
    type=bool,
    help="Use LSTM or not")
parser.add_argument(
    "--masking",
    default="mask",
    type=str,
    choices=[None, "mask"],
    help="Use masking or not")
parser.add_argument(
    "--file-no",
    default=1,
    type=int,
    help="Use masking or not")
parser.add_argument(
    "--local-mode",
    help="Init Ray in local mode for easier debugging.",
    action="store_true")
parser.add_argument(
    "--normalize-obs",
    default=True,
    type=bool,
    help="Normalize obs or not")
parser.add_argument(
    "--flat-obs",
    default=True,
    type=bool,
    help="flatting the obs or not")
parser.add_argument(
    "--left-shift",
    default=True,
    type=bool,
    help="Perform left shift if possible or not for the job operations")
parser.add_argument(
    "--action-mode",
    default='task',
    type=str,
    choices=['task', 'job'],
    help="Choose action mode for the env")
parser.add_argument(
    "--env-verbose",
    default=0,
    type=int,
    help="verbose for the env")
parser.add_argument(
    "--scaling-divisor",
    default=100,
    type=int,
    help="scaling the reward")
parser.add_argument(
    "--no-of-workers",
    default=0,
    type=int,
    help="number of actors")
parser.add_argument(
    "--scale-reward",
    default=True,
    type=bool,
    help="Use reward scaling or not")


def setup(algo, timestamp):
    Path(f'./plots/{algo}/{timestamp}').mkdir(parents=True, exist_ok=True)
    plots_save_path = './plots/' + algo + "/" + timestamp
    Path(f'./agents_runs/{algo}//{timestamp}').mkdir(parents=True, exist_ok=True)
    agent_save_path = './agents_runs/' + algo + '/' + timestamp
    best_agent_save_path = './agents_runs/' + algo + '/' + timestamp \
                           + '_best_agents'
    # best_agent_save_path = './agents_runs/ConveyorEnv_token_n/PPO_best_agents'
    Path(best_agent_save_path).mkdir(parents=True, exist_ok=True)

    return plots_save_path, agent_save_path, best_agent_save_path


def evaluate_instance(size, no):
    with open(f"./data/{size}.json") as f:
        data = json.load(f)
        instance_no = EVALUATE[no]
        opt_value = data["optimal_time"][str(instance_no)]
        jsp_data = data["jobs_data"][str(instance_no)]
        m = int(size[0])
        if m not in [3, 6, 8]:
            if m == 1:
                m = int(size[:2])
        machine = []
        duration = []
        for i in range(len(jsp_data)):
            c = 0
            for j in jsp_data[i]:
                if c % 2 == 0:
                    machine.append(j)
                else:
                    duration.append(j)
                c += 1
        machine = list(map(int, machine))
        duration = list(map(int, duration))
        # print(machine, duration)
        machine = np.array(machine).reshape(m, m)
        duration = np.array(duration).reshape(m, m)
        jsp = np.concatenate((machine, duration), axis=0, dtype=int).reshape(2, m, m)

    return jsp, opt_value


def evaluate(algo, algo_config: dir, plots_save_path):
    checkpoint_path = r"E:\Sachin\Final\PPO\New folder1\2023-02-06_best_agents\PPO"
    entries = os.listdir(checkpoint_path)
    f = []
    for entry in entries:
        if entry[:3] == "PPO":
            sub_entries = os.listdir(checkpoint_path + "/" + entry)
            for sub_entry in sub_entries:
                if sub_entry[:10] == "checkpoint":
                    checkpoint = os.listdir(checkpoint_path + "/" + entry + "/" + sub_entry)
                    for checkpoint_entry in checkpoint:
                        if checkpoint_entry[-3:] == "pkl":
                            f.append(checkpoint_path + "/" + entry + "/" + sub_entry + "/" + checkpoint_entry)
    # f = [r"E:\Sachin\agents_runs\agents_runs\PPO\2023-01-28_best_agents\PPO\PPO_Dis_jsp_6x6_31806_00004_4_reward_version=A,vf_loss_coeff=0.0000_2023-01-29_03-29-44\checkpoint_000050\algorithm_state.pkl"]
    for no, path in enumerate(f):
        if algo == 'PPO':
            agent = ppo.PPO(config=algo_config, env=f'Dis_jsp_{args.instance_size}')
        elif algo == 'A3C':
            agent = a3c.A3C(config=algo_config, env=f'Dis_jsp_{args.instance_size}')
        else:
            agent = dqn.DQN(config=algo_config, env=f'Dis_jsp_{args.instance_size}')
        agent.restore(path)
        logger.info(f"Evaluating algo: {algo} with jsp size: {args.instance_size}")
        curr_episode = 0
        max_episode = 100

        time_begin = time.time()
        score_episode = []
        optimal_value = []
        makespan = []
        while curr_episode < max_episode:
            jsp = evaluate_instance(args.instance_size, curr_episode)
            # print(jsp[0].shape)
            # logger.info(f"Evaluating episode: {curr_episode}")
            # logger.info(f"Jsp problem {jsp[0]}, with optimal value \n {jsp[1]}")
            env_config = {
                "jsp": jsp,
                "reward_version": "A",
                "scaling_divisor": args.scaling_divisor,
                "scale_reward": args.scale_reward,
                "perform_left_shift_if_possible": args.left_shift,
                "normalize_observation_space": args.normalize_obs,
                "flat_observation_space": args.flat_obs,
                "action_mode": args.action_mode,
                "env_transform": args.masking,
                "verbose": args.env_verbose,
                "dtype": "float32",
            }
            if "reward_version=A" in path:
                env_config["reward_version"] = "A"
            elif "reward_version=B" in path:
                env_config["reward_version"] = "B"
            elif "reward_version=C" in path:
                env_config["reward_version"] = "C"
            elif "reward_version=D" in path:
                env_config["reward_version"] = "D"
            elif "reward_version=F" in path:
                env_config["reward_version"] = "F"
            else:
                env_config["reward_version"] = "E"
            env = DisjunctiveGraphJspEnv(env_config=env_config)
            obs = env.reset(jsp)
            done = False
            score = 0
            step = 0
            while not done:
                score_episode.append(score)
                action = agent.compute_single_action(obs)
                obs, reward, done, info = env.step(action)
                score += reward
                step += 1
                if len(info) > 0 and done:
                    # logger.info(f"Details: {info}")
                    makespan.append(info["makespan"])
                    optimal_value.append(info["optimal_value"])

            # env.render(mode="human", show=["gantt_window", "gantt_console", "graph_window", "graph_console"])
            curr_episode += 1

        makespan_avg = sum(makespan)/max_episode
        optimal_value_avg = sum(optimal_value)/max_episode
        error = (makespan_avg - optimal_value_avg)/optimal_value_avg*100
        print(f"checkpoint-no: {path[-24:-20]}")
        print("makespan_avg: ", makespan_avg)
        print("optimal_value_avg: ", optimal_value_avg)
        print("Gap %: ", error)

        # plt.scatter(optimal_value.keys(), makespan, marker="o")
        # plt.scatter(optimal_value.keys(), optimal_value.values(), marker="^")
        # plt.savefig(f"{plots_save_path}/makespan_{args.instance_size}")
        # time.sleep(100)


class JspEnv_v2(gym.Env, ABC):
    def __init__(self, env_config):
        self.env_config = env_config
        self.env = DisjunctiveGraphJspEnv(env_config)
        self.name = "DisjunctiveGraphJspEnv"
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        jsp = self.env_config["jsp"]
        return self.env.reset(jsp)

    def step(self, action):

        return self.env.step(action)

    def render(self, mode, show):

        return self.env.render(mode=mode, show=show)

if __name__ == "__main__":
    args = parser.parse_args()
    logger, timestamp = configure_logger()
    print(f"Running with following CLI options: {args}")

    if args.instance_size != "any":
        if int(args.instance_size[0]) == 1:
            m = int(args.instance_size[:2])
            n = int(args.instance_size[3:])
        else:
            m = int(args.instance_size[0])
            n = int(args.instance_size[-1])
        if args.action_mode == 'task':
            action_space = m * n
        else:
            action_space = m

        if args.normalize_obs:
            observation_space_shape = (m * n,
                                       (m * n) + n + 1)
        else:
            observation_space_shape = (m * n, (m * n) + 2)

        if args.flat_obs:
            a, b = observation_space_shape
            observation_space_shape = (a * b,)
    else:
        observation_space_shape = (100,)
        action_space = 10
        raise ValueError()

    ray.init(local_mode=args.local_mode, object_store_memory=100000000)
    register_env(f'Dis_jsp_{args.instance_size}', lambda c: JspEnv_v2(c))

    if args.masking:
        if m == n == 3 and args.action_mode == "job":
            if args.action_mode == "job":
                ModelCatalog.register_custom_model(f'Dis_jsp_{args.instance_size}', TorchParametricActionsModelv1)
            else:
                ModelCatalog.register_custom_model(f'Dis_jsp_{args.instance_size}', TorchParametricActionsModelv2)
        elif m == n == 6:
            if args.action_mode == "job":
                ModelCatalog.register_custom_model(f'Dis_jsp_{args.instance_size}', TorchParametricActionsModelv3)
            else:
                ModelCatalog.register_custom_model(f'Dis_jsp_{args.instance_size}', TorchParametricActionsModelv4)
        elif m == n == 10:
            if args.action_mode == "job":
                ModelCatalog.register_custom_model(f'Dis_jsp_{args.instance_size}', TorchParametricActionsModelv5)
            else:
                ModelCatalog.register_custom_model(f'Dis_jsp_{args.instance_size}', TorchParametricActionsModelv6)
        else:
            if args.action_mode == "job":
                ModelCatalog.register_custom_model(f'Dis_jsp_{args.instance_size}', TorchParametricActionsModelv7)
            else:
                ModelCatalog.register_custom_model(f'Dis_jsp_{args.instance_size}', TorchParametricActionsModelv8)
    else:
        ModelCatalog.register_custom_model(f'Dis_jsp_{args.instance_size}', TorchParametricActionModel)

    if args.algo == 'DQN':
        cfg = {
            "hidden": [],
            "dueling": False
        }
    else:
        cfg = {}

    if args.algo == 'PPO' or args.algo == 'A3C':
        config = dict({
            "env": f'Dis_jsp_{args.instance_size}',
            "model": {
                "custom_model": f'Dis_jsp_{args.instance_size}',
                "vf_share_layers": True
            },
            "env_config": {
                "jsp": evaluate_instance(args.instance_size, 0),
                "reward_version": "A",
                "scaling_divisor": args.scaling_divisor,
                "scale_reward": args.scale_reward,
                "perform_left_shift_if_possible": args.left_shift,
                "normalize_observation_space": args.normalize_obs,
                "flat_observation_space": args.flat_obs,
                "action_mode": args.action_mode,
                "env_transform": args.masking,
                "verbose": args.env_verbose,
                "dtype": "float32",
            },
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": args.no_of_workers,  # parallelism
            "framework": 'torch',
            "rollout_fragment_length": 125,
            "train_batch_size": 4000,
            # "sgd_minibatch_size": 512,
            # "num_sgd_iter": 20,
            "vf_loss_coeff": 0.0005,
            # "vf_loss_coeff": tune.grid_search([0.0005, 0.0009]),
            # "vf_clip_param": 10,
            # "lr": tune.grid_search([0.001, 0.0001])
            "lr": 0.0001,
            # "callbacks": MyCallbacks,
            # "optimizer": "SGD",
            # "entropy_coeff": tune.grid_search([tune.uniform(0.0001, 0.001), tune.uniform(0.0001, 0.001),
            #                                    tune.uniform(0.0001, 0.001), tune.uniform(0.0001, 0.001),
            #                                    tune.uniform(0.0001, 0.001)]),
            # "num_envs_per_worker": 4,
            # "horizon": 32,
            # "timesteps_per_batch": 2048,
        },
            **cfg)
        if args.algo == 'PPO':
            algo_config = ppo.DEFAULT_CONFIG.copy()
        elif args.algo == 'A3C':
            algo_config = a3c.DEFAULT_CONFIG.copy()
        else:
            algo_config = dqn.DEFAULT_CONFIG.copy()
        algo_config.update(config)
        algo_config['model']['fcnet_activation'] = 'relu'
        if args.lstm:
            algo_config['model']['use_lstm'] = True
            algo_config['model']['lstm_cell_size'] = 64
        algo_config['evaluation_interval'] = args.eval_interval
        # algo_config['evaluation_duration'] = 10
        algo_config["evaluation_parallel_to_training"]: True
    else:
        algo_config = None

    stop = {
        "training_iteration": 500
        # "episode_reward_mean": 30 - (40 * args.no_of_jobs * 0.002),
    }
    plots_save_path, agent_save_path, best_agent_save_path = setup(args.algo, timestamp)

    # automated run with tune and grid search and Tensorboard
    print("Running manual train loop without Ray Tune")

    evaluate(args.algo, algo_config, plots_save_path)

    ray.shutdown()

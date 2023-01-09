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
from jsp_env.src.graph_jsp_env.disjunctive_graph_logger import log
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


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
    "--run-type",
    default="train",
    type=str,
    choices=["train", "test", "trial", "evaluate"],
    help="Evaluation while training TRUE/FALSE"
)
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
    default=40,
    type=int,
    help="scaling the reward")
parser.add_argument(
    "--no-of-workers",
    default=0,
    type=int,
    help="scaling the reward")
parser.add_argument(
    "--scale-reward",
    default=True,
    type=bool,
    help="Use reward scaling or not")


def setup(algo, timestamp):
    Path(f'./plots/{algo}').mkdir(parents=True, exist_ok=True)
    plots_save_path = './plots/' + algo + timestamp
    Path(f'./agents_runs/{algo}//{timestamp}').mkdir(parents=True, exist_ok=True)
    agent_save_path = './agents_runs/' + algo + '/' + timestamp
    best_agent_save_path = './agents_runs/' + algo + '/' + timestamp \
                           + '_best_agents'
    # best_agent_save_path = './agents_runs/ConveyorEnv_token_n/PPO_best_agents'
    Path(best_agent_save_path).mkdir(parents=True, exist_ok=True)

    return plots_save_path, agent_save_path, best_agent_save_path


def instance_creator(size):
    if args.run_type == "train":
        if size != "any":
            with open(f"./data/{size}.json") as f:
                data = json.load(f)
        else:
            size = np.random.choice(["3x3", "6x6", "8x8", "10x10", "15x15"])
            with open(f"./data/{size}.json") as f:
                data = json.load(f)

        m = int(size[0])
        if m not in [3, 6, 8]:
            if m == 1:
                m = int(size[:2])
        instance_no = str(np.random.randint(len(data["jssp_identification"])))
        name = data["jssp_identification"][instance_no][:-5]
        opt_value = data["optimal_time"][instance_no]
        jsp_data = data["jobs_data"][instance_no]
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
        machine = np.array(machine).reshape(m, m)
        duration = np.array(duration).reshape(m, m)
        jsp = np.concatenate((machine, duration), axis=0).reshape(2, m, m)
    else:
        jsp = np.array([
            [
                [1, 2, 0],  # job 0
                [0, 2, 1]  # job 1
            ],
            [
                [17, 12, 19],  # task durations of job 0
                [8, 6, 2]  # task durations of job 1
            ]
        ])
        name = "Trail"
        opt_value = 48

    return jsp, name, opt_value


class JspEnv_v1(gym.Env, ABC):
    def __init__(self, env_config):
        self.jps_instance = env_config['jps_instance']
        self.scaling_divisor = env_config['scaling_divisor']
        self.scale_reward = env_config['scale_reward']
        self.normalize_observation_space = env_config['normalize_observation_space']
        self.flat_observation_space = env_config['flat_observation_space']
        self.action_mode = env_config['action_mode']
        self.perform_left_shift_if_possible = env_config['perform_left_shift_if_possible']
        self.verbose = env_config['verbose']
        self.env_transform = env_config["env_transform"]

        self.env = DisjunctiveGraphJspEnv(jps_instance=self.jps_instance[0],
                                          scaling_divisor=self.scaling_divisor,
                                          scale_reward=self.scale_reward,
                                          perform_left_shift_if_possible=self.perform_left_shift_if_possible,
                                          normalize_observation_space=self.normalize_observation_space,
                                          flat_observation_space=self.flat_observation_space,
                                          action_mode=self.action_mode,
                                          env_transform=self.env_transform,
                                          verbose=self.verbose)
        self.name = "DisjunctiveGraphJspEnv"
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


def evaluate(algo, algo_config: dir, plots_save_path):
    f = [r"E:\Sachin\agents_runs\PPO\2023-01-08_best_agents\PPO\PPO_Dis_jsp_6x6_ea232_00000_0_2023-01-08_18-04-37\checkpoint_000500"]
    for no, path in enumerate(f):
        # if algo == 'PPO':
        #     agent = ppo.PPO(config=algo_config, env=f'Dis_jsp_{args.instance_size}')
        # elif algo == 'A3C':
        #     agent = a3c.A3C(config=algo_config, env=f'Dis_jsp_{args.instance_size}')
        # else:
        #     agent = dqn.DQN(config=algo_config, env=f'Dis_jsp_{args.instance_size}')
        agent = Algorithm.from_checkpoint(checkpoint=path)

    logger.info(f"Evaluating algo: {algo} with jsp size: {args.instance_size}")
    curr_episode = 1
    max_episode = 10

    time_begin = time.time()
    score_episode = []
    while curr_episode <= max_episode:
        jsp, name, opt_value = instance_creator(args.instance_size)
        logger.info(f"Evaluating episode: {curr_episode}")
        logger.info(f"Jsp problem {name}, with optimal value {opt_value}: \n {jsp}")
        env_config = {
            "jps_instance": jsp,
            "scaling_divisor": args.scaling_divisor,
            "scale_reward": args.scale_reward,
            "perform_left_shift_if_possible": args.left_shift,
            "normalize_observation_space": args.normalize_obs,
            "flat_observation_space": args.flat_obs,
            "action_mode": args.action_mode,
            "env_transform": args.masking,
            "verbose": args.env_verbose,
        }
        env = JspEnv_v1(env_config)
        obs = env.reset()
        done = False
        score = 0
        step = 0
        while not done:
            score_episode.append(score)
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            score += reward
            step += 1
            if len(info) > 0:
                logger.info(f"Details: {info}")


if __name__ == "__main__":
    args = parser.parse_args()
    logger, timestamp = configure_logger()
    print(f"Running with following CLI options: {args}")
    jsp = np.array([
        [
            [1, 2, 0],  # job 0
            [0, 2, 1]  # job 1
        ],
        [
            [17, 12, 19],  # task durations of job 0
            [8, 6, 2]  # task durations of job 1
        ]
    ])

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
    register_env(f'Dis_jsp_{args.instance_size}', lambda c: JspEnv_v1(
        {
            "jps_instance": instance_creator(args.instance_size),
            "scaling_divisor": args.scaling_divisor,
            "scale_reward": args.scale_reward,
            "perform_left_shift_if_possible": args.left_shift,
            "normalize_observation_space": args.normalize_obs,
            "flat_observation_space": args.flat_obs,
            "action_mode": args.action_mode,
            "env_transform": args.masking,
            "verbose": args.env_verbose,
        }
    ))
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
                "jps_instance": instance_creator(args.instance_size),
                "scaling_divisor": args.scaling_divisor,
                "scale_reward": args.scale_reward,
                "perform_left_shift_if_possible": args.left_shift,
                "normalize_observation_space": args.normalize_obs,
                "flat_observation_space": args.flat_obs,
                "action_mode": args.action_mode,
                "env_transform": args.masking,
                "verbose": args.env_verbose,
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

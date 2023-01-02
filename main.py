import argparse
import platform
import sys
import logging

import gym

from util import TorchParametricActionModel, TorchParametricActionsModelv1
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

import numpy as np
import pandas as pd

from pathlib import Path
import os
import matplotlib.pyplot as plt
import time
import random

import ray
from ray import tune
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms import a3c
from ray.rllib.algorithms import dqn
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv
from graph_jsp_env.disjunctive_graph_logger import log

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


BASE_PATH = '.'
RESULTS_PATH = './results/'
REWARD_RESULTS_PATH = '/reward-results/'
AVG_OVR_EP_PATH = '/avg_over_ep-results/'
CHECKPOINT_ROOT = './checkpoints'


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
    "--stop-iters",
    type=int,
    default=5000,
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
    default=100,
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
    default=False,
    type=bool,
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

    _, m, n = jsp.shape

    if args.action_mode == 'task':
        action_space = m*n
    else:
        action_space = m

    if args.normalize_obs:
        observation_space_shape = (m*n,
                                   (m * n) + n + 1)
    else:
        observation_space_shape = (m * n, (m * n) + 2)

    if args.flat_obs:
        a, b = observation_space_shape
        observation_space_shape = (a * b,)

    ray.init(local_mode=args.local_mode, object_store_memory=1000000000)
    register_env(f'Dis_jsp_{args.file_no}', lambda _: DisjunctiveGraphJspEnv(
                                                        jps_instance=jsp,
                                                        perform_left_shift_if_possible=args.left_shift,
                                                        normalize_observation_space=args.normalize_obs,
                                                        flat_observation_space=args.flat_obs,
                                                        action_mode=args.action_mode,
                                                        dtype='float32',
                                                        verbose=args.env_verbose
    ))
    ModelCatalog.register_custom_model("Torch_model", TorchParametricActionModel)

    if args.algo == 'DQN':
        cfg = {
            "hidden": [],
            "dueling": False
        }
    else:
        cfg = {}

    if args.algo == 'PPO' or args.algo == 'A3C':
        config = dict({
            "env": f'Dis_jsp_{args.file_no}',
            "model": {
                "custom_model": "Torch_model",
                "vf_share_layers": True
            },
            # Env config comes here !!!!
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": 1,  # parallelism
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
        algo_config['evaluation_interval'] = 100
        # algo_config['evaluation_duration'] = 10
        algo_config["evaluation_parallel_to_training"]: True
    else:
        algo_config = None

    stop = {
        "training_iteration": 100
        # "episode_reward_mean": 30 - (40 * args.no_of_jobs * 0.002),
    }
    plots_save_path, agent_save_path, best_agent_save_path = setup(args.algo, timestamp)

    # automated run with tune and grid search and Tensorboard
    print("Training with Ray Tune.")
    print('...............................................................................\n'
          '\n\n\t\t\t\t\t\t\t\t Training Starts Here\n\n\n......................................')
    # result = tune.run(curriculum_learning, config=algo_config, local_dir=best_agent_save_path, log_to_file=True,
    #                   checkpoint_at_end=True, checkpoint_freq=50, reuse_actors=False, verbose=3,
    #                   checkpoint_score_attr='min-episode_len_mean',
    #                   resources_per_trial=ppo.PPOTrainer.default_resource_request(algo_config))
    # , resources_per_trial = ppo.PPOTrainer.default_resource_request(algo_config)
    result = tune.run(args.algo, config=algo_config, local_dir=best_agent_save_path, log_to_file=True,
                      checkpoint_at_end=True, checkpoint_freq=50, reuse_actors=False, verbose=3,
                      checkpoint_score_attr='min-episode_len_mean', stop=stop)
    logger.info(result)
    print('...............................................................................\n'
          '\n\n\t\t\t\t\t\t\t\t Training Ends Here\n\n\n........................................')

    ray.shutdown()








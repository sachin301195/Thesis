import argparse
import platform
import sys

import gym

from util import TorchParametricModel, TorchParametricModelv1
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
from ray.rllib.agents import ppo
from ray.rllib.agents import a3c
from ray.rllib.agents import dqn
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env






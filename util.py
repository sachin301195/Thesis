from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
from gym.spaces import Dict, Discrete, Box, Tuple

torch, nn = try_import_torch()


class TorchParametricActionModel(DQNTorchModel):
    """
    : This network to be used without action_masking
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kwargs):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)

        self.action_model = TorchFC(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict,
                state,
                seq_lens):
        input_dict['obs'] = input_dict['obs'].float()
        fc_out, _ = self.action_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return self.action_model.value_function()


class TorchParametricActionsModelv1(DQNTorchModel):
    """
    : This network to be used with action_masking and trial env version
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(99,),
                 action_embed_size=3,
                 **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)

        self.action_model = TorchFC(
            obs_space=Box(0, 1, shape=true_obs_shape),  # oder Box(0, 1, ...) wie im medium Artikel
            action_space=action_space,
            num_outputs=action_embed_size,
            model_config=model_config,
            name=name + "_action_embed")

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        # avail_actions = input_dict["obs"]["avail_action"]
        action_mask = input_dict["obs"]["action_mask"]
        # print('action_mask', action_mask)

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        # Compute the predicted action embedding
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["state"]})

        # state is empty
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()


class TorchParametricActionsModelv2(DQNTorchModel):
    """
    : This network to be used with action_masking and trial env version
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(1368,),
                 action_embed_size=6,
                 **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)

        self.action_model = TorchFC(
            obs_space=Box(0, 1, shape=true_obs_shape),  # oder Box(0, 1, ...) wie im medium Artikel
            action_space=action_space,
            num_outputs=action_embed_size,
            model_config=model_config,
            name=name + "_action_embed")

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        # avail_actions = input_dict["obs"]["avail_action"]
        action_mask = input_dict["obs"]["action_mask"]
        # print('action_mask', action_mask)

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        # Compute the predicted action embedding
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["state"]})

        # state is empty
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()


class TorchParametricActionsModelv3(DQNTorchModel):
    """
    : This network to be used with action_masking and trial env version
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(10200, ),
                 action_embed_siz =10,
                 **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)

        self.action_model = TorchFC(
            obs_space=Box(0, 1, shape=true_obs_shape),  # oder Box(0, 1, ...) wie im medium Artikel
            action_space=action_space,
            num_outputs=action_embed_size,
            model_config=model_config,
            name=name + "_action_embed")

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        # avail_actions = input_dict["obs"]["avail_action"]
        action_mask = input_dict["obs"]["action_mask"]
        # print('action_mask', action_mask)

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        # Compute the predicted action embedding
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["state"]})

        # state is empty
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()


class TorchParametricActionsModelv4(DQNTorchModel):
    """
    : This network to be used with action_masking and trial env version
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(99,),
                 action_embed_size=9,
                 **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)

        self.action_model = TorchFC(
            obs_space=Box(0, 1, shape=true_obs_shape),  # oder Box(0, 1, ...) wie im medium Artikel
            action_space=action_space,
            num_outputs=action_embed_size,
            model_config=model_config,
            name=name + "_action_embed")

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        # avail_actions = input_dict["obs"]["avail_action"]
        action_mask = input_dict["obs"]["action_mask"]
        # print('action_mask', action_mask)

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        # Compute the predicted action embedding
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["state"]})

        # state is empty
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()

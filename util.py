from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.algorithms.alpha_zero.models.custom_torch_models import ActorCriticModel, DenseModel
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
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
                 true_obs_shape=(126,),
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
        # print("I reached here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(input_dict)
        # print(state)
        action_mask = input_dict["obs"]["action_mask"]
        # print("I reached here11111111111111111111111111111111111111111111111111111111111111111111111111111111111")
        # print(action_mask)
        # print('action_mask', action_mask)

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        # print("I reached here222222222222222222222222222222222222222222222222222222222222222222222222222222222222")
        # print(inf_mask)
        # print("wait")

        # Compute the predicted action embedding
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["obs"]})
        # print(action_embed)
        # print("Well I reached here as well........................................................................")

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
                 true_obs_shape=(126,),
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
        # print("I reached here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(input_dict)
        # print(state)
        action_mask = input_dict["obs"]["action_mask"]
        # print("I reached here11111111111111111111111111111111111111111111111111111111111111111111111111111111111")
        # print(action_mask)
        # print('action_mask', action_mask)

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        # print("I reached here222222222222222222222222222222222222222222222222222222222222222222222222222222222222")
        # print(inf_mask)
        # print("wait")

        # Compute the predicted action embedding
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["obs"]})
        # print(action_embed)
        # print("Well I reached here as well........................................................................")

        # state is empty
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()


class TorchParametricActionsModelv3(DQNTorchModel):
    """
    : This network is used for action mode JOB and size 6x6
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(1476,),
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
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["obs"]})

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
                 true_obs_shape=(1476,),
                 action_embed_size=36,
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
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["obs"]})

        # state is empty
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()


class TorchParametricActionsModelv5(DQNTorchModel):
    """
    : This network to be used with action_masking and trial env version
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(10500,),
                 action_embed_size=10,
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
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["obs"]})

        # state is empty
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()


class TorchParametricActionsModelv6(DQNTorchModel):
    """
    : This network to be used with action_masking and trial env version
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(10500,),
                 action_embed_size=100,
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
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["obs"]})

        # state is empty
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()


class TorchParametricActionsModelv7(DQNTorchModel):
    """
    : This network to be used with action_masking and trial env version
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(51750,),
                 action_embed_size=15,
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
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["obs"]})

        # state is empty
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()


class TorchParametricActionsModelv8(DQNTorchModel):
    """
    : This network to be used with action_masking and trial env version
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(51750,),
                 action_embed_size=225,
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
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["obs"]})

        # state is empty
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()



class TorchParametricActionsModelv9(DQNTorchModel):
    """
    : This network to be used with action_masking and trial env version
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(162000,),
                 action_embed_size=20,
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
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["obs"]})

        # state is empty
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()


class TorchParametricActionsModelv11(DQNTorchModel):
    """
    : This network to be used with action_masking and trial env version
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(162000,),
                 action_embed_size=400,
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
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["obs"]})

        # state is empty
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()


class TorchParametricActionsModelv12(DenseModel):
    pass


class TorchParametricActionsModelv13(SACTorchModel):
    pass

class TorchParametricActionsModelv14(DQNTorchModel):
    """
    : This network to be used with action_masking and trial env version
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(4352,),
                 action_embed_size=64,
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
        action_embed, _ = self.action_model({"obs": input_dict["obs"]["obs"]})

        # state is empty
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()
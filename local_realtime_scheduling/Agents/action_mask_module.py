import gymnasium as gym
from typing import Dict, Optional, Tuple, Union

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

from configs import dfjspt_params

torch, nn = try_import_torch()


class ActionMaskingRLModule(RLModule):
    """An RLModule that implements an action masking for safe RL.

    This RLModule implements action masking to avoid unsafe/unwanted actions
    dependent on the current state (observations). It does so by using an
    environment generated action mask defining which actions are allowed and
    which should be avoided. The action mask is extracted from the
    environment's `gymnasium.spaces.dict.Dict` observation and applied after
    the module's `forward`-pass to the action logits. The resulting action
    logits prevent unsafe/unwanted actions to be sampled from the corresponding
    action distribution.

    Note, this RLModule is implemented for the `PPO` algorithm only. It is not
    guaranteed to work with other algorithms. Furthermore, not that for this
    module to work it requires an environment with a `gymnasium.spaces.dict.Dict`
    observation space containing tow key, `"action_mask"` and `"observations"`.
    """

    @override(RLModule)
    # def __init__(self, config: RLModuleConfig):
    def __init__(
        self,
        config=-1,
        *,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        inference_only: Optional[bool] = None,
        learner_only: bool = False,
        model_config: Optional[Union[dict, DefaultModelConfig]] = None,
        catalog_class=None,
    ):
        # PPOTorchRLModule.__init__(self, config)
        # super().__init__(config)
        # If observation space is not of type `Dict` raise an error.
        # if not isinstance(config.observation_space, gym.spaces.dict.Dict):
        if not isinstance(observation_space, gym.spaces.dict.Dict):
            raise ValueError(
                "This RLModule requires the environment to provide a "
                "`gym.spaces.Dict` observation space of the form: \n"
                " {'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                "  'observation_space': self.observation_space}"
            )

        # While the environment holds an observation space that contains, both,
        # the action mask and the original observation space, the 'RLModule'
        # receives only the `"observation"` element of the space, but not the
        # action mask.
        self.observation_space_with_mask = observation_space
        observation_space = observation_space["observation"]

        # Keeps track if observation specs have been checked already.
        self._checked_observations = False

        # The PPORLModule, in its constructor will build networks for the original
        # observation space (i.e. without the action mask).
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
        )


class ActionMaskingTorchRLModule(ActionMaskingRLModule, PPOTorchRLModule):
    @override(PPOTorchRLModule)
    # def __init__(self, config: RLModuleConfig):
    #     ActionMaskingRLModule.__init__(self, config)
    #     PPOTorchRLModule.__init__(self, config)

    def setup(self):
        super().setup()
        # We need to reset here the observation space such that the
        # super`s (`PPOTorchRLModule`) observation space is the
        # original space (i.e. without the action mask) and `self`'s
        # observation space contains the action mask.
        self.observation_space = self.observation_space_with_mask

    # @override(PPOTorchRLModule)
    # def _forward(
    #         self, batch: Dict[str, TensorType], **kwargs
    # ) -> Dict[str, TensorType]:
    #     # Preprocess the original batch to extract the action mask.
    #     action_mask, batch = self._preprocess_batch(batch)
    #     # Run the forward pass.
    #     outs = super()._forward(batch, **kwargs)
    #     # outs = super()._forward({Columns.OBS: batch[Columns.OBS]}, **kwargs)
    #     # Mask the action logits and return.
    #     # return {Columns.ACTION_DIST_INPUTS: self._mask_action_logits(outs, batch['action_mask'])}
    #     return self._mask_action_logits(outs, action_mask)

    @override(PPOTorchRLModule)
    def _forward_inference(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        # Preprocess the original batch to extract the action mask.
        action_mask, batch = self._preprocess_batch(batch)
        # Run the forward pass.
        outs = super()._forward_inference(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, action_mask)

    @override(PPOTorchRLModule)
    def _forward_exploration(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        # Preprocess the original batch to extract the action mask.
        action_mask, batch = self._preprocess_batch(batch)
        # Run the forward pass.
        outs = super()._forward_exploration(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, action_mask)

    @override(PPOTorchRLModule)
    def _forward_train(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        # Run the forward pass.
        outs = super()._forward_train(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, batch["action_mask"])
        # return {Columns.ACTION_DIST_INPUTS: self._mask_action_logits(outs, batch['action_mask'])}

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, TensorType], embeddings=None):
        # Preprocess the batch to extract the `observations` to `Columns.OBS`.
        action_mask, batch = self._preprocess_batch(batch)
        # NOTE: Because we manipulate the batch we need to add the `action_mask`
        # to the batch to access them in `_forward_train`.
        batch["action_mask"] = action_mask
        # Call the super's method to compute values for GAE.
        return super().compute_values(batch, embeddings)

    def _preprocess_batch(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        """Extracts observations and action mask from the batch

        Args:
            batch: A dictionary containing tensors (at least `Columns.OBS`)

        Returns:
            A tuple with the action mask tensor and the modified batch containing
                the original observations.
        """
        # Check observation specs for action mask and observation keys.
        self._check_batch(batch)

        # # 检查 batch['obs'] 的类型是否为字典
        # if not isinstance(batch[Columns.OBS], dict):
        #     raise ValueError(
        #         f"Expected batch['{Columns.OBS}'] to be a dictionary, "
        #         f"but got {type(batch[Columns.OBS])} instead. \n"
        #         f"batch['{Columns.OBS}']: \n {batch[Columns.OBS]}"
        #     )

        if type(batch[Columns.OBS]) is dict:
            # Extract the available actions tensor from the observation.
            action_mask = batch[Columns.OBS].pop("action_mask")

            # Modify the batch for the `PPORLModule`'s `forward` method, i.e.
            # pass only `"obs"` into the `forward` method.
            # batch[Columns.OBS] = batch[Columns.OBS].pop("observation")

            # Extract the observation sub-dictionary
            observation = batch[Columns.OBS].pop("observation")

            # Prepare a list to hold all flattened components
            flattened_obs_parts = []

            # Iterate through each key-value pair in the observation dict
            for key, value in observation.items():
                # If value is a tensor, flatten it
                if isinstance(value, torch.Tensor):
                    if dfjspt_params.use_lstm:
                        flattened_obs_parts.append(value.flatten(start_dim=2))
                    else:
                        flattened_obs_parts.append(value.flatten(start_dim=1))
                else:
                    # Handle other cases, such as if value is a different type
                    raise ValueError(f"Unexpected value type in observation: {type(value)} for key {key}")

            # Concatenate all flattened components into a single tensor
            flattened_obs = torch.cat(flattened_obs_parts, dim=-1)

            # Put the concatenated tensor back into the batch
            batch[Columns.OBS] = flattened_obs
            batch['action_mask'] = action_mask

        else:
            action_mask = batch['action_mask']

        # Return the extracted action mask and the modified batch.
        return action_mask, batch

    def _mask_action_logits(
        self, batch: Dict[str, TensorType], action_mask: TensorType
    ) -> Dict[str, TensorType]:
        """Masks the action logits for the output of `forward` methods

        Args:
            batch: A dictionary containing tensors (at least action logits).
            action_mask: A tensor containing the action mask for the current
                observations.

        Returns:
            A modified batch with masked action logits for the action distribution
            inputs.
        """
        # Convert action mask into an `[0.0][-inf]`-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)

        # Mask the logits.
        batch[Columns.ACTION_DIST_INPUTS] += inf_mask

        # Return the batch with the masked action logits.
        return batch

    def _check_batch(self, batch: Dict[str, TensorType]) -> Optional[ValueError]:
        """Assert that the batch includes action mask and observations.

        Args:
            batch: A dicitonary containing tensors (at least `Columns.OBS`) to be
                checked.

        Raises:
            `ValueError` if the column `Columns.OBS`  does not contain observations
                and action mask.
        """
        if not self._checked_observations:
            if "action_mask" not in batch[Columns.OBS]:
                raise ValueError(
                    "No action mask found in observation. This `RLModule` requires "
                    "the environment to provide observations that include an "
                    "action mask (i.e. an observation space of the Dict space "
                    "type that looks as follows: \n"
                    "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                    "'observations': self.observation_space}"
                )
            if "observation" not in batch[Columns.OBS]:
                raise ValueError(
                    "No observations found in observation. This 'RLModule` requires "
                    "the environment to provide observations that include the original "
                    "observations under a key `'observations'` in a dict (i.e. an "
                    "observation space of the Dict space type that looks as follows: \n"
                    "{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),"
                    "'observations': <observation_space>}"
                )
            self._checked_observations = True


## Changes to source code:

### 1. 
Add the following code into `/python3.10/site-packages/ray/rllib/core/models/catalog.py`:

1) Before Class `catalog`:

```
from gymnasium import spaces
```

```
class NestedModelConfig(ModelConfig):
    """Custom ModelConfig for handling nested observation spaces (Dict)."""

    def __init__(self, observation_space: spaces.Space, model_config_dict: dict):

        if not isinstance(observation_space, spaces.Dict):
            raise ValueError(f"Expected a Dict observation space, but got {type(observation_space)}")

        self.input_dims = self.flatten_and_concat(observation_space)

        self.mlp_config = MLPEncoderConfig(
            input_dims=(self.input_dims,),  
            hidden_layer_dims=model_config_dict.get("fcnet_hiddens", [256, 256]),
            hidden_layer_use_bias=model_config_dict.get("fcnet_use_bias", True),
            hidden_layer_activation=model_config_dict.get("fcnet_activation", "relu")
        )

    def flatten_and_concat(self, observation_space: spaces.Dict) -> int:
        total_dims = 0
        for key, space in observation_space.spaces.items():
            if isinstance(space, spaces.Box):
                total_dims += np.prod(space.shape)
            else:
                raise ValueError(f"Unsupported space type {type(space)} for key {key}")
        return total_dims

    def build(self, framework: str = "torch"):

        return self.mlp_config.build(framework=framework)

    @property
    def output_dims(self):

        return self.mlp_config.output_dims
```

2) in `_get_encoder_config()`:

```
    # NestedModelConfig
    encoder_config = NestedModelConfig(
        observation_space=observation_space,
        model_config_dict=model_config_dict
    )
    # raise ValueError(
    #     f"No default encoder config for obs space={observation_space},"
    #     f" lstm={use_lstm} found."
    # )
```


### 2.

Change method `tokenize` in `/python3.10/site-packages/ray/rllib/core/models/base.py`:

```
@DeveloperAPI
def tokenize(tokenizer: Encoder, inputs: dict, framework: str) -> dict:
    """Tokenizes the observations from the input dict.

    Args:
        tokenizer: The tokenizer to use.
        inputs: The input dict.

    Returns:
        The output dict.
    """
    # Tokenizer may depend solely on observations.
    obs = inputs[Columns.OBS]
    tokenizer_inputs = {Columns.OBS: obs}
    size = list(obs.size() if framework == "torch" else obs.shape)
    b_dim, t_dim = size[:2]
    fold, unfold = get_fold_unfold_fns(b_dim, t_dim, framework=framework)
    # Push through the tokenizer encoder.
    fold_input = fold(tokenizer_inputs)
    out = tokenizer(fold_input)
    out = out[ENCODER_OUT]
    # Then unfold batch- and time-dimensions again.
    unfold_output = unfold(out)
    return unfold_output
    # return unfold(out)
```

Change class `TorchLSTMEncoder` in `/python3.10/site-packages/ray/rllib/core/models/torch/encoder.py`:

```
    def __init__(self, config: RecurrentEncoderConfig) -> None:
        TorchModel.__init__(self, config)

        # Maybe create a tokenizer
        # if config.tokenizer_config is not None:
        #     self.tokenizer = config.tokenizer_config.build(framework="torch")
        #     lstm_input_dims = config.tokenizer_config.output_dims
        # else:
        #     self.tokenizer = None
        #     lstm_input_dims = config.input_dims
        self.tokenizer = None
        lstm_input_dims = (config.tokenizer_config.input_dims,)
```










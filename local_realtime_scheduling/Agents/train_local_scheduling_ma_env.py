from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
import os
import pickle
from local_realtime_scheduling.Environment.LocalSchedulingMultiAgentEnv_v2 import LocalSchedulingMultiAgentEnv
from local_realtime_scheduling.Agents.args_parser import add_default_args
from local_realtime_scheduling.Agents.action_mask_module import ActionMaskingTorchRLModule
from local_realtime_scheduling.Agents.training_pipeline import train_with_tune_pipeline
from local_realtime_scheduling.Agents.customized_callback import MyCallbacks
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, Local_Job_schedule


parser = add_default_args(
    default_iters=500,
    default_reward=1000,
)


if __name__ == "__main__":
    from configs import dfjspt_params
    args = parser.parse_args()

    if args.algo != "PPO":
        raise ValueError("This example only supports PPO. Please use --algo=PPO.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_schedule_file = os.path.dirname(current_dir) + \
                          "/InterfaceWithGlobal/local_schedules/local_schedule_" + \
                          f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}" \
                          + f"I0_window{dfjspt_params.time_window_size}/window_{dfjspt_params.current_window}.pkl"

    with open(local_schedule_file,
              "rb") as file:
        local_schedule = pickle.load(file)

    # print(vars(local_schedule))

    env_config = {
        "n_machines": dfjspt_params.n_machines,
        "n_jobs": dfjspt_params.n_jobs,
        "n_transbots": dfjspt_params.n_transbots,
        "local_schedule": local_schedule
    }

    example_env = LocalSchedulingMultiAgentEnv(env_config)
    example_env.reset()

    train_batch_size = 4 * (example_env.num_machines + example_env.num_transbots) * int(example_env.time_upper_bound)

    model_config = {}
    if dfjspt_params.use_lstm:
        model_config = {
            "vf_share_layers": False,
            "use_lstm": True,
            "max_seq_len": (example_env.num_machines + example_env.num_transbots) * 5,
        }

    base_config = (
        PPOConfig()
        .environment(
            env=LocalSchedulingMultiAgentEnv,
            env_config=env_config,
            disable_env_checking=True,
        )
        .env_runners(
            num_env_runners=0,
            num_envs_per_env_runner=1,
            batch_mode="complete_episodes",
            rollout_fragment_length="auto",
            sample_timeout_s=600,
            observation_filter="MeanStdFilter",
        )
        .training(
            train_batch_size_per_learner=train_batch_size,
            minibatch_size=(example_env.num_machines + example_env.num_transbots) * int(example_env.time_upper_bound),
            entropy_coeff=[
                [0, 0.005],
                [train_batch_size * 10, 0.001],
                [train_batch_size * 50, 0],
            ],
            num_epochs=10,
            # lr=1e-5,
            lr=[
                [0, 1e-4],
                [train_batch_size * 5, 5e-5],
                [train_batch_size * 20, 1e-5],
            ],
        )
        .learners(
            num_learners=0,
            num_cpus_per_learner=1,
            num_gpus_per_learner=0,
        )
        .checkpointing(
            checkpoint_trainable_policies_only=True,
        )
        .rl_module(
            # We need to explicitly specify here RLModule to use and
            # the catalog needed to build it.
            # rl_module_spec=RLModuleSpec(
            #     module_class=ActionMaskingTorchRLModule,
            #     # model_config={
            #     #     "head_fcnet_hiddens": [64, 64],
            #     #     "head_fcnet_activation": "relu",
            #     # },
            # ),
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    # f"p_{agent_i}": RLModuleSpec(
                    #     module_class=ActionMaskingTorchRLModule,
                    #     # model_config={
                    #     #     # "head_fcnet_hiddens": [64, 64],
                    #     #     # "head_fcnet_activation": "relu",
                    #     # },
                    #     observation_space=example_env.observation_spaces[agent_i],
                    #     action_space=example_env.action_spaces[agent_i],
                    # ) for agent_i in example_env.agents
                    "p_machine": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        observation_space=example_env.observation_spaces[example_env.machine_agents[0]],
                        action_space=example_env.action_spaces[example_env.machine_agents[0]],
                        model_config=model_config,
                    ),
                    "p_transbot": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        observation_space=example_env.observation_spaces[example_env.transbot_agents[0]],
                        action_space=example_env.action_spaces[example_env.transbot_agents[0]],
                        model_config=model_config,
                    ),
                },
            ),
        )
        .multi_agent(
            # Use a simple set of policy IDs. Spaces for the individual policies
            # are inferred automatically using reverse lookup via the
            # `policy_mapping_fn` and the env provided spaces for the different
            # agents. Alternatively, you could use:
            # policies: {main0: PolicySpec(...), main1: PolicySpec}
            # policies={f"p_{agent_i}" for agent_i in example_env.agents},
            # policy_mapping_fn=lambda aid, *a, **kw: f"p_{aid}",
            policies={"p_machine", "p_transbot"},
            policy_mapping_fn=lambda agent_id, *a, **kw: "p_machine" if agent_id.startswith("machine") else "p_transbot",
        )
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_interval=10,
            # Run evaluation parallel to training to speed up the example.
            evaluation_parallel_to_training=False,
            # evaluation_config=PPOConfig.overrides(explore=False),
        )
        .callbacks(MyCallbacks)
    )

    # base_config["enable_rl_module_and_learner"] = False
    # base_config["enable_env_runner_and_connector_v2"] = False
    # base_config["num_env_runners"] = 0
    # base_config["inference_only"] = False

    args.no_tune = True

    print(vars(base_config))
    print(vars(args))
    print(f"num_learners = {base_config.num_learners}")
    print(f"train_batch_size_per_learner = {base_config.train_batch_size_per_learner}")
    print(f"total_train_batch_size = {base_config.total_train_batch_size}")
    print(f"sample_timeout_s = {base_config.sample_timeout_s}")

    # Run the example (with Tune).
    train_with_tune_pipeline(base_config, args)


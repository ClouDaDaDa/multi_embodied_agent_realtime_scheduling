# import json
import os
import pickle
import time
# import pandas as pd
import numpy as np
from pathlib import Path
import torch
from ray.rllib.utils.numpy import convert_to_numpy, softmax
from ray.rllib.utils.torch_utils import FLOAT_MIN

# from memory_profiler import profile
# @profile
def func(content: str):
    print(content)

try:
    import gymnasium as gym
    gymnasium = True
except Exception:
    import gym
    gymnasium = False
import ray
from ray.rllib.core.rl_module import RLModule
from local_realtime_scheduling.Agents.action_mask_module import ActionMaskingTorchRLModule
from local_realtime_scheduling.Environment.LocalSchedulingMultiAgentEnv_v2 import LocalSchedulingMultiAgentEnv
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, Local_Job_schedule
from configs import dfjspt_params


# Example usage
if __name__ == "__main__":
    ray.init(local_mode=False)

    time0 = time.time()

    # checkpoint_dir = '/home/rglab/ray_results/PPO_2025-03-04_02-09-14/PPO_LocalSchedulingMultiAgentEnv_9d698_00000_0_2025-03-04_02-09-14/checkpoint_000055'
    checkpoint_dir = '/home/rglab/ray_results/PPO_2025-03-03_22-05-14/PPO_LocalSchedulingMultiAgentEnv_87812_00000_0_2025-03-03_22-05-14/checkpoint_000003'

    machine_rl_module_checkpoint_dir = Path(checkpoint_dir) / "learner_group" / "learner" / "rl_module" / "p_machine"
    transbot_rl_module_checkpoint_dir = Path(checkpoint_dir) / "learner_group" / "learner" / "rl_module" / "p_transbot"

    # Now that you have the correct subdirectory, create the actual RLModule.
    machine_rl_module = RLModule.from_checkpoint(machine_rl_module_checkpoint_dir)
    transbot_rl_module = RLModule.from_checkpoint(transbot_rl_module_checkpoint_dir)

    time1 = time.time()
    print(f"Time for loading policies is {time1-time0}.")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    local_schedule_file = os.path.dirname(os.path.dirname(current_dir)) + \
                          "/local_realtime_scheduling/InterfaceWithGlobal/local_schedules/local_schedule_" + \
                          f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}" \
                          + f"I0_window{dfjspt_params.time_window_size}/window_{dfjspt_params.current_window}.pkl"

    result_file_name = os.path.dirname(os.path.dirname(current_dir)) + \
                       "/local_realtime_scheduling/local_results/local_result_" + \
                       f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}" \
                       + f"I0_window{dfjspt_params.time_window_size}/window_{dfjspt_params.current_window}.pkl"

    init_schedule_result_file_name = os.path.dirname(os.path.dirname(current_dir)) + \
                       "/local_results/local_result_" + \
                       f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}" \
                       + f"I0_window{dfjspt_params.time_window_size}/window_{dfjspt_params.current_window}_init_schedule.pkl"

    with open(local_schedule_file,
              "rb") as file:
        local_schedule = pickle.load(file)

    # print(vars(local_schedule))

    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_jobs": dfjspt_params.n_jobs,
        "n_transbots": dfjspt_params.n_transbots,
        "local_schedule": local_schedule,
        # "local_result_file": result_file_name,
        # "local_result_file": init_schedule_result_file_name,
        # "render_mode": "human",
    }

    scheduling_env = LocalSchedulingMultiAgentEnv(config)

    time2 = time.time()
    print(f"Time for initializing env is {time2 - time1}.")

    num_episodes = 100
    makespans = []
    times = []

    for episode in range(num_episodes):
        print(f"\nStarting episode {episode + 1}")

        begin_time = time.time()

        decision_count = 0
        observations, infos = scheduling_env.reset()
        # scheduling_env.render()
        # print(f"decision_count = {decision_count}")
        decision_count += 1
        done = {'__all__': False}
        truncated = {'__all__': False}
        total_rewards = {}
        for agent in scheduling_env.agents:
            total_rewards[agent] = 0.0

        while (not done['__all__']) and (not truncated['__all__']):
            actions = {}
            for agent_id, obs in observations.items():
                # print(f"current agent = {agent_id}")

                if agent_id.startswith("machine"):
                    input_dict = {
                        "obs": {
                            "action_mask": torch.tensor(obs["action_mask"]).unsqueeze(0),
                            "observation": {
                                "job_features": torch.tensor(obs["observation"]["job_features"]).unsqueeze(0),
                                "time_to_makespan": torch.tensor(obs["observation"]["time_to_makespan"]).unsqueeze(0),
                                "machine_features": torch.tensor(obs["observation"]["machine_features"]).unsqueeze(0),
                            }
                        }
                    }
                    rl_module_out = machine_rl_module.forward_inference(input_dict)

                elif agent_id.startswith("transbot"):
                    input_dict = {
                        "obs": {
                            "action_mask": torch.tensor(obs["action_mask"]).unsqueeze(0),
                            "observation": {
                                "job_features": torch.tensor(obs["observation"]["job_features"]).unsqueeze(0),
                                "time_to_makespan": torch.tensor(obs["observation"]["time_to_makespan"]).unsqueeze(0),
                                "transbot_features": torch.tensor(obs["observation"]["transbot_features"]).unsqueeze(0),
                            }
                        }
                    }
                    rl_module_out = transbot_rl_module.forward_inference(input_dict)

                logits = convert_to_numpy(rl_module_out['action_dist_inputs'])[0]

                action_prob = softmax(logits)
                action_prob[action_prob <= 1e-6] = 0.0
                actions[agent_id] = np.random.choice(len(logits), p=action_prob)
                # actions[agent_id] = np.random.choice(np.where(logits == np.max(logits))[0])

            observations, rewards, done, truncated, info = scheduling_env.step(actions)
            # scheduling_env.render()
            # print(f"decision_count = {decision_count}")
            decision_count += 1

            for agent, reward in rewards.items():
                total_rewards[agent] += reward

            # print(f"Actions: {actions}")
            # print(f"Rewards: {rewards}")
            # print(f"Done: {done}")
        # scheduling_env.close()
        # for job_id in range(scheduling_env.num_jobs):
        #     print(f"job {job_id}: {scheduling_env.scheduling_instance.jobs[job_id].scheduled_results}")
        #
        # for machine_id in range(scheduling_env.num_machines):
        #     print(f"machine {machine_id}: {scheduling_env.factory_instance.machines[machine_id].scheduled_results}")
        #     # print(scheduling_env.factory_instance.machines[machine_id].reliability_history)
        # for transbot_id in range(scheduling_env.num_transbots):
        #     print(f"transbot {transbot_id}: {scheduling_env.factory_instance.agv[transbot_id].scheduled_results}")
        #     # print(scheduling_env.factory_instance.agv[transbot_id].battery.soc_history)

        print(f"Actual makespan = {scheduling_env.current_time_after_step}")
        print(f"Estimated makespan = {scheduling_env.initial_estimated_makespan}")
        print(f"Total reward for episode {episode + 1}: {total_rewards['machine0']}")

        makespans.append(scheduling_env.current_time_after_step)

        end_time = time.time()
        print(f"Running time for episode {episode + 1} is {end_time - begin_time}")
        times.append(end_time - begin_time)

        func("Local Scheduling completed.")

        print(f"Min makespan up to now is {np.min(makespans)}.")
        print(f"Average makespan up to now is {np.average(makespans)}.")


    print(f"\nMin makespan across {num_episodes} episodes is {np.min(makespans)}.")
    print(f"Average makespan across {num_episodes} episodes is {np.average(makespans)}.")
    print(f"Average running time across {num_episodes} episodes is {np.average(times)}.")



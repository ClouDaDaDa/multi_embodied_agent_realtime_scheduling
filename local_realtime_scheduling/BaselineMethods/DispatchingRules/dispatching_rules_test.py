# from memory_profiler import profile
# @profile
import numpy as np

from configs import dfjspt_params


def func(content: str):
    print(content)

import os
import pickle
# import numpy as np
from local_realtime_scheduling.Environment.LocalSchedulingMultiAgentEnv_v2 import LocalSchedulingMultiAgentEnv
from local_realtime_scheduling.BaselineMethods.DispatchingRules.machine_agent_heuristics import machine_agent_heuristics
from local_realtime_scheduling.BaselineMethods.DispatchingRules.transbot_agent_heuristics import transbot_agent_heuristics
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, Local_Job_schedule
from local_realtime_scheduling.Environment.InitialScheduleEnv import InitialScheduleEnv

# Example usage
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    local_schedule_file = os.path.dirname(os.path.dirname(current_dir)) + \
                          "/InterfaceWithGlobal/local_schedules/local_schedule_" + \
                          f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}" \
                          + f"I0_window{dfjspt_params.time_window_size}/window_{dfjspt_params.current_window}.pkl"

    result_file_name = os.path.dirname(os.path.dirname(current_dir)) + \
                       "/local_results/local_result_" + \
                       f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}" \
                       + f"I0_window{dfjspt_params.time_window_size}/window_{dfjspt_params.current_window}.pkl"

    with open(local_schedule_file,
              "rb") as file:
        local_schedule = pickle.load(file)

    print(vars(local_schedule))

    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_jobs": dfjspt_params.n_jobs,
        "n_transbots": dfjspt_params.n_transbots,
        "local_schedule": local_schedule,
        # "local_result_file": result_file_name,
        # "render_mode": "human",
    }

    # scheduling_env = LocalSchedulingMultiAgentEnv(config)
    scheduling_env = InitialScheduleEnv(config)

    func("Env instance created.")

    num_episodes = 10

    makespans = []

    for episode in range(num_episodes):
        print(f"\nStarting episode {episode + 1}")
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
                    actions[agent_id] = machine_agent_heuristics(machine_obs=obs)

                elif agent_id.startswith("transbot"):
                    actions[agent_id] = transbot_agent_heuristics(transbot_obs=obs)

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

        func("Local Scheduling completed.")

        print(f"Min makespan up to now is {np.min(makespans)}.")
        print(f"Average makespan up to now is {np.average(makespans)}.")


    print(f"\nMin makespan across {num_episodes} episodes is {np.min(makespans)}.")
    print(f"Average makespan across {num_episodes} episodes is {np.average(makespans)}.")
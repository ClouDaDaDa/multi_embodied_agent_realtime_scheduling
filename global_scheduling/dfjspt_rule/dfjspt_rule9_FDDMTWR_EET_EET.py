import copy

from matplotlib import pyplot as plt

# from DFJSPT_generalize.dfjspt_env import DfjsptMaEnv
# from DFJSPT_generalize import dfjspt_params
import numpy as np
import time
import json

from dfjspt_rule.job_selection_rules import job_FDD_MTWR_action
from dfjspt_rule.machine_selection_rules import machine_EET_action, transbot_EET_action
from env_for_rule import DfjsptMaEnv_for_rule

def rule9_mean_makespan(
    instance_id,
    train_or_eval_or_test=None,
):

    config = {
        "train_or_eval_or_test": train_or_eval_or_test,
    }
    env = DfjsptMaEnv(config)
    makespan_list = []
    for _ in range(100):
        observation, info = env.reset(
            # options={"instance_id": instance_id,}
        )
        # env.render()
        done = False
        count = 0
        stage = next(iter(info["agent0"].values()), None)
        total_reward = 0

        while not done:
            if stage == 0:
                legal_job_actions = copy.deepcopy(observation["agent0"]["action_mask"])
                real_job_attrs = copy.deepcopy(observation["agent0"]["observation"])
                # real_job_attrs = copy.deepcopy(observation["agent0"]["observation"]["job_features"])
                # job_actions_mask = (1 - legal_job_actions) * 1e8
                # jobs_cumulative_finished_work = real_job_attrs[:, 5]
                # jobs_remain_work = real_job_attrs[:, 7]
                # if np.any(jobs_remain_work == 0):
                #     jobs_remain_work[jobs_remain_work == 0] = 0.001
                # jobs_ratio = 1.0 * jobs_cumulative_finished_work / jobs_remain_work
                # jobs_ratio += job_actions_mask
                # FDD_MTWR_job_action = {
                #     "agent0": np.argmin(jobs_ratio)
                # }
                FDD_MTWR_job_action = job_FDD_MTWR_action(legal_job_actions=legal_job_actions,
                                                          real_job_attrs=real_job_attrs)
                observation, reward, terminated, truncated, info = env.step(FDD_MTWR_job_action)
                stage = next(iter(info["agent1"].values()), None)

            elif stage == 1:
                legal_machine_actions = copy.deepcopy(observation["agent1"]["action_mask"])
                real_machine_attrs = copy.deepcopy(observation["agent1"]["observation"])
                # # real_machine_attrs = copy.deepcopy(observation["agent1"]["observation"]["machine_features"])
                # machine_actions_mask = (1 - legal_machine_actions) * 1e8
                # machine_last_finish_time = real_machine_attrs[:, 3]
                # machine_last_finish_time += machine_actions_mask
                # EET_machine_action = {
                #     "agent1": np.argmin(machine_last_finish_time)
                # }
                EET_machine_action = machine_EET_action(legal_machine_actions=legal_machine_actions,
                                                        real_machine_attrs=real_machine_attrs)
                observation, reward, terminated, truncated, info = env.step(EET_machine_action)
                stage = next(iter(info["agent2"].values()), None)

            else:
                legal_transbot_actions = copy.deepcopy(observation["agent2"]["action_mask"])
                real_transbot_attrs = copy.deepcopy(observation["agent2"]["observation"])
                # # real_transbot_attrs = copy.deepcopy(observation["agent2"]["transbot_features"])
                # transbot_last_finish_time = real_transbot_attrs[:, 3]
                # EET_transbot_action = {
                #     "agent2": np.argmin(transbot_last_finish_time)
                # }
                EET_transbot_action = transbot_EET_action(legal_transbot_actions=legal_transbot_actions,
                                                          real_transbot_attrs=real_transbot_attrs)
                observation, reward, terminated, truncated, info = env.step(EET_transbot_action)
                stage = next(iter(info["agent0"].values()), None)
                done = terminated["__all__"]
                count += 1
                total_reward += reward["agent0"]

        make_span = env.final_makespan
        # env.render()
        # time.sleep(15)
        makespan_list.append(make_span)
    mean_makespan = np.mean(makespan_list)
    return makespan_list, mean_makespan
    # print(make_span)
    # print(count)
    # print(total_reward)



def rule9_single_makespan(
    instance_id,
    train_or_eval_or_test=None,
):

    config = {
        "train_or_eval_or_test": train_or_eval_or_test,
    }
    env = DfjsptMaEnv(config)

    observation, info = env.reset(options={
        "instance_id": instance_id,

    })
    # env.render()
    done = False
    count = 0
    stage = next(iter(info["agent0"].values()), None)
    total_reward = 0

    while not done:
        if stage == 0:
            legal_job_actions = copy.deepcopy(observation["agent0"]["action_mask"])
            real_job_attrs = copy.deepcopy(observation["agent0"]["observation"])
            # job_actions_mask = (1 - legal_job_actions) * 1e8
            # jobs_cumulative_finished_work = real_job_attrs[:, 5]
            # jobs_remain_work = real_job_attrs[:, 7]
            # if np.any(jobs_remain_work == 0):
            #     jobs_remain_work[jobs_remain_work == 0] = 0.001
            # jobs_ratio = 1.0 * jobs_cumulative_finished_work / jobs_remain_work
            # jobs_ratio += job_actions_mask
            # FDD_MTWR_job_action = {
            #     "agent0": np.argmin(jobs_ratio)
            # }
            FDD_MTWR_job_action = job_FDD_MTWR_action(legal_job_actions=legal_job_actions,
                                                      real_job_attrs=real_job_attrs)
            observation, reward, terminated, truncated, info = env.step(FDD_MTWR_job_action)
            stage = next(iter(info["agent1"].values()), None)

        elif stage == 1:
            legal_machine_actions = copy.deepcopy(observation["agent1"]["action_mask"])
            real_machine_attrs = copy.deepcopy(observation["agent1"]["observation"])
            # machine_actions_mask = (1 - legal_machine_actions) * 1e8
            # machine_last_finish_time = real_machine_attrs[:, 3]
            # # machine_last_finish_time = real_machine_attrs
            # machine_last_finish_time += machine_actions_mask
            # EET_machine_action = {
            #     "agent1": np.argmin(machine_last_finish_time)
            # }
            EET_machine_action = machine_EET_action(legal_machine_actions=legal_machine_actions,
                                                    real_machine_attrs=real_machine_attrs)
            observation, reward, terminated, truncated, info = env.step(EET_machine_action)
            stage = next(iter(info["agent2"].values()), None)

        else:
            legal_transbot_actions = copy.deepcopy(observation["agent2"]["action_mask"])
            real_transbot_attrs = copy.deepcopy(observation["agent2"]["observation"])
            # transbot_last_finish_time = real_transbot_attrs[:, 3]
            # # transbot_last_finish_time = real_transbot_attrs
            # EET_transbot_action = {
            #     "agent2": np.argmin(transbot_last_finish_time)
            # }
            EET_transbot_action = transbot_EET_action(legal_transbot_actions=legal_transbot_actions, real_transbot_attrs=real_transbot_attrs)
            observation, reward, terminated, truncated, info = env.step(EET_transbot_action)
            stage = next(iter(info["agent0"].values()), None)
            done = terminated["__all__"]
            count += 1
            total_reward += reward["agent0"]

    make_span = env.final_makespan
    # complexity = env.final_makespan * env.n_jobs / np.sum(env.mean_processing_time_of_operations)
    # env.render()
    # time.sleep(5)
    # return make_span, complexity
    return make_span

def rule9_a_makespan(
    instance,
):

    config = {
        "instance": instance,
    }
    env = DfjsptMaEnv_for_rule(config)

    observation, info = env.reset()
    # env.render()
    done = False
    count = 0
    stage = next(iter(info["agent0"].values()), None)
    total_reward = 0

    while not done:
        if stage == 0:
            legal_job_actions = copy.deepcopy(observation["agent0"]["action_mask"])
            real_job_attrs = copy.deepcopy(observation["agent0"]["observation"])
            # job_actions_mask = (1 - legal_job_actions) * 1e8
            # jobs_cumulative_finished_work = real_job_attrs[:, 5]
            # jobs_remain_work = real_job_attrs[:, 7]
            # if np.any(jobs_remain_work == 0):
            #     jobs_remain_work[jobs_remain_work == 0] = 0.001
            # jobs_ratio = 1.0 * jobs_cumulative_finished_work / jobs_remain_work
            # jobs_ratio += job_actions_mask
            # FDD_MTWR_job_action = {
            #     "agent0": np.argmin(jobs_ratio)
            # }
            FDD_MTWR_job_action = job_FDD_MTWR_action(legal_job_actions=legal_job_actions,
                                                      real_job_attrs=real_job_attrs)
            observation, reward, terminated, truncated, info = env.step(FDD_MTWR_job_action)
            stage = next(iter(info["agent1"].values()), None)

        elif stage == 1:
            legal_machine_actions = copy.deepcopy(observation["agent1"]["action_mask"])
            real_machine_attrs = copy.deepcopy(observation["agent1"]["observation"])
            # machine_actions_mask = (1 - legal_machine_actions) * 1e8
            # machine_last_finish_time = real_machine_attrs[:, 3]
            # # machine_last_finish_time = real_machine_attrs
            # machine_last_finish_time += machine_actions_mask
            # EET_machine_action = {
            #     "agent1": np.argmin(machine_last_finish_time)
            # }
            EET_machine_action = machine_EET_action(legal_machine_actions=legal_machine_actions,
                                                    real_machine_attrs=real_machine_attrs)
            observation, reward, terminated, truncated, info = env.step(EET_machine_action)
            stage = next(iter(info["agent2"].values()), None)

        else:
            legal_transbot_actions = copy.deepcopy(observation["agent2"]["action_mask"])
            real_transbot_attrs = copy.deepcopy(observation["agent2"]["observation"])
            # transbot_last_finish_time = real_transbot_attrs[:, 3]
            # # transbot_last_finish_time = real_transbot_attrs
            # EET_transbot_action = {
            #     "agent2": np.argmin(transbot_last_finish_time)
            # }
            EET_transbot_action = transbot_EET_action(legal_transbot_actions=legal_transbot_actions, real_transbot_attrs=real_transbot_attrs)
            observation, reward, terminated, truncated, info = env.step(EET_transbot_action)
            stage = next(iter(info["agent0"].values()), None)
            done = terminated["__all__"]
            count += 1
            total_reward += reward["agent0"]

    make_span = env.final_makespan
    # complexity = env.final_makespan * env.n_jobs / np.sum(env.mean_processing_time_of_operations)
    # env.render()
    # time.sleep(5)
    # return make_span, complexity
    return make_span


if __name__ == '__main__':
    import os
    folder_name = os.path.dirname(os.path.dirname(__file__)) + "/dfjspt_data/my_data/pool_J" + str(dfjspt_params.n_jobs) + "_M" + str(dfjspt_params.n_machines) + "_T" + str(dfjspt_params.n_transbots)
    # pool_name = folder_name + "/instance" + str(dfjspt_params.n_instances) + "_Omin" + str(dfjspt_params.min_n_operations) + "_Omax" + str(dfjspt_params.max_n_operations)
    # with open(pool_name, "r") as fp:
    #     loaded_pool = json.load(fp)

    makespan_list = []
    complexity_list = []
    for j in range(dfjspt_params.n_instances_for_testing):
        # makespan, complexity = rule9_single_makespan(
        #     instance_id=j,
        #     train_or_eval_or_test="test",
        # )
        _, makespan = rule9_mean_makespan(
            instance_id=j,
            train_or_eval_or_test="test",
        )
        makespan_list.append(makespan)
        # complexity_list.append(complexity)
    average_makespan = np.mean(makespan_list)
    # complexity_sorted_indices = [complexity_list.index(x) for x in sorted(complexity_list)]
    makespan_sorted_indices = [makespan_list.index(x) for x in sorted(makespan_list)]
    print(makespan_list)
    print(average_makespan)

    # normalized_makespan_list = [makes / np.min(makespan_list) for makes in makespan_list]
    # normalized_complexity_list = [comp / np.min(complexity_list) for comp in complexity_list]
    #
    # plt.figure(figsize=(15, 8))
    # plt.plot(range(dfjspt_params.n_instances_for_testing), normalized_makespan_list,
    #          marker='o', color='g', label='makespan')
    # plt.plot(range(dfjspt_params.n_instances_for_testing), normalized_complexity_list,
    #          marker='*', color='r', label='complexity')
    # plt.legend()
    # plt.show()

    # rule_makespan_results = []
    # time_0 = time.time()
    # for i in range(dfjspt_params.n_instances):
    #     makespan_i = rule9_single_makespan(
    #         instance_id=i,
    #         train_or_eval_or_test=None,
    #     )
    #     rule_makespan_results.append(makespan_i)
    #     print(f"Makespan of instance {i} = {makespan_i}.")
    # time_1 = time.time()
    # total_time = time_1 - time_0
    # print(f"Total time = {total_time}.")
    # print(f"Average running time per instance = {total_time / dfjspt_params.n_instances}")
    # average = 0 if not rule_makespan_results else (sum(rule_makespan_results) / len(rule_makespan_results))
    # print(f"Average makespan = {average}.")
    # #
    # rule_makespan_results_name = folder_name + "/RuleMakespan_Omax" + str(dfjspt_params.max_n_operations) + "_Omin" + str(
    #     dfjspt_params.min_n_operations) + "_Instance" + str(dfjspt_params.n_instances)
    #
    # with open(rule_makespan_results_name, "w") as fp:
    #     json.dump(rule_makespan_results, fp)



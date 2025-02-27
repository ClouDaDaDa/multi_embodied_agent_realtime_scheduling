import numpy as np
import copy


def machine_agent_heuristics(machine_obs):
    """
    :param machine_obs = {
                "action_mask": machine_action_mask,
                "observation": {
                    "job_features": job_features,
                    "time_to_makespan": self.current_time_after_step - self.initial_estimated_makespan,
                    "machine_features": machine_features,
                }
            }
    :return: machine_action
    """

    action_mask = copy.deepcopy(machine_obs["action_mask"])
    num_jobs = len(action_mask) - 5

    # Heuristics:
    # Only perform CM after the machine is failed;
    # if machine_obs["observation"]["machine_features"][0] != 3:
    #     action_mask[num_jobs:num_jobs+4] = 0
    invalid_action_penalties = (1 - action_mask) * 1e8
    if machine_obs["observation"]["machine_features"][0] != 3:
        invalid_action_penalties[num_jobs:num_jobs+3] = 1e4
    # Cannot do nothing when idling
    if machine_obs["observation"]["machine_features"][0] == 0:
        invalid_action_penalties[num_jobs + 4] = 1e3
    # Choose the job with the shortest processing time (SPT)
    processing_time = np.zeros((len(action_mask),))
    processing_time[:num_jobs] = machine_obs["observation"]["job_features"][:, 3]
    action_score = processing_time + invalid_action_penalties
    machine_action = np.random.choice(np.where(action_score == action_score.min())[0])

    if num_jobs <= machine_action < num_jobs + 4:
        if machine_obs["observation"]["machine_features"][0] == 1 or machine_obs["observation"]["machine_features"][0] == 2:
            raise ValueError(f"Invalid action!")

    return machine_action

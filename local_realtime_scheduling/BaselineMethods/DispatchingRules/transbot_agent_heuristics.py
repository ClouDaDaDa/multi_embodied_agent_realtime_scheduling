import numpy as np
import copy


def transbot_agent_heuristics(transbot_obs):
    """
    :param transbot_obs = {
                "action_mask": transbot_action_mask,
                "observation": {
                    "job_features": job_features,
                    "time_to_makespan": self.current_time_after_step - self.initial_estimated_makespan,
                    "transbot_features": transbot_features,
                }
            }
    :return: transbot_action
    """

    action_mask = copy.deepcopy(transbot_obs["action_mask"])
    num_jobs = len(action_mask) - 2

    # Heuristics:
    # Only go to charging after the transbot is in low battery;
    # if transbot_obs["observation"]["transbot_features"][0] != 4:
    #     action_mask[num_jobs] = 0
    invalid_action_penalties = (1 - action_mask) * 1e8
    if transbot_obs["observation"]["transbot_features"][0] != 4:
        invalid_action_penalties[num_jobs] = 1e4
    # Cannot do nothing when idling
    if transbot_obs["observation"]["transbot_features"][0] == 0:
        invalid_action_penalties[num_jobs + 1] = 1e3
    # Choose the job with the shortest transporting time (STT)
    transporting_time = np.zeros((len(action_mask),))
    transporting_time[:num_jobs] = transbot_obs["observation"]["job_features"][:, 4]
    action_score = transporting_time + invalid_action_penalties
    transbot_action = np.random.choice(np.where(action_score == action_score.min())[0])

    return transbot_action
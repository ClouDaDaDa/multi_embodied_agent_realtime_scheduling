from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.annotations import override


class MyCallbacks(DefaultCallbacks):

    # def on_train_result(self, algorithm, result):
    #     # Check if the policy network was updated in the current training iteration
    #     if "policy_updated" in result:
    #         policy_updated = result["policy_updated"]
    #         print(f"Policy network updated: {policy_updated}")
    @override(DefaultCallbacks)
    def on_episode_end(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ):
        # metrics_logger.stats["total_makespan"] = env.current_time_after_step
        # episode.custom_metrics["total_cost"] = episode.worker.env.total_cost
        print(env.current_time_after_step)
        metrics_logger.log_value(key="actual_makespan", value=env.current_time_after_step,
                                 reduce="mean")





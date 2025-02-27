# import math
# import random
# import heapq
# import matplotlib.pyplot as plt
import numpy as np
from configs import dfjspt_params
from System.FactoryGraph import FactoryGraph


class SchedulingInstance:
    def __init__(self,
                 seed,
                 n_jobs,
                 n_machines,
                 # n_transbots,
                 ):

        np.random.seed(seed)

        self.factory_graph = FactoryGraph(n_machines)

        # Whether the number of jobs in this scheduling instance is fixed (n_jobs) or random
        if dfjspt_params.n_jobs_is_fixed:
            self.n_jobs_for_this_instance = n_jobs
        else:
            self.n_jobs_for_this_instance = np.random.randint(10, n_jobs + 1)

        # Whether the number of operations for each job is fixed (n_machines) or random
        if dfjspt_params.n_operations_is_n_machines:
            self.n_operations_for_jobs = n_machines * np.ones((self.n_jobs_for_this_instance,), dtype=int)
        else:
            self.n_operations_for_jobs = np.random.randint(low=dfjspt_params.min_n_operations,
                                                      high=dfjspt_params.max_n_operations + 1,
                                                      size=self.n_jobs_for_this_instance, dtype=int)

        self.job_arrival_time = np.zeros(shape=(n_jobs,), dtype=int)
        if dfjspt_params.consider_job_insert and dfjspt_params.new_arrival_jobs > 0:
            for new_job in range(dfjspt_params.new_arrival_jobs):
                self.job_arrival_time[n_jobs - new_job - 1] = np.random.randint(
                    low=dfjspt_params.earliest_arrive_time,
                    high=dfjspt_params.latest_arrive_time + 1
                )

        self.job_due_date = 1e6 * np.ones(shape=(n_jobs,), dtype=int)

        # processing time for each operation
        self.processing_time_baseline = np.random.randint(low=dfjspt_params.min_prcs_time,
                                                     high=dfjspt_params.max_prcs_time,
                                                     size=(n_jobs, max(self.n_operations_for_jobs)))
        self.processing_time_matrix = -1 * np.ones((n_jobs, max(self.n_operations_for_jobs), n_machines))
        if dfjspt_params.is_fully_flexible:
            self.n_compatible_machines_for_operations = n_machines * np.ones(shape=(n_jobs, max(self.n_operations_for_jobs)),
                                                                        dtype=int)
        else:
            self.n_compatible_machines_for_operations = np.random.randint(1, n_machines + 1,
                                                                     size=(n_jobs, max(self.n_operations_for_jobs)))

        for job_id in range(n_jobs):
            for operation_id in range(self.n_operations_for_jobs[job_id]):
                if dfjspt_params.time_for_compatible_machines_are_same:
                    self.processing_time_matrix[job_id, operation_id, :] = self.processing_time_baseline[job_id, operation_id]
                else:
                    # the same operation may have different time in different machines
                    self.processing_time_matrix[job_id, operation_id, :] = np.random.randint(
                        max(dfjspt_params.min_prcs_time,
                            self.processing_time_baseline[job_id, operation_id] - dfjspt_params.time_viration_range),
                        min(dfjspt_params.max_prcs_time,
                            self.processing_time_baseline[job_id, operation_id] + dfjspt_params.time_viration_range + 1),
                        size=(n_machines,)
                    )
                # some machine may cannot process this operation
                zero_columns = np.random.choice(n_machines,
                                                n_machines - self.n_compatible_machines_for_operations[job_id, operation_id],
                                                replace=False)
                self.processing_time_matrix[job_id, operation_id, zero_columns] = -1

        # # quality of each resource (quality<1.0 may cause longer time than expected)
        # if dfjspt_params.all_machines_are_perfect:
        #     self.machine_quality = np.ones((1, n_machines), dtype=float)
        #     self.transbot_quality = np.ones((1, n_transbots), dtype=float)
        # else:
        #     self.num_perfect_machine = np.random.randint(0, n_machines + 1)
        #     orig_machine_quality = np.round(np.random.uniform(dfjspt_params.min_quality, 1.0, size=(n_machines,)), 1)
        #     machine_mask = np.random.choice(n_machines, self.num_perfect_machine, replace=False)
        #     orig_machine_quality[machine_mask] = 1.0
        #     self.machine_quality = orig_machine_quality.reshape((1, n_machines))
        #
        #     self.num_perfect_transbot = np.random.randint(0, n_transbots + 1)
        #     orig_transbot_quality = np.round(np.random.uniform(dfjspt_params.min_quality, 1.0, size=(n_transbots,)), 1)
        #     transbot_mask = np.random.choice(n_transbots, self.num_perfect_transbot, replace=False)
        #     orig_transbot_quality[transbot_mask] = 1.0
        #     self.transbot_quality = orig_transbot_quality.reshape((1, n_transbots))


# Example Usage:
if __name__ == "__main__":

    n_jobs = 10
    n_machines = 50
    n_transbots = 5

    # Initialize instance
    instance = SchedulingInstance(
        seed=42,
        n_jobs=n_jobs,
        n_machines=n_machines,
        # n_transbots=n_transbots,
    )

    print(instance.factory_graph.unload_transport_time_matrix)





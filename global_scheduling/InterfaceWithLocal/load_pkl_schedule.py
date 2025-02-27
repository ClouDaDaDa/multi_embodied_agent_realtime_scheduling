# from memory_profiler import profile
# @profile
import os

def func(text):
    print(text)

func("before import:")

import pickle
from configs import dfjspt_params
import numpy as np

func("after import:")


# Function to convert loaded GlobalSchedule back to Numpy data
def convert_class_to_numpy(global_schedule):
    # Extract makespan
    makespan = global_schedule.makespan

    # Determine the number of jobs and maximum number of operations
    n_jobs = len(global_schedule.jobs)
    max_n_operations = max(len(job.operations) for job in global_schedule.jobs)

    # Initialize Numpy arrays
    job_arrival_time = np.zeros(n_jobs)
    job_due_date = np.zeros(n_jobs)
    result_start_time_for_jobs = np.zeros((n_jobs, max_n_operations, 2))
    result_finish_time_for_jobs = np.zeros((n_jobs, max_n_operations, 2))

    # Initialize routes
    machine_routes = {}
    transbot_routes = {}

    # Populate Numpy arrays and routes
    for job in global_schedule.jobs:
        job_id = job.job_id
        job_arrival_time[job_id] = job.arrival_time
        job_due_date[job_id] = job.due_date

        for operation in job.operations:
            op_type = operation.type
            operation_id = operation.operation_id

            if op_type == "Transport":
                # Transport operation details
                if operation.transbot_assigned not in transbot_routes:
                    transbot_routes[operation.transbot_assigned] = []
                transbot_routes[operation.transbot_assigned].append([
                    operation.job_id,
                    operation.operation_id,
                    operation.transbot_source,
                    operation.job_location,
                    operation.destination,
                ])

                # Transport times
                result_start_time_for_jobs[job_id, operation_id, 0] = operation.estimated_start_time
                result_finish_time_for_jobs[job_id, operation_id, 0] = operation.estimated_end_time

            elif op_type == "Processing":
                # Processing operation details
                if operation.machine_assigned not in machine_routes:
                    machine_routes[operation.machine_assigned] = []
                machine_routes[operation.machine_assigned].append([
                    operation.job_id,
                    operation.operation_id,
                ])

                # Processing times
                result_start_time_for_jobs[job_id, operation_id, 1] = operation.estimated_start_time
                result_finish_time_for_jobs[job_id, operation_id, 1] = operation.estimated_end_time

    return (
        makespan,
        job_arrival_time,
        job_due_date,
        result_start_time_for_jobs,
        result_finish_time_for_jobs,
        machine_routes,
        transbot_routes,
    )


global_schedule_file = os.path.dirname(os.path.abspath(__file__)) \
                       + "/global_schedules/global_schedule_" \
                       + f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}" \
                       + f"T{dfjspt_params.n_transbots}I0.pkl"

func("before load:")

with open(global_schedule_file, "rb") as file:
    loaded_schedule = pickle.load(file)
# print(loaded_schedule)

func("after load:")

makespan, job_arrival_time, job_due_date, result_start_time_for_jobs, result_finish_time_for_jobs, machine_routes, transbot_routes = convert_class_to_numpy(loaded_schedule)

func("after convert to numpy:")





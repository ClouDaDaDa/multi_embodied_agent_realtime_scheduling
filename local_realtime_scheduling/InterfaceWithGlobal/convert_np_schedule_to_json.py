import os

import numpy as np
import json
import random


# def convert_schedule_to_json(result_start_time_for_jobs, result_finish_time_for_jobs):
#     n_jobs, max_n_operations, _ = result_start_time_for_jobs.shape
#     global_schedule = {"Global_Schedule": {"Jobs": []}}
#
#     # Iterate through all jobs
#     for job_idx in range(n_jobs):
#         job_data = {
#             "Job_ID": f"Job_{job_idx + 1:03}",
#             "Operations": []
#         }
#
#         # Iterate through all operations for the job
#         for op_idx in range(max_n_operations):
#             transport_start = result_start_time_for_jobs[job_idx, op_idx, 0]
#             transport_end = result_finish_time_for_jobs[job_idx, op_idx, 0]
#             processing_start = result_start_time_for_jobs[job_idx, op_idx, 1]
#             processing_end = result_finish_time_for_jobs[job_idx, op_idx, 1]
#
#             if transport_start == 0 and transport_end == 0 and processing_start == 0 and processing_end == 0:
#                 continue  # Skip unassigned operations
#
#             # Transport operation
#             transport_operation = {
#                 "Operation_ID": f"Op_{job_idx + 1:03}_{op_idx * 2 + 1:02}",
#                 "Type": "Transport",
#                 "Robot_Assigned": f"Transbot_{(op_idx % 3) + 1}",  # Example assignment logic
#                 "Unload_Transport": {
#                     "Source": "Placeholder_Source",
#                     "Destination": "Placeholder_Destination",
#                     "Path_Plan": [],
#                     "Distance": 0,
#                     "Estimated_Start_Time": transport_start,
#                     "Estimated_Duration": transport_end - transport_start,
#                     "Estimated_End_Time": transport_end
#                 },
#                 "Loaded_Transport": {
#                     "Source": "Placeholder_Source",
#                     "Destination": "Placeholder_Destination",
#                     "Path_Plan": [],
#                     "Distance": 0,
#                     "Estimated_Start_Time": transport_start,
#                     "Estimated_Duration": transport_end - transport_start,
#                     "Estimated_End_Time": transport_end
#                 }
#             }
#             job_data["Operations"].append(transport_operation)
#
#             # Processing operation
#             processing_operation = {
#                 "Operation_ID": f"Op_{job_idx + 1:03}_{op_idx * 2 + 2:02}",
#                 "Type": "Processing",
#                 "Machine_Assigned": f"Machine_{(op_idx % 3) + 1}",  # Example assignment logic
#                 "Estimated_Start_Time": processing_start,
#                 "Estimated_Duration": processing_end - processing_start,
#                 "Estimated_End_Time": processing_end
#             }
#             job_data["Operations"].append(processing_operation)
#
#         global_schedule["Global_Schedule"]["Jobs"].append(job_data)
#
#     # Output to JSON file
#     with open("global_schedule.json", "w") as json_file:
#         json.dump(global_schedule, json_file, indent=4)

def convert_schedule_to_json(result_start_time_for_jobs, result_finish_time_for_jobs, machine_routes, transbot_routes):
    n_jobs, max_n_operations, _ = result_start_time_for_jobs.shape

    global_schedule = {"Global_Schedule": {"Jobs": []}}

    # 遍历每个作业
    for job_id in range(n_jobs):
        job_schedule = {"Job_ID": f"Job_{job_id + 1:03d}", "Operations": []}

        # 遍历该作业的所有工序
        for operation_id in range(max_n_operations):
            start_times = result_start_time_for_jobs[job_id, operation_id]
            finish_times = result_finish_time_for_jobs[job_id, operation_id]

            # 跳过无效的工序
            if start_times[0] == 0 and finish_times[0] == 0 and start_times[1] == 0 and finish_times[1] == 0:
                continue

            # 运输工序 (Transport)
            transport_operation = next(
                (task for transbot_id, tasks in transbot_routes.items()
                 for task in tasks if task[0] == job_id and task[1] == operation_id),
                None
            )
            if transport_operation:
                transbot_id, _, transbot_source, job_location, destination = transport_operation
                transport_schedule = {
                    "Operation_ID": f"Op_{job_id + 1:03d}_{operation_id + 1:02d}",
                    "Type": "Transport",
                    "Robot_Assigned": f"Transbot_{transbot_id + 1}",
                    "Unload_Transport": {
                        "Source": transbot_source,
                        "Destination": job_location,
                        "Path_Plan": [],  # Placeholder for now
                        "Distance": 0,  # Placeholder for now
                        "Estimated_Start_Time": start_times[0],
                        "Estimated_Duration": finish_times[0] - start_times[0],
                        "Estimated_End_Time": finish_times[0]
                    },
                    "Loaded_Transport": {
                        "Source": job_location,
                        "Destination": destination,
                        "Path_Plan": [],  # Placeholder for now
                        "Distance": 0,  # Placeholder for now
                        "Estimated_Start_Time": start_times[0],  # Assumes same as Unload End Time
                        "Estimated_Duration": finish_times[0] - start_times[0],
                        "Estimated_End_Time": finish_times[0]
                    }
                }
                job_schedule["Operations"].append(transport_schedule)

            # 加工工序 (Processing)
            processing_operation = next(
                (task for machine_id, tasks in machine_routes.items()
                 for task in tasks if task[0] == job_id and task[1] == operation_id),
                None
            )
            if processing_operation:
                machine_id, _ = processing_operation
                processing_schedule = {
                    "Operation_ID": f"Op_{job_id + 1:03d}_{operation_id + 1:02d}",
                    "Type": "Processing",
                    "Machine_Assigned": f"Machine_{machine_id + 1}",
                    "Estimated_Start_Time": start_times[1],
                    "Estimated_Duration": finish_times[1] - start_times[1],
                    "Estimated_End_Time": finish_times[1]
                }
                job_schedule["Operations"].append(processing_schedule)

        global_schedule["Global_Schedule"]["Jobs"].append(job_schedule)

    # 转换为 JSON 格式
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = current_dir + "/simulate_global_data/global_schedule.json"
    with open(file_name, "w") as json_file:
        json.dump(global_schedule, json_file, indent=4)

    return global_schedule



# Simulate real data for testing
# def simulate_real_data(n_jobs, max_operations, n_machines, n_transbots):
#     result_start_time_for_jobs = np.zeros((n_jobs, max_operations, 2), dtype=float)
#     result_finish_time_for_jobs = np.zeros((n_jobs, max_operations, 2), dtype=float)
#
#     for job_idx in range(n_jobs):
#         current_time = 0
#         for op_idx in range(max_operations):
#             # Transport operation
#             transport_duration = random.randint(1, 10)  # Random transport duration
#             transport_start = current_time
#             transport_end = transport_start + transport_duration
#             result_start_time_for_jobs[job_idx, op_idx, 0] = transport_start
#             result_finish_time_for_jobs[job_idx, op_idx, 0] = transport_end
#             current_time = transport_end
#
#             # Processing operation
#             processing_duration = random.randint(5, 20)  # Random processing duration
#             processing_start = current_time
#             processing_end = processing_start + processing_duration
#             result_start_time_for_jobs[job_idx, op_idx, 1] = processing_start
#             result_finish_time_for_jobs[job_idx, op_idx, 1] = processing_end
#             current_time = processing_end
#
#     return result_start_time_for_jobs, result_finish_time_for_jobs

def simulate_real_data(n_jobs, max_operations, n_machines, n_transbots):
    n_operations_for_jobs = [random.randint(1, max_operations) for _ in range(n_jobs)]
    result_start_time_for_jobs = np.zeros((n_jobs, max_operations, 2), dtype=float)
    result_finish_time_for_jobs = np.zeros((n_jobs, max_operations, 2), dtype=float)

    # Generate start and finish times
    for job_idx in range(n_jobs):
        current_time = 0
        for op_idx in range(n_operations_for_jobs[job_idx]):
            # Transport operation
            transport_duration = random.randint(1, 10)
            transport_start = current_time
            transport_end = transport_start + transport_duration
            result_start_time_for_jobs[job_idx, op_idx, 0] = transport_start
            result_finish_time_for_jobs[job_idx, op_idx, 0] = transport_end
            current_time = transport_end

            # Processing operation
            processing_duration = random.randint(5, 20)
            processing_start = current_time
            processing_end = processing_start + processing_duration
            result_start_time_for_jobs[job_idx, op_idx, 1] = processing_start
            result_finish_time_for_jobs[job_idx, op_idx, 1] = processing_end
            current_time = processing_end

    # Generate machine routes and transbot routes
    machine_routes = {m_id: np.array([], dtype=int) for m_id in range(n_machines)}
    transbot_routes = {t_id: np.array([], dtype=int) for t_id in range(n_transbots)}

    for job_idx in range(n_jobs):
        for op_idx in range(n_operations_for_jobs[job_idx]):
            tspt_task_id = sum(n_operations_for_jobs[:job_idx]) + op_idx
            prcs_task_id = tspt_task_id

            # Randomly assign machine and transbot
            assigned_machine = random.randint(0, n_machines - 1)
            assigned_transbot = random.randint(0, n_transbots - 1)

            machine_routes[assigned_machine] = np.append(machine_routes[assigned_machine], prcs_task_id)
            transbot_routes[assigned_transbot] = np.append(transbot_routes[assigned_transbot], tspt_task_id)

    return result_start_time_for_jobs, result_finish_time_for_jobs, machine_routes, transbot_routes



# Example Usage
if __name__ == "__main__":
    # Example usage
    n_jobs = 3
    max_operations = 8
    n_machines = 5
    n_transbots = 3

    # result_start, result_finish = simulate_real_data(n_jobs, max_operations, n_machines, n_transbots)
    # convert_schedule_to_json(result_start, result_finish)
    result_start, result_finish, machine_routes, transbot_routes = simulate_real_data(
        n_jobs, max_operations, n_machines, n_transbots
    )
    convert_schedule_to_json(result_start, result_finish, machine_routes, transbot_routes)






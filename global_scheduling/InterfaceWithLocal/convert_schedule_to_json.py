import json
import numpy as np


def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):  # 如果是 NumPy 数组，转换为列表
        return obj.tolist()
    elif isinstance(obj, np.int64):  # 如果是 NumPy int64，转换为 Python int
        return int(obj)
    elif isinstance(obj, np.float64):  # 如果是 NumPy float64，转换为 Python float
        return float(obj)
    elif isinstance(obj, dict):  # 如果是字典，递归处理键和值
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):  # 如果是列表，递归处理每个元素
        return [convert_numpy_types(element) for element in obj]
    else:  # 如果是其他类型，直接返回
        return obj


def convert_schedule_to_json(
        file_name,
        result_start_time_for_jobs,
        result_finish_time_for_jobs,
        machine_routes,
        transbot_routes):
    n_jobs, max_n_operations, _ = result_start_time_for_jobs.shape

    global_schedule = {"Global_Schedule": {"Jobs": []}}

    # 遍历每个作业
    for job_id in range(n_jobs):
        job_schedule = {"Job_ID": job_id, "Operations": []}
        # job_schedule = {"Job_ID": f"Job_{job_id}", "Operations": []}

        # 遍历该作业的所有工序
        for operation_id in range(max_n_operations):
            start_times = result_start_time_for_jobs[job_id, operation_id]
            finish_times = result_finish_time_for_jobs[job_id, operation_id]

            # 跳过无效的工序
            if start_times[0] == 0 and finish_times[0] == 0 and start_times[1] == 0 and finish_times[1] == 0:
                continue

            # 运输工序 (Transport)
            transport_operation = next(
                ((transbot_id, task) for transbot_id, tasks in transbot_routes.items()
                 for task in tasks if task[0] == job_id and task[1] == operation_id),
                None
            )

            if transport_operation is not None:
                transbot_id, task = transport_operation
                job_id, operation_id, transbot_source, job_location, destination = task
                transport_schedule = {
                    "Type": "Transport",
                    "Job_ID": job_id,
                    "Operation_ID": operation_id,
                    "Transbot_Assigned": transbot_id,
                    "Transbot_Source": transbot_source,
                    "Job_Location": job_location,
                    "Destination": destination,
                    "Estimated_Start_Time": start_times[0],
                    "Estimated_Duration": finish_times[0] - start_times[0],
                    "Estimated_End_Time": finish_times[0]
                }
                # transport_schedule = {
                #     "Type": "Transport",
                #     "Job_ID": job_id,
                #     "Operation_ID": operation_id,
                #     # "Operation_ID": f"Op_{job_id}_{2*operation_id}",
                #     # "Robot_Assigned": f"Transbot_{transbot_id}",
                #     "Transbot_Assigned": transbot_id,
                #     "Unload_Transport": {
                #         "Source": transbot_source,
                #         "Destination": job_location,
                #         "Path_Plan": [],  # Placeholder for now
                #         "Distance": 0,  # Placeholder for now
                #         "Estimated_Start_Time": start_times[0],
                #         "Estimated_Duration": 0,
                #         "Estimated_End_Time": start_times[0]
                #     },
                #     "Loaded_Transport": {
                #         "Source": job_location,
                #         "Destination": destination,
                #         "Path_Plan": [],  # Placeholder for now
                #         "Distance": 0,  # Placeholder for now
                #         "Estimated_Start_Time": start_times[0],
                #         "Estimated_Duration": finish_times[0] - start_times[0],
                #         "Estimated_End_Time": finish_times[0]
                #     }
                # }
                job_schedule["Operations"].append(transport_schedule)

            # 加工工序 (Processing)
            processing_operation = next(
                ((machine_id, task) for machine_id, tasks in machine_routes.items()
                 for task in tasks if task[0] == job_id and task[1] == operation_id),
                None
            )
            if processing_operation is not None:
                machine_id, task = processing_operation
                job_id, operation_id = task
                processing_schedule = {
                    "Type": "Processing",
                    "Job_ID": job_id,
                    "Operation_ID": operation_id,
                    # "Operation_ID": f"Op_{job_id}_{2*operation_id+1}",
                    # "Machine_Assigned": f"Machine_{machine_id}",
                    "Machine_Assigned": machine_id,
                    "Estimated_Start_Time": start_times[1],
                    "Estimated_Duration": finish_times[1] - start_times[1],
                    "Estimated_End_Time": finish_times[1]
                }
                job_schedule["Operations"].append(processing_schedule)

        global_schedule["Global_Schedule"]["Jobs"].append(job_schedule)

    # 转换 NumPy 数据类型
    global_schedule = convert_numpy_types(global_schedule)

    # 转换为 JSON 格式
    with open(file_name, "w") as json_file:
        json.dump(global_schedule, json_file, indent=4)

    return global_schedule


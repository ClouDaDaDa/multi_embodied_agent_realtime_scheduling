import json
import os
from typing import List, Dict, Optional


# Define classes to represent the Global Schedule structure

class Operation:
    def __init__(self, operation_id, operation_type, resource_assigned, estimated_start, estimated_duration,
                 estimated_end, source=None, destination = None, path_plan = None, distance = None):
        self.operation_id = operation_id
        self.operation_type = operation_type  # "Transport" or "Processing"
        self.resource_assigned = resource_assigned  # Machine or Transbot assigned
        self.source = source
        self.destination = destination
        self.path_plan = path_plan
        self.distance = distance
        self.estimated_start = estimated_start
        self.estimated_duration = estimated_duration
        self.estimated_end = estimated_end

    def to_dict(self):
        if self.operation_type == "Processing":
            operation_dict = {
                "Operation_ID": self.operation_id,
                "Type": self.operation_type,
                "Resource_Assigned": self.resource_assigned,
                "Estimated_Start_Time": self.estimated_start,
                "Estimated_Duration": self.estimated_duration,
                "Estimated_End_Time": self.estimated_end
            }
        else:
            operation_dict = {
                "Operation_ID": self.operation_id,
                "Type": self.operation_type,
                "Resource_Assigned": self.resource_assigned,
                "Source": self.source,
                "Destination": self.destination,
                "Path_Plan": self.path_plan,
                "Distance": self.distance,
                "Estimated_Start_Time": self.estimated_start,
                "Estimated_Duration": self.estimated_duration,
                "Estimated_End_Time": self.estimated_end
            }
        # if self.unload_transport:
        #     operation_dict["Unload_Transport"] = self.unload_transport.to_dict()
        # if self.loaded_transport:
        #     operation_dict["Loaded_Transport"] = self.loaded_transport.to_dict()
        return operation_dict


class Job:
    def __init__(self, job_id, operations: List[Operation]):
        self.job_id = job_id
        self.operations = operations

    def to_dict(self):
        return {
            "Job_ID": self.job_id,
            "Operations": [op.to_dict() for op in self.operations]
        }


class GlobalSchedule:
    def __init__(self, jobs: List[Job]):
        self.jobs = jobs

    def to_dict(self):
        return {
            "Global_Schedule": {
                "Jobs": [job.to_dict() for job in self.jobs]
            }
        }

    def get_max_completion_time(self):
        max_time = 0
        for job in self.jobs:
            for operation in job.operations:
                if operation.estimated_end > max_time:
                    max_time = operation.estimated_end
        return max_time


# Helper function to read a global schedule from a JSON file
def read_global_schedule(filepath: str) -> GlobalSchedule:
    with open(filepath, 'r') as file:
        data = json.load(file)

    jobs = []
    for job_data in data["Global_Schedule"]["Jobs"]:
        operations = []
        for op_data in job_data["Operations"]:
            # Handling the optional transport details for transport operations
            # unload_transport = None
            # loaded_transport = None

            # Create the Operation object based on the type and details
            if op_data["Type"] == "Processing":
                operation = Operation(
                    operation_id=op_data["Operation_ID"],
                    operation_type=op_data["Type"],
                    resource_assigned=op_data.get("Machine_Assigned"),
                    estimated_start=op_data["Estimated_Start_Time"],
                    estimated_duration=op_data["Estimated_Duration"],
                    estimated_end=op_data["Estimated_End_Time"],
                )
                operations.append(operation)

            elif op_data["Type"] == "Transport":
                if "Unload_Transport" in op_data:
                    operation = Operation(
                        operation_id=op_data["Operation_ID"],
                        operation_type="Unload_Transport",
                        resource_assigned=op_data.get("Robot_Assigned"),
                        source=op_data["Unload_Transport"]["Source"],
                        destination=op_data["Unload_Transport"]["Destination"],
                        path_plan=op_data["Unload_Transport"]["Path_Plan"],
                        distance=op_data["Unload_Transport"]["Distance"],
                        estimated_start=op_data["Unload_Transport"]["Estimated_Start_Time"],
                        estimated_duration=op_data["Unload_Transport"]["Estimated_Duration"],
                        estimated_end=op_data["Unload_Transport"]["Estimated_End_Time"],
                    )
                    operations.append(operation)
                if "Loaded_Transport" in op_data:
                    operation = Operation(
                        operation_id=op_data["Operation_ID"],
                        operation_type="Loaded_Transport",
                        resource_assigned=op_data.get("Robot_Assigned"),
                        source=op_data["Loaded_Transport"]["Source"],
                        destination=op_data["Loaded_Transport"]["Destination"],
                        path_plan=op_data["Loaded_Transport"]["Path_Plan"],
                        distance=op_data["Loaded_Transport"]["Distance"],
                        estimated_start=op_data["Loaded_Transport"]["Estimated_Start_Time"],
                        estimated_duration=op_data["Loaded_Transport"]["Estimated_Duration"],
                        estimated_end=op_data["Loaded_Transport"]["Estimated_End_Time"],
                    )
                    operations.append(operation)
        job = Job(job_id=job_data["Job_ID"], operations=operations)
        jobs.append(job)

    return GlobalSchedule(jobs=jobs)


# Helper function to divide global schedule into time windows
def divide_schedule_into_time_windows(global_schedule: GlobalSchedule, time_window_size: int) -> List[Dict]:
    """
    Divides the global schedule into local schedules based on a specified time window size.

    Parameters:
        global_schedule (GlobalSchedule): The full schedule containing all jobs and operations.
        time_window_size (int): The duration of each time window.

    Returns:
        List[Dict]: A list of dictionaries, each representing the local schedule for a time window.
    """
    max_completion_time = global_schedule.get_max_completion_time()
    time_windows = []

    # Iterate over each time window
    for start_time in range(0, int(max_completion_time) + 1, time_window_size):
        end_time = start_time + time_window_size
        local_schedule = {"Time_Window": [start_time, end_time], "Operations_In_Window": []}

        # Check each job and its operations to see if it falls within the current time window
        for job in global_schedule.jobs:
            for operation in job.operations:
                # If operation overlaps with the current time window, add it to the local schedule
                if (operation.estimated_start < end_time and operation.estimated_end > start_time):
                    local_schedule["Operations_In_Window"].append(operation.to_dict())

        time_windows.append(local_schedule)

    return time_windows


# Helper function to save local schedules as JSON files
def save_local_schedules(local_schedules: List[Dict], output_folder: str):
    """
    Saves each local schedule as a JSON file.

    Parameters:
        local_schedules (List[Dict]): List of local schedules divided by time windows.
        output_folder (str): Directory where the JSON files will be saved.
    """
    for idx, local_schedule in enumerate(local_schedules):
        filename = f"{output_folder}/local_schedule_window_{idx}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as file:
            json.dump(local_schedule, file, indent=4)
        print(f"Local schedule for time window {idx} saved to {filename}")


# Helper function to organize operations by resource within each time window
def organize_operations_by_resource(local_schedules: List[Dict]) -> List[Dict]:
    """
    Organizes operations within each time window by the resource assigned.

    Parameters:
        local_schedules (List[Dict]): List of local schedules divided by time windows.

    Returns:
        List[Dict]: A list of dictionaries, each representing a resource-organized schedule for a time window.
    """
    organized_schedules = []

    # Iterate over each time window schedule
    for window_schedule in local_schedules:
        start_time, end_time = window_schedule["Time_Window"]
        resource_schedule = {"Time_Window": [start_time, end_time], "Resource_Operations": {}}

        # Organize operations by resource
        for operation in window_schedule["Operations_In_Window"]:
            resource = operation["Resource_Assigned"]
            if resource not in resource_schedule["Resource_Operations"]:
                resource_schedule["Resource_Operations"][resource] = []
            resource_schedule["Resource_Operations"][resource].append(operation)

        # Sort each resource's operations by their estimated start time for better scheduling clarity
        for resource, ops in resource_schedule["Resource_Operations"].items():
            resource_schedule["Resource_Operations"][resource] = sorted(
                ops, key=lambda x: x["Estimated_Start_Time"]
            )

        organized_schedules.append(resource_schedule)

    return organized_schedules


# Updated function to save resource-organized local schedules as JSON files
def save_organized_schedules(organized_schedules: List[Dict], output_folder: str):
    """
    Saves each resource-organized local schedule as a JSON file.

    Parameters:
        organized_schedules (List[Dict]): List of local schedules organized by resources.
        output_folder (str): Directory where the JSON files will be saved.
    """
    for idx, organized_schedule in enumerate(organized_schedules):
        filename = f"{output_folder}/organized_schedule_window_{idx}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as file:
            json.dump(organized_schedule, file, indent=4)
        print(f"Organized schedule for time window {idx} saved to {filename}")


# Main execution
if __name__ == "__main__":
    # Define the path to the global schedule JSON file and the output folder for local schedules
    global_schedule_filepath = "global_schedule.json"
    output_folder = "local_schedules"

    # Define the time window size (e.g., 100 time units)
    time_window_size = 100

    # Read the global schedule from JSON
    global_schedule = read_global_schedule(global_schedule_filepath)

    # Divide the global schedule into time windows
    local_schedules = divide_schedule_into_time_windows(global_schedule, time_window_size)

    # # Save the local schedules to individual JSON files
    # save_local_schedules(local_schedules, output_folder)

    # Organize operations by resource within each time window
    organized_schedules = organize_operations_by_resource(local_schedules)

    # Save the organized schedules to individual JSON files
    save_organized_schedules(organized_schedules, output_folder)









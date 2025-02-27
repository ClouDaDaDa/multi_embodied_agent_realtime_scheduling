import pickle
import os
from typing import List


class Local_Job_schedule:
    def __init__(self, job_id):
        self.job_id = job_id
        self.operations = {}
        self.available_time = None
        self.estimated_finish_time = None

    def add_operation(self, operation):
        self.operations[operation.operation_id] = operation

    def __repr__(self):
        return f"Job(Job_ID={self.job_id}, Operations={self.operations})"


class LocalSchedule:
    def __init__(self):
        self.jobs = {}
        self.local_makespan = None
        self.time_window_start = None
        self.time_window_end = None

    def add_job(self, job):
        self.jobs[job.job_id] = job

    def __repr__(self):
        return f"LocalSchedule(Jobs={self.jobs})"


# Function to divide global schedule into local schedules based on time windows
def divide_schedule_into_time_windows(
        global_schedule,
        time_window_size
) -> List[LocalSchedule]:
    max_completion_time = global_schedule.makespan
    local_schedules = []

    for start_time in range(0, int(max_completion_time) + 1, time_window_size):
        end_time = min(start_time + time_window_size, max_completion_time)
        local_schedule = LocalSchedule()
        local_schedule.time_window_start = start_time
        local_schedule.time_window_end = end_time
        local_schedule.local_makespan = end_time

        for job in global_schedule.jobs:
            # Check if the job should be included in this time window
            relevant_job = None
            for operation in job.operations:
                if operation.scheduled_start_processing_time < end_time \
                        and operation.scheduled_finish_processing_time >= start_time:
                    # If the job is not in the local schedule, add it
                    if relevant_job is None:
                        relevant_job = Local_Job_schedule(job_id=job.job_id)
                        if operation.scheduled_start_processing_time >= start_time:
                            relevant_job.available_time = start_time
                        else:
                            relevant_job.available_time = operation.scheduled_finish_processing_time
                        local_schedule.add_job(relevant_job)
                    # Add the operation to the relevant job
                    relevant_job.add_operation(operation)
                    if operation.scheduled_finish_processing_time > local_schedule.local_makespan:
                        local_schedule.local_makespan = operation.scheduled_finish_processing_time
            if relevant_job is not None:
                relevant_job.estimated_finish_time = max(op.scheduled_finish_processing_time for _, op in relevant_job.operations.items())
        local_schedules.append(local_schedule)

    return local_schedules


# Function to save LocalSchedule objects as .pkl files
def save_local_schedules(local_schedules: List[LocalSchedule], output_folder: str):
    for idx, local_schedule in enumerate(local_schedules):
        filename = os.path.join(output_folder, f"window_{idx}.pkl")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "wb") as file:
            pickle.dump(local_schedule, file)
        print(f"Local schedule for time window {idx} saved to {filename}")


# Main execution
if __name__ == "__main__":
    from configs import dfjspt_params

    # Define the path to the GlobalSchedule file and the output folder for local schedules
    global_schedule_filepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) \
                                + "/global_scheduling/InterfaceWithLocal/global_schedules/global_schedule_" \
                                + f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}" \
                                + f"T{dfjspt_params.n_transbots}I0.pkl"

    output_folder = os.path.dirname(os.path.abspath(__file__)) + \
                    "/local_schedules/local_schedule_" + \
                    f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}I0" \
                    + f"_window{dfjspt_params.time_window_size}"

    # Load the GlobalSchedule from the pkl file
    with open(global_schedule_filepath, "rb") as file:
        global_schedule = pickle.load(file)

    # Divide the global schedule into local schedules based on time windows
    local_schedules = divide_schedule_into_time_windows(global_schedule, dfjspt_params.time_window_size)

    # Save each local schedule as a .pkl file
    save_local_schedules(local_schedules, output_folder)

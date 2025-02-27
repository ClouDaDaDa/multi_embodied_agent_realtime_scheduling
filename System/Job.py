import numpy as np
import logging
logging.basicConfig(level=logging.INFO)


class Job:
    def __init__(self, job_id: int, operations_matrix, arrival_time=0, due_date=1e8):
        """
        :param job_id: Unique identifier for the job
        :param operations_matrix: size of (n_operations, n_machines),
        where the (i,j)-th element is the processing duration of operation i in machine j.
        If machine j cannot process operation i, this value is -1.0.
        :param due_date: The due date of the job
        """
        # Static features of the job
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.due_date = due_date
        self.operations_matrix = np.array(operations_matrix, dtype=float)
        self.num_total_processing_operations = len(operations_matrix)

        # Dynamic global features of the job
        self.current_location = 'warehouse'
        self.job_status = 0  # 0: idle, 1: processing, 2: transporting, 3: completed
        self.job_progress = 0.0
        self.is_completed = False
        self.cumulative_processing_time = 0.0
        self.cumulative_transporting_time = 0.0
        # Store (operation type, operation id, machine/transbot id, start/end time) for all operations
        self.scheduled_results = []

        self.current_processing_operation = 0  # to be processed
        self.current_transporting_operation = 0
        self.assigned_machine = None  # refers to the job's next operation (not start yet)
        self.assigned_transbot = None

        # Dynamic local features of the job
        self.available_time_for_current_time_window = None
        self.estimated_finish_time_for_current_time_window = None
        self.processing_operations_for_current_time_window = []
        self.transporting_operations_for_current_time_window = []
        self.num_processing_operations_for_current_time_window = 0
        self.job_progress_for_current_time_window = 1.0
        self.is_completed_for_current_time_window = True
        self.estimated_remaining_time_for_current_task = 0.0

        # logging.info(f"Job {self.job_id} created with due date {self.due_date}, it has {self.num_operations} operations.")

    def assigned_to_machine(self, machine_id):
        self.assigned_machine = machine_id

    def start_processing(self, start_time, estimated_duration):
        self.job_status = 1
        self.estimated_remaining_time_for_current_task = estimated_duration
        # (operation type, operation id, machine id, start time)
        self.scheduled_results.append(
            ("Processing", self.current_processing_operation, f"machine{self.assigned_machine}", start_time))
        # As long as this operation starts, the progress will increase by one.
        self.current_processing_operation += 1
        self.assigned_machine = None
        self.get_overall_progress()
        # self.check_completed()

    def update_processing(self):
        """
        update processing for 1 time step
        """
        self.cumulative_processing_time += 1.0
        self.estimated_remaining_time_for_current_task -= 1.0

    def finish_processing(self, finish_time):
        if self.job_status != 3:
            self.job_status = 0
        self.estimated_remaining_time_for_current_task = 0.0
        # (operation type, operation id, machine id, finish time)
        self.scheduled_results.append((self.scheduled_results[-1][:3] + (finish_time,)))
        self.check_completed()

    def assigned_to_transbot(self, transbot_id):
        self.assigned_transbot = transbot_id
        # print(f"Job {self.job_id} is assigned to transbot {transbot_id}.")

    def start_transporting(self, start_time, estimated_duration):
        self.job_status = 2
        self.estimated_remaining_time_for_current_task = estimated_duration
        # (operation type, operation id, transbot id, start time)
        self.scheduled_results.append(
            ("Transport", self.current_processing_operation, f"transbot{self.assigned_transbot}", start_time))
        self.assigned_transbot = None

    def update_transporting(self):
        """
        update transporting for 1 time step
        """
        self.cumulative_transporting_time += 1.0
        self.estimated_remaining_time_for_current_task -= 1.0

    def finish_transporting(self, finish_time, current_location):
        self.job_status = 0
        self.current_location = current_location
        self.estimated_remaining_time_for_current_task = 0.0
        # (operation type, operation id, transbot id, finish time)
        self.scheduled_results.append((self.scheduled_results[-1][:3] + (finish_time,)))
        # print(f"Job {self.job_id} arrived at {self.current_location}.")

    def get_overall_progress(self):
        self.job_progress = 1.0 * self.current_processing_operation / self.num_total_processing_operations
        self.job_progress_for_current_time_window = 1.0 * self.current_processing_operation / self.num_processing_operations_for_current_time_window

    def check_completed(self):
        if (self.num_processing_operations_for_current_time_window - self.current_processing_operation == 0) and (self.job_status == 0):
            self.job_status = 3
            self.is_completed_for_current_time_window = True
        if (self.num_total_processing_operations - self.current_processing_operation == 0) and (self.job_status == 0):
            self.is_completed = True

    def reset_job(self):
        # Dynamic global features of the job
        self.current_location = 'warehouse'
        self.job_status = 0  # 0: idle, 1: processing, 2: transporting, 3: completed
        self.job_progress = 0.0
        self.is_completed = False
        self.cumulative_processing_time = 0.0
        self.cumulative_transporting_time = 0.0
        # Store (operation type, operation id, machine/transbot id, start/end time) for all operations
        self.scheduled_results = []

        self.current_processing_operation = 0  # to be processed
        self.current_transporting_operation = 0
        self.assigned_machine = None  # refers to the job's next operation (not start yet)
        self.assigned_transbot = None

        self.reset_job_for_current_time_window()

    def reset_job_for_current_time_window(self):
        # Dynamic local features of the job
        self.available_time_for_current_time_window = None
        self.estimated_finish_time_for_current_time_window = None
        self.processing_operations_for_current_time_window = []
        self.transporting_operations_for_current_time_window = []
        self.num_processing_operations_for_current_time_window = 0
        self.job_progress_for_current_time_window = 1.0
        self.is_completed_for_current_time_window = True
        self.estimated_remaining_time_for_current_task = 0.0


# Example usage
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    operations = [
        [10.0, 20.0, -1.0, 40.0, -1.0],
        [20.0, 10.0, 30.0, -1.0, 40.0],
        [-1.0, 30.0, 20.0, 10.0, -1.0]
    ]
    job = Job(job_id=0, operations_matrix=operations, due_date=1000.0)

    # for i in range(len(operations)):
    #     available_machines = [j for j in range(len(operations[i])) if operations[i][j] != -1]
    #     machine_index = np.random.choice(available_machines)
    #     start_time = job.cumulative_total_time
    #     processing_duration = job.operations_matrix[i, machine_index]
    #     job.start_processing()
    #     job.update_processing(machine_id=machine_index, start_time=start_time, processing_duration=processing_duration)
    #     job.finish_processing()
    #
    # print(f"Final job status: {job.job_status}, cumulative time: {job.cumulative_processing_time}, is completed: {job.is_completed}")
    #
    # # Plot the Gantt chart
    # fig, gnt = plt.subplots()
    # gnt.set_ylim(0, 20)
    # gnt.set_xlim(0, 120)
    # gnt.set_xlabel('Time')
    # gnt.set_ylabel('Operation')
    #
    # gnt.set_yticks([5])
    # gnt.set_yticklabels(['Operation'])
    #
    # colors = ['tab:blue', 'tab:orange', 'tab:green']
    # for i, (operation, machine_id, start_time, end_time) in enumerate(job.scheduled_results):
    #     gnt.broken_barh([(start_time, end_time - start_time)], (5, 4), facecolors=(colors[operation % len(colors)]))
    #
    # # Adding legend
    # handles = [mpatches.Patch(color=colors[i], label=f'Operation {i}') for i in range(len(colors))]
    # plt.legend(handles=handles, loc='upper right')
    #
    # plt.title('Gantt Chart of Job Processing')
    # plt.grid(True)
    # plt.show()

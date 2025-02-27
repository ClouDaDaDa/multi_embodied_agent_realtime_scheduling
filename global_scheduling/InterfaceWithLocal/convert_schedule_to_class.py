import pickle


class Operation_schedule:
    def __init__(self, job_id, operation_id, **kwargs):
        self.job_id = job_id
        self.operation_id = operation_id
        self.scheduled_start_transporting_time = None
        self.scheduled_finish_transporting_time = None
        self.assigned_transbot = None
        self.scheduled_start_processing_time = None
        self.scheduled_finish_processing_time = None
        self.assigned_machine = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"Job_ID={self.job_id}, Operation_ID={self.operation_id}, {vars(self)})"


class Job_schedule:
    def __init__(self, job_id):
        self.job_id = job_id
        self.operations = []
        self.arrival_time = None
        self.due_date = None

    def add_operation(self, operation):
        self.operations.append(operation)

    def __repr__(self):
        return f"Job(Job_ID={self.job_id}, Operations={self.operations})"


class GlobalSchedule:
    def __init__(self):
        self.jobs = []
        self.makespan = None

    def add_job(self, job):
        self.jobs.append(job)

    def __repr__(self):
        return f"GlobalSchedule(Jobs={self.jobs})"


def convert_schedule_to_class(
        file_name,
        makespan,
        job_arrival_time,
        job_due_date,
        result_start_time_for_jobs,
        result_finish_time_for_jobs,
        machine_routes,
        transbot_routes
):
    n_jobs, max_n_operations, _ = result_start_time_for_jobs.shape

    global_schedule = GlobalSchedule()
    global_schedule.makespan = makespan

    for job_id in range(n_jobs):
        job = Job_schedule(job_id)
        job.arrival_time = job_arrival_time[job_id]
        job.due_date = job_due_date[job_id]

        for operation_id in range(max_n_operations):
            start_times = result_start_time_for_jobs[job_id, operation_id]
            finish_times = result_finish_time_for_jobs[job_id, operation_id]

            if start_times[0] == 0 and finish_times[0] == 0 and start_times[1] == 0 and finish_times[1] == 0:
                continue

            operation = Operation_schedule(
                job_id=job_id,
                operation_id=operation_id,
            )
            job.add_operation(operation)

            # Transport Operation
            transport_operation = next(
                ((transbot_id, task) for transbot_id, tasks in transbot_routes.items()
                 for task in tasks if task[0] == job_id and task[1] == operation_id),
                None
            )
            if transport_operation is not None:
                job.operations[operation_id].assigned_transbot = transport_operation[0]
                job.operations[operation_id].scheduled_start_transporting_time = start_times[0]
                job.operations[operation_id].scheduled_finish_transporting_time = finish_times[0]

            # if transport_operation is not None:
            #     transbot_id, task = transport_operation
            #     job_id, operation_id, transbot_source, job_location, destination = task
            #     transport_schedule = Operation_schedule(
            #         op_type="Transport",
            #         job_id=job_id,
            #         operation_id=operation_id,
            #         transbot_assigned=transbot_id,
            #         transbot_source=transbot_source,
            #         job_location=job_location,
            #         destination=destination,
            #         estimated_start_time=start_times[0],
            #         estimated_duration=finish_times[0] - start_times[0],
            #         estimated_end_time=finish_times[0]
            #     )
            #     job.add_operation(transport_schedule)

            # Processing Operation
            processing_operation = next(
                ((machine_id, task) for machine_id, tasks in machine_routes.items()
                 for task in tasks if task[0] == job_id and task[1] == operation_id),
                None
            )
            if processing_operation is not None:
                job.operations[operation_id].assigned_machine = processing_operation[0]
                job.operations[operation_id].scheduled_start_processing_time = start_times[1]
                job.operations[operation_id].scheduled_finish_processing_time = finish_times[1]

            # if processing_operation is not None:
            #     machine_id, task = processing_operation
            #     job_id, operation_id = task
            #     processing_schedule = Operation_schedule(
            #         op_type="Processing",
            #         job_id=job_id,
            #         operation_id=operation_id,
            #         machine_assigned=machine_id,
            #         estimated_start_time=start_times[1],
            #         estimated_duration=finish_times[1] - start_times[1],
            #         estimated_end_time=finish_times[1]
            #     )
            #     job.add_operation(processing_schedule)

        global_schedule.add_job(job)

        with open(file_name, "wb") as file:
            pickle.dump(global_schedule, file)

    return global_schedule

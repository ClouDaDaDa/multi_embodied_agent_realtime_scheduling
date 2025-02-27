# from memory_profiler import profile
# @profile
from local_realtime_scheduling.Environment.ExecutionResult import LocalResult, Local_Job_result
from local_realtime_scheduling.Environment.path_planning import a_star_search


def func(content: str):
    print(content)


import pickle
from gymnasium import spaces

func("import part 1")

# from System.Order import Order
# from System.Machine import Machine
# from System.AGV import AGV
# from System.Job import Job
from System.SchedulingInstance import SchedulingInstance
from System.FactoryInstance import FactoryInstance
from configs import dfjspt_params

import logging
logging.basicConfig(level=logging.INFO)


func("import part 2")

from ray.rllib.env.multi_agent_env import MultiAgentEnv

func("import part 3")


def get_direction_from_coordinates(location1, location2):
    # Define the reverse direction map
    reverse_direction_map = {
        (1, 0): 2,  # Right
        (-1, 0): 3,  # Left
        (0, 1): 4,  # Up
        (0, -1): 5  # Down
    }

    # Calculate the difference
    d_location = location2 - location1

    # Return the direction key
    return reverse_direction_map.get(d_location, None)


class LocalSchedulingMultiAgentEnv(MultiAgentEnv):
    """
    A Multi-agent Environment for Integrated Production, Transportation and Maintenance Real-time Scheduling.
    """

    def __init__(self,
                 config,
                 ):
        """
        :param config: including:
        n_machines
        n_trasnbots
        n_jobs
        local_schedule
        """
        super(LocalSchedulingMultiAgentEnv, self).__init__()

        func("Env initialized.")

        # Initialize parameters
        self.num_jobs = config["n_jobs"]
        self.num_machines = config["n_machines"]
        self.num_transbots = config["n_transbots"]

        self.factory_instance = FactoryInstance(
            seed=42,
            n_machines=self.num_machines,
            n_transbots=self.num_transbots,
        )

        self.scheduling_instance = SchedulingInstance(
            seed=52,
            n_jobs=self.num_jobs,
            n_machines=self.num_machines,
        )

        self.local_schedule = config["local_schedule"]
        self.local_result = None

        # Light Maintenance (LM), Middle Maintenance (MM), Overhaul (OH), and Corrective Maintenance (CM)
        self.num_maintenance_methods = 4

        self.maintenance_costs = [20.0, 50.0, 80.0, 100.0]
        self.storage_unit_cost = 1.0
        self.tardiness_unit_cost = 1.0
        self.costs_baseline = self.num_jobs * self.storage_unit_cost * 30.0

        self.MAX_DURATION = dfjspt_params.max_prcs_time
        self.max_operations = None
        self.MAX_MAINTENANCE_COUNTS = None
        self.initial_estimated_makespan = None

        self.maintenance_cost_for_machines = np.zeros((self.num_machines,), dtype=float)
        self.maintenance_counts_for_machines = np.zeros((self.num_machines,), dtype=int)

        self.current_time_before_step = 0.0
        self.current_time_after_step = 0.0
        self.reward_this_step = 0.0
        self.schedule_done = False
        self.real_makespan = None
        self.chosen_machine = None
        self.chosen_task = None
        self.total_cost = 0.0
        self.estimated_makespan = self.initial_estimated_makespan
        self.prev_std_deviation = 0.0
        self.prev_time_to_due_date = 0.0
        self.decision_stage = 0  # 0 for machines and 1 for transbots

        # Define agent IDs
        self.machine_agents = [f"machine{i}" for i in range(self.num_machines)]
        self.transbot_agents = [f"transbot{i}" for i in range(self.num_transbots)]
        self.agents = self.possible_agents = self.machine_agents + self.transbot_agents

        self.terminateds = set()
        self.truncateds = set()
        self.resetted = False

        # Define observation and action spaces for each agent type
        self.observation_spaces = {}
        self.action_spaces = {}

        # Machines: Observation space and action space
        for machine_agent_id in self.machine_agents:
            # possible actions:
            # [0,num_jobs-1] for start processing, [num_jobs,num_jobs+3] for start maintenance, num_jobs+4 for do nothing
            self.num_machine_actions = self.num_jobs + 4 + 1
            # states:
            # [0] machine internal status, [1] reliability, [2] number of remaining tasks,
            # [3] if next job in queue is available, [4] time to next job in queue is available
            # [5] time to finish the current task
            self.num_machine_states = 6
            self.observation_spaces[machine_agent_id] = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.num_machine_actions,), dtype=np.int64),
                "observation": spaces.Dict({
                    "job_features": spaces.Box(low=-1.0, high=float('inf'), shape=(self.num_jobs, 5)),
                    "time_to_makespan": spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,)),
                    "machine_features": spaces.Box(low=-1.0, high=float('inf'), shape=(3,)),
                }),
            })
            self.action_spaces[machine_agent_id] = spaces.Discrete(self.num_machine_actions)

        # Transbots: Observation space and action space
        for transbot_agent_id in self.transbot_agents:
            # possible actions:
            # [0,num_jobs-1] for start transportation, num_jobs for start charging, num_jobs+1 for do nothing
            self.num_transbot_actions = self.num_jobs + 2
            # states:
            #
            self.num_transbot_states = 10
            self.observation_spaces[transbot_agent_id] = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.num_transbot_actions,), dtype=np.int64),
                "observation": spaces.Dict({
                    "job_features": spaces.Box(low=-1.0, high=float('inf'), shape=(self.num_jobs, 5)),
                    "time_to_makespan": spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,)),
                    "transbot_features": spaces.Box(low=-1.0, high=float('inf'), shape=(3,)),
                }),
            })
            self.action_spaces[transbot_agent_id] = spaces.Discrete(self.num_transbot_actions)

    def _initialize_state(self):
        obs = {}
        for machine_agent_id in self.machine_agents:
            obs[machine_agent_id] = self._get_machine_obs(machine_agent_id)
        # for transbot_agent_id in self.transbot_agents:
        #     obs[transbot_agent_id] = self._get_transbot_obs(transbot_agent_id)
        return obs

    def _get_info(self):
        return {
            "current_decision_stage": self.decision_stage
        }

    def _get_machine_obs(self, machine_agent_id):
        machine_index = int(machine_agent_id.lstrip("machine"))
        machine = self.factory_instance.machines[machine_index]

        # dynamic action masking (DAM) logic for the machine:
        # 1 for valid and 0 for invalid action
        machine_action_mask = np.zeros((self.num_machine_actions,), dtype=int)
        # idling: can choose a job, or perform a maintenance except CM, or do nothing
        if machine.machine_status == 0:
            machine_action_mask[self.num_jobs:self.num_jobs+3] = 1
            machine_action_mask[self.num_jobs + 4] = 1
        elif machine.machine_status == 1 or machine.machine_status == 2:
        # processing or under maintenance: can only do nothing
            machine_action_mask[self.num_jobs + 4] = 1
        elif machine.machine_status == 3:
        # faulty: can perform CM or do nothing
            machine_action_mask[self.num_jobs + 3:] = 1

        job_features = -1 * np.ones((self.num_jobs, 5), dtype=float)
        for job_id in range(self.num_jobs):
            job_features[job_id, 0] = job_id  # [0] job_id

            if job_id in self.local_schedule.jobs:  # the job is in this problem
                this_job = self.scheduling_instance.jobs[job_id]

                if this_job.job_status != 3:  # the job still has operation to be processed

                    if this_job.assigned_machine is None:  # the job's next pending operation has not been assigned to a machine

                        # this machine can handle the job's next pending operation
                        if this_job.operations_matrix[this_job.current_processing_operation, machine_index] > 0:

                            # [1] job's internal status
                            job_features[job_id, 1] = this_job.job_status
                            # [2] job's progress
                            job_features[job_id, 2] = this_job.job_progress_for_current_time_window
                            # [3] processing time for this machine to handle this operation
                            job_features[job_id, 3] = this_job.operations_matrix[this_job.current_processing_operation, machine_index]
                            # [4] distance from this machine to this job
                            job_location_index = self.factory_instance.factory_graph.location_index_map[this_job.current_location]
                            job_features[job_id, 4] = self.factory_instance.factory_graph.unload_transport_time_matrix[
                                job_location_index, machine_index  # machine_location_index == machine_index
                            ]

                            if machine.machine_status == 0:
                                machine_action_mask[job_id] = 1

        machine_features = np.array([
            # [0] machine status
            machine.machine_status,
            # [1] reliability
            machine.reliability,
            # [2] time to finish the current task
            machine.estimated_remaining_time_to_finish,
        ])

        return {
            "action_mask": machine_action_mask,
            "observation": {
                "job_features": job_features,
                "time_to_makespan": self.current_time_after_step - self.initial_estimated_makespan,
                "machine_features": machine_features,
            }
        }

    def _get_transbot_obs(self, transbot_agent_id):
        transbot_index = int(transbot_agent_id.lstrip("transbot"))
        transbot = self.factory_instance.agv[transbot_index]

        # dynamic action masking logic for the transbot: 1 for valid and 0 for invalid action
        transbot_action_mask = np.zeros((self.num_transbot_actions,), dtype=int)
        # idling (0): can choose a job, or go to charge, or do nothing
        if transbot.agv_status == 0:
            transbot_action_mask[self.num_jobs:] = 1
        # unload transporting (1): can change its task or insist the current task
        elif transbot.agv_status == 1:
            transbot_action_mask[self.num_jobs:] = 1
        # loaded transporting (2) or charging (3): can only do nothing
        elif transbot.agv_status == 2 or transbot.agv_status == 3:
            transbot_action_mask[self.num_jobs + 1] = 1
        # low battery (4): can only go to charge
        elif transbot.agv_status == 4:
            transbot_action_mask[self.num_jobs] = 1

        job_features = -1 * np.ones((self.num_jobs, 5), dtype=float)
        for job_id in range(self.num_jobs):
            job_features[job_id, 0] = job_id  # [0] job_id

            if job_id in self.local_schedule.jobs:  # the job is in this problem
                this_job = self.scheduling_instance.jobs[job_id]

                if this_job.job_status != 3:  # the job still has operation to be processed

                    if this_job.assigned_machine is not None:  # the job's next pending operation has been assigned to a machine
                        # job_features[job_id, 3] = 1  # [3] next operation has been assigned to a machine: 1 for yes and -1 for no

                        if this_job.job_status != 2:  # if the job is in transporting, this transbot don't consider it
                            job_location_index = self.factory_instance.factory_graph.location_index_map[
                                this_job.current_location]
                            job_to_machine = self.factory_instance.factory_graph.unload_transport_time_matrix[
                                job_location_index, this_job.assigned_machine  # machine_location_index == machine_index
                            ]

                            if job_to_machine > 0:  # the job is not at its destination and needs to be transported
                                if this_job.assigned_transbot is None:  # it hasn't been assigned to another transbot

                                    job_location = self.factory_instance.factory_graph.pickup_dropoff_points[this_job.current_location]
                                    transbot_to_job = abs(transbot.current_location[0] - job_location[0])\
                                                      + abs(transbot.current_location[1] - job_location[1])

                                    job_features[job_id, 1] = this_job.job_status  # [1] job's internal status
                                    job_features[job_id, 2] = this_job.job_progress_for_current_time_window  # [2] job's progress
                                    # [3] next operation has been assigned to a machine: 1 for yes and -1 for no
                                    job_features[job_id, 3] = 1
                                    # [4] transport time for this transbot to handle this operation
                                    job_features[job_id, 4] = transbot_to_job + job_to_machine

                                    if (transbot.agv_status == 0 or transbot.agv_status == 1) and not transbot.is_for_charging:
                                        transbot_action_mask[job_id] = 1

        transbot_features = np.array([
            # [0] transbot status
            transbot.agv_status,
            # [1] SOC
            transbot.battery.soc,
            # [2] time to finish the current task
            transbot.estimated_remaining_time_to_finish,
            # [3] distance to the transbot’s current target
        ])

        return {
            "action_mask": transbot_action_mask,
            "observation": {
                "job_features": job_features,
                "time_to_makespan": self.current_time_after_step - self.initial_estimated_makespan,
                "transbot_features": transbot_features,
            }
        }

    def reset(self, seed=None, options=None):

        func("Env reset.")

        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.max_operations = max(job.operations_matrix.shape[0] for job in self.scheduling_instance.jobs)
        self.MAX_MAINTENANCE_COUNTS = sum(job.num_total_processing_operations for job in self.scheduling_instance.jobs)
        self.initial_estimated_makespan = self.local_schedule.local_makespan
        self.current_time_before_step = self.local_schedule.time_window_start
        self.current_time_after_step = self.local_schedule.time_window_start
        self.reward_this_step = 0.0
        self.schedule_done = False
        self.real_makespan = None
        self.chosen_machine = None
        self.chosen_task = None
        self.total_cost = 0.0
        self.estimated_makespan = self.initial_estimated_makespan
        self.prev_std_deviation = 0.0
        self.prev_time_to_due_date = 0.0
        self.decision_stage = 0

        self.local_result = LocalResult()

        # Reset all machines and jobs
        for machine in self.factory_instance.machines:
            machine.reset_machine_for_current_time_window()

        for agv in self.factory_instance.agv:
            agv.reset_agv_for_current_time_window()

        for job in self.scheduling_instance.jobs:
            job.reset_job_for_current_time_window()

        for job_id in self.local_schedule.jobs:
            local_schedule_job = self.local_schedule.jobs[job_id]
            this_job = self.scheduling_instance.jobs[job_id]
            this_job.available_time_for_current_time_window = local_schedule_job.available_time
            this_job.estimated_finish_time_for_current_time_window = local_schedule_job.estimated_finish_time
            this_job.is_completed_for_current_time_window = False

            for operation_id in local_schedule_job.operations:
                local_schedule_job_operation = local_schedule_job.operations[operation_id]
                if local_schedule_job_operation.type == "Processing":
                    this_job.processing_operations_for_current_time_window.append(int(operation_id))
                    self.factory_instance.machines[local_schedule_job_operation.machine_assigned].processing_tasks_queue_for_current_time_window.append((job_id, operation_id))
                else:
                    this_job.transporting_operations_for_current_time_window.append(int(operation_id))
                    self.factory_instance.agv[local_schedule_job_operation.transbot_assigned].transport_tasks_queue_for_current_time_window.append((job_id, operation_id))

            if len(this_job.processing_operations_for_current_time_window) > 0:
                this_job.current_processing_operation = this_job.processing_operations_for_current_time_window[0]
                this_job.num_processing_operations_for_current_time_window = len(this_job.processing_operations_for_current_time_window)
                this_job.job_progress_for_current_time_window = 0.0
            if len(this_job.transporting_operations_for_current_time_window) > 0:
                this_job.current_transporting_operation = this_job.transporting_operations_for_current_time_window[0]

            self.local_result.add_job(Local_Job_result(job_id=job_id))

        observations = self._initialize_state()
        infos = self._get_info()

        return observations, infos

    def step(self, action_dict):
        observations, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}
        self.reward_this_step = 0.0
        self.resetted = False

        # Decision Stage 0: Machines should make decisions
        if self.decision_stage == 0:
            # for each machine agent that take an action at this step:
            for agent_id, action in action_dict.items():

                if agent_id.startswith("transbot"):
                    raise Exception(f"Only machines can action in decision_stage {self.decision_stage}!")

                if agent_id.startswith("machine"):
                    machine_index = int(agent_id.lstrip("machine"))
                    current_machine = self.factory_instance.machines[machine_index]

                    # perform a processing task
                    if 0 <= action < self.num_jobs:
                        # Check the validity of the processing action
                        self._check_machine_processing_action(machine_index=machine_index,
                                                              processing_action=action)

                        current_machine.current_processing_task = action
                        self.scheduling_instance.jobs[action].assigned_to_machine(machine_index)

                    # perform maintenance
                    elif self.num_jobs <= action < self.num_jobs + 4:
                        # Check the validity of the maintenance action
                        maintenance_method = action - self.num_jobs
                        self._check_machine_maintenance_action(machine_index=machine_index,
                                                               maintenance_method=maintenance_method)

                        current_machine.start_maintenance(maintenance_method=maintenance_method,
                                                          start_time=self.current_time_before_step)
                        self.maintenance_cost_for_machines[machine_index] += self.maintenance_costs[maintenance_method]
                        self.maintenance_counts_for_machines[machine_index] += 1

                        logging.info(f"Machine {machine_index} starts maintenance ({action - self.num_jobs}) at time {self.current_time_before_step}.")

                    # do-nothing
                    elif action == self.num_jobs + 4:
                        pass
                    else:
                        raise Exception(f"Invalid action ({action}) for machine {machine_index}!")

            # Transbots get new observations:
            for transbot_agent_id in self.transbot_agents:
                observations[transbot_agent_id] = self._get_transbot_obs(transbot_agent_id)
                # todo: design a suitable reward function for each transbot
                rewards[transbot_agent_id] = 0.0
                terminated[transbot_agent_id] = self._check_done()
                if terminated[transbot_agent_id]:
                    self.terminateds.add(transbot_agent_id)
                truncated[transbot_agent_id] = False

        # Decision Stage 1: Transbots should make decisions
        else:

            # for each machine agent that take an action at this step:
            for agent_id, action in action_dict.items():

                if agent_id.startswith("machine"):
                    raise Exception(f"Only transbots can action in decision_stage {self.decision_stage}!")

                if agent_id.startswith("transbot"):
                    transbot_index = int(agent_id.lstrip("transbot"))
                    current_transbot = self.factory_instance.agv[transbot_index]

                    # perform the transporting task
                    if 0 <= action < self.num_jobs:
                        # Check the validity of the transporting action
                        self._check_transbot_transporting_action(transbot_index=transbot_index,
                                                                 transporting_action=action)

                        current_transbot.current_task = action
                        self.scheduling_instance.jobs[action].assigned_to_transbot(transbot_index)

                    # (move for) charging
                    elif action == self.num_jobs:
                        # Check the validity of the charging action
                        self._check_transbot_charging_action(transbot_index=transbot_index,
                                                             charging_action=action)

                        current_transbot.current_task = -1
                        charging_station_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                            self.factory_instance.factory_graph.nearest_charging_station(
                                current_transbot.current_location
                            )
                        ]
                        unload_path = a_star_search(
                            graph=self.factory_instance.factory_graph,
                            start=current_transbot.current_location,
                            goal=charging_station_location
                        )
                        current_transbot.start_unload_transporting(
                            target_location=charging_station_location,
                            unload_path=unload_path,
                            start_time=self.current_time_before_step
                        )

                    # do-nothing
                    elif action == self.num_jobs + 1:
                        if current_transbot.agv_status == 4:
                            raise Exception(f"...!")

                    else:
                        raise Exception(f"Invalid action ({action}) for transbot {transbot_index}!")

            # Environment moves forward for 1 time step

            # step the env according to actions:
            self.current_time_before_step = self.current_time_after_step
            self.current_time_after_step += 1.0

            for machine_agent_id in self.machine_agents:
                # Each machine moves forward for 1 time step
                machine_index = int(machine_agent_id.lstrip("machine"))
                self._step_a_machine_for_one_step(machine_index=machine_index)

                # Machines get new observations:
                observations[machine_agent_id] = self._get_machine_obs(machine_agent_id)
                # todo: design a suitable reward function for each machine
                rewards[machine_agent_id] = 0.0
                terminated[machine_agent_id] = self._check_done()
                if terminated[machine_agent_id]:
                    self.terminateds.add(machine_agent_id)
                truncated[machine_agent_id] = False

            for transbot_agent_id in self.transbot_agents:
                # Each transbot moves forward for 1 time step
                transbot_index = int(transbot_agent_id.lstrip("transbot"))
                self._step_a_transbot_for_one_step(transbot_index=transbot_index)


        self.decision_stage = 1 - self.decision_stage
        infos = self._get_info()
        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = False

        return observations, rewards, terminated, truncated, infos

    def _step_a_machine_for_one_step(self, machine_index):
        current_machine = self.factory_instance.machines[machine_index]

        # Promote different evolutions according to different internal status
        # 0 (Idling): waiting for 1 time step
        if current_machine.machine_status == 0:
            # Checks whether it can start its scheduled task
            if current_machine.current_processing_task is not None:
                current_job = self.scheduling_instance.jobs[current_machine.current_processing_task]
                # Check whether the job is currently processable
                if current_job.job_status == 0 and current_machine.machine_status == 0 and current_job.current_location == current_machine.location:
                    # Get the processing time of the operation on machine
                    processing_duration = current_job.operations_matrix[
                        current_job.current_processing_operation, machine_index]
                    estimated_processing_duration = processing_duration / current_machine.reliability

                    # Update internal status of the job
                    current_job.start_processing(start_time=self.current_time_before_step,
                                                 estimated_duration=estimated_processing_duration)

                    # Update internal status of the machine
                    current_machine.start_processing(start_time=self.current_time_before_step,
                                                     estimated_processing_duration=estimated_processing_duration)
                    # logging.info(
                    #     f"Machine {machine_index} starts processing Job {current_job.job_id} at time {self.current_time_before_step}.")

                else:
                    current_machine.update_waiting_process(waiting_time=1.0)
            else:
                current_machine.update_waiting_process(waiting_time=1.0)

        # 1 (Processing): degrading for 1 time step
        elif current_machine.machine_status == 1:
            current_job = self.scheduling_instance.jobs[current_machine.current_processing_task]
            current_job.update_processing()
            current_machine.update_degradation_process()

            if self._check_machine_finish_task(machine_id=machine_index):
                current_machine.finish_processing(finish_time=self.current_time_after_step)
                current_job.finish_processing(finish_time=self.current_time_after_step)

        # 2 (Maintenance): maintaining for 1 time step
        elif current_machine.machine_status == 2:
            current_machine.update_maintenance_process()

            if self._check_machine_finish_task(machine_id=machine_index):
                current_machine.finish_maintenance(finish_time=self.current_time_after_step)

        # 3 (Faulty): broken for 1 time step
        elif current_machine.machine_status == 3:
            pass

    def _step_a_transbot_for_one_step(self, transbot_index):
        current_transbot = self.factory_instance.agv[transbot_index]

        # Promote different evolutions according to different internal status
        # 0 (Idling): waiting for 1 time step
        if current_transbot.agv_status == 0:
            # Checks whether it can start its scheduled task
            if current_transbot.current_task is not None:
                current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                job_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                    current_job.current_location]
                machine_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                    self.factory_instance.machines[current_job.assigned_machine].location]
                transbot_to_job = abs(current_transbot.current_location[0] - job_location[0]) \
                                  + abs(current_transbot.current_location[1] - job_location[1])

                # Check whether the job is currently transportable
                if transbot_to_job == 0:  # if the transbot is at the same location with the job:
                    if current_job.job_status == 0:
                        # Loaded transport can start immediately
                        loaded_path = a_star_search(
                            graph=self.factory_instance.factory_graph,
                            start=job_location,
                            goal=machine_location
                        )
                        current_transbot.start_loaded_transporting(
                            target_location=machine_location,
                            loaded_path=loaded_path,
                            start_time=self.current_time_before_step
                        )
                        # Update internal status of the job
                        current_job.start_transporting(start_time=self.current_time_before_step,
                                                       estimated_duration=len(loaded_path) - 1)
                    else:
                        current_transbot.idling_process()
                else:
                    if current_job.estimated_remaining_time_for_current_task <= transbot_to_job:
                        # the transbot can start to go to the job
                        unload_path = a_star_search(
                            graph=self.factory_instance.factory_graph,
                            start=current_transbot.current_location,
                            goal=job_location
                        )
                        current_transbot.start_unload_transporting(
                            target_location=job_location,
                            unload_path=unload_path,
                            start_time=self.current_time_before_step
                        )
                    else:
                        current_transbot.idling_process()
            else:
                current_transbot.idling_process()

        # 1 (Unload Transporting): moving for 1 time step
        elif current_transbot.agv_status == 1:
            # If the congestion exceeds 10.0, an error message will be displayed
            if current_transbot.congestion_time >= 10.0:
                raise Exception(f"Transbot {transbot_index} has been congested for 10 time steps!")

            # Check if next_location is walkable
            next_location = current_transbot.unload_path[0]
            if self.factory_instance.factory_graph.is_walkable(x=next_location[0], y=next_location[1]):
                del current_transbot.unload_path[0]
                current_transbot.congestion_time = 0.0
                direction = next_location - current_transbot.current_location
                # Mark the old location of the transbot as walkable
                self.factory_instance.factory_graph.set_walkable(location=current_transbot.current_location)
                # transbot move for one step
                current_transbot.moving_one_step(direction=direction, load=0)
                if self._check_transbot_finish_transporting(transbot_id=transbot_index):
                    current_transbot.finish_unload_transporting(finish_time=self.current_time_after_step)
                    if current_transbot.current_task >= 0:
                        # Check whether the job is currently transportable
                        current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                        job_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                            current_job.current_location]
                        machine_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                            self.factory_instance.machines[current_job.assigned_machine].location]
                        transbot_to_job = abs(current_transbot.current_location[0] - job_location[0]) \
                                          + abs(current_transbot.current_location[1] - job_location[1])
                        if transbot_to_job > 0:
                            raise Exception(
                                f"Transbot {transbot_index} hasn't get job {current_transbot.current_task}!")
                        if current_job.job_status == 0:
                            # Loaded transport can start immediately
                            loaded_path = a_star_search(
                                graph=self.factory_instance.factory_graph,
                                start=job_location,
                                goal=machine_location
                            )
                            current_transbot.start_loaded_transporting(
                                target_location=machine_location,
                                loaded_path=loaded_path,
                                start_time=self.current_time_after_step
                            )
                            # Update internal status of the job
                            current_job.start_transporting(start_time=self.current_time_after_step,
                                                           estimated_duration=len(loaded_path) - 1)
                    elif current_transbot.current_task == -1:
                        current_transbot.start_charging(start_time=self.current_time_after_step)
                else:
                    # Mark the new position of the transbot as an obstacle
                    self.factory_instance.factory_graph.set_obstacle(location=current_transbot.current_location)
            else:
                # todo: wait or re-plan the path?
                # waiting
                current_transbot.congestion_time += 1.0
                current_transbot.moving_one_step(direction=(0, 0), load=0)

        # 2 (Loaded Transporting): moving for 1 time step
        elif current_transbot.agv_status == 2:
            # If the congestion exceeds 10.0, an error message will be displayed
            if current_transbot.congestion_time >= 10.0:
                raise Exception(f"Transbot {transbot_index} has been congested for 10 time steps!")

            # Check if next_location is walkable
            next_location = current_transbot.loaded_path[0]
            if self.factory_instance.factory_graph.is_walkable(x=next_location[0], y=next_location[1]):
                del current_transbot.loaded_path[0]
                current_transbot.congestion_time = 0.0
                direction = next_location - current_transbot.current_location
                # Mark the old location of the transbot as walkable
                self.factory_instance.factory_graph.set_walkable(location=current_transbot.current_location)
                # transbot move for one step
                current_transbot.moving_one_step(direction=direction, load=1)
                if self._check_transbot_finish_transporting(transbot_id=transbot_index):
                    current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                    current_job.finish_transporting(finish_time=self.current_time_after_step,
                                                    current_location=f"machine_{current_job.assigned_machine}")
                    current_transbot.finish_loaded_transporting(finish_time=self.current_time_after_step)
                else:
                    # Mark the new position of the transbot as an obstacle
                    self.factory_instance.factory_graph.set_obstacle(location=current_transbot.current_location)
            else:
                # todo: wait or re-plan the path?
                # waiting
                current_transbot.congestion_time += 1.0
                current_transbot.moving_one_step(direction=(0, 0), load=1)

        # 3 (Charging): charging for 1 time step
        elif current_transbot.agv_status == 3:
            current_transbot.update_charging_process()
            self._check_transbot_finish_charging(transbot_id=transbot_index)

        # 4 (Low battery):
        elif current_transbot.agv_status == 4:
            pass

    def _check_machine_processing_action(self, machine_index, processing_action):
        # Check what status is the machine currently in, must be 0 (idling) to continue
        if self.factory_instance.machines[machine_index].machine_status != 0:
            raise Exception(f"...!")

        # Check whether the job is in the current problem
        if processing_action not in self.local_schedule.jobs:  # the job is in this problem
            raise Exception(f"Job {processing_action} is not in the current problem!")

        current_job = self.scheduling_instance.jobs[processing_action]
        current_operation = current_job.current_processing_operation

        # Check whether the job has been finished
        if current_job.job_status == 3:
            raise Exception(f"Job {processing_action} has been finished!")

        # Check whether the job has already assigned to another machine
        if current_job.assigned_machine is not None:
            raise Exception(
                f"Job {processing_action}'s operation {current_operation} has already assigned to machine {current_job.assigned_machine}!")

        # Check whether the job and operation can be processed by the machine
        if current_job.operations_matrix[current_operation, machine_index] <= 0:
            raise Exception(f"Machine {machine_index} cannot process job {processing_action}'s operation {current_operation}!")

    def _check_machine_maintenance_action(self, machine_index, maintenance_method):
        current_machine = self.factory_instance.machines[machine_index]
        # Check what status is the machine currently in, must be 0 (idling) or 3 (failed) to continue
        if current_machine.machine_status == 1 or current_machine.machine_status == 2:
            raise Exception(f"...!")
        if current_machine.machine_status == 0 and maintenance_method == 3:
            raise Exception(f"machine {machine_index} is not failed, so cannot choose CM ({maintenance_method})!")
        if current_machine.machine_status == 4 and maintenance_method != 3:
            raise Exception(f"machine {machine_index} is failed, so can only choose CM (not {maintenance_method})!")

    def _check_machine_finish_task(self, machine_id):
        current_machine = self.factory_instance.machines[machine_id]
        if current_machine.current_time_after_step >= current_machine.start_time_of_the_task + current_machine.actual_processing_duration:
            return True
        else:
            return False

    def _check_transbot_finish_transporting(self, transbot_id):
        current_transbot = self.factory_instance.agv[transbot_id]
        if current_transbot.current_location == current_transbot.target_location:
            return True
        else:
            return False

    def _check_transbot_transporting_action(self, transbot_index, transporting_action):
        current_transbot = self.factory_instance.agv[transbot_index]
        # Check what status is the transbot currently in, must be 0 (idling) or 1 (unload trans) to continue
        if current_transbot.agv_status != 0 and current_transbot.agv_status != 1:
            raise Exception(f"...!")

        # Check whether the job is in the current problem
        if transporting_action not in self.local_schedule.jobs:  # the job is in this problem
            raise Exception(f"Job {transporting_action} is not in the current problem!")

        current_job = self.scheduling_instance.jobs[transporting_action]

        # Check whether the job has been finished
        if current_job.job_status == 3:
            raise Exception(f"Job {transporting_action} has been finished!")

        # Check whether the job has assigned to a machine
        if current_job.assigned_machine is None:
            raise Exception(f"Job {transporting_action} hasn't assigned to a machine, so the target is None!")

        # Check whether the job needs transportation
        if current_job.job_status == 2:  # if the job is in transporting, this transbot don't consider it
            raise Exception(f"Job {transporting_action} is in transporting!")
        job_location_index = self.factory_instance.factory_graph.location_index_map[
            current_job.current_location]
        job_to_machine = self.factory_instance.factory_graph.unload_transport_time_matrix[
            job_location_index, current_job.assigned_machine  # machine_location_index == machine_index
        ]
        if job_to_machine == 0:  # the job is at its destination and doesn't need to be transported
            raise Exception(f"Job {transporting_action} doesn't need to be transported!")

        # Check whether the job has already assigned to another transbot
        if current_job.assigned_transbot is not None:
            raise Exception(f"Job {transporting_action} has already assigned to another transbot!")

    def _check_transbot_charging_action(self, transbot_index, charging_action):
        current_transbot = self.factory_instance.agv[transbot_index]
        # Check what status is the transbot currently in,
        # must be 0 (idling), 1 (unload trans) or 4 (low battery) to continue
        if current_transbot.agv_status != 0 and current_transbot.agv_status != 1 and current_transbot.agv_status != 4:
            raise Exception(f"...!")

    def _check_transbot_finish_charging(self, transbot_id):
        current_transbot = self.factory_instance.agv[transbot_id]
        if current_transbot.charging_time <= 0:
            return True
        else:
            return False

    def _check_done(self):
        if all(job.is_completed_for_current_time_window for job in self.scheduling_instance.jobs):
            return True
        else:
            return False


# Example usage
if __name__ == "__main__":
    import os
    import sys
    import numpy as np

    func("main function begin.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir) + "/InterfaceWithGlobal")

    with open(os.path.dirname(current_dir) + "/InterfaceWithGlobal/local_schedules_J6M4T2I0/local_schedule_window_0.pkl", "rb") as file:
        local_schedule = pickle.load(file)

    print(vars(local_schedule))

    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_jobs": dfjspt_params.n_jobs,
        "n_transbots": dfjspt_params.n_transbots,
        "local_schedule": local_schedule
    }

    scheduling_env = LocalSchedulingMultiAgentEnv(config)

    func("Env instance created.")

    num_episodes = 5

    for episode in range(num_episodes):
        print(f"\nStarting episode {episode + 1}")
        observations, infos = scheduling_env.reset()
        done = False
        total_reward = 0

        while not done:
            actions = {}
            for agent_id, obs in observations.items():
                action_mask = obs['action_mask']
                valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
                if valid_actions:
                    actions[agent_id] = np.random.choice(valid_actions)
                else:
                    raise Exception(f"No valid actions for agent {agent_id}!")
                    # actions[agent_id] = 0  # Default to a no-op if no valid actions

            observations, rewards, done, truncated, info = scheduling_env.step(actions)

            for reward in rewards.values():
                total_reward += reward

            print(f"Actions: {actions}")
            print(f"Rewards: {rewards}")
            print(f"Done: {done}")

        print(f"Total reward for episode {episode + 1}: {total_reward}")

    func("Local Scheduling completed.")




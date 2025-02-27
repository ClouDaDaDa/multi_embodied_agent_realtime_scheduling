# from memory_profiler import profile
# @profile
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import matplotlib.cm as cm
from local_realtime_scheduling.Environment.ExecutionResult import LocalResult, Local_Job_result, Operation_result
from local_realtime_scheduling.Environment.path_planning import a_star_search
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, Local_Job_schedule
# from global_scheduling.InterfaceWithLocal.convert_schedule_to_class import convert_schedule_to_class

def func(content: str):
    print(content)


import random
from gymnasium import spaces
import numpy as np


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


class LocalSchedulingMultiAgentEnv(MultiAgentEnv):
    """
    A Multi-agent Environment for Integrated Production, Transportation and Maintenance Real-time Scheduling.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

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
        self.local_result_file = config["local_result_file"] if "local_result_file" in config else None

        # todo: read the FactoryInstance and SchedulingInstance from config
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

        self.max_operations = None
        self.MAX_MAINTENANCE_COUNTS = None
        self.initial_estimated_makespan = None
        self.time_upper_bound = None

        self.maintenance_cost_for_machines = np.zeros((self.num_machines,), dtype=np.float32)
        self.maintenance_counts_for_machines = np.zeros((self.num_machines,), dtype=np.int32)

        self.current_time_before_step = 0.0
        self.current_time_after_step = 0.0
        self.reward_this_step = 0.0
        self.total_cost = 0.0
        self.estimated_makespan = self.initial_estimated_makespan

        # Define agent IDs
        self.machine_agents = [f"machine{i}" for i in range(self.num_machines)]
        self.transbot_agents = [f"transbot{i}" for i in range(self.num_transbots)]
        self.agents = self.possible_agents = self.machine_agents + self.transbot_agents

        # Initialize the index lists
        self.machine_index = list(range(self.num_machines))
        self.transbot_index = list(range(self.num_transbots))
        self.machine_action_agent_count = 0
        self.transbot_action_agent_count = 0
        self.all_agents_have_made_decisions = False
        self.current_actor = None

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
            # observations:
            self.observation_spaces[machine_agent_id] = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.num_machine_actions,), dtype=np.int32),
                "observation": spaces.Dict({
                    "job_features": spaces.Box(low=-1.0, high=float('inf'), shape=(self.num_jobs, 5)),
                    "time_to_makespan": spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,)),
                    "machine_features": spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,)),
                }),
            })
            self.action_spaces[machine_agent_id] = spaces.Discrete(self.num_machine_actions)

        # Transbots: Observation space and action space
        for transbot_agent_id in self.transbot_agents:
            # possible actions:
            # [0,num_jobs-1] for start transportation, num_jobs for start charging, num_jobs+1 for do nothing
            self.num_transbot_actions = self.num_jobs + 2
            # observations:
            self.observation_spaces[transbot_agent_id] = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.num_transbot_actions,), dtype=np.int32),
                "observation": spaces.Dict({
                    "job_features": spaces.Box(low=-1.0, high=float('inf'), shape=(self.num_jobs, 5)),
                    "time_to_makespan": spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,)),
                    "transbot_features": spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,)),
                }),
            })
            self.action_spaces[transbot_agent_id] = spaces.Discrete(self.num_transbot_actions)

        # Rendering settings
        self.render_mode = config["render_mode"] if "render_mode" in config else None
        if self.render_mode in ["human", "rgb_array"]:
            self.fig = None
            self.ax = None

            # Adjust figure size dynamically based on the factory graph size
            self.scale_factor = 1.0  # Scale factor for dynamic figure sizing
            self.figsize = (self.factory_instance.factory_graph.width * self.scale_factor,
                            self.factory_instance.factory_graph.height * self.scale_factor)

            self.machine_color_ids = set(
                f"Machine {machine.machine_id}"
                for machine in self.factory_instance.machines
            )
            if self.num_machines <= 20:
                self.machine_colormap = plt.colormaps["tab20"]
            else:
                self.machine_colormap = cm.get_cmap("hsv", self.num_machines)
            self.machine_color_map = {resource: self.machine_colormap(i / self.num_machines) for i, resource in
                                       enumerate(self.machine_color_ids)}

            self.transbot_color_ids = set(
                f"Transbot {transbot.agv_id}"
                for transbot in self.factory_instance.agv
            )
            if self.num_transbots <= 12:
                self.transbot_colormap = plt.colormaps["Set3"]
            else:
                self.transbot_colormap = cm.get_cmap("Purples", self.num_transbots)
            self.transbot_color_map = {resource: self.transbot_colormap(i / self.num_transbots) for i, resource in
                                       enumerate(self.transbot_color_ids)}


    def _initialize_state(self, agent_id):
        obs = {}
        obs[agent_id] = self._get_machine_obs(agent_id)
        return obs

    def _get_info(self, agent_id):
        return {
            agent_id: {"current_action_agent": agent_id}
        }

    def _get_machine_obs(self, machine_agent_id):
        machine_index = int(machine_agent_id.lstrip("machine"))
        machine = self.factory_instance.machines[machine_index]

        # dynamic action masking (DAM) logic for the machine:
        # 1 for valid and 0 for invalid action
        machine_action_mask = np.zeros((self.num_machine_actions,), dtype=np.int32)
        # idling: can choose a job, or perform a maintenance except CM, or do nothing
        if machine.machine_status == 0:
            machine_action_mask[self.num_jobs:self.num_jobs+3] = 1
            machine_action_mask[self.num_jobs + 4] = 1
        elif machine.machine_status == 1 or machine.machine_status == 2:
        # processing or under maintenance: can choose a job, or do nothing
            machine_action_mask[self.num_jobs + 4] = 1
        elif machine.machine_status == 3:
        # faulty: can perform CM or do nothing
            machine_action_mask[self.num_jobs + 3:] = 1

        if machine.reliability >= 0.8:
            machine_action_mask[self.num_jobs:self.num_jobs+4] = 0

        job_features = -1 * np.ones((self.num_jobs, 5), dtype=np.float32)
        for job_id in range(self.num_jobs):
            job_features[job_id, 0] = job_id  # [0] job_id

            if job_id in self.local_schedule.jobs:  # the job is in this problem
                this_job = self.scheduling_instance.jobs[job_id]

                # if this_job.job_status != 3:  # the job still has operation to be processed
                if (this_job.num_processing_operations_for_current_time_window - this_job.current_processing_operation != 0):

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

                            if (machine.machine_status in (0, 1, 2)) and machine.current_processing_task is None:
                                machine_action_mask[job_id] = 1

        machine_features = np.array([
            # [0] machine status
            machine.machine_status,
            # [1] reliability
            machine.reliability,
            # [2] time to finish the current task
            machine.estimated_remaining_time_to_finish,
        ], dtype=np.float32)

        return {
            "action_mask": machine_action_mask,
            "observation": {
                "job_features": job_features,
                "time_to_makespan": np.array([self.initial_estimated_makespan - self.current_time_after_step], dtype=np.float32),
                "machine_features": machine_features,
            }
        }

    def _get_transbot_obs(self, transbot_agent_id):
        transbot_index = int(transbot_agent_id.lstrip("transbot"))
        transbot = self.factory_instance.agv[transbot_index]

        # dynamic action masking logic for the transbot: 1 for valid and 0 for invalid action
        transbot_action_mask = np.zeros((self.num_transbot_actions,), dtype=np.int32)
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

        if transbot.battery.soc >= 0.8:
            transbot_action_mask[self.num_jobs] = 0

        job_features = -1 * np.ones((self.num_jobs, 5), dtype=np.float32)
        for job_id in range(self.num_jobs):
            job_features[job_id, 0] = job_id  # [0] job_id

            if job_id in self.local_schedule.jobs:  # the job is in this problem
                this_job = self.scheduling_instance.jobs[job_id]

                if this_job.job_status != 3:  # the job still has operation to be processed

                    if this_job.assigned_machine is not None:  # the job's next pending operation has been assigned to a machine

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
        ], dtype=np.float32)

        return {
            "action_mask": transbot_action_mask,
            "observation": {
                "job_features": job_features,
                "time_to_makespan": np.array([self.initial_estimated_makespan - self.current_time_after_step], dtype=np.float32),
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
        self.time_upper_bound = self.initial_estimated_makespan * 2
        self.current_time_before_step = self.local_schedule.time_window_start
        self.current_time_after_step = self.local_schedule.time_window_start
        self.reward_this_step = 0.0
        self.total_cost = 0.0
        self.estimated_makespan = self.initial_estimated_makespan

        # Randomly shuffle the index lists
        random.shuffle(self.machine_index)
        random.shuffle(self.transbot_index)
        self.machine_action_agent_count = 0
        self.transbot_action_agent_count = 0
        self.all_agents_have_made_decisions = False

        self.local_result = LocalResult()
        self.local_result.time_window_start = self.current_time_before_step

        # Reset all machines and jobs
        for machine in self.factory_instance.machines:
            # machine.reset_machine_for_current_time_window()
            machine.reset_machine()

        for agv in self.factory_instance.agv:
            # agv.reset_agv_for_current_time_window()
            agv.reset_agv()

        for job in self.scheduling_instance.jobs:
            # job.reset_job_for_current_time_window()
            job.reset_job()

        for job_id in self.local_schedule.jobs:
            self.local_result.add_job_result(Local_Job_result(job_id=job_id))
            local_schedule_job = self.local_schedule.jobs[job_id]
            this_job = self.scheduling_instance.jobs[job_id]
            this_job.available_time_for_current_time_window = local_schedule_job.available_time
            this_job.estimated_finish_time_for_current_time_window = local_schedule_job.estimated_finish_time
            this_job.is_completed_for_current_time_window = False

            for operation_id in local_schedule_job.operations:
                local_schedule_job_operation = local_schedule_job.operations[operation_id]

                if local_schedule_job_operation.type == "Processing":
                    self.local_result.jobs[job_id].add_operation_result(Operation_result(
                        job_id=job_id,
                        operation_id=operation_id,
                        # actual_start_transporting_time=None,
                        # actual_finish_transporting_time=None,
                        # assigned_transbot=None,
                        # actual_start_processing_time=None,
                        # actual_finish_processing_time=None,
                        # assigned_machine=None,
                    ))
                    this_job.processing_operations_for_current_time_window.append(int(operation_id))
                    self.factory_instance.machines[local_schedule_job_operation.machine_assigned].processing_tasks_queue_for_current_time_window.append((job_id, operation_id))

            if len(this_job.processing_operations_for_current_time_window) > 0:
                this_job.current_processing_operation = this_job.processing_operations_for_current_time_window[0]
                this_job.num_processing_operations_for_current_time_window = len(this_job.processing_operations_for_current_time_window)
                this_job.job_progress_for_current_time_window = 0.0



        self.current_actor = self.machine_agents[self.machine_index[self.machine_action_agent_count]]
        self.machine_action_agent_count += 1
        observations = self._initialize_state(agent_id=self.current_actor)
        infos = self._get_info(agent_id=self.current_actor)

        return observations, infos

    def step(self, action_dict):
        observations, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}
        self.reward_this_step = 0.0
        self.resetted = False

        # print(action_dict)
        if len(action_dict) != 1:  # Only one agent make action in one step()
            raise ValueError("The dict must contain exactly one key-value pair!")

        # for the agent that take an action at this step:
        for agent_id, action in action_dict.items():

            if agent_id != self.current_actor:
                raise ValueError(f"The agent_id {agent_id} mismatch current_actor{self.current_actor}!")

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

                    current_machine.current_maintenance_method = maintenance_method

                # do-nothing
                elif action == self.num_jobs + 4:
                    pass
                else:
                    raise Exception(f"Invalid action ({action}) for machine {machine_index}!")

                if self.machine_action_agent_count < self.num_machines:
                    self.current_actor = self.machine_agents[self.machine_index[self.machine_action_agent_count]]
                    self.machine_action_agent_count += 1
                    # observations[self.current_actor] = self._get_machine_obs(machine_agent_id=self.current_actor)

                else:
                    self.machine_action_agent_count = 0
                    if self.transbot_action_agent_count != 0:
                        raise ValueError(f"transbot_action_agent_count ({self.transbot_action_agent_count}) is not 0!")
                    self.current_actor = self.transbot_agents[self.transbot_index[self.transbot_action_agent_count]]
                    self.transbot_action_agent_count += 1
                    # observations[self.current_actor] = self._get_transbot_obs(transbot_agent_id=self.current_actor)

                infos = self._get_info(agent_id=self.current_actor)


            elif agent_id.startswith("transbot"):
                transbot_index = int(agent_id.lstrip("transbot"))
                current_transbot = self.factory_instance.agv[transbot_index]

                # perform the transporting task
                if 0 <= action < self.num_jobs:
                    # Check the validity of the transporting action
                    self._check_transbot_transporting_action(transbot_index=transbot_index,
                                                             transporting_action=action)

                    # If transbot changes its decision, release the binding relationship with the previous job
                    if current_transbot.current_task is not None:
                        old_job = self.scheduling_instance.jobs[current_transbot.current_task]
                        old_job.assigned_transbot = None
                        current_transbot.scheduled_results.append(
                            ("Unload Transporting", current_transbot.current_task, self.current_time_after_step))

                        current_transbot.current_task = action
                        new_job = self.scheduling_instance.jobs[action]
                        new_job.assigned_to_transbot(transbot_index)

                        if current_transbot.agv_status == 1:
                            new_job_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                                new_job.current_location]

                            unload_path = a_star_search(
                                graph=self.factory_instance.factory_graph,
                                start=current_transbot.current_location,
                                goal=new_job_location
                            )
                            # print(
                            #     f"from {current_transbot.current_location} to {charging_station_location}, unload_path = {unload_path}")
                            current_transbot.start_unload_transporting(
                                target_location=new_job_location,
                                unload_path=unload_path,
                                start_time=self.current_time_after_step
                            )
                    else:
                        current_transbot.current_task = action
                        self.scheduling_instance.jobs[action].assigned_to_transbot(transbot_index)

                # (move for) charging
                elif action == self.num_jobs:
                    # Check the validity of the charging action
                    self._check_transbot_charging_action(transbot_index=transbot_index,
                                                         charging_action=action)

                    # If transbot changes its decision, release the binding relationship with the previous job
                    if current_transbot.current_task is not None:
                        this_job = self.scheduling_instance.jobs[current_transbot.current_task]
                        this_job.assigned_transbot = None

                    current_transbot.current_task = -1

                # do-nothing
                elif action == self.num_jobs + 1:
                    if current_transbot.agv_status == 4:
                        raise Exception(f"...!")

                else:
                    raise Exception(f"Invalid action ({action}) for transbot {transbot_index}!")

                if self.transbot_action_agent_count < self.num_transbots:
                    self.current_actor = self.transbot_agents[self.transbot_index[self.transbot_action_agent_count]]
                    self.transbot_action_agent_count += 1
                    # observations[self.current_actor] = self._get_transbot_obs(transbot_agent_id=self.current_actor)

                else:
                    self.transbot_action_agent_count = 0
                    self.all_agents_have_made_decisions = True
                    if self.machine_action_agent_count != 0:
                        raise ValueError(f"machine_action_agent_count ({self.machine_action_agent_count}) is not 0!")
                    self.current_actor = self.machine_agents[self.machine_index[self.machine_action_agent_count]]
                    self.machine_action_agent_count += 1
                    # observations[self.current_actor] = self._get_machine_obs(machine_agent_id=self.current_actor)

                infos = self._get_info(agent_id=self.current_actor)


        if self.all_agents_have_made_decisions:
            # Environment moves forward for 1 time step

            # step the env according to actions:
            self.current_time_before_step = self.current_time_after_step
            self.current_time_after_step += 1.0

            # if self.current_time_after_step > 600:
            #     print("too many steps!")

            # if self.current_time_before_step == 0:
            #     self.reward_this_step += 1.0
            # self.reward_this_step -= self.current_time_after_step / self.initial_estimated_makespan
            self.reward_this_step -= 1.0 / self.initial_estimated_makespan

            for machine_agent_id in self.machine_agents:
                # Each machine moves forward for 1 time step
                machine_index = int(machine_agent_id.lstrip("machine"))
                self._step_a_machine_for_one_step(machine_index=machine_index)

            for transbot_agent_id in self.transbot_agents:
                # Each transbot moves forward for 1 time step
                transbot_index = int(transbot_agent_id.lstrip("transbot"))
                self._step_a_transbot_for_one_step(transbot_index=transbot_index)

            for machine_agent_id in self.machine_agents:
                # Machines get new rewards:
                rewards[machine_agent_id] = self.reward_this_step
                terminated[machine_agent_id] = self._check_done()
                if terminated[machine_agent_id]:
                    self.terminateds.add(machine_agent_id)
                    rewards[machine_agent_id] += 1.0
                    observations[machine_agent_id] = self._get_machine_obs(machine_agent_id=machine_agent_id)
                truncated[machine_agent_id] = self._check_truncated()
                if truncated[machine_agent_id]:
                    self.truncateds.add(machine_agent_id)
                    observations[machine_agent_id] = self._get_machine_obs(machine_agent_id=machine_agent_id)

            for transbot_agent_id in self.transbot_agents:
                # Transbots get new rewards:
                rewards[transbot_agent_id] = self.reward_this_step
                terminated[transbot_agent_id] = self._check_done()
                if terminated[transbot_agent_id]:
                    self.terminateds.add(transbot_agent_id)
                    rewards[transbot_agent_id] += 1.0
                    observations[transbot_agent_id] = self._get_transbot_obs(transbot_agent_id=transbot_agent_id)
                truncated[transbot_agent_id] = self._check_truncated()
                if truncated[transbot_agent_id]:
                    self.truncateds.add(transbot_agent_id)
                    observations[transbot_agent_id] = self._get_transbot_obs(transbot_agent_id=transbot_agent_id)

            # Randomly shuffle the index lists
            random.shuffle(self.machine_index)
            random.shuffle(self.transbot_index)
            # self.all_agents_have_made_decisions = False

        else:
            for machine_agent_id in self.machine_agents:
                rewards[machine_agent_id] = 0.0
            for transbot_agent_id in self.transbot_agents:
                rewards[transbot_agent_id] = 0.0

        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)

        if terminated["__all__"]:
            self.local_result.actual_local_makespan = self.current_time_after_step

            if self.local_result_file:
                os.makedirs(os.path.dirname(self.local_result_file), exist_ok=True)
                with open(self.local_result_file,
                          "wb") as file:
                    pickle.dump(self.local_result, file)

        if self.current_actor.startswith("machine"):
            observations[self.current_actor] = self._get_machine_obs(machine_agent_id=self.current_actor)
        elif self.current_actor.startswith("transbot"):
            observations[self.current_actor] = self._get_transbot_obs(transbot_agent_id=self.current_actor)

        return observations, rewards, terminated, truncated, infos

    def _step_a_machine_for_one_step(self, machine_index):
        current_machine = self.factory_instance.machines[machine_index]

        # Promote different evolutions according to different internal status
        # 0 (Idling): waiting for 1 time step
        if current_machine.machine_status == 0:
            # Checks whether it can start its scheduled task

            if current_machine.current_maintenance_method is not None:
                current_machine.start_maintenance(start_time=self.current_time_before_step)
                self.maintenance_cost_for_machines[machine_index] += self.maintenance_costs[
                    current_machine.current_maintenance_method]
                self.maintenance_counts_for_machines[machine_index] += 1

                current_machine.update_maintenance_process()

                # logging.info(f"Machine {machine_index} starts maintenance ({action - self.num_jobs}) at time {self.current_time_before_step}.")

            elif current_machine.current_processing_task is not None:
                current_job = self.scheduling_instance.jobs[current_machine.current_processing_task]
                # Check whether the job is currently processable
                if current_job.job_status == 0 and current_machine.machine_status == 0 and current_job.current_location == current_machine.location:
                    # Get the processing time of the operation on machine
                    processing_duration = current_job.operations_matrix[
                        current_job.current_processing_operation, machine_index]
                    estimated_processing_duration = int(processing_duration / current_machine.reliability)
                    noise = random.randint(-5, 5)
                    actual_processing_duration = max(0, estimated_processing_duration + noise)

                    # Update internal status of the job
                    # print(f"job{current_job.job_id},op{current_job.current_processing_operation},machine{current_machine.machine_id},time{self.current_time_before_step}")
                    self.local_result.jobs[current_job.job_id].operations[
                        current_job.current_processing_operation
                    ].actual_start_processing_time = self.current_time_before_step
                    self.local_result.jobs[current_job.job_id].operations[
                        current_job.current_processing_operation
                    ].assigned_machine = current_machine.machine_id
                    current_job.start_processing(start_time=self.current_time_before_step,
                                                 estimated_duration=estimated_processing_duration)

                    # Update internal status of the machine
                    current_machine.start_processing(start_time=self.current_time_before_step,
                                                     estimated_processing_duration=estimated_processing_duration,
                                                     actual_processing_duration=actual_processing_duration)
                    # logging.info(
                    #     f"Machine {machine_index} starts processing Job {current_job.job_id} at time {self.current_time_before_step}.")
                    current_machine.update_degradation_process()

                else:
                    current_machine.update_waiting_process()

            else:
                current_machine.update_waiting_process()

        # 1 (Processing): degrading for 1 time step
        elif current_machine.machine_status == 1:
            current_job = self.scheduling_instance.jobs[current_machine.current_processing_task]
            current_job.update_processing()
            current_machine.update_degradation_process()

            if self._check_machine_finish_task(machine_id=machine_index):
                current_machine.finish_processing(finish_time=self.current_time_after_step)
                current_job.finish_processing(finish_time=self.current_time_after_step)
                self.local_result.jobs[current_job.job_id].operations[
                    current_job.current_processing_operation - 1
                ].actual_finish_processing_time = self.current_time_after_step

        # 2 (Maintenance): maintaining for 1 time step
        elif current_machine.machine_status == 2:
            current_machine.update_maintenance_process()

            if self._check_machine_finish_task(machine_id=machine_index):
                current_machine.finish_maintenance(finish_time=self.current_time_after_step)

        # 3 (Faulty): broken for 1 time step
        elif current_machine.machine_status == 3:
            current_machine.update_waiting_process()

    def _step_a_transbot_for_one_step(self, transbot_index):
        current_transbot = self.factory_instance.agv[transbot_index]

        # Promote different evolutions according to different internal status
        # 0 (Idling): waiting for 1 time step
        if current_transbot.agv_status == 0:
            # Checks whether it can start its scheduled task
            if current_transbot.current_task is not None:
                # for charging
                if current_transbot.current_task == -1:
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
                    # print(
                    #     f"from {current_transbot.current_location} to {charging_station_location}, unload_path = {unload_path}")
                    current_transbot.start_unload_transporting(
                        target_location=charging_station_location,
                        unload_path=unload_path,
                        start_time=self.current_time_before_step
                    )

                    self._handle_transbot_unload_move(current_transbot=current_transbot)

                # for transporting
                else:
                    current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                    job_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                        current_job.current_location]
                    # print(f"Transbot {transbot_index}: (job {current_job.job_id})'s location = {current_job.current_location}")
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
                            # print(f"from {job_location} to {machine_location}, loaded_path = {loaded_path}")
                            current_transbot.start_loaded_transporting(
                                target_location=machine_location,
                                loaded_path=loaded_path,
                                start_time=self.current_time_before_step
                            )
                            # Update internal status of the job
                            # print(
                            #     f"job{current_job.job_id},op{current_job.current_processing_operation},transbot{current_transbot.agv_id},time{self.current_time_before_step}")
                            self.local_result.jobs[current_job.job_id].operations[
                                current_job.current_processing_operation
                            ].actual_start_transporting_time = self.current_time_before_step
                            self.local_result.jobs[current_job.job_id].operations[
                                current_job.current_processing_operation
                            ].assigned_transbot = current_transbot.agv_id
                            current_job.start_transporting(start_time=self.current_time_before_step,
                                                           estimated_duration=len(loaded_path) - 1)

                            self._handle_transbot_loaded_move(current_transbot=current_transbot)

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
                            # print(f"from {current_transbot.current_location} to {job_location}, unload_path = {unload_path}")
                            current_transbot.start_unload_transporting(
                                target_location=job_location,
                                unload_path=unload_path,
                                start_time=self.current_time_before_step
                            )

                            self._handle_transbot_unload_move(current_transbot=current_transbot)

                        else:
                            current_transbot.idling_process()
            else:
                current_transbot.idling_process()

        # 1 (Unload Transporting): moving for 1 time step
        elif current_transbot.agv_status == 1:
            # If the congestion exceeds 10.0, an error message will be displayed
            if current_transbot.congestion_time >= 10.0:
                # raise Exception(f"Transbot {transbot_index} has been congested for 10 time steps!")
                if len(current_transbot.unload_path) > 0:
                    if current_transbot.is_for_charging:  # go to a charging station
                        goal = self.factory_instance.factory_graph.pickup_dropoff_points[
                            self.factory_instance.factory_graph.nearest_charging_station(
                                current_transbot.current_location
                            )
                        ]
                    else:  # go to a job
                        goal = self.factory_instance.factory_graph.pickup_dropoff_points[
                            self.scheduling_instance.jobs[
                                current_transbot.current_task
                            ].current_location
                        ]
                    replan_path = a_star_search(
                        graph=self.factory_instance.factory_graph,
                        start=current_transbot.current_location,
                        goal=goal,
                    )
                    current_transbot.unload_path = replan_path
                else:
                    pass

            self._handle_transbot_unload_move(current_transbot=current_transbot)

        # 2 (Loaded Transporting): moving for 1 time step
        elif current_transbot.agv_status == 2:
            # If the congestion exceeds 10.0, an error message will be displayed
            if current_transbot.congestion_time >= 10.0:
                # raise Exception(f"Transbot {transbot_index} has been congested for 10 time steps!")
                if len(current_transbot.loaded_path) > 0:
                    goal = self.factory_instance.factory_graph.pickup_dropoff_points[
                        self.factory_instance.machines[
                            self.scheduling_instance.jobs[
                                current_transbot.current_task
                            ].assigned_machine
                        ].location
                    ]
                    replan_path = a_star_search(
                        graph=self.factory_instance.factory_graph,
                        start=current_transbot.current_location,
                        goal=goal,
                    )
                    current_transbot.loaded_path = replan_path
                else:
                    pass

            self._handle_transbot_loaded_move(current_transbot=current_transbot)

        # 3 (Charging): charging for 1 time step
        elif current_transbot.agv_status == 3:
            current_transbot.update_charging_process()
            self._check_transbot_finish_charging(transbot_id=transbot_index)

        # 4 (Low battery):
        elif current_transbot.agv_status == 4:
            if current_transbot.current_task != -1:
                raise ValueError(f"The transbot {current_transbot.agv_id} should go to charge!")
            else:
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
                # print(
                #     f"from {current_transbot.current_location} to {charging_station_location}, unload_path = {unload_path}")
                current_transbot.start_unload_transporting(
                    target_location=charging_station_location,
                    unload_path=unload_path,
                    start_time=self.current_time_before_step
                )

                self._handle_transbot_unload_move(current_transbot=current_transbot)

    def _transbot_move_to_the_next_location(self, current_transbot, next_location, load: int):
        current_transbot.congestion_time = 0.0
        direction = (next_location[0] - current_transbot.current_location[0],
                     next_location[1] - current_transbot.current_location[1])
        # Mark the old location of the transbot as walkable
        self.factory_instance.factory_graph.set_walkable(location=current_transbot.current_location)
        # transbot move for one step
        current_transbot.moving_one_step(direction=direction, load=load)
        if load > 0:
            current_job = self.scheduling_instance.jobs[current_transbot.current_task]
            current_job.update_transporting(current_location=current_transbot.current_location)

    def _move_to_another_walkable_location(self, current_transbot, next_location, load: int):
        walkable_next_direction = self.factory_instance.factory_graph.check_adjacent_positions_walkable(
            current_location=current_transbot.current_location,
            occupied_location=next_location
        )
        if walkable_next_direction is None:
            # waiting
            current_transbot.congestion_time += 1.0
            current_transbot.moving_one_step(direction=(0, 0), load=load)
            if load > 0:
                current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                current_job.update_transporting(current_location=current_transbot.current_location)
        else:
            current_transbot.congestion_time = 0.0
            current_transbot.loaded_path.insert(0, current_transbot.current_location)
            # Mark the old location of the transbot as walkable
            self.factory_instance.factory_graph.set_walkable(location=current_transbot.current_location)
            # transbot move for one step
            current_transbot.moving_one_step(direction=walkable_next_direction, load=load)
            if load > 0:
                current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                current_job.update_transporting(current_location=current_transbot.current_location)
            # Mark the new position of the transbot as an obstacle
            self.factory_instance.factory_graph.set_obstacle(location=current_transbot.current_location)

    def _check_walkable_path(self, current_transbot):
        if len(current_transbot.unload_path) == 0:
            # raise ValueError(f"Transbot {current_transbot.agv_id}'s unload path is empty!")
            # Try to re-plan a walkable path
            start = current_transbot.current_location
            if current_transbot.agv_status == 1:  # Unload transporting
                if current_transbot.is_for_charging:  # go to a charging station
                    goal = self.factory_instance.factory_graph.pickup_dropoff_points[
                        self.factory_instance.factory_graph.nearest_charging_station(
                            current_transbot.current_location
                        )
                    ]
                else:  # go to a job
                    goal = self.factory_instance.factory_graph.pickup_dropoff_points[
                        self.scheduling_instance.jobs[
                            current_transbot.current_task
                        ].current_location
                    ]
            elif current_transbot.agv_status == 2:  # Loaded transporting
                goal = self.factory_instance.factory_graph.pickup_dropoff_points[
                            self.factory_instance.machines[
                                self.scheduling_instance.jobs[
                                    current_transbot.current_task
                                ].assigned_machine
                            ].location
                        ]
            else:
                raise ValueError(f"Incorrect transbot status {current_transbot.agv_status}!")

            replan_path = a_star_search(
                graph=self.factory_instance.factory_graph,
                start=start,
                goal=goal,
            )
            if len(replan_path) > 0:
                if current_transbot.agv_status == 1:
                    current_transbot.unload_path = replan_path
                elif current_transbot.agv_status == 2:
                    current_transbot.loaded_path = replan_path
                return True

            else:
                return False

        else:
            return True

    def _handle_transbot_unload_move(self, current_transbot):
        # Check whether current_transbot has a walkable path
        if self._check_walkable_path(current_transbot=current_transbot):

            # Check if next_location is walkable
            next_location = current_transbot.unload_path[0]
            if self.factory_instance.factory_graph.is_walkable(x=next_location[0], y=next_location[1]):
                del current_transbot.unload_path[0]

                self._transbot_move_to_the_next_location(current_transbot=current_transbot,
                                                         next_location=next_location,
                                                         load=0)

                if self._check_transbot_finish_transporting(transbot_id=current_transbot.agv_id):
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
                                f"Transbot {current_transbot.agv_id} hasn't get job {current_transbot.current_task}!")
                        if current_job.job_status == 0:
                            # Loaded transport can start immediately
                            loaded_path = a_star_search(
                                graph=self.factory_instance.factory_graph,
                                start=job_location,
                                goal=machine_location
                            )
                            # print(f"from {job_location} to {machine_location}, loaded_path = {loaded_path}")
                            current_transbot.start_loaded_transporting(
                                target_location=machine_location,
                                loaded_path=loaded_path,
                                start_time=self.current_time_after_step
                            )
                            # Update internal status of the job
                            # print(
                            #     f"job{current_job.job_id},op{current_job.current_processing_operation},transbot{current_transbot.agv_id},time{self.current_time_after_step}")
                            self.local_result.jobs[current_job.job_id].operations[
                                current_job.current_processing_operation
                            ].actual_start_transporting_time = self.current_time_before_step
                            self.local_result.jobs[current_job.job_id].operations[
                                current_job.current_processing_operation
                            ].assigned_transbot = current_transbot.agv_id
                            current_job.start_transporting(start_time=self.current_time_after_step,
                                                           estimated_duration=len(loaded_path) - 1)
                    elif current_transbot.current_task == -1:
                        current_transbot.start_charging(start_time=self.current_time_after_step)
                else:
                    # Mark the new position of the transbot as an obstacle
                    self.factory_instance.factory_graph.set_obstacle(location=current_transbot.current_location)

            else:
                if current_transbot.congestion_time > 1:
                    # Randomly move in other passable directions
                    self._move_to_another_walkable_location(current_transbot=current_transbot,
                                                            next_location=next_location,
                                                            load=0)
                else:
                    # waiting
                    current_transbot.congestion_time += 1.0
                    current_transbot.moving_one_step(direction=(0, 0), load=0)
        else:
            # waiting
            current_transbot.congestion_time += 1.0
            current_transbot.moving_one_step(direction=(0, 0), load=0)

    def _handle_transbot_loaded_move(self, current_transbot):
        # Check whether current_transbot has a walkable path
        if self._check_walkable_path(current_transbot=current_transbot):
            # Check if next_location is walkable
            next_location = current_transbot.loaded_path[0]
            if self.factory_instance.factory_graph.is_walkable(x=next_location[0], y=next_location[1]):
                del current_transbot.loaded_path[0]

                self._transbot_move_to_the_next_location(current_transbot=current_transbot,
                                                         next_location=next_location,
                                                         load=1)

                if self._check_transbot_finish_transporting(transbot_id=current_transbot.agv_id):
                    current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                    if self.factory_instance.factory_graph.pickup_dropoff_points[
                        f"machine_{current_job.assigned_machine}"] != current_transbot.current_location:
                        raise ValueError(f"job's location mismatch machine's location!")
                    current_job.finish_transporting(finish_time=self.current_time_after_step,
                                                    current_location=f"machine_{current_job.assigned_machine}")
                    self.local_result.jobs[current_job.job_id].operations[
                        current_job.current_processing_operation
                    ].actual_finish_transporting_time = self.current_time_after_step
                    current_transbot.finish_loaded_transporting(finish_time=self.current_time_after_step)
                else:
                    # Mark the new position of the transbot as an obstacle
                    self.factory_instance.factory_graph.set_obstacle(location=current_transbot.current_location)

            else:
                if current_transbot.congestion_time > 1:
                    # Randomly move in other passable directions
                    self._move_to_another_walkable_location(current_transbot=current_transbot,
                                                            next_location=next_location,
                                                            load=1)
                else:
                    # waiting
                    current_transbot.congestion_time += 1.0
                    current_transbot.moving_one_step(direction=(0, 0), load=1)
                    current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                    current_job.update_transporting(current_location=current_transbot.current_location)
        else:
            # waiting
            current_transbot.congestion_time += 1.0
            current_transbot.moving_one_step(direction=(0, 0), load=1)
            current_job = self.scheduling_instance.jobs[current_transbot.current_task]
            current_job.update_transporting(current_location=current_transbot.current_location)



    def _check_machine_processing_action(self, machine_index, processing_action):
        # Check what status is the machine currently in, must be 0 (idling) or 1 or 2 to continue
        if self.factory_instance.machines[machine_index].machine_status not in (0, 1, 2):
            raise Exception(f"Only idling machine can choose a job!")

        # Check whether the job is in the current problem
        if processing_action not in self.local_schedule.jobs:  # the job is not in this problem
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
        if self.current_time_after_step >= current_machine.start_time_of_the_task + current_machine.actual_processing_duration:
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

    def _check_truncated(self):
        if self.current_time_after_step >= self.time_upper_bound:
            return True
        else:
            return False

    def render(self):
        """
        Render the factory layout for the current time step.
        This method visualizes the current state of the factory, including machines, transbots, and jobs.
        """

        if self.render_mode is None:
            logging.warning("You are calling render method without specifying any render mode.")
            return None

        if self.render_mode not in ["human", "rgb_array"]:
            raise NotImplementedError(f"Render mode '{self.render_mode}' is not supported.")

        if self.resetted:
            plt.ion()  # Enable interactive mode
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            self.ax.set_xlim(-1, self.factory_instance.factory_graph.width + 1)
            self.ax.set_ylim(-1, self.factory_instance.factory_graph.height + 1)
            # self.plot_size = 100 * (self.factory_instance.factory_graph.width + self.factory_instance.factory_graph.height) / 2
            # self.plot_size = 25 * np.pi
            self.ax.set_xlabel("X Position")
            self.ax.set_ylabel("Y Position")
            x_ticks = range(-1, self.factory_instance.factory_graph.width + 1, 1)
            y_ticks = range(-1, self.factory_instance.factory_graph.height + 1, 1)
            self.ax.set_xticks(x_ticks)
            self.ax.set_yticks(y_ticks)
            self.ax.set_aspect('equal', adjustable='box')
            # self.ax.grid(True, linestyle='--', linewidth=0.5)
            self.fig.subplots_adjust(right=0.6)

            # # Initialize lists for each category's handles and labels
            # self.job_handles, self.job_labels = [], []
            # self.machine_handles, self.machine_labels = [], []
            # self.transbot_handles, self.transbot_labels = [], []

        # Update dynamic elements: transbots and jobs
        if self.all_agents_have_made_decisions or self.resetted:
            self._update_dynamic_elements()

        if self.render_mode == "human":
            # Redraw and pause for real-time display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(1 / self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            raise NotImplementedError(f"Render mode '{self.render_mode}' is not supported.")

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            plt.ioff()


    def _plot_static_elements(self):
        """
        Plot static elements: obstacles, pickup/dropoff points (machines, charging stations, warehouse).
        """
        # Plot obstacles
        for x, y in self.factory_instance.factory_graph.obstacles:
            self.ax.add_patch(pch.Rectangle((x - 0.5, y - 0.5), 1, 1, color='dimgray'))
        # for x in range(self.factory_instance.factory_graph.width):
        #     for y in range(self.factory_instance.factory_graph.height):
        #         if not self.factory_instance.factory_graph.is_walkable(x, y):
        #             self.ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='dimgray'))

        # Plot pickup/dropoff points (machines, charging stations, warehouse)
        for point_name, location in self.factory_instance.factory_graph.pickup_dropoff_points.items():
            x, y = location
            if point_name == "warehouse" or "charging" in point_name:
                self.ax.add_patch(pch.Circle((x, y), radius=0.45, alpha=0.6, color='lightgray', label=f'{point_name}'))
                # self.ax.scatter(x, y, color='lightgray', alpha=0.6, s=self.pickup_dropoff_size, label=f'{point_name}')
            elif "machine" in point_name:
                machine_id = int(point_name.split('_')[-1])
                machine_color = self.machine_color_map[f"Machine {machine_id}"]
                machine_handle = self.ax.add_patch(pch.Circle((x, y), radius=0.45, alpha=0.6, color=machine_color,
                                             label=f'Machine {machine_id} ({x}, {y})'))
                # self.machine_handles.append(machine_handle)
                # self.machine_labels.append(f'Machine {machine_id} ({x}, {y})')
            # if "machine" in point_name or "charging" in point_name or point_name == "warehouse":
            #     self.ax.scatter(x, y, color='lightgray', alpha=0.6, s=600, label=f'{point_name}')

    def _update_dynamic_elements(self):
        """
        Update and plot dynamic elements: transbots and jobs.
        This method clears the dynamic elements and redraws them for the current time step.
        """
        # Clear previous dynamic elements
        self.ax.cla()
        # self.ax.grid(True)
        x_ticks = range(-1, self.factory_instance.factory_graph.width + 1, 1)
        y_ticks = range(-1, self.factory_instance.factory_graph.height + 1, 1)
        self.ax.set_xticks(x_ticks)
        self.ax.set_yticks(y_ticks)
        self.ax.grid(True, linestyle='--', linewidth=0.5)

        # Replot the static elements (obstacles, pickup/dropoff points)
        self._plot_static_elements()

        # Display current time on the canvas
        current_time_text = f"Time: {self.current_time_after_step:.1f}"
        self.ax.text(0.95, 0.95, current_time_text, transform=self.ax.transAxes,
                     fontsize=12, color='black', ha='right', va='top', fontweight='normal',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.1'))

        # Plot transbots' current positions with dynamic updates
        for transbot in self.factory_instance.agv:
            x, y = transbot.current_location
            transbot_color = self.transbot_color_map[f"Transbot {transbot.agv_id}"]
            # color = np.random.rand(3,)  # Random color for each transbot
            transbot_handle = self.ax.add_patch(pch.Circle((x, y), radius=0.3, color=transbot_color,
                                         label=f'Transbot {transbot.agv_id} ({x}, {y})'))
            # self.transbot_handles.append(transbot_handle)
            # self.transbot_labels.append(f'Transbot {transbot.agv_id} ({x}, {y})')

        # Plot jobs' current positions with dynamic updates
        for job in self.scheduling_instance.jobs:
            if job.moving_location is None:
                x, y = self.factory_instance.factory_graph.pickup_dropoff_points[job.current_location]
            else:
                x, y = job.moving_location

            job_handle = self.ax.add_patch(pch.RegularPolygon((x, y), 3, radius=0.15, color='yellow',
                                         label=f'Job {job.job_id} ({x}, {y})'))
            # self.job_handles.append(job_handle)
            # self.job_labels.append(f'Job {job.job_id} ({x}, {y})')

        # handles = self.job_handles + self.machine_handles + self.transbot_handles
        # labels = self.job_labels + self.machine_labels + self.transbot_labels

        # Remove duplicate labels from the legend (only show the first occurrence)
        handles, labels = self.ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if any(x in label for x in ['charging', 'warehouse']):
                continue
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)

        self.ax.legend(unique_handles, unique_labels, loc='upper left',
                       bbox_to_anchor=(1.01, 1), borderaxespad=0., ncol=3)

    #     """
    #     Render the current state of the environment, showing key features of machines,
    #     transbots, and jobs over time.
    #     """
    #
    #     time_steps = np.arange(0, self.current_time_after_step)
    #
    #     # machine_status
    #     machine_status_history = {machine_id: [] for machine_id in self.machine_agents}
    #     for machine_agent_id in self.machine_agents:
    #         machine_index = int(machine_agent_id.lstrip("machine"))
    #         current_machine = self.factory_instance.machines[machine_index]
    #         machine_status_history[machine_agent_id] = current_machine.status_history
    #
    #     # agv_status
    #     transbot_status_history = {transbot_id: [] for transbot_id in self.transbot_agents}
    #     for transbot_agent_id in self.transbot_agents:
    #         transbot_index = int(transbot_agent_id.lstrip("transbot"))
    #         current_transbot = self.factory_instance.agv[transbot_index]
    #         transbot_status_history[transbot_agent_id] = current_transbot.status_history
    #
    #     # battery.soc
    #     transbot_battery_soc_history = {transbot_id: [] for transbot_id in self.transbot_agents}
    #     for transbot_agent_id in self.transbot_agents:
    #         transbot_index = int(transbot_agent_id.lstrip("transbot"))
    #         current_transbot = self.factory_instance.agv[transbot_index]
    #         transbot_battery_soc_history[transbot_agent_id] = current_transbot.battery.soc_history
    #
    #     # job_status
    #     job_status_history = {job_id: [] for job_id in self.scheduling_instance.jobs}
    #     for job_id, job in self.scheduling_instance.jobs.items():
    #         job_status_history[job_id] = job.status_history
    #
    #     # job_progress_for_current_time_window
    #     job_progress_history = {job_id: [] for job_id in self.scheduling_instance.jobs}
    #     for job_id, job in self.scheduling_instance.jobs.items():
    #         job_progress_history[job_id] = job.progress_history
    #
    #     fig, axes = plt.subplots(3, 2, figsize=(15, 15))  # 创建 3x2 的子图网格
    #
    #     # 1. machine_status
    #     ax = axes[0, 0]
    #     for machine_agent_id, status_history in machine_status_history.items():
    #         ax.plot(time_steps[:len(status_history)], status_history, label=f'Machine {machine_agent_id}')
    #     ax.set_title('Machine Status')
    #     ax.set_xlabel('Time Step')
    #     ax.set_ylabel('Machine Status')
    #     ax.legend(loc='upper right')
    #
    #     # 2. transbot_status
    #     ax = axes[0, 1]
    #     for transbot_agent_id, status_history in transbot_status_history.items():
    #         ax.plot(time_steps[:len(status_history)], status_history, label=f'Transbot {transbot_agent_id}')
    #     ax.set_title('Transbot Status')
    #     ax.set_xlabel('Time Step')
    #     ax.set_ylabel('Transbot Status')
    #     ax.legend(loc='upper right')
    #
    #     # 3. battery.soc
    #     ax = axes[1, 0]
    #     for transbot_agent_id, battery_soc_history in transbot_battery_soc_history.items():
    #         ax.plot(time_steps[:len(battery_soc_history)], battery_soc_history, label=f'Transbot {transbot_agent_id}')
    #     ax.set_title('Transbot Battery SOC')
    #     ax.set_xlabel('Time Step')
    #     ax.set_ylabel('Battery SOC')
    #     ax.legend(loc='upper right')
    #
    #     # 4. job_status
    #     ax = axes[1, 1]
    #     for job_id, status_history in job_status_history.items():
    #         ax.plot(time_steps[:len(status_history)], status_history, label=f'Job {job_id}')
    #     ax.set_title('Job Status')
    #     ax.set_xlabel('Time Step')
    #     ax.set_ylabel('Job Status')
    #     ax.legend(loc='upper right')
    #
    #     # 5. job_progress_for_current_time_window
    #     ax = axes[2, 0]
    #     for job_id, progress_history in job_progress_history.items():
    #         ax.plot(time_steps[:len(progress_history)], progress_history, label=f'Job {job_id}')
    #     ax.set_title('Job Progress')
    #     ax.set_xlabel('Time Step')
    #     ax.set_ylabel('Progress (%)')
    #     ax.legend(loc='upper right')
    #
    #     plt.tight_layout()
    #     plt.show()
    #     plt.pause(0.1)


# Example usage
if __name__ == "__main__":

    func("main function begin.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_schedule_file = os.path.dirname(current_dir) + \
                          "/InterfaceWithGlobal/local_schedules/local_schedule_" + \
                          f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}" \
                          + f"I0_window{dfjspt_params.time_window_size}/window_{dfjspt_params.current_window}.pkl"

    result_file_name = os.path.dirname(current_dir) + \
                       "/local_results/local_result_" + \
                       f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}" \
                       + f"I0_window{dfjspt_params.time_window_size}/window_{dfjspt_params.current_window}.pkl"

    with open(local_schedule_file,
              "rb") as file:
        local_schedule = pickle.load(file)

    print(vars(local_schedule))

    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_jobs": dfjspt_params.n_jobs,
        "n_transbots": dfjspt_params.n_transbots,
        "local_schedule": local_schedule,
        # "local_result_file": result_file_name,
        "render_mode": "human",
    }

    scheduling_env = LocalSchedulingMultiAgentEnv(config)

    func("Env instance created.")

    num_episodes = 1

    for episode in range(num_episodes):
        print(f"\nStarting episode {episode + 1}")
        decision_count = 0
        observations, infos = scheduling_env.reset()
        scheduling_env.render()
        print(f"decision_count = {decision_count}")
        decision_count += 1
        done = {'__all__': False}
        truncated = {'__all__': False}
        total_rewards = {}
        for agent in scheduling_env.agents:
            total_rewards[agent] = 0.0

        while (not done['__all__']) and (not truncated['__all__']):
            actions = {}
            for agent_id, obs in observations.items():
                # print(f"current agent = {agent_id}")
                action_mask = obs['action_mask']
                valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
                if valid_actions:
                    actions[agent_id] = np.random.choice(valid_actions)
                else:
                    raise Exception(f"No valid actions for agent {agent_id}!")
                    # actions[agent_id] = 0  # Default to a no-op if no valid actions

            observations, rewards, done, truncated, info = scheduling_env.step(actions)
            scheduling_env.render()
            # print(f"decision_count = {decision_count}")
            decision_count += 1

            for agent, reward in rewards.items():
                total_rewards[agent] += reward

        scheduling_env.close()
            # print(f"Actions: {actions}")
            # print(f"Rewards: {rewards}")
            # print(f"Done: {done}")
        for job_id in range(scheduling_env.num_jobs):
            print(f"job {job_id}: {scheduling_env.scheduling_instance.jobs[job_id].scheduled_results}")
        for machine_id in range(scheduling_env.num_machines):
            print(f"machine {machine_id}: {scheduling_env.factory_instance.machines[machine_id].scheduled_results}")
        for transbot_id in range(scheduling_env.num_transbots):
            print(f"transbot {transbot_id}: {scheduling_env.factory_instance.agv[transbot_id].scheduled_results}")

        print(f"Actual makespan = {scheduling_env.current_time_after_step}")
        print(f"Estimated makespan = {scheduling_env.initial_estimated_makespan}")
        print(f"Total reward for episode {episode + 1}: {total_rewards}")

        func("Local Scheduling completed.")




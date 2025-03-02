from System.Battery import Battery


class AGV:
    def __init__(self,
                 agv_id: int,):
        # Static features of the transbot
        self.agv_id = agv_id
        self.battery = Battery()

        # Dynamic global features of the transbot
        self.agv_status = 0  # 0 for idling, 1 for unload moving, 2 for loaded moving, 3 for charging, 4 for low_battery
        self.is_failed = False
        self.has_a_job = False
        self.is_for_charging = False
        self.current_location = (0, 0)
        self.target_location = None
        self.current_task = None
        self.cumulative_total_time = 0.0  # True agv life, including all waiting, running and charging time
        self.cumulative_work_time = 0.0  # True agv working time, including all task running time (running time for charging not included)
        self.scheduled_results = []  # Store (task type, task id, start/end time) for all tasks
        self.congestion_time = 0.0
        self.charging_time = 0.0
        self.unload_path = []
        self.loaded_path = []
        self.estimated_remaining_time_to_finish = 0.0

        # Dynamic local features of the transbot
        self.transport_tasks_queue_for_current_time_window = []

    def is_low_battery(self):
        if self.battery.soc < self.battery.low_power_threshold:
            self.is_failed = True
            self.agv_status = 4

    def update_soc_history(self, time_point: float, soc: float):
        self.battery.soc_history[time_point] = soc

    def start_charging(self, start_time: float, target_soc=1.0):
        if start_time != self.cumulative_total_time:
            raise Exception(f"Error in time synchronization with the environment: {start_time} != {self.cumulative_total_time}")
        self.agv_status = 3
        self.is_failed = False
        # self.remaining_time_to_finish = self.charging_time
        self.battery.update_soc(max(target_soc - self.battery.soc, 0))
        self.charging_time = self.battery.charge_exponential(target_soc)
        self.estimated_remaining_time_to_finish = self.charging_time
        self.scheduled_results.append(("Charging", target_soc, start_time))

    def update_charging_process(self, charging_time=1.0):
        self.charging_time -= charging_time
        self.estimated_remaining_time_to_finish -= charging_time
        self.cumulative_total_time += charging_time

    def finish_charging(self, finish_time: float):
        if finish_time != self.cumulative_total_time:
            raise Exception(f"Error in time synchronization with the environment: {finish_time} != {self.cumulative_total_time}")
        self.agv_status = 0
        self.is_for_charging = False
        self.current_task = None
        # self.remaining_time_to_finish = 0.0
        self.estimated_remaining_time_to_finish = 0.0
        self.charging_time = 0.0
        self.scheduled_results.append((self.scheduled_results[-1][:2] + (finish_time,)))
        self.update_soc_history(self.cumulative_total_time, self.battery.soc)

    def idling_process(self, idling_time=1.0):
        soc_change = self.battery.discharge_idling(idling_time)
        self.battery.update_soc(-soc_change)
        self.is_low_battery()
        self.cumulative_total_time += idling_time
        self.update_soc_history(self.cumulative_total_time, self.battery.soc)

    # def moving_process(self, start_time, moving_time, job_id, load=0):
    #     soc_change = self.battery.discharge_moving(moving_time, load)
    #     self.battery.update_soc(-soc_change)
    #     # self.is_low_battery()
    #     self.cumulative_total_time += moving_time
    #     self.check_reach_target()
    #     self.scheduled_results.append((2, job_id, start_time, self.cumulative_total_time))
    #     self.update_soc_history(self.cumulative_total_time, self.battery.soc)

    def start_unload_transporting(self,
                                  target_location: tuple,
                                  unload_path: list,
                                  start_time: float):
        if start_time != self.cumulative_total_time:
            raise Exception(f"Error in time synchronization with the environment: {start_time} != {self.cumulative_total_time}")
        # if self.current_location != unload_path.pop(0):
        #     raise Exception(f"Transbot is not currently at the starting point of the planned path!")
        self.agv_status = 1
        if self.current_task >= 0:
            pass
        elif self.current_task == -1:
            self.is_for_charging = True
        else:
            raise Exception(f"Invalid task {self.current_task}! It must be -1 for charging or 0-jobs for transporting.")
        self.target_location = target_location
        self.unload_path = unload_path
        self.estimated_remaining_time_to_finish = len(unload_path)
        self.scheduled_results.append(("Unload Transporting", self.current_task, start_time))
        # print(f"Transbot {self.agv_id} is going to {self.target_location} for task {self.current_task}, from {self.current_location}")

    def finish_unload_transporting(self, finish_time: float):
        if finish_time != self.cumulative_total_time:
            raise Exception(f"Error in time synchronization with the environment: {finish_time} != {self.cumulative_total_time}")
        self.agv_status = 0
        self.unload_path = []
        self.scheduled_results.append((self.scheduled_results[-1][:2] + (finish_time,)))

    def start_loaded_transporting(self,
                                  target_location: tuple,
                                  loaded_path: list,
                                  start_time: float):
        if start_time != self.cumulative_total_time:
            raise Exception(f"Error in time synchronization with the environment: {start_time} != {self.cumulative_total_time}")
        # if self.current_location != loaded_path.pop(0):
        #     raise Exception(f"Transbot is not currently at the starting point of the planned path!")
        if self.current_location == target_location:
            raise Exception(f"Transbot {self.agv_id} is already in target {target_location}!")
        self.agv_status = 2
        self.has_a_job = True
        self.target_location = target_location
        self.loaded_path = loaded_path
        self.estimated_remaining_time_to_finish = len(loaded_path)
        self.scheduled_results.append(("Loaded Transporting", self.current_task, start_time))
        # print(
        #     f"Transbot {self.agv_id} is going to {self.target_location} with job {self.current_task}, from {self.current_location}")

    def finish_loaded_transporting(self, finish_time: float):
        if finish_time != self.cumulative_total_time:
            raise Exception(f"Error in time synchronization with the environment: {finish_time} != {self.cumulative_total_time}")
        self.agv_status = 0
        self.has_a_job = False
        self.current_task = None
        self.loaded_path = []
        self.is_low_battery()
        self.scheduled_results.append((self.scheduled_results[-1][:2] + (finish_time,)))

    def moving_one_step(self, direction: tuple, load: int, moving_time=1.0):
        # # Define direction map
        # direction_map = {
        #     0: (1, 0),  # Right
        #     1: (-1, 0),  # Left
        #     2: (0, 1),  # Up
        #     3: (0, -1),  # Down
        #     4: (0, 0)  # Stay
        # }

        # Update current_location based on direction
        self.current_location = (self.current_location[0] + direction[0], self.current_location[1] + direction[1])

        soc_change = self.battery.discharge_moving(moving_time, load=load)
        self.battery.update_soc(-soc_change)
        self.cumulative_total_time += moving_time
        self.estimated_remaining_time_to_finish -= moving_time

        # self.check_reach_target()
        self.update_soc_history(self.cumulative_total_time, self.battery.soc)

    # def check_reach_target(self):
    #     """
    #     Check if the AGV has reached its target location and update its status accordingly.
    #     """
    #     if self.current_location == self.target_location:
    #         if self.current_task >= 0:
    #             self.agv_status = 0  # Set to idling
    #             self.is_low_battery()
    #         elif self.current_task == -1:
    #             self.agv_status = 3  # Set to charging
    #         else:
    #             raise Exception(f"...!")
    #
    #         self.target_location = None
    #         # self.current_transport_task = None
    #         self.unload_path = []
    #         self.loaded_path = []

    def reset_agv(self):
        self.battery.reset_battery()
        # Dynamic global features of the transbot
        self.agv_status = 0  # 0 for idling, 1 for unload moving, 2 for loaded moving, 3 for charging, 4 for low_battery
        self.is_failed = False
        self.has_a_job = False
        self.is_for_charging = False
        self.current_location = (0, 0)
        self.target_location = None
        self.current_task = None
        self.cumulative_total_time = 0.0  # True agv life, including all waiting, running and charging time
        self.cumulative_work_time = 0.0  # True agv working time, including all task running time (running time for charging not included)
        self.scheduled_results = []  # Store (task type, task id, start/end time) for all tasks
        self.congestion_time = 0.0
        self.charging_time = 0.0
        self.unload_path = []
        self.loaded_path = []
        self.estimated_remaining_time_to_finish = 0.0

        self.reset_agv_for_current_time_window()

    def reset_agv_for_current_time_window(self):
        # Dynamic local features of the transbot
        self.transport_tasks_queue_for_current_time_window = []


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Initialize AGV
    agv = AGV(agv_id=1)

    # Simulate AGV activities
    # time_history, soc_history = simulate_agv(battery, activities)
    soc_history = []
    time_history = []
    total_time = 0

    # # Example activities
    # activities = [
    #     {'action': 'moving', 'time': 30, 'load': 0},
    #     {'action': 'idling', 'time': 10},
    #     {'action': 'pickdrop', 'load': 80},
    #     {'action': 'moving', 'time': 30, 'load': 80},
    #     # {'action': 'charge_linear', 'load': 0.8},  # Charging to 80% SOC
    #     {'action': 'idling', 'time': 10},
    #     {'action': 'moving', 'time': 50, 'load': 0},
    #     {'action': 'moving', 'time': 20, 'load': 60},
    #     {'action': 'pickdrop', 'load': 60},
    #     {'action': 'charge_exponential', 'target_soc': 0.8},  # Charging to 100% SOC
    #     {'action': 'idling', 'time': 20},
    #     {'action': 'moving', 'time': 30, 'load': 0},
    #     {'action': 'moving', 'time': 20, 'load': 20},
    #     {'action': 'pickdrop', 'load': 20},
    #     {'action': 'idling', 'time': 50},
    # ]

    # for activity in activities:

    #     total_time += agv.perform_activity(activity)

    #     soc_history.append(agv.battery.soc)
    #     time_history.append(total_time)

    # # Plot the results
    # plt.plot(time_history, soc_history, marker='o')
    # plt.xlabel('Time (minutes)')
    # plt.ylabel('State of Charge (SOC)')
    # plt.title('AGV Battery SOC over Time')
    # # plt.grid(True)
    # plt.show()

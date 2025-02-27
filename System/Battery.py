import numpy as np


class Battery:
    def __init__(self,
                 capacity=200.0,
                 soc=1.0,
                 low_power_threshold=0.3,
                 voltage=24.0,
                 charge_rate=100.0,
                 charge_efficiency=0.9,
                 self_discharge_rate=0.01,
                 temperature=25.0,
                 temp_coefficient=0.005
                 ):
        self.capacity = capacity  # Battery Capacity (Ah)
        self.soc = soc  # State of Charge in percentage (1.0 for 100%)
        self.low_power_threshold = low_power_threshold  # Low battery warning threshold (20%)
        self.voltage = voltage  # (V)
        self.charge_rate = charge_rate  # charge rate for linear charging model (A)
        self.charge_efficiency = charge_efficiency  # Charge efficiency
        self.self_discharge_rate = self_discharge_rate  # Self-discharge rate per hour
        self.temperature = temperature  # Ambient temperature (C)
        self.temp_coefficient = temp_coefficient  # Temperature coefficient affecting capacity

        # Exponential charging model parameters
        # self.a1 = 1.0
        # 50 = a1 * e^(ln120 * 0.7) = a1 * 120^0.7
        # a1 = 50 / (120^0.7) = 1.752
        self.a1 = 1.752

        self.a2 = np.log(60.0 * self.capacity / self.charge_rate)

        # Energy draw parameters
        self.ampere_draw_idling = 5.0  # (A)
        self.ampere_draw_moving_empty = 40.0  # (A)
        self.ampere_draw_moving_load_param = 10.0  # (A/kg)
        self.ampere_draw_pickdrop_empty = 20.0  # (A)
        self.ampere_draw_pickdrop_load_param = 10.0  # (A/kg)

        self.soc_history = {}

    def update_soc(self, change_soc):
        self.soc += change_soc
        self.soc = max(0.0, min(self.soc, 1.0))  # Clamp SOC between 0% and 100%

    def apply_self_discharge(self, hours):
        self_discharge = self.self_discharge_rate * hours
        self.update_soc(-self_discharge)

    def adjust_capacity_for_temperature(self):
        # Adjust capacity based on temperature
        adjusted_capacity = self.capacity * (1 - self.temp_coefficient * (self.temperature - 25.0))
        return adjusted_capacity

    def charge_linear(self, target_soc):
        if target_soc > self.soc:
            required_charge = (target_soc - self.soc) * self.adjust_capacity_for_temperature()
            charging_time = 60.0 * required_charge / (self.charge_rate * self.charge_efficiency)  # hours to minutes
        else:
            charging_time = 0
        return charging_time

    def charge_exponential(self, target_soc):
        if target_soc > self.soc:
            # a1 * e^(a2 * (target_soc - soc))
            charging_time = self.a1 * (np.exp(self.a2 * target_soc) - np.exp(self.a2 * self.soc))

        else:
            charging_time = 0
        return charging_time

    def discharge_idling(self, time):
        # The input parameter "time" is in minutes
        # discharge_ah = self.ampere_draw_idling * (time / 60.0)  # minutes to hours

        # time = 1.0:
        # discharge_soc = 5 * (1.0 * x) / 200 = 0.0001 (soc from 1.0 to 0.3 requires 7000 time)
        # 5 * x = 0.2, x = 0.04
        discharge_ah = self.ampere_draw_idling * (time * 0.004)

        discharge_soc = discharge_ah / self.adjust_capacity_for_temperature()
        return discharge_soc

    def discharge_moving(self, time, load):
        # The input parameter "time" is in minutes
        # The input parameter "load" is in kg
        # discharge_ah = (self.ampere_draw_moving_empty + self.ampere_draw_moving_load_param * load) * (time / 60.0)

        # time = 1.0:
        # discharge_soc = 50 * (1.0 * 0.004) / 200 = 0.001 (soc from 1.0 to 0.3 requires 700 time)
        # 5 * x = 0.02, x = 0.004
        discharge_ah = (self.ampere_draw_moving_empty + self.ampere_draw_moving_load_param * load) * (time * 0.004)

        discharge_soc = discharge_ah / self.adjust_capacity_for_temperature()
        return discharge_soc

    def discharge_picking_and_dropping(self, load, time=1.0):
        # The input parameter "time" is in minutes
        # The input parameter "load" is in kg
        discharge_ah = (self.ampere_draw_pickdrop_empty + self.ampere_draw_pickdrop_load_param * load) * (time / 60.0)
        discharge_soc = discharge_ah / self.adjust_capacity_for_temperature()
        return discharge_soc

    def reset_battery(self):
        self.soc = 1.0
        self.soc_history = {}


    # Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    battery_capacities = [100.0, 200.0, 300.0, 400.0, 500.0, 1000.0]
    batteries = []
    for capacity_k in battery_capacities:
        battery_k = Battery(capacity=capacity_k)
        battery_k.soc = 0.0
        batteries.append(battery_k)

    soc_values = np.linspace(0.2, 1.0, 81)
    linear_charging_times = [batteries[0].charge_linear(target_soc=soc) for soc in soc_values]
    exponential_charging_times = [batteries[0].charge_exponential(target_soc=soc) for soc in soc_values]

    plt.figure(figsize=(12, 6))
    plt.plot(soc_values, linear_charging_times, label='Linear Charging Model')
    plt.plot(soc_values, exponential_charging_times, label='Exponential Charging Model')
    plt.xlabel('State of Charge (SOC) %')
    plt.ylabel('Charging Time (minutes)')
    plt.title('Charging Process from 20% to 100% SOC')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    for battery_k in batteries:
        linear_charging_times = [battery_k.charge_linear(target_soc=soc) for soc in soc_values]
        plt.plot(soc_values, linear_charging_times, label=f'{battery_k.capacity} Ah')
    plt.xlabel('State of Charge (SOC) %')
    plt.ylabel('Charging Time (minutes)')
    plt.title('Linear Charging Model with Different Battery Capacities')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    for battery_k in batteries:
        linear_charging_times = [battery_k.charge_linear(target_soc=soc) for soc in soc_values]
        exponential_charging_times = [battery_k.charge_exponential(target_soc=soc) for soc in soc_values]
        plt.plot(soc_values, exponential_charging_times, label=f'{battery_k.capacity} Ah')
    plt.xlabel('State of Charge (SOC) %')
    plt.ylabel('Charging Time (minutes)')
    plt.title('Exponential Charging Model with Different Battery Capacities')
    plt.legend()
    plt.grid(True)
    plt.show()




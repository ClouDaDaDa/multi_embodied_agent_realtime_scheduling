import numpy as np
# from scipy.stats import gamma, fatiguelife, lognorm


class DegradationModel:
    def __init__(self, degradation_type: str, degradation_parameters: dict):
        """
        Initialize the degradation model with specified type and parameters.

        :param degradation_type: Type of degradation model ('weibull', 'lognormal', 'gamma', 'fatigue', etc.)
        :param degradation_parameters: Dictionary of parameters specific to the chosen degradation model.
        """
        self.degradation_type = degradation_type
        self.degradation_parameters = degradation_parameters

    def degradation_function(self, current_life: float) -> float:
        """
        Calculate the degradation level of the machine at a given time (current_life).

        :param current_life: Current life or time point to calculate degradation level.
        :return: Degradation level (unreliability) at the given time point.
        """
        if self.degradation_type == "weibull":
            shape = self.degradation_parameters["shape"]
            scale = self.degradation_parameters["scale"]
            return 1 - np.exp(- (current_life / scale) ** shape)

            # 0.7 = 1 - 0.3
            # 0.3 = e^(- (current_life / scale) ^ shape)

        # elif self.degradation_type == "lognormal":
        #     mean = self.degradation_parameters["mean"]
        #     sigma = self.degradation_parameters["sigma"]
        #     return lognorm.cdf(current_life, s=sigma, scale=np.exp(mean))
        #
        # elif self.degradation_type == "gamma":
        #     shape = self.degradation_parameters["shape"]
        #     rate = self.degradation_parameters["rate"]
        #     return gamma.cdf(current_life, a=shape, scale=1/rate)
        #
        # elif self.degradation_type == "fatigue":
        #     alpha = self.degradation_parameters["alpha"]
        #     beta = self.degradation_parameters["beta"]
        #     return fatiguelife.cdf(current_life, c=alpha, scale=beta)

        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")

    def failure_rate(self, current_life: float) -> float:
        """
        Calculate the instantaneous failure rate (hazard rate) of the machine at a given time.

        :param current_life: Current life or time point to calculate failure rate.
        :return: Failure rate (instantaneous probability of failure at the given time).
        """

        if current_life == 0.0:
            return 0.0
        else:
            if self.degradation_type == "weibull":
                shape = self.degradation_parameters["shape"]
                scale = self.degradation_parameters["scale"]
                return (shape / scale) * (current_life / scale) ** (shape - 1)

            # elif self.degradation_type == "lognormal":
            #     mean = self.degradation_parameters["mean"]
            #     sigma = self.degradation_parameters["sigma"]
            #     pdf_value = lognorm.pdf(current_life, s=sigma, scale=np.exp(mean))
            #     cdf_value = lognorm.cdf(current_life, s=sigma, scale=np.exp(mean))
            #     return pdf_value / (1 - cdf_value) if (1 - cdf_value) > 0 else 1.0
            #
            # elif self.degradation_type == "gamma":
            #     shape = self.degradation_parameters["shape"]
            #     rate = self.degradation_parameters["rate"]
            #     pdf_value = gamma.pdf(current_life, a=shape, scale=1 / rate)
            #     cdf_value = gamma.cdf(current_life, a=shape, scale=1 / rate)
            #     return (pdf_value / (1 - cdf_value)) if (1 - cdf_value) > 0 else 1.0
            #
            # elif self.degradation_type == "fatigue":
            #     alpha = self.degradation_parameters["alpha"]
            #     beta = self.degradation_parameters["beta"]
            #     pdf_value = fatiguelife.pdf(current_life, alpha, scale=beta)
            #     cdf_value = fatiguelife.cdf(current_life, alpha, scale=beta)
            #     return pdf_value / (1 - cdf_value) if (1 - cdf_value) > 0 else 1.0

            else:
                raise ValueError(f"Unknown degradation type: {self.degradation_type}")


# Example usage:
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    # from lifelines.datasets import load_rossi

    # Example parameters for each degradation model
    weibull_params = {"shape": 1.5, "scale": 100}
    lognormal_params = {"mean": 3.0, "sigma": 0.4}
    gamma_params = {"shape": 2.0, "rate": 0.1}
    fatigue_params = {"alpha": 0.5, "beta": 100}

    # # Example parameters for Cox Proportional Hazards Model
    # rossi_data = load_rossi()
    # cox_params = {"data": rossi_data}

    # Create instances of DegradationModel for each model
    model_weibull = DegradationModel("weibull", weibull_params)
    model_lognormal = DegradationModel("lognormal", lognormal_params)
    model_gamma = DegradationModel("gamma", gamma_params)
    model_fatigue = DegradationModel("fatigue", fatigue_params)

    # Simulation parameters
    num_steps = 500
    current_time = np.linspace(0, num_steps, num_steps + 1)
    delta_time = 1.0

    # Arrays to store results
    weibull_degradation_levels = np.zeros(num_steps + 1)
    lognormal_degradation_levels = np.zeros(num_steps + 1)
    gamma_degradation_levels = np.zeros(num_steps + 1)
    fatigue_degradation_levels = np.zeros(num_steps + 1)

    weibull_failure_rates = np.zeros(num_steps + 1)
    lognormal_failure_rates = np.zeros(num_steps + 1)
    gamma_failure_rates = np.zeros(num_steps + 1)
    fatigue_failure_rates = np.zeros(num_steps + 1)

    # Simulate degradation process for each model
    for i in range(num_steps + 1):
        current_life = current_time[i]

        weibull_degradation_levels[i] = model_weibull.degradation_function(current_life)
        lognormal_degradation_levels[i] = model_lognormal.degradation_function(current_life)
        gamma_degradation_levels[i] = model_gamma.degradation_function(current_life)
        fatigue_degradation_levels[i] = model_fatigue.degradation_function(current_life)

        weibull_failure_rates[i] = model_weibull.failure_rate(current_life)
        lognormal_failure_rates[i] = model_lognormal.failure_rate(current_life)
        gamma_failure_rates[i] = model_gamma.failure_rate(current_life)
        fatigue_failure_rates[i] = model_fatigue.failure_rate(current_life)

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 18))

    # Degradation level subplot
    axes[0].plot(current_time, weibull_degradation_levels, label='Weibull')
    axes[0].plot(current_time, lognormal_degradation_levels, label='Lognormal')
    axes[0].plot(current_time, gamma_degradation_levels, label='Gamma')
    axes[0].plot(current_time, fatigue_degradation_levels, label='Fatigue Life')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Degradation Level')
    axes[0].legend()

    # Failure rate subplot
    axes[1].plot(current_time, weibull_failure_rates, label='Weibull')
    axes[1].plot(current_time, lognormal_failure_rates, label='Lognormal')
    axes[1].plot(current_time, gamma_failure_rates, label='Gamma')
    axes[1].plot(current_time, fatigue_failure_rates, label='Fatigue Life')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Failure Rate')
    axes[1].legend()

    plt.tight_layout()
    plt.show()




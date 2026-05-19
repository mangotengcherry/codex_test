from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    conf_level: float = 0.95

    # Outlier removal (per-axis quantile cuts)
    metro_outlier_q_low: float = 0.005
    metro_outlier_q_high: float = 0.995
    eds_outlier_q_high: float = 0.9995

    # GPR hyperparameters
    gpr_length_scale: float = 10.0
    gpr_n_restarts: int = 5
    gpr_grid_points: int = 500

    # Window existence / y_target
    # y_target = GPR_min + sigma_factor * GPR_std (analogous to original good_mean + 0.25*good_std)
    y_target_sigma_factor: float = 0.25

    # Bootstrap PWI CI
    bootstrap_n: int = 1000
    bootstrap_seed: int = 0

    @property
    def alpha(self) -> float:
        return 1.0 - self.conf_level

from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    n_groups: int = 10
    conf_level: float = 0.95
    min_group_count: int = 15
    min_valid_groups: int = 3
    skew_threshold: float = 1.3
    metro_outlier_q_low: float = 0.005
    metro_outlier_q_high: float = 0.995
    eds_outlier_q_high: float = 0.9995
    y_target_sigma_factor: float = 0.25
    imag_tolerance: float = 1e-10

    @property
    def alpha(self) -> float:
        return 1.0 - self.conf_level

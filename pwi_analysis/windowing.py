import numpy as np
import pandas as pd
from sklearn.utils import resample

from .config import AnalysisConfig


def compute_window_gpr(
    x_grid: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_target: float,
) -> tuple[float, float, float, float]:
    """Find window boundaries where the GPR prediction curve crosses y_target.

    Accepts pre-computed grid predictions so the caller can reuse them without
    a redundant GPR inference pass.

    Returns (window_low, window_high, window_low_std, window_high_std).
    The *_std values are the GPR posterior standard deviation at each crossing,
    providing a first-order uncertainty estimate for the boundary location.
    """
    crossings = np.where(np.diff(np.sign(y_pred - y_target)))[0]

    left_idx = int(crossings[0])
    right_idx = int(crossings[-1])

    return (
        float(x_grid[left_idx]),
        float(x_grid[right_idx]),
        float(y_std[left_idx]),
        float(y_std[right_idx]),
    )


def compute_pwi(df: pd.DataFrame, window_low: float, window_high: float) -> float:
    """PWI percentage: fraction of data points within the window."""
    within = df["item_value"].between(window_low, window_high).sum()
    return round(within / len(df) * 100, 2)


def compute_pwi_with_ci(
    df: pd.DataFrame,
    window_low: float,
    window_high: float,
    cfg: AnalysisConfig,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for PWI index.

    Returns (pwi_mean, ci_low, ci_high).
    """
    rng = np.random.default_rng(cfg.bootstrap_seed)
    boot_pwi = [
        compute_pwi(
            resample(df, random_state=int(rng.integers(0, 2**31))),
            window_low,
            window_high,
        )
        for _ in range(cfg.bootstrap_n)
    ]
    return (
        round(float(np.mean(boot_pwi)), 2),
        round(float(np.percentile(boot_pwi, 2.5)), 2),
        round(float(np.percentile(boot_pwi, 97.5)), 2),
    )

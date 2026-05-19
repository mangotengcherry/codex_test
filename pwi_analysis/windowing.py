import numpy as np
import pandas as pd

from .config import AnalysisConfig


def compute_window(
    df: pd.DataFrame,
    good_groups: list,
    cfg: AnalysisConfig,
) -> tuple[float, float]:
    """Compute window_low / window_high via separate left and right regressions.

    Strategy
    --------
    1. Mark good/bad groups.
    2. Compute y_target = good mean + sigma_factor * good std.
    3. Split data at the left/right boundaries of the good region.
    4. Fit a quadratic to each side and find where it crosses y_target.
    """
    df = df.copy()
    df["logic"] = (~df["group"].isin(good_groups)).astype(int)

    good_mask = df["logic"] == 0
    good_bin = df.loc[good_mask, "bin_value"]
    y_target = good_bin.mean() + cfg.y_target_sigma_factor * good_bin.std()

    # Split at the centre of the good region so each side captures the full
    # U-half-shape (good low-bin region → bad high-bin region). Splitting at
    # the good boundary (min/max) leaves only monotone "bad" data on each
    # side, which produces complex roots when y_target is below those values.
    x_center = df.loc[good_mask, "item_value"].mean()

    left_df = df[df["item_value"] <= x_center]
    right_df = df[df["item_value"] >= x_center]

    _check_min_points(left_df, "left", n=3)
    _check_min_points(right_df, "right", n=3)

    window_low = _fit_root(
        left_df["item_value"].values, left_df["bin_value"].values,
        y_target, side="left", cfg=cfg,
    )
    window_high = _fit_root(
        right_df["item_value"].values, right_df["bin_value"].values,
        y_target, side="right", cfg=cfg,
    )

    if window_low >= window_high:
        raise ValueError(
            f"Invalid window: low={window_low:.4f} >= high={window_high:.4f}"
        )

    return window_low, window_high


def compute_pwi(df_filtered: pd.DataFrame, window_low: float, window_high: float) -> float:
    """PWI percentage based on the outlier-filtered (pre-group-filter) population."""
    within = df_filtered["item_value"].between(window_low, window_high).sum()
    return round(within / len(df_filtered) * 100, 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_root(
    x: np.ndarray,
    y: np.ndarray,
    y_target: float,
    side: str,
    cfg: AnalysisConfig,
) -> float:
    coeffs = np.polyfit(x, y, 2)
    # Shift constant term so roots are where the curve equals y_target
    shifted_coeffs = [coeffs[0], coeffs[1], coeffs[2] - y_target]
    roots = np.roots(shifted_coeffs)

    if np.any(np.abs(roots.imag) > cfg.imag_tolerance):
        raise ValueError(
            f"Complex roots on {side} side — window boundary cannot be determined"
        )

    real_roots = roots.real
    # Left side: take the root closest to (and below) the good region
    # Right side: take the root closest to (and above) the good region
    return float(real_roots.min() if side == "left" else real_roots.max())


def _check_min_points(df: pd.DataFrame, side: str, n: int) -> None:
    if len(df) < n:
        raise ValueError(f"Insufficient data for {side} regression: {len(df)} < {n}")

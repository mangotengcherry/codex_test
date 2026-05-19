import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .modeling import fit_gpr, validate_window_exists
from .preprocess import preprocess
from .windowing import compute_pwi, compute_pwi_with_ci, compute_window_gpr

logger = logging.getLogger(__name__)


@dataclass
class PWIResult:
    window_low: float
    window_high: float
    window_low_std: float    # GPR posterior std at left crossing
    window_high_std: float   # GPR posterior std at right crossing
    pwi_index: float
    pwi_ci_low: float        # Bootstrap 95% CI lower bound
    pwi_ci_high: float       # Bootstrap 95% CI upper bound
    r2: float                # GPR R² (reference quality indicator)
    y_target: float          # Threshold used to find crossing points
    window_depth: float      # (y_target - GPR_min) / GPR_std — effect-size-like window prominence


def pwi_analysis(
    metro_data: pd.DataFrame,
    eds_data: pd.DataFrame,
    cfg: Optional[AnalysisConfig] = None,
) -> tuple[Optional[PWIResult], str]:
    """Run the full PWI analysis pipeline.

    Returns (PWIResult, "Success") on success, or (None, reason) on early exit.
    All expected failure modes surface as (None, message) rather than exceptions.
    """
    if cfg is None:
        cfg = AnalysisConfig()

    try:
        df = preprocess(metro_data, eds_data, cfg)

        gpr, r2 = fit_gpr(df, cfg)
        logger.info("GPR fit: R²=%.4f  kernel=%s", r2, gpr.kernel_)

        # Single prediction pass shared by all downstream steps
        x_grid = np.linspace(df["item_value"].min(), df["item_value"].max(), cfg.gpr_grid_points)
        y_pred, y_std = gpr.predict(x_grid.reshape(-1, 1), return_std=True)

        gpr_std = float(y_pred.std())
        y_target = float(y_pred.min() + cfg.y_target_sigma_factor * gpr_std)
        min_idx = int(np.argmin(y_pred))

        # window_depth: how prominently the GPR curve dips below y_target
        # Analogous to effect size — higher = more pronounced process window
        window_depth = float((y_target - y_pred.min()) / gpr_std) if gpr_std > 0 else 0.0

        if not validate_window_exists(y_pred, y_target, min_idx):
            return None, (
                f"No process window found: GPR curve does not cross y_target={y_target:.4f} "
                f"on both sides of the minimum (R²={r2:.4f}, window_depth={window_depth:.4f})"
            )

        window_low, window_high, wl_std, wh_std = compute_window_gpr(
            x_grid, y_pred, y_std, y_target
        )
        pwi_mean, pwi_ci_low, pwi_ci_high = compute_pwi_with_ci(df, window_low, window_high, cfg)

        result = PWIResult(
            window_low=window_low,
            window_high=window_high,
            window_low_std=wl_std,
            window_high_std=wh_std,
            pwi_index=pwi_mean,
            pwi_ci_low=pwi_ci_low,
            pwi_ci_high=pwi_ci_high,
            r2=r2,
            y_target=y_target,
            window_depth=window_depth,
        )
        return result, "Success"

    except ValueError as exc:
        logger.warning("Analysis stopped: %s", exc)
        return None, str(exc)
    except Exception as exc:
        logger.exception("Unexpected error in pwi_analysis")
        return None, f"Unexpected error: {exc}"

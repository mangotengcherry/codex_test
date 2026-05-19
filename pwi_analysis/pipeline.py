import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .config import AnalysisConfig
from .grouping import assign_groups
from .hypothesis import run_hypothesis_test
from .posthoc import identify_good_groups
from .preprocess import preprocess
from .windowing import compute_pwi, compute_window

logger = logging.getLogger(__name__)


@dataclass
class PWIResult:
    window_low: float
    window_high: float
    pwi_index: float
    p_value: float
    is_normal: bool
    good_groups: list = field(default_factory=list)
    bad_groups: list = field(default_factory=list)


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
        df_filtered = preprocess(metro_data, eds_data, cfg)
        df_grouped = assign_groups(df_filtered, cfg)

        p_value, is_normal = run_hypothesis_test(df_grouped, cfg)
        logger.info("Hypothesis test: p=%.4f, normal=%s", p_value, is_normal)

        if p_value >= cfg.alpha:
            return None, f"No significant difference (p={p_value:.4f})"

        good_groups, bad_groups = identify_good_groups(df_grouped, is_normal, cfg)

        window_low, window_high = compute_window(df_grouped, good_groups, cfg)

        # Denominator uses the pre-group-filter population for an unbiased PWI
        pwi_index = compute_pwi(df_filtered, window_low, window_high)

        result = PWIResult(
            window_low=window_low,
            window_high=window_high,
            pwi_index=pwi_index,
            p_value=p_value,
            is_normal=is_normal,
            good_groups=sorted(good_groups),
            bad_groups=sorted(bad_groups),
        )
        return result, "Success"

    except ValueError as exc:
        logger.warning("Analysis stopped: %s", exc)
        return None, str(exc)
    except Exception as exc:
        logger.exception("Unexpected error in pwi_analysis")
        return None, f"Unexpected error: {exc}"

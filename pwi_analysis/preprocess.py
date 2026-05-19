import pandas as pd

from .config import AnalysisConfig

_REQUIRED_METRO_COLS = {"root_lot_id", "wafer_id", "item_value"}
_REQUIRED_EDS_COLS = {"root_lot_id", "wafer_id", "bin_value"}


def preprocess(
    metro_data: pd.DataFrame,
    eds_data: pd.DataFrame,
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    """Merge metro and EDS data, then remove outliers from both axes.

    Returns the cleaned DataFrame. Raises ValueError when the result is empty.
    """
    _validate_columns(metro_data, _REQUIRED_METRO_COLS, "metro_data")
    _validate_columns(eds_data, _REQUIRED_EDS_COLS, "eds_data")

    merged = pd.merge(metro_data, eds_data, on=["root_lot_id", "wafer_id"], how="inner")
    if merged.empty:
        raise ValueError("No matching data after merge on root_lot_id / wafer_id")

    q_metro_lo, q_metro_hi = merged["item_value"].quantile(
        [cfg.metro_outlier_q_low, cfg.metro_outlier_q_high]
    )
    q_eds_hi = merged["bin_value"].quantile(cfg.eds_outlier_q_high)

    df = merged[
        merged["item_value"].between(q_metro_lo, q_metro_hi, inclusive="both")
        & (merged["bin_value"] < q_eds_hi)  # strict upper cut for EDS top 0.05%
    ].copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("No data remaining after outlier removal")

    return df


def _validate_columns(df: pd.DataFrame, required: set, name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

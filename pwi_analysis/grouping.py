import pandas as pd

from .config import AnalysisConfig


def assign_groups(df: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Assign equal-frequency groups to item_value and drop under-populated groups.

    Uses rank(method='first') before qcut so duplicate values don't create NaN groups.
    Raises ValueError when too few valid groups remain for statistical testing.
    """
    df = df.copy()
    labels = [f"g{i:02d}" for i in range(1, cfg.n_groups + 1)]

    df["group"] = pd.qcut(
        df["item_value"].rank(method="first"),
        cfg.n_groups,
        labels=labels,
    )

    group_counts = df["group"].value_counts()
    valid_groups = group_counts[group_counts >= cfg.min_group_count].index
    df = df[df["group"].isin(valid_groups)].copy()
    df["group"] = df["group"].cat.remove_unused_categories()

    n_valid = df["group"].nunique()
    if n_valid < cfg.min_valid_groups:
        raise ValueError(
            f"Insufficient valid groups after size filter: {n_valid} < {cfg.min_valid_groups}"
        )

    return df

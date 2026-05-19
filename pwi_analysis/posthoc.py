import pandas as pd
import scikit_posthocs as sp

from .config import AnalysisConfig


def identify_good_groups(
    df: pd.DataFrame,
    is_normal: bool,
    cfg: AnalysisConfig,
) -> tuple[list, list]:
    """Identify good groups via Games-Howell (normal) or Dunn's test (non-normal).

    Good groups = not significantly different from the reference group
                  (the group with the lowest mean bin_value).
    Returns (good_groups, bad_groups).
    """
    posthoc_df = _run_posthoc(df, is_normal)

    group_means = df.groupby("group", observed=True)["bin_value"].mean()
    ref_group = group_means.idxmin()

    p_vals = posthoc_df.loc[ref_group]
    mask = p_vals >= cfg.alpha
    mask[ref_group] = True  # ref_group always belongs to good groups

    good_groups = posthoc_df.columns[mask].tolist()
    all_groups = df["group"].cat.categories.tolist()
    bad_groups = [g for g in all_groups if g not in good_groups]

    return good_groups, bad_groups


def _run_posthoc(df: pd.DataFrame, is_normal: bool) -> pd.DataFrame:
    if is_normal:
        # Tamhane's T2: parametric post-hoc that does not assume equal variances
        # (equivalent role to Games-Howell; posthoc_gameshowell absent in scikit-posthocs)
        return sp.posthoc_tamhane(df, val_col="bin_value", group_col="group")
    return sp.posthoc_dunn(
        df, val_col="bin_value", group_col="group", p_adjust="bonferroni"
    )

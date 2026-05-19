import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

from .config import AnalysisConfig


def run_hypothesis_test(df: pd.DataFrame, cfg: AnalysisConfig) -> tuple[float, bool]:
    """Run ANOVA (normal) or Kruskal-Wallis (non-normal) across groups.

    Normality is determined by the skewness of bin_value.
    Returns (p_value, is_normal).
    """
    skewness = df["bin_value"].skew()
    is_normal = abs(skewness) <= cfg.skew_threshold

    if is_normal:
        p_value = _run_anova(df)
    else:
        p_value = _run_kruskal(df)

    return p_value, is_normal


def _run_anova(df: pd.DataFrame) -> float:
    model = ols("bin_value ~ C(group)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    p_col = next(c for c in anova_table.columns if "PR" in c)
    return float(anova_table[p_col].iloc[0])


def _run_kruskal(df: pd.DataFrame) -> float:
    group_arrays = [
        grp["bin_value"].values
        for _, grp in df.groupby("group", observed=True)
    ]
    _, p_value = stats.kruskal(*group_arrays)
    return float(p_value)

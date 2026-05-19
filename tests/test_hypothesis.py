import numpy as np
import pandas as pd
import pytest

from pwi_analysis.config import AnalysisConfig
from pwi_analysis.hypothesis import run_hypothesis_test


def _make_grouped_df(group_means: dict, n_per_group: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, mean in group_means.items():
        vals = rng.normal(mean, 0.2, n_per_group)
        rows.append(pd.DataFrame({"group": label, "bin_value": vals}))
    df = pd.concat(rows, ignore_index=True)
    df["group"] = pd.Categorical(df["group"], categories=sorted(group_means.keys()))
    return df


def test_anova_detects_significant_difference(cfg):
    df = _make_grouped_df({"g01": 1.0, "g02": 1.0, "g03": 5.0})
    p, is_normal = run_hypothesis_test(df, cfg)
    assert p < cfg.alpha
    assert bool(is_normal) is True


def test_anova_no_difference(cfg):
    df = _make_grouped_df({"g01": 1.0, "g02": 1.0, "g03": 1.0})
    p, is_normal = run_hypothesis_test(df, cfg)
    assert p >= cfg.alpha


def test_kruskal_triggered_for_skewed_data(cfg):
    """Inject a highly skewed distribution to force Kruskal path."""
    rng = np.random.default_rng(7)
    n = 100
    skewed = np.concatenate([rng.exponential(1, n), rng.exponential(5, n), rng.exponential(10, n)])
    groups = np.repeat(["g01", "g02", "g03"], n)
    df = pd.DataFrame({"group": groups, "bin_value": skewed})
    df["group"] = pd.Categorical(df["group"], categories=["g01", "g02", "g03"])

    p, is_normal = run_hypothesis_test(df, cfg)
    assert bool(is_normal) is False
    assert isinstance(p, float)


def test_returns_float_p_value(grouped_data, cfg):
    p, _ = run_hypothesis_test(grouped_data, cfg)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0

import numpy as np
import pandas as pd
import pytest

from pwi_analysis.posthoc import identify_good_groups


def _make_grouped_df(group_means: dict, n_per_group: int = 80, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, mean in group_means.items():
        vals = rng.normal(mean, 0.3, n_per_group)
        rows.append(pd.DataFrame({"group": label, "bin_value": vals}))
    df = pd.concat(rows, ignore_index=True)
    df["group"] = pd.Categorical(df["group"], categories=sorted(group_means.keys()))
    return df


def test_good_groups_include_low_mean_groups(cfg):
    df = _make_grouped_df({"g01": 1.0, "g02": 1.1, "g03": 5.0, "g04": 5.2})
    good, bad = identify_good_groups(df, is_normal=True, cfg=cfg)
    assert "g01" in good
    assert "g03" in bad or "g04" in bad


def test_ref_group_always_in_good(cfg):
    df = _make_grouped_df({"g01": 1.0, "g02": 3.0, "g03": 5.0})
    good, _ = identify_good_groups(df, is_normal=True, cfg=cfg)
    # g01 has lowest mean → must be in good_groups
    assert "g01" in good


def test_no_overlap_between_good_and_bad(cfg):
    df = _make_grouped_df({"g01": 1.0, "g02": 1.1, "g03": 5.0})
    good, bad = identify_good_groups(df, is_normal=True, cfg=cfg)
    assert set(good) & set(bad) == set()


def test_all_groups_covered(cfg):
    df = _make_grouped_df({"g01": 1.0, "g02": 3.0, "g03": 5.0})
    good, bad = identify_good_groups(df, is_normal=True, cfg=cfg)
    all_groups = set(df["group"].cat.categories)
    assert set(good) | set(bad) == all_groups


def test_nonparametric_path_runs(cfg):
    df = _make_grouped_df({"g01": 1.0, "g02": 3.0, "g03": 5.0})
    good, bad = identify_good_groups(df, is_normal=False, cfg=cfg)
    assert len(good) >= 1
    assert len(bad) >= 0

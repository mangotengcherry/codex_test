import numpy as np
import pandas as pd
import pytest

from pwi_analysis.config import AnalysisConfig
from pwi_analysis.windowing import compute_pwi, compute_window


def _make_u_shaped_df(n: int = 300, seed: int = 2) -> tuple[pd.DataFrame, list, list]:
    """U-shaped data: groups g01/g05 are bad (high bin_value), g02-g04 are good."""
    rng = np.random.default_rng(seed)
    means = {"g01": 5.0, "g02": 1.0, "g03": 0.8, "g04": 1.0, "g05": 5.0}
    item_centers = {"g01": 75, "g02": 88, "g03": 100, "g04": 112, "g05": 125}
    rows = []
    for label, bm in means.items():
        iv = rng.normal(item_centers[label], 3, n // 5)
        bv = rng.normal(bm, 0.2, n // 5)
        rows.append(pd.DataFrame({"group": label, "item_value": iv, "bin_value": bv}))
    df = pd.concat(rows, ignore_index=True)
    df["group"] = pd.Categorical(df["group"], categories=sorted(means.keys()))
    good_groups = ["g02", "g03", "g04"]
    bad_groups = ["g01", "g05"]
    return df, good_groups, bad_groups


def test_window_low_less_than_high(cfg):
    df, good, _ = _make_u_shaped_df()
    low, high = compute_window(df, good, cfg)
    assert low < high


def test_window_contains_good_region(cfg):
    df, good, _ = _make_u_shaped_df()
    low, high = compute_window(df, good, cfg)
    good_x = df[df["group"].isin(good)]["item_value"]
    assert low <= good_x.mean() <= high


def test_pwi_between_0_and_100():
    df = pd.DataFrame({"item_value": np.linspace(80, 120, 200)})
    pwi = compute_pwi(df, window_low=90, window_high=110)
    assert 0.0 <= pwi <= 100.0


def test_pwi_full_window_is_100():
    df = pd.DataFrame({"item_value": np.linspace(90, 110, 100)})
    pwi = compute_pwi(df, window_low=90, window_high=110)
    assert pwi == 100.0


def test_pwi_empty_window_is_0():
    df = pd.DataFrame({"item_value": np.linspace(90, 110, 100)})
    pwi = compute_pwi(df, window_low=200, window_high=300)
    assert pwi == 0.0


def test_insufficient_data_raises(cfg):
    """Fewer than 3 rows per side → polyfit cannot proceed."""
    # 2 rows total: x_center will be 100, left has 1 point, right has 1 point
    tiny_df = pd.DataFrame(
        {
            "item_value": [90.0, 110.0],
            "bin_value":  [2.0,  2.0],
            "group": pd.Categorical(
                ["g02", "g04"],
                categories=["g01", "g02", "g03", "g04", "g05"],
            ),
        }
    )
    with pytest.raises(ValueError, match="Insufficient data"):
        compute_window(tiny_df, ["g02", "g04"], cfg)


def test_returns_floats(cfg):
    df, good, _ = _make_u_shaped_df()
    low, high = compute_window(df, good, cfg)
    assert isinstance(low, float)
    assert isinstance(high, float)

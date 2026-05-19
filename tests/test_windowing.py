import numpy as np
import pandas as pd
import pytest

from pwi_analysis.config import AnalysisConfig
from pwi_analysis.modeling import fit_gpr
from pwi_analysis.windowing import compute_pwi, compute_pwi_with_ci, compute_window_gpr


def _predict_grid(gpr, df, cfg):
    x_grid = np.linspace(df["item_value"].min(), df["item_value"].max(), cfg.gpr_grid_points)
    y_pred, y_std = gpr.predict(x_grid.reshape(-1, 1), return_std=True)
    y_target = float(y_pred.min() + cfg.y_target_sigma_factor * y_pred.std())
    return x_grid, y_pred, y_std, y_target


def test_window_low_less_than_high(preprocessed_data, cfg):
    gpr, _ = fit_gpr(preprocessed_data, cfg)
    x_grid, y_pred, y_std, y_target = _predict_grid(gpr, preprocessed_data, cfg)
    low, high, _, _ = compute_window_gpr(x_grid, y_pred, y_std, y_target)
    assert low < high


def test_window_contains_center_of_data(preprocessed_data, cfg):
    """The window should straddle the item_value center (~100 in the U-shaped fixture)."""
    gpr, _ = fit_gpr(preprocessed_data, cfg)
    x_grid, y_pred, y_std, y_target = _predict_grid(gpr, preprocessed_data, cfg)
    low, high, _, _ = compute_window_gpr(x_grid, y_pred, y_std, y_target)
    center = preprocessed_data["item_value"].mean()
    assert low < center < high


def test_window_returns_floats(preprocessed_data, cfg):
    gpr, _ = fit_gpr(preprocessed_data, cfg)
    x_grid, y_pred, y_std, y_target = _predict_grid(gpr, preprocessed_data, cfg)
    low, high, low_std, high_std = compute_window_gpr(x_grid, y_pred, y_std, y_target)
    for v in (low, high, low_std, high_std):
        assert isinstance(v, float)


def test_window_std_nonnegative(preprocessed_data, cfg):
    gpr, _ = fit_gpr(preprocessed_data, cfg)
    x_grid, y_pred, y_std, y_target = _predict_grid(gpr, preprocessed_data, cfg)
    _, _, low_std, high_std = compute_window_gpr(x_grid, y_pred, y_std, y_target)
    assert low_std >= 0.0
    assert high_std >= 0.0


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


def test_pwi_with_ci_bounds_ordered(preprocessed_data, cfg):
    pwi, ci_low, ci_high = compute_pwi_with_ci(preprocessed_data, 95.0, 105.0, cfg)
    assert ci_low <= pwi <= ci_high


def test_pwi_with_ci_in_valid_range(preprocessed_data, cfg):
    pwi, ci_low, ci_high = compute_pwi_with_ci(preprocessed_data, 95.0, 105.0, cfg)
    assert 0.0 <= ci_low <= 100.0
    assert 0.0 <= ci_high <= 100.0

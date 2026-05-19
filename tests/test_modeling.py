import numpy as np
import pytest

from pwi_analysis.modeling import fit_gpr, validate_window_exists


def test_fit_gpr_returns_r2(preprocessed_data, cfg):
    gpr, r2 = fit_gpr(preprocessed_data, cfg)
    assert 0.0 <= r2 <= 1.0


def test_fit_gpr_r2_positive_for_u_shape(preprocessed_data, cfg):
    """U-shaped data should yield a GPR with positive R²."""
    _, r2 = fit_gpr(preprocessed_data, cfg)
    assert r2 > 0.0


def test_validate_window_exists_true_for_u_shape(preprocessed_data, cfg):
    import numpy as np
    from pwi_analysis.modeling import fit_gpr, validate_window_exists

    gpr, _ = fit_gpr(preprocessed_data, cfg)
    x_grid = np.linspace(
        preprocessed_data["item_value"].min(),
        preprocessed_data["item_value"].max(),
        cfg.gpr_grid_points,
    )
    y_pred, _ = gpr.predict(x_grid.reshape(-1, 1), return_std=True)
    y_target = float(y_pred.min() + cfg.y_target_sigma_factor * y_pred.std())
    min_idx = int(np.argmin(y_pred))
    assert validate_window_exists(y_pred, y_target, min_idx)


def test_validate_window_exists_false_when_flat():
    """Flat prediction (all equal) never dips below y_target."""
    y_pred = np.ones(100) * 1.0
    y_target = 0.5
    min_idx = 50
    assert not validate_window_exists(y_pred, y_target, min_idx)


def test_validate_window_exists_false_when_min_above_target():
    """If the minimum is above y_target, no window."""
    x = np.linspace(0, 10, 100)
    y_pred = (x - 5) ** 2 + 2.0  # minimum = 2.0 at x=5
    y_target = 1.0  # below the minimum
    min_idx = int(np.argmin(y_pred))
    assert not validate_window_exists(y_pred, y_target, min_idx)


def test_validate_window_exists_false_when_monotone():
    """Monotonically decreasing curve: minimum at the right boundary, not interior."""
    y_pred = np.linspace(5.0, 0.1, 100)
    y_target = 1.0
    min_idx = int(np.argmin(y_pred))  # = 99 (rightmost)
    assert not validate_window_exists(y_pred, y_target, min_idx)

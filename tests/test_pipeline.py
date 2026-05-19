import numpy as np
import pandas as pd

from pwi_analysis.config import AnalysisConfig
from pwi_analysis.pipeline import PWIResult, pwi_analysis


def test_success_returns_result(sample_data, cfg):
    metro, eds = sample_data
    result, msg = pwi_analysis(metro, eds, cfg)
    assert msg == "Success"
    assert isinstance(result, PWIResult)


def test_result_fields_are_valid(sample_data, cfg):
    metro, eds = sample_data
    result, _ = pwi_analysis(metro, eds, cfg)
    assert result is not None
    assert result.window_low < result.window_high
    assert 0.0 <= result.pwi_index <= 100.0
    assert result.pwi_ci_low <= result.pwi_index <= result.pwi_ci_high
    assert result.window_low_std >= 0.0
    assert result.window_high_std >= 0.0
    assert 0.0 <= result.r2 <= 1.0
    assert isinstance(result.y_target, float)
    assert result.window_depth > 0.0


def test_no_window_returns_none_for_flat_data(cfg):
    """Completely flat bin_value → GPR sees no relationship → no window."""
    rng = np.random.default_rng(99)
    n = 300
    lot_ids = rng.integers(100, 115, n)
    wafer_ids = rng.integers(1, 26, n)
    item_vals = rng.normal(100, 10, n)

    metro = pd.DataFrame(
        {
            "root_lot_id": lot_ids,
            "wafer_id": wafer_ids,
            "item_value": item_vals,
            "m_key2": "KEY_001",
        }
    )
    # bin_value is constant with tiny noise → no U-shape → no window
    eds = pd.DataFrame(
        {
            "root_lot_id": lot_ids,
            "wafer_id": wafer_ids,
            "bin_value": rng.normal(1.0, 0.001, n),
            "bin_id": "BIN_001",
        }
    )

    result, msg = pwi_analysis(metro, eds, cfg)
    assert result is None
    assert "No process window found" in msg


def test_empty_data_returns_none(cfg):
    metro = pd.DataFrame(columns=["root_lot_id", "wafer_id", "item_value", "m_key2"])
    eds = pd.DataFrame(columns=["root_lot_id", "wafer_id", "bin_value", "bin_id"])
    result, msg = pwi_analysis(metro, eds, cfg)
    assert result is None
    assert msg


def test_missing_column_returns_none(sample_data, cfg):
    metro, eds = sample_data
    result, msg = pwi_analysis(metro.drop(columns=["item_value"]), eds, cfg)
    assert result is None
    assert "missing columns" in msg


def test_default_config_used_when_none(sample_data):
    metro, eds = sample_data
    result, msg = pwi_analysis(metro, eds, cfg=None)
    assert isinstance(msg, str)

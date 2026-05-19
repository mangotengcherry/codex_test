import pandas as pd
import pytest

from pwi_analysis.preprocess import preprocess


def test_returns_merged_columns(sample_data, cfg):
    metro, eds = sample_data
    df = preprocess(metro, eds, cfg)
    assert {"item_value", "bin_value", "root_lot_id", "wafer_id"}.issubset(df.columns)


def test_metro_outliers_removed(sample_data, cfg):
    metro, eds = sample_data
    metro = metro.copy()
    metro.loc[0, "item_value"] = 1e6
    df = preprocess(metro, eds, cfg)
    assert df["item_value"].max() < 1e6


def test_eds_outliers_removed(sample_data, cfg):
    metro, eds = sample_data
    eds = eds.copy()
    eds.loc[0, "bin_value"] = 1e6
    df = preprocess(metro, eds, cfg)
    assert df["bin_value"].max() < 1e6


def test_no_matching_rows_raises(cfg):
    metro = pd.DataFrame(
        {"root_lot_id": [1], "wafer_id": [1], "item_value": [100.0], "m_key2": ["K"]}
    )
    eds = pd.DataFrame(
        {"root_lot_id": [99], "wafer_id": [99], "bin_value": [0.5], "bin_id": ["B"]}
    )
    with pytest.raises(ValueError, match="No matching data"):
        preprocess(metro, eds, cfg)


def test_missing_column_raises(sample_data, cfg):
    metro, eds = sample_data
    with pytest.raises(ValueError, match="missing columns"):
        preprocess(metro.drop(columns=["item_value"]), eds, cfg)


def test_result_has_no_nulls(sample_data, cfg):
    metro, eds = sample_data
    df = preprocess(metro, eds, cfg)
    assert df[["item_value", "bin_value"]].isna().sum().sum() == 0


def test_row_count_reduced_by_outlier_removal(sample_data, cfg):
    metro, eds = sample_data
    df = preprocess(metro, eds, cfg)
    merged_len = len(pd.merge(metro, eds, on=["root_lot_id", "wafer_id"], how="inner"))
    assert len(df) <= merged_len

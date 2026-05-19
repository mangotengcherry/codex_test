import pytest

from pwi_analysis.config import AnalysisConfig
from pwi_analysis.grouping import assign_groups


def test_group_column_created(preprocessed_data, cfg):
    df = assign_groups(preprocessed_data, cfg)
    assert "group" in df.columns


def test_group_count_at_most_n(preprocessed_data, cfg):
    df = assign_groups(preprocessed_data, cfg)
    assert df["group"].nunique() <= cfg.n_groups


def test_all_groups_meet_min_count(preprocessed_data, cfg):
    df = assign_groups(preprocessed_data, cfg)
    counts = df["group"].value_counts()
    assert (counts >= cfg.min_group_count).all()


def test_raises_when_too_few_groups(sample_data):
    """Force failure by requiring impossibly many groups from small data."""
    from pwi_analysis.preprocess import preprocess

    metro, eds = sample_data
    # min_group_count so high that almost no group survives
    strict_cfg = AnalysisConfig(n_groups=5, min_group_count=500, min_valid_groups=3)
    df = preprocess(metro, eds, strict_cfg)
    with pytest.raises(ValueError, match="Insufficient valid groups"):
        assign_groups(df, strict_cfg)


def test_labels_are_sorted_strings(preprocessed_data, cfg):
    df = assign_groups(preprocessed_data, cfg)
    labels = sorted(df["group"].cat.categories.tolist())
    assert labels == sorted(labels)


def test_no_nan_groups(preprocessed_data, cfg):
    df = assign_groups(preprocessed_data, cfg)
    assert df["group"].isna().sum() == 0

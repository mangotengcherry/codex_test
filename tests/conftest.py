import numpy as np
import pandas as pd
import pytest

from pwi_analysis.config import AnalysisConfig


@pytest.fixture
def cfg() -> AnalysisConfig:
    return AnalysisConfig(gpr_n_restarts=1, bootstrap_n=100)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def sample_data(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlated metro/EDS data with a U-shaped bin_value response.

    item_value is centered at 100; bin_value is minimised near 100 (good region)
    and increases toward the extremes. This produces a clear process window.
    """
    n = 600
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

    # U-shape: quadratic + small noise, so the centre has low bin_value
    bin_vals = 0.005 * (item_vals - 100) ** 2 + rng.normal(0, 0.05, n) + 0.5
    eds = pd.DataFrame(
        {
            "root_lot_id": lot_ids,
            "wafer_id": wafer_ids,
            "bin_value": bin_vals,
            "bin_id": "BIN_001",
        }
    )

    return metro, eds


@pytest.fixture
def preprocessed_data(sample_data, cfg):
    from pwi_analysis.preprocess import preprocess

    metro, eds = sample_data
    return preprocess(metro, eds, cfg)

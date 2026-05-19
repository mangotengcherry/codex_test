import logging
from typing import Optional

import pandas as pd
from joblib import Parallel, delayed

from .config import AnalysisConfig
from .pipeline import PWIResult, pwi_analysis

logger = logging.getLogger(__name__)


def run_parallel_pwi(
    metro_all: pd.DataFrame,
    eds_all: pd.DataFrame,
    cfg: Optional[AnalysisConfig] = None,
    n_jobs: int = -1,
) -> list[dict]:
    """Run pwi_analysis for every (m_key2, bin_id) combination in parallel.

    Returns a list of dicts with keys: m_key2, bin_id, result, message.
    """
    if cfg is None:
        cfg = AnalysisConfig()

    m_keys = metro_all["m_key2"].unique().tolist()
    bin_ids = eds_all["bin_id"].unique().tolist()
    tasks = [(key, bid) for key in m_keys for bid in bin_ids]

    logger.info(
        "Launching %d tasks (%d keys × %d bins), n_jobs=%d",
        len(tasks), len(m_keys), len(bin_ids), n_jobs,
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_one)(metro_all, eds_all, key, bid, cfg)
        for key, bid in tasks
    )
    return results


def _process_one(
    metro_all: pd.DataFrame,
    eds_all: pd.DataFrame,
    m_key: str,
    bin_id: str,
    cfg: AnalysisConfig,
) -> dict:
    m_temp = metro_all[metro_all["m_key2"] == m_key]
    e_temp = eds_all[eds_all["bin_id"] == bin_id]
    result, message = pwi_analysis(m_temp, e_temp, cfg)
    return {"m_key2": m_key, "bin_id": bin_id, "result": result, "message": message}

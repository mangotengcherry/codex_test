import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from .config import AnalysisConfig


def fit_gpr(df: pd.DataFrame, cfg: AnalysisConfig) -> tuple[GaussianProcessRegressor, float]:
    """Fit a GPR model: bin_value ~ f(item_value).

    Returns (fitted_model, R²).
    """
    X = df[["item_value"]].values
    y = df["bin_value"].values

    kernel = ConstantKernel(1.0) * RBF(length_scale=cfg.gpr_length_scale) + WhiteKernel()
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=cfg.gpr_n_restarts,
        normalize_y=True,
    )
    gpr.fit(X, y)
    r2 = float(gpr.score(X, y))
    return gpr, r2


def validate_window_exists(y_pred: np.ndarray, y_target: float, min_idx: int) -> bool:
    """Check that the GPR curve dips below y_target with crossings on both sides.

    Three conditions must hold:
    1. The curve minimum is below y_target.
    2. At least two sign-change crossings exist (left and right of minimum).
    3. The minimum lies strictly between the two outermost crossings.
    """
    if y_pred[min_idx] >= y_target:
        return False

    crossings = np.where(np.diff(np.sign(y_pred - y_target)))[0]
    if len(crossings) < 2:
        return False

    left_cross, right_cross = int(crossings[0]), int(crossings[-1])
    return left_cross < min_idx < right_cross

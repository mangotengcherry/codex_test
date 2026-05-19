"""Visualize the GPR-based PWI analysis pipeline.

Generates four plots saved to docs/:
  1. pipeline_overview.png    — GPR fit, window, uncertainty, PWI inclusion
  2. window_validation.png    — three failure modes of validate_window_exists
  3. bootstrap_pwi.png        — Bootstrap distribution of PWI with 95% CI
  4. old_vs_new.png           — Side-by-side: quadratic fit vs GPR fit on same data
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import resample

from pwi_analysis import AnalysisConfig, pwi_analysis
from pwi_analysis.modeling import fit_gpr
from pwi_analysis.preprocess import preprocess
from pwi_analysis.windowing import compute_pwi

OUTDIR = ROOT / "docs"
OUTDIR.mkdir(exist_ok=True)


def make_u_shaped_data(seed: int = 42, n: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    lot_ids = rng.integers(100, 115, n)
    wafer_ids = rng.integers(1, 26, n)
    item_vals = rng.normal(100, 10, n)
    metro = pd.DataFrame(
        {"root_lot_id": lot_ids, "wafer_id": wafer_ids, "item_value": item_vals}
    )
    bin_vals = 0.005 * (item_vals - 100) ** 2 + rng.normal(0, 0.05, n) + 0.5
    eds = pd.DataFrame(
        {"root_lot_id": lot_ids, "wafer_id": wafer_ids, "bin_value": bin_vals}
    )
    return metro, eds


# ---------------------------------------------------------------------------
# Plot 1: Pipeline overview — what the GPR-based PWI analysis produces
# ---------------------------------------------------------------------------

def plot_pipeline_overview(cfg: AnalysisConfig) -> None:
    metro, eds = make_u_shaped_data()
    df = preprocess(metro, eds, cfg)
    gpr, r2 = fit_gpr(df, cfg)

    x_grid = np.linspace(df["item_value"].min(), df["item_value"].max(), cfg.gpr_grid_points)
    y_pred, y_std = gpr.predict(x_grid.reshape(-1, 1), return_std=True)
    y_target = float(y_pred.min() + cfg.y_target_sigma_factor * y_pred.std())

    crossings = np.where(np.diff(np.sign(y_pred - y_target)))[0]
    window_low = float(x_grid[crossings[0]])
    window_high = float(x_grid[crossings[-1]])

    fig, ax = plt.subplots(figsize=(10, 6))

    inside = df["item_value"].between(window_low, window_high)
    ax.scatter(
        df.loc[~inside, "item_value"], df.loc[~inside, "bin_value"],
        s=10, alpha=0.4, color="#c0392b", label="outside window",
    )
    ax.scatter(
        df.loc[inside, "item_value"], df.loc[inside, "bin_value"],
        s=10, alpha=0.5, color="#27ae60", label="inside window",
    )

    ax.plot(x_grid, y_pred, color="#2c3e50", lw=2, label="GPR mean")
    ax.fill_between(
        x_grid, y_pred - 2 * y_std, y_pred + 2 * y_std,
        alpha=0.15, color="#2c3e50", label="GPR 95% band",
    )

    ax.axhline(y_target, color="#e67e22", lw=1.5, ls="--", label=f"y_target = {y_target:.3f}")
    ax.axvline(window_low, color="#8e44ad", lw=1.5, ls=":", label=f"window low = {window_low:.2f}")
    ax.axvline(window_high, color="#8e44ad", lw=1.5, ls=":", label=f"window high = {window_high:.2f}")

    ax.axvspan(window_low, window_high, alpha=0.07, color="#27ae60")

    pwi = compute_pwi(df, window_low, window_high)
    title = (
        f"GPR-based Process Window\n"
        f"PWI = {pwi:.1f}%   |   R² = {r2:.3f}   |   "
        f"window depth = {(y_target - y_pred.min()) / y_pred.std():.2f}"
    )
    ax.set_title(title)
    ax.set_xlabel("item_value (Metro)")
    ax.set_ylabel("bin_value (EDS, lower = better)")
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTDIR / "pipeline_overview.png", dpi=130)
    plt.close(fig)
    print(f"  ✓ docs/pipeline_overview.png")


# ---------------------------------------------------------------------------
# Plot 2: validate_window_exists — three failure modes
# ---------------------------------------------------------------------------

def plot_window_validation() -> None:
    x = np.linspace(0, 10, 200)

    cases = [
        ("U-shape (window EXISTS)", (x - 5) ** 2 / 25 + 0.3, 0.5, True),
        ("Flat curve (no relationship)", np.ones_like(x) * 1.0, 0.5, False),
        ("Monotone decreasing (no interior min)", np.linspace(5.0, 0.1, 200), 1.0, False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for ax, (title, y, y_target, valid) in zip(axes, cases):
        ax.plot(x, y, color="#2c3e50", lw=2)
        ax.axhline(y_target, color="#e67e22", lw=1.3, ls="--", label=f"y_target = {y_target}")

        min_idx = int(np.argmin(y))
        ax.scatter(x[min_idx], y[min_idx], color="#8e44ad", s=70, zorder=5, label="GPR min")

        crossings = np.where(np.diff(np.sign(y - y_target)))[0]
        for c in crossings:
            ax.axvline(x[c], color="#27ae60" if valid else "#c0392b", lw=1.2, ls=":")

        outcome = "✓ window exists" if valid else "✗ no window"
        color = "#27ae60" if valid else "#c0392b"
        ax.set_title(f"{title}\n{outcome}", color=color, fontweight="bold")
        ax.set_xlabel("item_value")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("bin_value")

    fig.suptitle(
        "validate_window_exists: geometric criteria for process window",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTDIR / "window_validation.png", dpi=130)
    plt.close(fig)
    print(f"  ✓ docs/window_validation.png")


# ---------------------------------------------------------------------------
# Plot 3: Bootstrap PWI distribution + 95% CI
# ---------------------------------------------------------------------------

def plot_bootstrap_pwi(cfg: AnalysisConfig) -> None:
    metro, eds = make_u_shaped_data()
    result, msg = pwi_analysis(metro, eds, cfg)
    assert result is not None, msg

    df = preprocess(metro, eds, cfg)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    boot = [
        compute_pwi(
            resample(df, random_state=int(rng.integers(0, 2**31))),
            result.window_low,
            result.window_high,
        )
        for _ in range(cfg.bootstrap_n)
    ]
    boot = np.array(boot)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(boot, bins=30, alpha=0.75, color="#3498db", edgecolor="white")
    ax.axvline(result.pwi_index, color="#27ae60", lw=2.2, label=f"PWI mean = {result.pwi_index:.2f}%")
    ax.axvline(result.pwi_ci_low, color="#e67e22", lw=1.6, ls="--",
               label=f"CI low = {result.pwi_ci_low:.2f}%")
    ax.axvline(result.pwi_ci_high, color="#e67e22", lw=1.6, ls="--",
               label=f"CI high = {result.pwi_ci_high:.2f}%")

    ax.set_xlabel("PWI (%)")
    ax.set_ylabel("Bootstrap frequency")
    ax.set_title(
        f"Bootstrap PWI distribution (n={cfg.bootstrap_n})\n"
        f"95% CI = [{result.pwi_ci_low:.2f}, {result.pwi_ci_high:.2f}]"
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTDIR / "bootstrap_pwi.png", dpi=130)
    plt.close(fig)
    print(f"  ✓ docs/bootstrap_pwi.png")


# ---------------------------------------------------------------------------
# Plot 4: old (quadratic) vs new (GPR) on the same data
# ---------------------------------------------------------------------------

def plot_old_vs_new(cfg: AnalysisConfig) -> None:
    metro, eds = make_u_shaped_data()
    df = preprocess(metro, eds, cfg)
    gpr, r2 = fit_gpr(df, cfg)

    x_grid = np.linspace(df["item_value"].min(), df["item_value"].max(), cfg.gpr_grid_points)
    y_pred, y_std = gpr.predict(x_grid.reshape(-1, 1), return_std=True)
    y_target_gpr = float(y_pred.min() + cfg.y_target_sigma_factor * y_pred.std())
    cross_gpr = np.where(np.diff(np.sign(y_pred - y_target_gpr)))[0]
    win_lo_gpr = float(x_grid[cross_gpr[0]])
    win_hi_gpr = float(x_grid[cross_gpr[-1]])

    coeffs = np.polyfit(df["item_value"].values, df["bin_value"].values, 2)
    y_quad = np.polyval(coeffs, x_grid)
    y_target_quad = y_quad.min() + cfg.y_target_sigma_factor * y_quad.std()
    cross_quad = np.where(np.diff(np.sign(y_quad - y_target_quad)))[0]
    win_lo_quad = float(x_grid[cross_quad[0]]) if len(cross_quad) >= 2 else np.nan
    win_hi_quad = float(x_grid[cross_quad[-1]]) if len(cross_quad) >= 2 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    for ax, (y_curve, title, win_lo, win_hi, y_t, color, extra) in zip(
        axes,
        [
            (y_quad, "Old: single quadratic fit", win_lo_quad, win_hi_quad, y_target_quad, "#c0392b",
             "rigid shape; no uncertainty band"),
            (y_pred, "New: GPR (non-parametric)", win_lo_gpr, win_hi_gpr, y_target_gpr, "#2c3e50",
             f"R² = {r2:.3f}; posterior band shown"),
        ],
    ):
        ax.scatter(df["item_value"], df["bin_value"], s=8, alpha=0.35, color="#7f8c8d")
        ax.plot(x_grid, y_curve, color=color, lw=2)
        if title.startswith("New"):
            ax.fill_between(
                x_grid, y_pred - 2 * y_std, y_pred + 2 * y_std,
                alpha=0.18, color=color,
            )
        ax.axhline(y_t, color="#e67e22", lw=1.3, ls="--")
        if np.isfinite(win_lo) and np.isfinite(win_hi):
            ax.axvline(win_lo, color="#8e44ad", lw=1.3, ls=":")
            ax.axvline(win_hi, color="#8e44ad", lw=1.3, ls=":")
            ax.axvspan(win_lo, win_hi, alpha=0.07, color="#27ae60")
            ax.set_title(f"{title}\nwindow = [{win_lo:.2f}, {win_hi:.2f}]\n{extra}")
        else:
            ax.set_title(f"{title}\nwindow = N/A (no crossing)\n{extra}")
        ax.set_xlabel("item_value")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("bin_value")

    fig.suptitle("Old pipeline (quadratic) vs New pipeline (GPR)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTDIR / "old_vs_new.png", dpi=130)
    plt.close(fig)
    print(f"  ✓ docs/old_vs_new.png")


def main() -> None:
    cfg = AnalysisConfig(gpr_n_restarts=1, bootstrap_n=300)
    print("Generating visualizations...")
    plot_pipeline_overview(cfg)
    plot_window_validation()
    plot_bootstrap_pwi(cfg)
    plot_old_vs_new(cfg)
    print("Done.")


if __name__ == "__main__":
    main()

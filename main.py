"""
PWI Analysis — quick-start entry point.

Run modes
---------
  python main.py               # single analysis with demo data
  python main.py --parallel    # parallel run across multiple m_key2 × bin_id pairs
  python main.py --n 8         # change number of groups (default: 10)
"""

import argparse
import logging

import numpy as np
import pandas as pd

from pwi_analysis import AnalysisConfig, PWIResult, pwi_analysis, run_parallel_pwi

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------

def make_demo_data(
    n: int = 800,
    n_keys: int = 3,
    n_bins: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic metro + EDS data with a U-shaped bin_value response.

    item_value is centred at 100; bin_value is minimised near 100 (the "good"
    process window) and rises toward the extremes — mimicking real yield data.
    """
    rng = np.random.default_rng(seed)

    lot_ids   = rng.integers(100, 120, n)
    wafer_ids = rng.integers(1, 26, n)
    item_vals = rng.normal(100, 10, n)

    metro_frames, eds_frames = [], []

    for key_idx in range(n_keys):
        m_key = f"KEY_{key_idx + 1:03d}"
        # Slight centre shift per key to simulate real process variation
        offset = rng.uniform(-3, 3)
        iv = item_vals + offset

        metro_frames.append(
            pd.DataFrame(
                {
                    "root_lot_id": lot_ids,
                    "wafer_id":    wafer_ids,
                    "item_value":  iv,
                    "m_key2":      m_key,
                }
            )
        )

        for bin_idx in range(n_bins):
            bin_id = f"BIN_{bin_idx + 1:03d}"
            # Each bin has a slightly different noise level
            noise_scale = 0.05 + bin_idx * 0.02
            bv = 0.005 * (iv - 100) ** 2 + rng.normal(0, noise_scale, n) + 0.5

            eds_frames.append(
                pd.DataFrame(
                    {
                        "root_lot_id": lot_ids,
                        "wafer_id":    wafer_ids,
                        "bin_value":   bv,
                        "bin_id":      bin_id,
                    }
                )
            )

    metro = pd.concat(metro_frames, ignore_index=True)
    eds   = pd.concat(eds_frames,   ignore_index=True)
    return metro, eds


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_result(result: PWIResult | None, message: str, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    if result is None:
        print(f"{prefix}→ Stopped: {message}")
        return

    print(
        f"{prefix}✓  window=[{result.window_low:.3f}±{result.window_low_std:.3f},"
        f" {result.window_high:.3f}±{result.window_high_std:.3f}]"
        f"  PWI={result.pwi_index:.1f}% (95% CI [{result.pwi_ci_low:.1f}, {result.pwi_ci_high:.1f}])"
        f"  R²={result.r2:.4f}"
        f"  depth={result.window_depth:.3f}"
        f"  y_target={result.y_target:.4f}"
    )


def print_parallel_summary(results: list[dict]) -> None:
    success = [r for r in results if r["result"] is not None]
    failed  = [r for r in results if r["result"] is None]

    print(f"\n{'─' * 60}")
    print(f"Parallel run: {len(results)} tasks  |  {len(success)} success  |  {len(failed)} skipped")
    print(f"{'─' * 60}")

    for r in success:
        print_result(r["result"], r["message"], label=f"{r['m_key2']} × {r['bin_id']}")

    if failed:
        print("\nSkipped:")
        for r in failed:
            print(f"  [{r['m_key2']} × {r['bin_id']}] {r['message']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PWI Analysis demo")
    parser.add_argument("--parallel", action="store_true", help="Run parallel multi-key analysis")
    parser.add_argument("--samples",  type=int, default=800, help="Rows per key (default: 800)")
    parser.add_argument("--keys",     type=int, default=3,   help="Number of m_key2 values (default: 3)")
    parser.add_argument("--bins",     type=int, default=2,   help="Number of bin_id values (default: 2)")
    return parser.parse_args()


def run_single(metro: pd.DataFrame, eds: pd.DataFrame, cfg: AnalysisConfig) -> None:
    # Use only the first key and first bin for a single clean demo
    m = metro[metro["m_key2"] == metro["m_key2"].iloc[0]]
    e = eds[eds["bin_id"]   == eds["bin_id"].iloc[0]]

    print("\n── Single analysis ──────────────────────────────────────")
    print(f"  metro rows : {len(m):,}  |  eds rows : {len(e):,}")
    print(f"  conf_level : {cfg.conf_level}  |  y_target_sigma_factor : {cfg.y_target_sigma_factor}")

    result, message = pwi_analysis(m, e, cfg)
    print_result(result, message)


def run_parallel(metro: pd.DataFrame, eds: pd.DataFrame, cfg: AnalysisConfig) -> None:
    print("\n── Parallel analysis ────────────────────────────────────")
    print(f"  m_key2 values : {sorted(metro['m_key2'].unique())}")
    print(f"  bin_id values : {sorted(eds['bin_id'].unique())}")

    results = run_parallel_pwi(metro, eds, cfg=cfg, n_jobs=-1)
    print_parallel_summary(results)


def main() -> None:
    args = parse_args()

    cfg = AnalysisConfig()

    print("Generating demo data …")
    metro, eds = make_demo_data(
        n=args.samples,
        n_keys=args.keys,
        n_bins=args.bins,
    )
    print(f"  metro : {len(metro):,} rows  |  eds : {len(eds):,} rows")

    if args.parallel:
        run_parallel(metro, eds, cfg)
    else:
        run_single(metro, eds, cfg)


if __name__ == "__main__":
    main()

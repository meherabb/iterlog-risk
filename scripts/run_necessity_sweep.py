#!/usr/bin/env python3
r"""Reproduces Table 14 (necessity vs. calibration size, full numbers) and the in-text
necessity table in Section 5.2 (Theorem 2, made empirical). CPU-only.

Usage
-----
    python scripts/run_necessity_sweep.py --config configs/necessity.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bands.lil_lower_bound import necessity_sweep
from src.utils.script_helpers import ensure_output_dir, load_config, print_run_header, save_json, save_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/necessity.yaml")
    parser.add_argument("--quick", action="store_true", help="reduced reps for a fast smoke test")
    args = parser.parse_args()

    config = load_config(args.config)
    print_run_header("run_necessity_sweep", config)
    out_dir = ensure_output_dir(config)

    c_values = tuple(config["c_values"])
    t0 = time.time()

    print(f"Main-body arm (Section 5.2's in-text table, n={config['main_body_arm']['n']}):")
    n_reps_main = 20 if args.quick else config["main_body_arm"]["n_reps"]
    main_body = necessity_sweep(
        n=config["main_body_arm"]["n"],
        p=config["p"],
        delta=config["delta"],
        k_min=config["k_min"],
        c_values=c_values,
        n_reps=n_reps_main,
        master_seed=config["master_seed"],
    )
    for r in main_body:
        print(
            f"  c={r.c:<5} rate={r.violation_rate:.4f}  95% CI [{r.ci95[0]:.4f}, {r.ci95[1]:.4f}]"
            f"  ({r.n_violations}/{r.n_reps})"
        )

    print(f"\nFull sweep across n in {config['n_values']} (Table 14):")
    n_reps_sweep = 20 if args.quick else config["n_reps"]
    sweep_results = {}
    for n in config["n_values"]:
        results = necessity_sweep(
            n=n,
            p=config["p"],
            delta=config["delta"],
            k_min=config["k_min"],
            c_values=c_values,
            n_reps=n_reps_sweep,
            master_seed=config["master_seed"],
        )
        sweep_results[n] = results
        rates_str = "  ".join(f"c={r.c}: {r.violation_rate:.4f}" for r in results)
        print(f"  n={n:>6}: {rates_str}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")

    # Flatten into arrays for the .npz checkpoint
    n_arr = np.array(config["n_values"])
    c_arr = np.array(c_values)
    rate_matrix = np.array([[r.violation_rate for r in sweep_results[n]] for n in config["n_values"]])
    save_npz(
        out_dir, config["output_prefix"], "sweep", n_values=n_arr, c_values=c_arr, violation_rate=rate_matrix
    )

    save_json(
        out_dir,
        config["output_prefix"],
        "summary",
        {
            "main_body_arm": [
                {"c": r.c, "rate": r.violation_rate, "ci95": list(r.ci95), "n_reps": r.n_reps}
                for r in main_body
            ],
            "sweep": {
                str(n): [{"c": r.c, "rate": r.violation_rate} for r in results]
                for n, results in sweep_results.items()
            },
            "elapsed_seconds": elapsed,
        },
    )
    print(f"Results written to {out_dir}/{config['output_prefix']}_{{sweep.npz,summary.json}}")


if __name__ == "__main__":
    main()

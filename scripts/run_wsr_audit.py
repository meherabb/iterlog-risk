#!/usr/bin/env python3
r"""Reproduces Table 11 (betting bound before/after the Lemma 1 patch), Appendix D.3.

With --synthetic (the default, and the only mode runnable without GPU/dataset access),
runs the same heterogeneous-risk construction validated in tests/test_wsr_betting.py: a
linear risk ramp that triggers the documented silent-margin defect far more often than
i.i.d. data does. Without --synthetic, this script expects real deployment data from
scripts/run_deployment.py's output (see docs/REPRODUCING.md) -- that path requires the
same models/data as configs/deployment.yaml and is not runnable in this repository's own
CPU-only development environment.

Usage
-----
    python scripts/run_wsr_audit.py --config configs/wsr_audit.yaml --synthetic
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import wsr_betting
from src.utils.determinism import derive_seed
from src.utils.script_helpers import ensure_output_dir, load_config, print_run_header, save_json, save_npz


def calibration_side_audit(n_items, eta_range, delta, n_trials, master_seed):
    """The paper's own diagnostic: a single pre-registered k0, no selection, no transfer --
    if a calibration-side certificate is exact, it should violate at close to (not above)
    delta here."""
    raw_violations = 0
    patched_violations = 0
    for trial in range(n_trials):
        rng = np.random.default_rng(derive_seed("wsr_audit/calibration_side", trial, master_seed))
        eta = np.linspace(eta_range[0], eta_range[1], n_items)
        data = rng.binomial(1, eta).astype(float)
        true_mean = eta.mean()

        raw = wsr_betting.upper_bound_buggy(data, delta=delta).upper_bound
        patched = wsr_betting.patched_upper_bound(data, delta=delta)

        if raw < true_mean:
            raw_violations += 1
        if patched < true_mean:
            patched_violations += 1

    return {
        "raw_rate": raw_violations / n_trials,
        "patched_rate": patched_violations / n_trials,
        "n_trials": n_trials,
    }


def bug_trigger_rate_comparison(n_items, eta_range, delta, n_trials, master_seed):
    """The core mechanism check: does the silent-margin shortcut fire more often on
    heterogeneous (sorted, non-stationary) data than on i.i.d. data of the same size?"""

    def iid_trial(rng):
        return rng.binomial(1, np.mean(eta_range), size=n_items).astype(float)

    def heterogeneous_trial(rng):
        eta = np.linspace(eta_range[0], eta_range[1], n_items)
        return rng.binomial(1, eta).astype(float)

    iid_fires = het_fires = 0
    for trial in range(n_trials):
        rng = np.random.default_rng(derive_seed("wsr_audit/bug_rate", trial, master_seed))
        if wsr_betting.upper_bound_buggy(iid_trial(rng), delta=delta).used_buggy_shortcut:
            iid_fires += 1
        if wsr_betting.upper_bound_buggy(heterogeneous_trial(rng), delta=delta).used_buggy_shortcut:
            het_fires += 1

    return {
        "iid_rate": iid_fires / n_trials,
        "heterogeneous_rate": het_fires / n_trials,
        "n_trials": n_trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wsr_audit.yaml")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="use the synthetic heterogeneous-risk construction (default; "
        "the real-data path needs scripts/run_deployment.py's output)",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    print_run_header("run_wsr_audit", config)
    out_dir = ensure_output_dir(config)

    if not args.synthetic:
        print("Real-data mode requires scripts/run_deployment.py's output; see docs/REPRODUCING.md.")
        print("Falling back to --synthetic.")

    sc = config["synthetic_fallback"]
    n_trials = 30 if args.quick else sc["n_trials"]

    t0 = time.time()
    print("Calibration-side audit (no selection, no transfer)...")
    cal_audit = calibration_side_audit(
        sc["n_items"], sc["eta_range"], config["delta"], n_trials, config["master_seed"]
    )
    print(f"  Raw violation rate:     {cal_audit['raw_rate']:.3f}")
    print(f"  Patched violation rate: {cal_audit['patched_rate']:.3f}")

    print("\nBug-trigger-rate comparison (i.i.d. vs. heterogeneous)...")
    trigger = bug_trigger_rate_comparison(
        sc["n_items"], sc["eta_range"], config["delta"], n_trials, config["master_seed"]
    )
    print(f"  i.i.d. bug rate:            {trigger['iid_rate']:.3f}  (paper: ~0.01)")
    print(
        f"  heterogeneous bug rate:     {trigger['heterogeneous_rate']:.3f}  (paper: ~0.32; "
        f"this synthetic ramp is a more extreme stress test, so a higher rate is expected)"
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")

    save_npz(
        out_dir,
        config["output_prefix"],
        "synthetic",
        cal_raw_rate=cal_audit["raw_rate"],
        cal_patched_rate=cal_audit["patched_rate"],
        iid_bug_rate=trigger["iid_rate"],
        heterogeneous_bug_rate=trigger["heterogeneous_rate"],
    )
    save_json(
        out_dir,
        config["output_prefix"],
        "summary",
        {"calibration_side_audit": cal_audit, "bug_trigger_comparison": trigger, "elapsed_seconds": elapsed},
    )
    print(f"Results written to {out_dir}/{config['output_prefix']}_{{synthetic.npz,summary.json}}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""Reproduces Table 7 (exchangeability stress test) and Figure 6.

**Requires CIFAR-10/CIFAR-10-C + a trained ResNet-18** to get the real per-severity exact
conditional risk. The statistical protocol itself (calibrate on clean, deploy on shifted,
check violation rate and the mean conditional-risk gap) is pure NumPy; --smoke-test
exercises it against a synthetic stand-in for increasing distribution shift.

Usage
-----
    python scripts/run_exchangeability.py --config configs/exchangeability.yaml --smoke-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bands import sorted_filtration, uniform_band
from src.bands.validity import clopper_pearson_ci_on_rate, is_violated
from src.eval.vision_tier_a import exchangeability_violation_gap
from src.utils.determinism import derive_seed
from src.utils.script_helpers import ensure_output_dir, load_config, print_run_header, save_json, save_npz


def run_severity(severity, eta_clean, scores_clean, eta_shift_fn, n_cal, delta, k_min, n_reps, master_seed):
    r"""Calibrate a band on clean data, then check it against a *shifted* eta at the same
    items -- the paper's own construction for "calibrate on clean, deploy on corrupted"
    without literally re-scoring images at every severity (real CIFAR-10-C would re-run the
    model at each severity; here the shift is injected directly into eta as the stand-in).
    """
    violations = 0
    for rep in range(n_reps):
        rng = np.random.default_rng(derive_seed(f"exchangeability/severity{severity}", rep, master_seed))
        idx = rng.choice(len(eta_clean), size=n_cal, replace=False)
        eta_cal = eta_clean[idx]
        scores_cal = scores_clean[idx]

        # Calibrate the band on the CLEAN calibration data.
        losses_cal = rng.binomial(1, eta_cal).astype(float)
        pi = sorted_filtration.build_permutation(scores_cal, rng)
        curves = sorted_filtration.realized_and_empirical_curves(losses_cal, pi, eta=eta_cal)
        U_k = uniform_band._band_from_curve(curves.Rhat_k, delta=delta)

        # Deploy: check the SAME band against the shifted eta at severity > 0. Re-derive
        # the realized curve under the *same* permutation pi used for calibration, since
        # exchangeability violation means the sort order no longer matches the shifted
        # risk landscape.
        eta_shifted = eta_shift_fn(eta_cal, severity)
        Rbar_shifted_sorted = np.cumsum(eta_shifted[pi]) / np.arange(1, n_cal + 1)

        if is_violated(Rbar_shifted_sorted, U_k, k_min=k_min).violated:
            violations += 1

    lo, hi = clopper_pearson_ci_on_rate(violations, n_reps)
    return {"violations": violations, "n_reps": n_reps, "rate": violations / n_reps, "ci95": [lo, hi]}


def linear_shift(eta, severity, max_gap=0.35):
    """A synthetic stand-in for CIFAR-10-C's severity-indexed shift: risk increases
    linearly with severity, matching the paper's own reported gap growing from +0.063
    (severity 1) to +0.354 (severity 5)."""
    if severity == 0:
        return eta.copy()
    gap = max_gap * (severity / 5.0)
    return np.clip(eta + gap, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exchangeability.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    print_run_header("run_exchangeability", config)
    out_dir = ensure_output_dir(config)

    if not args.smoke_test:
        print(
            f"Full run needs {config['model']} trained on CIFAR-10 and evaluated across "
            "CIFAR-10-C's severities, which this environment cannot produce (no GPU/dataset "
            "access). Run with --smoke-test for a synthetic-shift check of the statistical "
            "protocol itself."
        )
        return

    rng = np.random.default_rng(0)
    n_items = 5000
    eta_clean = rng.beta(2, 8, size=n_items)  # mean ~0.2, roughly matching a well-trained classifier
    scores_clean = -eta_clean + rng.normal(scale=0.1, size=n_items)

    results = []
    for severity in config["severities"]:
        r = run_severity(
            severity,
            eta_clean,
            scores_clean,
            linear_shift,
            config["n_cal"],
            config["delta"],
            100,
            n_reps=50,
            master_seed=config["master_seed"],
        )
        gap = exchangeability_violation_gap(eta_clean, linear_shift(eta_clean, severity))
        r["mean_eta_gap"] = gap
        results.append(r)
        print(f"  severity {severity}: violation rate={r['rate']:.3f}  mean-eta gap={gap:+.3f}")

    save_npz(
        out_dir,
        config["output_prefix"],
        "smoke_test",
        severities=np.array(config["severities"]),
        violation_rate=np.array([r["rate"] for r in results]),
        mean_eta_gap=np.array([r["mean_eta_gap"] for r in results]),
    )
    save_json(out_dir, config["output_prefix"], "smoke_test_summary", {"results": results})


if __name__ == "__main__":
    main()

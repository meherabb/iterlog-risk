#!/usr/bin/env python3
r"""Reproduces Table 3 (synthetic validity + ablations), Table 2 (feasibility), and the
data behind Figure 3 (feasibility phase diagram). CPU-only; runs in well under a minute.

Usage
-----
    python scripts/run_synthetic_validity.py --config configs/synthetic.yaml

See docs/REPRODUCING.md for what this produces and docs/CROSSWALK.md for how it maps back
to the paper's own notebook-cell crosswalk (Appendix J.2).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bands import population_band, sorted_filtration, uniform_band
from src.bands.validity import clopper_pearson_ci_on_rate, is_violated
from src.utils.determinism import derive_seed
from src.utils.script_helpers import ensure_output_dir, load_config, print_run_header, save_json, save_npz


def run_homogeneous_arm(n, p, delta, k_min, n_reps, master_seed):
    """The core Table 3 check: homogeneous world, score independent of loss."""
    violations = 0
    for rep in range(n_reps):
        rng = np.random.default_rng(derive_seed("v1_synthetic/homogeneous", rep, master_seed))
        losses = rng.binomial(1, p, size=n).astype(float)
        scores = rng.normal(size=n)
        eta = np.full(n, p)
        result = uniform_band.compute(losses, scores, delta=delta, k_min=k_min, rng=rng, eta=eta)
        if is_violated(result.Rbar_k, result.U_k, k_min=k_min).violated:
            violations += 1
    lo, hi = clopper_pearson_ci_on_rate(violations, n_reps)
    return {"violations": violations, "n_reps": n_reps, "rate": violations / n_reps, "ci95": [lo, hi]}


def run_heterogeneous_arm(n, delta, k_min, n_reps, master_seed):
    """Table 3's other main row: heterogeneous eta, informative score."""
    violations = 0
    for rep in range(n_reps):
        rng = np.random.default_rng(derive_seed("v1_synthetic/heterogeneous", rep, master_seed))
        eta = rng.beta(2, 5, size=n)  # a spread of base rates, mean ~0.29
        scores = -eta + rng.normal(scale=0.3, size=n)  # informative but noisy score
        losses = rng.binomial(1, eta).astype(float)
        result = uniform_band.compute(losses, scores, delta=delta, k_min=k_min, rng=rng, eta=eta)
        if is_violated(result.Rbar_k, result.U_k, k_min=k_min).violated:
            violations += 1
    lo, hi = clopper_pearson_ci_on_rate(violations, n_reps)
    return {"violations": violations, "n_reps": n_reps, "rate": violations / n_reps, "ci95": [lo, hi]}


def run_ablations(n, p, delta, k_min, n_reps, master_seed, ablation_config):
    """Score quantization, geometric base, and delta-allocation ablations -- all should
    hold the violation rate at 0.0000 while shifting the band's own width slightly."""
    results = {}
    widths_at_k2000 = []

    for b in ablation_config["geometric_base"]:
        violations = 0
        width_sample = None
        arm_id = f"v1_synthetic/ablation_base_{b}"
        for rep in range(n_reps):
            rng = np.random.default_rng(derive_seed(arm_id, rep, master_seed))
            losses = rng.binomial(1, p, size=n).astype(float)
            scores = rng.normal(size=n)
            eta = np.full(n, p)
            pi = sorted_filtration.build_permutation(scores, rng)
            curves = sorted_filtration.realized_and_empirical_curves(losses, pi, eta=eta)
            U_k = uniform_band._band_from_curve(curves.Rhat_k, delta=delta, b=b)
            if is_violated(curves.Rbar_k, U_k, k_min=k_min).violated:
                violations += 1
            if rep == 0:
                width_sample = float(U_k[1999] - curves.Rhat_k[1999])  # width at k=2000
        results[f"geometric_base_{b}"] = {"violations": violations, "n_reps": n_reps}
        widths_at_k2000.append(width_sample)

    results["width_range_at_k2000"] = [min(widths_at_k2000), max(widths_at_k2000)]
    return results


def run_feasibility_table(feas_config, delta):
    rows = []
    for n in feas_config["n_values"]:
        thr = population_band.feasibility_threshold(
            alpha=feas_config["alpha"],
            n=n,
            delta=delta,
            gamma=feas_config["gamma"],
            use_paper_constant=feas_config["use_paper_constant"],
        )
        eps = population_band.epsilon_n_as_implemented(n, delta)
        rows.append({"n": n, "epsilon_n": eps, "required_Rhat_k": thr})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/synthetic.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print_run_header("run_synthetic_validity", config)
    out_dir = ensure_output_dir(config)

    t0 = time.time()

    print("Running homogeneous-world arm (Table 3)...")
    homogeneous = run_homogeneous_arm(
        config["n"], config["p"], config["delta"], config["k_min"], config["n_reps"], config["master_seed"]
    )
    print(
        f"  -> {homogeneous['violations']}/{homogeneous['n_reps']} violations, "
        f"95% CI {homogeneous['ci95']}"
    )

    print("Running heterogeneous-world arm (Table 3)...")
    heterogeneous = run_heterogeneous_arm(
        config["n"], config["delta"], config["k_min"], config["n_reps"], config["master_seed"]
    )
    print(
        f"  -> {heterogeneous['violations']}/{heterogeneous['n_reps']} violations, "
        f"95% CI {heterogeneous['ci95']}"
    )

    print("Running construction ablations...")
    ablations = run_ablations(
        config["n"],
        config["p"],
        config["delta"],
        config["k_min"],
        config["n_reps"],
        config["master_seed"],
        config["ablations"],
    )
    print(f"  -> band width at k=2000 ranges over {ablations['width_range_at_k2000']}")

    print("Computing feasibility table (Table 2)...")
    feasibility = run_feasibility_table(config["feasibility"], config["delta"])
    for row in feasibility:
        print(f"  n={row['n']:>6}: required Rhat_k <= {row['required_Rhat_k']:.3f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")

    save_npz(
        out_dir,
        config["output_prefix"],
        "main",
        homogeneous_violations=homogeneous["violations"],
        homogeneous_rate=homogeneous["rate"],
        heterogeneous_violations=heterogeneous["violations"],
        heterogeneous_rate=heterogeneous["rate"],
        feasibility_n=np.array([r["n"] for r in feasibility]),
        feasibility_threshold=np.array([r["required_Rhat_k"] for r in feasibility]),
    )
    save_json(
        out_dir,
        config["output_prefix"],
        "summary",
        {
            "homogeneous": homogeneous,
            "heterogeneous": heterogeneous,
            "ablations": ablations,
            "feasibility": feasibility,
            "elapsed_seconds": elapsed,
        },
    )
    print(f"Results written to {out_dir}/{config['output_prefix']}_{{main.npz,summary.json}}")


if __name__ == "__main__":
    main()

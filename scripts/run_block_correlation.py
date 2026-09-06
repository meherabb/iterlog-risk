#!/usr/bin/env python3
r"""Reproduces Table 13 (block-correlation sweep, full numbers) and the data behind
Figure 2. CPU-only.

Runs the paper's own two block structures (Section 5.4) at synthetic scale: the "llm_pool"
structure uses 4 blocks with heterogeneous marginal risk (mirroring the four source
datasets' different base rates); "cifar10" uses 10 equal-sized blocks (mirroring the ten
class labels). Both use the Gaussian copula (src/data/copula.py) to induce correlation
while holding each item's marginal eta exactly fixed.

Usage
-----
    python scripts/run_block_correlation.py --config configs/block_correlation.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bands import block_robust, sorted_filtration, uniform_band
from src.bands.validity import is_violated
from src.data.copula import correlated_losses
from src.utils.determinism import derive_seed
from src.utils.script_helpers import ensure_output_dir, load_config, print_run_header, save_json, save_npz


def run_one_cell(
    structure_name, n_blocks, approx_block_sizes, rho, delta, item_k_min, block_k_min, n_reps, master_seed
):
    """One (block_structure, rho) cell: item-level vs. block-level violation rate."""
    item_violations = 0
    block_violations = 0
    arm_id = f"blockcorr/{structure_name}/rho{rho}"

    for rep in range(n_reps):
        rng = np.random.default_rng(derive_seed(arm_id, rep, master_seed))

        block_eta_true = rng.uniform(0.1, 0.4, size=n_blocks)
        block_of_item = np.concatenate([np.full(size, b) for b, size in enumerate(approx_block_sizes)])
        eta = block_eta_true[block_of_item]
        n = len(eta)
        scores = rng.normal(size=n) - 2.0 * eta  # score informatively (anti-)correlated with risk
        losses = correlated_losses(eta, block_of_item, rho, rng)

        # Item level (ignores block structure)
        pi = sorted_filtration.build_permutation(scores, rng)
        curves = sorted_filtration.realized_and_empirical_curves(losses, pi, eta=eta)
        U_k_item = uniform_band._band_from_curve(curves.Rhat_k, delta=delta)
        if is_violated(curves.Rbar_k, U_k_item, k_min=item_k_min).violated:
            item_violations += 1

        # Block level (Theorem 4)
        assignment = block_robust.assign_blocks(block_of_item)
        block_scores = np.array([scores[block_of_item == b].mean() for b in range(n_blocks)])
        block_losses, _, block_eta_obs = block_robust.block_level_curves(
            losses, block_scores, assignment, eta=eta
        )
        res = block_robust.compute(
            block_losses, block_scores, delta=delta, k_min=block_k_min, rng=rng, block_eta=block_eta_obs
        )
        if is_violated(res.Rbar_k, res.U_k, k_min=block_k_min).violated:
            block_violations += 1

    return item_violations / n_reps, block_violations / n_reps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/block_correlation.yaml")
    parser.add_argument("--quick", action="store_true", help="reduced reps for a fast smoke test")
    args = parser.parse_args()

    config = load_config(args.config)
    print_run_header("run_block_correlation", config)
    out_dir = ensure_output_dir(config)
    n_reps = 30 if args.quick else config["n_reps"]

    t0 = time.time()
    all_results = {}

    for structure_name, structure_cfg in config["block_structures"].items():
        print(f"\nBlock structure: {structure_name} ({structure_cfg['n_blocks']} blocks)")
        cell_results = []
        for rho in config["rho_values"]:
            item_rate, block_rate = run_one_cell(
                structure_name,
                structure_cfg["n_blocks"],
                structure_cfg["approx_block_sizes"],
                rho,
                config["delta"],
                config["item_level_k_min"],
                config["block_level_k_min"],
                n_reps,
                config["master_seed"],
            )
            cell_results.append({"rho": rho, "item_rate": item_rate, "block_rate": block_rate})
            print(f"  rho={rho}: item={item_rate:.3f}  block={block_rate:.3f}")
        all_results[structure_name] = cell_results

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")

    rho_arr = np.array(config["rho_values"])
    save_npz(
        out_dir,
        config["output_prefix"],
        "sweep",
        rho_values=rho_arr,
        llm_pool_item=np.array([r["item_rate"] for r in all_results["llm_pool"]]),
        llm_pool_block=np.array([r["block_rate"] for r in all_results["llm_pool"]]),
        cifar10_item=np.array([r["item_rate"] for r in all_results["cifar10"]]),
        cifar10_block=np.array([r["block_rate"] for r in all_results["cifar10"]]),
    )
    save_json(out_dir, config["output_prefix"], "summary", {**all_results, "elapsed_seconds": elapsed})
    print(f"Results written to {out_dir}/{config['output_prefix']}_{{sweep.npz,summary.json}}")


if __name__ == "__main__":
    main()

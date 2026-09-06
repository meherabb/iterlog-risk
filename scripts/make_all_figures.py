#!/usr/bin/env python3
r"""Regenerate every figure from already-computed result files, without re-running any
experiment. CPU-only and fast -- use this after running the arms in docs/REPRODUCING.md
once, to iterate on figure styling without paying the experiment cost again.

Usage
-----
    python scripts/make_all_figures.py --results-dir results/ --figure necessity
    python scripts/make_all_figures.py --results-dir results/ --figure all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; no display needed to write PNG/PDF files
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def figure_necessity(results_dir: Path, out_dir: Path) -> Path | None:
    """Figure 1: the necessity sweep, band-shrinkage-factor vs. violation rate."""
    path = results_dir / "v3_necessity_sweep.npz"
    if not path.exists():
        print(f"  [skip] {path} not found -- run scripts/run_necessity_sweep.py first")
        return None
    data = np.load(path)
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, n in enumerate(data["n_values"]):
        ax.plot(data["c_values"], data["violation_rate"][i], marker="o", label=f"n={n}")
    ax.set_xlabel("width fraction c")
    ax.set_ylabel("violation rate")
    ax.set_title("Theorem 2: the iterated logarithm is not slack")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "figure_necessity.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def figure_blockcorr(results_dir: Path, out_dir: Path) -> Path | None:
    """Figure 2: item- vs. block-level violation rate against induced correlation."""
    path = results_dir / "blockcorr_sweep.npz"
    if not path.exists():
        print(f"  [skip] {path} not found -- run scripts/run_block_correlation.py first")
        return None
    data = np.load(path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, prefix, title in zip(axes, ["llm_pool", "cifar10"], ["LLM pool", "CIFAR-10"]):
        ax.plot(data["rho_values"], data[f"{prefix}_item"], marker="o", label="item-level")
        ax.plot(data["rho_values"], data[f"{prefix}_block"], marker="s", label="block-level")
        ax.set_xlabel(r"induced correlation $\rho$")
        ax.set_title(title)
    axes[0].set_ylabel("violation rate")
    axes[0].legend()
    fig.suptitle("Theorem 4: block-robust certification under induced correlation")
    fig.tight_layout()
    out_path = out_dir / "figure_blockcorr.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def figure_feasibility_phase(results_dir: Path, out_dir: Path) -> Path | None:
    """Figure 3: the feasibility phase diagram (Corollary 2)."""
    path = results_dir / "v1_synthetic_main.npz"
    if not path.exists():
        print(f"  [skip] {path} not found -- run scripts/run_synthetic_validity.py first")
        return None
    data = np.load(path)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(data["feasibility_n"], data["feasibility_threshold"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("calibration size n")
    ax.set_ylabel(r"required $\hat R_k$")
    ax.set_title("Corollary 2: feasibility threshold vs. calibration size")
    fig.tight_layout()
    out_path = out_dir / "figure_feasibility_phase.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


FIGURE_REGISTRY = {
    "necessity": figure_necessity,
    "blockcorr": figure_blockcorr,
    "feasibility_phase": figure_feasibility_phase,
    # fig-band, fig-heatmap, fig-price, fig-realdata, fig-robust, fig-fragility, fig-transfer,
    # fig-utility all require the real LLM/vision results from scripts/run_tier_a.py,
    # run_deployment.py, and run_exchangeability.py, which need GPU/dataset access this
    # repository's own development environment did not have -- their plotting functions
    # follow the identical registry pattern above and are straightforward to add once you
    # have those results.
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--out-dir", default="results/figures/")
    parser.add_argument("--figure", default="all", choices=["all", *FIGURE_REGISTRY.keys()])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures_to_make = FIGURE_REGISTRY.keys() if args.figure == "all" else [args.figure]
    print(f"Reading results from {results_dir}, writing figures to {out_dir}\n")
    for name in figures_to_make:
        print(f"{name}:")
        path = FIGURE_REGISTRY[name](results_dir, out_dir)
        if path is not None:
            print(f"  -> {path}")


if __name__ == "__main__":
    main()

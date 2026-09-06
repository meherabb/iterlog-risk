#!/usr/bin/env python3
r"""Reproduces Appendix I (the confounded n_cal=20,000 scaling arm), including its own
power calculation showing the result is confounded by an undersized holdout rather than
conclusive either way.

**Requires the real Phi-3.5-mini-instruct deployment run at n_cal=20,000** to get the
actual baseline overrun rates. The power calculation itself -- comparing the held-out
standard error against each certificate's width -- is pure arithmetic and is exercised
below directly, since it needs only the widths, not a live model.

Usage
-----
    python scripts/run_scaling_experiment.py --config configs/scaling.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.script_helpers import ensure_output_dir, load_config, print_run_header, save_json


def held_out_standard_error(n_selected: int, base_error_rate: float = 0.5) -> float:
    r"""The standard error of a held-out error-rate *measurement* itself, among the
    ``n_selected`` items actually above the chosen threshold (Appendix I: "about 2,000 of
    them above threshold" -- this selected-subset size, not the full held-out pool size, is
    the correct denominator here; using the full pool size overstates n and cannot
    reproduce the paper's own reported SE=0.0107, since sqrt(p(1-p)/4000) has no solution
    for p(1-p) <= 0.25 that reaches 0.0107, while sqrt(p(1-p)/2000) does).
    ``base_error_rate=0.5`` is the worst case / largest possible SE for a fixed n, used as a
    conservative stand-in when the real selected-subset rate isn't available."""
    return float(np.sqrt(base_error_rate * (1 - base_error_rate) / n_selected))


def exceeded_by_noise_alone(certificate_width: float, standard_error: float) -> float:
    r"""If a certificate's own width were *exactly* tight (no slack at all), what fraction
    of held-out measurements would exceed it purely from sampling noise around the true
    rate? Modeled as a one-sided normal tail at ``certificate_width / standard_error``
    standard errors out -- the same logic Appendix I uses to argue the $n_{\mathrm{cal}}=
    20{,}000$ arm's elevated rates are explainable by an undersized holdout alone.
    """
    from scipy import stats

    z = certificate_width / standard_error
    return float(1.0 - stats.norm.cdf(z))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scaling.yaml")
    parser.add_argument(
        "--certificate-widths",
        nargs="*",
        type=float,
        default=[0.0080, 0.0122, 0.0139],
        help="widths to run the power calculation against (default: the paper's own "
        "reported Clopper-Pearson/Hoeffding/empirical-Bernstein widths at this scale)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print_run_header("run_scaling_experiment", config)
    out_dir = ensure_output_dir(config)

    print(
        f"Full run needs {config['model']} scored at n_cal={config['n_cal']}, which this "
        "environment cannot produce (no GPU/Hugging Face Hub access).\n"
        "The power calculation below only needs the held-out size and certificate widths, "
        "and reproduces Appendix I's own argument that this arm is confounded, not "
        "conclusive:\n"
    )

    se = held_out_standard_error(config["selected_above_threshold"])
    print(f"Held-out standard error at n_selected={config['selected_above_threshold']}: {se:.4f}")
    print(f"{'certificate width':>18} {'exceeded-by-noise-alone rate':>30}")

    rows = []
    for width in args.certificate_widths:
        rate = exceeded_by_noise_alone(width, se)
        print(f"{width:>18.4f} {rate:>30.3f}")
        rows.append({"width": width, "exceeded_by_noise_alone_rate": rate})

    print(
        "\nIf these noise-alone rates land within a couple of points of the certificates' "
        "*observed* overrun rates at this scale (Appendix I reports 0.229/0.127/0.098 "
        "against observed 0.244/0.144/0.118), the experiment cannot separate an invalid "
        "certificate from a valid one measured with an undersized holdout -- which is "
        "exactly the paper's own conclusion here, not a numerical coincidence this script "
        "can independently verify without the real per-model data."
    )

    save_json(
        out_dir,
        config["output_prefix"],
        "power_calculation",
        {"selected_above_threshold": config["selected_above_threshold"], "standard_error": se, "rows": rows},
    )


if __name__ == "__main__":
    main()

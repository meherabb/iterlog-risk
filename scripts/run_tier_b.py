#!/usr/bin/env python3
r"""Reproduces Table 5 (Tier-B validity under genuine decoding noise).

**Requires GPU + Hugging Face Hub access** to genuinely decode 32 samples/question at
T=1 on TriviaQA for Qwen2.5-1.5B-Instruct and Phi-3.5-mini-instruct. The leave-one-out
statistical logic itself (``src/eval/tier_protocols.py``) is pure NumPy and fully tested
in ``tests/test_scoring_and_eval.py`` against synthetic decode-correctness arrays.

Usage
-----
    python scripts/run_tier_b.py --config configs/tier_b.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval import tier_protocols
from src.utils.determinism import derive_seed
from src.utils.script_helpers import ensure_output_dir, load_config, print_run_header, save_json


def genuine_decode(model, tokenizer, question: str, n_samples: int, temperature: float, max_new_tokens: int):
    """The one GPU-dependent step: decode n_samples genuine completions. Not implemented
    further here -- see the module docstring."""
    raise NotImplementedError("genuine_decode needs a live model; see the module docstring.")


def run_arm(model_name, score_family, is_wrong_matrix, delta, k_min, master_seed):
    r"""One Tier-B arm: for every (item, held-out-sample) pair, check the held-out draw
    against the leave-one-out $\hat\eta$ curve built from the other 31 samples.

    ``is_wrong_matrix`` shape ``(n_items, n_samples)`` stands in for genuine decode
    correctness -- in the real script this comes from ``genuine_decode``; here it lets the
    leave-one-out/scoring logic itself be exercised and checked end to end without a model.
    """
    rotations = tier_protocols.tier_b_all_rotations(is_wrong_matrix)
    n_items, n_samples = is_wrong_matrix.shape
    # Treat each (item, held-out-sample) pair as one "replication" of a length-1 held-out
    # check against the estimated eta -- consistent with Table 5's per-arm interval being
    # reported over the pooled n_items * n_samples draws.
    n_reps = n_items * n_samples
    violations = int(
        np.sum(rotations.eta_hat < rotations.raw_loss)
    )  # eta_hat underestimated an actual wrong answer
    return {"arm_id": f"tierB/{model_name}/{score_family}", "violations": violations, "n_reps": n_reps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tier_b.yaml")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="exercise the leave-one-out logic on synthetic decode data (no GPU needed)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print_run_header("run_tier_b", config)
    out_dir = ensure_output_dir(config)

    if not args.smoke_test:
        print(
            f"Full Tier-B decoding requires GPU + Hugging Face Hub access for {config['models']} "
            f"on {config['dataset']}, which this environment does not have.\n"
            "Run with --smoke-test to exercise the leave-one-out statistical logic on "
            "synthetic decode data instead."
        )
        return

    rng = np.random.default_rng(derive_seed("tierB/smoke_test", 0, config["master_seed"]))
    n_items, n_samples = 500, config["decoding"]["n_samples_per_question"]
    true_error_rate = 0.3
    is_wrong = rng.binomial(1, true_error_rate, size=(n_items, n_samples)).astype(float)

    result = run_arm(
        "smoke-test-model", "maxprob", is_wrong, config["delta"], config["k_min"], config["master_seed"]
    )
    rate = result["violations"] / result["n_reps"]
    print(
        f"Smoke test: {result['violations']}/{result['n_reps']} = {rate:.4f} "
        f"(leave-one-out estimate disagreeing with the true held-out draw; not a Theorem-1 "
        f"violation rate on its own -- see docs/REPRODUCING.md for how this feeds Table 5)"
    )

    save_json(out_dir, config["output_prefix"], "smoke_test", result)


if __name__ == "__main__":
    main()

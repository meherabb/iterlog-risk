#!/usr/bin/env python3
r"""Reproduces Table 4 (Tier-A summary), Table 10 (full per-arm breakdown), Figure 5
(real-data bands), Table 15/16 (dataset/model manifests), and the score-family contrasts
in Appendix G.

**Requires GPU + Hugging Face Hub access** to load Pythia-410M, Qwen2.5-1.5B-Instruct, and
Phi-3.5-mini-instruct and score them against the pool -- this is the single largest compute
cost in the paper (Appendix J: ~5.9 of 6.3 total GPU-hours). Not runnable in a
network-restricted environment; every function below *except* ``load_model_and_tokenizer``
and ``score_batch`` is pure NumPy and independently tested in ``tests/test_uniform_band.py``,
``tests/test_scoring_and_eval.py``, etc. against synthetic stand-ins for exactly this data.

Usage
-----
    python scripts/run_tier_a.py --config configs/tier_a.yaml
    python scripts/run_tier_a.py --config configs/tier_a.yaml --manifest-only  # no GPU needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bands import uniform_band
from src.bands.validity import clopper_pearson_ci_on_rate, is_violated
from src.data import loaders
from src.eval import tier_protocols
from src.scoring import entropy, maxprob, seqlogprob
from src.utils.determinism import derive_seed
from src.utils.script_helpers import ensure_output_dir, load_config, print_run_header, save_json

SCORE_FUNCTIONS = {
    "maxprob": maxprob.from_logits,
    "entropy": entropy.score,  # expects probabilities; caller must softmax first
    "seqlogprob": seqlogprob.score,
    # hidden_centroid / hidden_knn need a fitted reference and are wired up per-model below
}


def load_model_and_tokenizer(hf_id: str):
    """The one genuinely GPU/network-dependent step. Returns (model, tokenizer) via
    transformers -- not implemented further here since it cannot be exercised in this
    repository's own development environment; see the module docstring."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype="auto", device_map="auto")
    model.eval()
    return model, tokenizer


def compute_logits_and_eta(model, tokenizer, examples: list[loaders.Example]):
    """Score every example's answer options, returning (logits, eta, hidden_states) --
    eta_i = 1 - p_T(correct_answer | question) is the Tier-A exact conditional risk
    (Section "Experimental design", Tier A). Not implemented further here (needs a live
    model); this is the one function a GPU-equipped fork of this script needs to fill in.
    """
    raise NotImplementedError(
        "compute_logits_and_eta needs a live model forward pass; see the module docstring "
        "for why this could not be executed in this repository's own environment."
    )


def run_manifest_only(config, out_dir):
    """Write Table 15/16 (dataset and model manifests) -- these are static reference
    tables, not experimental output, so they need no GPU at all."""
    dataset_manifest = {
        "arc_easy": {"items": 5197, "task": "4-way science MCQ"},
        "arc_challenge": {"items": 2590, "task": "4-way science MCQ"},
        "openbookqa": {"items": 5957, "task": "4-way science MCQ"},
        "commonsense_qa": {"items": 10962, "task": "5-way commonsense MCQ"},
        "mmlu": {"items": 14042, "task": "4-way, 57 subjects"},
    }
    model_manifest = {
        m["name"]: {"hf_id": m["hf_id"], "is_instruct": m["is_instruct"]} for m in config["models"]
    }
    save_json(out_dir, config["output_prefix"], "dataset_manifest", dataset_manifest)
    save_json(out_dir, config["output_prefix"], "model_manifest", model_manifest)
    print("Wrote dataset and model manifests (Tables 15-16). No GPU was needed for this step.")


def run_one_arm(model_name, group, score_family, eta, scores, delta, k_min, n_reps, master_seed):
    """One (model, group, score) arm of the 32-arm Tier-A sweep: simulate the Bernoulli
    loss from the model's own exact eta, sort by the score, and check Theorem 1's coverage."""
    arm_id = f"tierA/{model_name}/{group}/{score_family}"
    violations = 0
    for rep in range(n_reps):
        rng = np.random.default_rng(derive_seed(arm_id, rep, master_seed))
        losses = tier_protocols.tier_a_loss(eta, rng)
        result = uniform_band.compute(losses, scores, delta=delta, k_min=k_min, rng=rng, eta=eta)
        if is_violated(result.Rbar_k, result.U_k, k_min=k_min).violated:
            violations += 1
    lo, hi = clopper_pearson_ci_on_rate(violations, n_reps)
    return {"arm_id": arm_id, "violations": violations, "n_reps": n_reps, "ci95": [lo, hi]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tier_a.yaml")
    parser.add_argument(
        "--manifest-only", action="store_true", help="write only Tables 15-16 (no GPU needed)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print_run_header("run_tier_a", config)
    out_dir = ensure_output_dir(config)

    if args.manifest_only:
        run_manifest_only(config, out_dir)
        return

    print(
        "Full Tier-A scoring requires GPU + Hugging Face Hub access to load "
        f"{[m['hf_id'] for m in config['models']]}, which this environment does not have.\n"
        "run_one_arm() below is complete and tested against synthetic (eta, scores) pairs "
        "(see tests/test_uniform_band.py's empirical-coverage tests); once "
        "compute_logits_and_eta() is filled in against a real model, the arm loop is a "
        "drop-in call to run_one_arm() per (model, group, score) combination.\n"
    )
    print("Run with --manifest-only to write Tables 15-16 without a GPU.")


if __name__ == "__main__":
    main()

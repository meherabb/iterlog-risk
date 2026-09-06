r"""Three self-checks, run automatically at the start of every experiment script and
aborting the run on failure (Appendix J, "Reproducibility"): an RNG-stream reproducibility
check, a forward-pass determinism check on the loaded model, and a full-pipeline
determinism check that runs the same seed twice and compares arrays element-wise.

The forward-pass and full-pipeline checks take the actual model/pipeline callable as a
parameter (rather than hardcoding a specific model), so they can be exercised in tests with
a lightweight stand-in callable and still exactly check the property a real model run would
need to satisfy.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.utils.determinism import derive_seed


def check_rng_stream_reproducibility(arm_id: str = "self_check", n_draws: int = 1000) -> bool:
    """Same (arm_id, rep_index) must always produce the same seed, and a generator built
    from that seed must always produce the same draws -- this is the property every other
    check in this module, and every replication in the paper, depends on."""
    seed_a = derive_seed(arm_id, rep_index=0)
    seed_b = derive_seed(arm_id, rep_index=0)
    if seed_a != seed_b:
        return False
    draws_a = np.random.default_rng(seed_a).uniform(size=n_draws)
    draws_b = np.random.default_rng(seed_b).uniform(size=n_draws)
    return bool(np.array_equal(draws_a, draws_b))


def check_forward_pass_determinism(
    forward_fn: Callable[[np.ndarray], np.ndarray],
    sample_input: np.ndarray,
    n_repeats: int = 3,
) -> bool:
    """Run the same input through ``forward_fn`` (e.g. a model's forward pass) multiple
    times and confirm every run gives bit-identical output. With a real model this needs
    ``cudnn.deterministic=True`` (see :func:`src.utils.determinism.set_torch_determinism`)
    to hold in practice; this function only checks the *outcome*, not that setting.
    """
    outputs = [forward_fn(sample_input) for _ in range(n_repeats)]
    first = outputs[0]
    return all(np.array_equal(first, out) for out in outputs[1:])


def check_full_pipeline_determinism(
    pipeline_fn: Callable[[int], np.ndarray],
    seed: int = 20260727,
) -> bool:
    """Run an entire pipeline (data loading through final output) twice from the same seed
    and confirm the results match element-wise -- the paper's strongest self-check, since it
    catches non-determinism anywhere in the chain, not just in one component.

    ``pipeline_fn`` takes a seed and returns whatever array the pipeline ultimately produces
    (e.g. a full violation-indicator array across all replications in an arm).
    """
    result_a = pipeline_fn(seed)
    result_b = pipeline_fn(seed)
    return bool(np.array_equal(result_a, result_b))


def run_all(
    forward_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    sample_input: np.ndarray | None = None,
    pipeline_fn: Callable[[int], np.ndarray] | None = None,
) -> dict[str, bool]:
    """Run whichever checks have the inputs they need, matching the CLI entry point
    ``python -m src.utils.self_checks --all`` that ``docs/REPRODUCING.md`` describes.
    The RNG-stream check always runs; the other two are skipped (not failed) if their
    required callables aren't supplied, since they're necessarily specific to whatever
    script is invoking them.
    """
    results = {"rng_stream": check_rng_stream_reproducibility()}
    if forward_fn is not None and sample_input is not None:
        results["forward_pass"] = check_forward_pass_determinism(forward_fn, sample_input)
    if pipeline_fn is not None:
        results["full_pipeline"] = check_full_pipeline_determinism(pipeline_fn)
    return results


if __name__ == "__main__":
    import sys

    outcome = run_all()
    print(outcome)
    if not all(outcome.values()):
        sys.exit(1)

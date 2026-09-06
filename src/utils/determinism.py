"""Deterministic, counter-based seeding.

Every replication in the paper is identified by (arm_id, rep_index) and is reproducible
in isolation (Appendix J, "Reproducibility"): seeds derive deterministically from a single
master seed via a counter-based generator, rather than from a single global stream that
would make replication k depend on how many draws replications 1..k-1 happened to consume.

The paper's master seed is 20260727 (Appendix J.1, "Seeds and hardware").
"""

from __future__ import annotations

import hashlib

import numpy as np

MASTER_SEED_DEFAULT = 20260727


def derive_seed(arm_id: str, rep_index: int, master_seed: int = MASTER_SEED_DEFAULT) -> int:
    """Deterministically derive a 63-bit seed for one (arm, replication) pair.

    Same (arm_id, rep_index, master_seed) always produces the same seed, independent of
    execution order — this is what makes a single replication reproducible in isolation
    (Appendix J, "Reproducibility").

    Parameters
    ----------
    arm_id:
        A short, stable identifier for the experimental arm, e.g. ``"tierA/qwen/pool/maxprob"``
        or ``"e4_deployment/r1"``. Two different arms must never share an arm_id, or their
        replication streams will collide.
    rep_index:
        The replication number within the arm, starting at 0.
    master_seed:
        The single seed the whole study is keyed on.

    Returns
    -------
    int
        A non-negative integer suitable for ``numpy.random.default_rng``.
    """
    payload = f"{master_seed}:{arm_id}:{rep_index}".encode()
    digest = hashlib.sha256(payload).digest()
    # Take 63 bits so the result is always a valid non-negative Python/NumPy seed.
    return int.from_bytes(digest[:8], byteorder="big") & 0x7FFF_FFFF_FFFF_FFFF


def rng_for(arm_id: str, rep_index: int, master_seed: int = MASTER_SEED_DEFAULT) -> np.random.Generator:
    """Convenience wrapper: build a ready-to-use ``Generator`` for one replication."""
    return np.random.default_rng(derive_seed(arm_id, rep_index, master_seed))


def set_torch_determinism() -> None:
    """Enable ``cudnn.deterministic=True`` and disable benchmark mode, matching Appendix J.1.

    Kept as a separate opt-in call (rather than a module-level side effect) so that
    importing this package never silently changes global PyTorch state. No-ops cleanly
    if torch is not installed, so this module has no hard torch dependency.
    """
    try:
        import torch
    except ImportError:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

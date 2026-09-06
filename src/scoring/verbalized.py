r"""Verbalized confidence (Tian & Justask; Kadavath et al.) -- a confidence number elicited
directly from the model itself, parsed from its own text output.

Theorem 1 requires no assumption about how a score was produced, which is exactly why this
family is includable at all: the model could be lying, miscalibrated, or simply making the
number up, and the certificate's guarantee is unaffected -- only the *width* changes,
depending on how informative the score turns out to be (Appendix G quantifies this).
"""

from __future__ import annotations

import re

import numpy as np

_NUMBER_PATTERN = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%?")


def parse_confidence(generated_text: str, max_new_tokens_hint: int | None = None) -> float | None:
    """Extract a confidence number from the model's own generated text (matching the
    paper's protocol of a short, constrained generation -- up to 4 new tokens, Appendix
    J.1). Returns a value in ``[0, 1]``, or ``None`` if no parseable number was found (the
    caller should treat that item as having a missing score, not a score of 0 -- see
    ``src/eval/tier_protocols.py`` for how missing verbalized scores are excluded from an
    arm rather than silently defaulting to a specific value).
    """
    match = _NUMBER_PATTERN.search(generated_text)
    if match is None:
        return None
    value = float(match.group(1))
    # Accept either a 0-1 fraction or a 0-100 percentage, inferred from magnitude.
    if value > 1.0:
        value = value / 100.0
    return float(np.clip(value, 0.0, 1.0))


def score(generated_texts: list[str]) -> np.ndarray:
    """``generated_texts``: one model completion per item. Returns shape ``(n_items,)``;
    unparseable items are assigned ``np.nan`` so downstream code can filter them explicitly
    rather than silently treating a parse failure as zero confidence.
    """
    parsed = [parse_confidence(t) for t in generated_texts]
    return np.array([p if p is not None else np.nan for p in parsed])

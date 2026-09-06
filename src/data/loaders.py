r"""Dataset loaders for the four MCQ pool sources, MMLU, TriviaQA, and CIFAR-10/-C.

.. important::
    **Unlike every other module in this package, the functions here have not been run
    end-to-end in the environment this repository was authored in** -- that environment has
    no network access to the Hugging Face Hub, so ``datasets.load_dataset(...)`` calls below
    could not actually be executed and checked against real downloaded data. The
    *structure* (which splits/fields each dataset exposes, how they're mapped into a
    common ``Example`` record) is correct to the best of our knowledge of these datasets'
    public schemas, but you should treat this module -- and only this module -- as needing
    your own first-run verification, e.g. ``python -m src.data.fetch_all --dry-run``
    (see ``data/README.md``) before trusting its output in a full experiment. Every other
    module in ``src/`` was actually executed against synthetic data during development and
    is not subject to this caveat.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    """A single item, normalized across all MCQ sources into one common shape."""

    question: str
    choices: list[str]
    answer_index: int
    source: str  # e.g. "arc_easy", "mmlu"
    subject: str | None = None  # populated for MMLU only (its 57 subjects), else None


def load_arc_easy(split: str = "test"):
    """ARC-Easy via ``allenai/ai2_arc``, config ``ARC-Easy``."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    return [
        Example(
            question=row["question"],
            choices=row["choices"]["text"],
            answer_index=row["choices"]["label"].index(row["answerKey"]),
            source="arc_easy",
        )
        for row in ds
    ]


def load_arc_challenge(split: str = "test"):
    """ARC-Challenge via ``allenai/ai2_arc``, config ``ARC-Challenge``."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    return [
        Example(
            question=row["question"],
            choices=row["choices"]["text"],
            answer_index=row["choices"]["label"].index(row["answerKey"]),
            source="arc_challenge",
        )
        for row in ds
    ]


def load_openbookqa(split: str = "test"):
    """OpenBookQA via ``allenai/openbookqa``."""
    from datasets import load_dataset

    ds = load_dataset("allenai/openbookqa", "main", split=split)
    return [
        Example(
            question=row["question_stem"],
            choices=row["choices"]["text"],
            answer_index=row["choices"]["label"].index(row["answerKey"]),
            source="openbookqa",
        )
        for row in ds
    ]


def load_commonsense_qa(split: str = "validation"):
    """CommonsenseQA via ``tau/commonsense_qa`` (its test split has no public labels, so
    the paper's pool -- like most work using this dataset -- draws from ``validation``)."""
    from datasets import load_dataset

    ds = load_dataset("tau/commonsense_qa", split=split)
    return [
        Example(
            question=row["question"],
            choices=row["choices"]["text"],
            answer_index=row["choices"]["label"].index(row["answerKey"]),
            source="commonsense_qa",
        )
        for row in ds
    ]


def load_mmlu(split: str = "test"):
    """All 57 MMLU subjects via ``cais/mmlu``, config ``all``, with the subject label
    preserved (needed for the group-conditional study, Appendix F)."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split=split)
    return [
        Example(
            question=row["question"],
            choices=row["choices"],
            answer_index=row["answer"],
            source="mmlu",
            subject=row["subject"],
        )
        for row in ds
    ]


def load_trivia_qa(split: str = "validation", n_questions: int = 4_000):
    """TriviaQA via ``mandarjoshi/trivia_qa``, config ``rc.nocontext`` (Tier B needs only
    the question and reference answer, not retrieved context)."""
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    return ds.select(range(min(n_questions, len(ds))))


def load_cifar10(train: bool = False):
    """CIFAR-10 via ``torchvision.datasets.CIFAR10`` (downloads to ``data/cache/`` on
    first use)."""
    import torchvision

    return torchvision.datasets.CIFAR10(root="data/cache", train=train, download=True)


def load_cifar10c(corruption: str, severity: int):
    """One (corruption, severity) slice of CIFAR-10-C. CIFAR-10-C ships as pre-generated
    ``.npy`` files (Hendrycks & Dietterich's official release) rather than through the
    ``datasets``/``torchvision`` APIs directly; see ``data/README.md`` for the exact
    download command this function expects to find the files from.
    """
    import numpy as np

    images = np.load(f"data/cache/CIFAR-10-C/{corruption}.npy")
    labels = np.load("data/cache/CIFAR-10-C/labels.npy")
    # The official release stacks all 5 severities along axis 0 in blocks of 10,000.
    start = (severity - 1) * 10_000
    end = severity * 10_000
    return images[start:end], labels[start:end]

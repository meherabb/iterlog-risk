"""Every comparison certificate used in the paper, implemented from scratch against its
own original reference (no external statistical packages) -- see docs/ARCHITECTURE.md.
"""

from src.baselines import (
    clopper_pearson,
    conformal_risk_control,
    empirical_bernstein,
    hoeffding,
    learn_then_test,
    transfer_corrected,
    wsr_betting,
)

__all__ = [
    "clopper_pearson",
    "hoeffding",
    "empirical_bernstein",
    "wsr_betting",
    "learn_then_test",
    "conformal_risk_control",
    "transfer_corrected",
]

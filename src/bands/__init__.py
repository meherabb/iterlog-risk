"""Theorems 1-4 and Corollaries 1-3: the paper's core contribution.

See docs/ARCHITECTURE.md for the complete theorem-to-function mapping. Re-exported here so
common entry points don't require reaching into individual submodules, e.g.:

    from src.bands import uniform_band, lil_lower_bound, population_band, block_robust
"""

from src.bands import (
    block_robust,
    lil_lower_bound,
    population_band,
    sorted_filtration,
    uniform_band,
    validity,
)

__all__ = [
    "sorted_filtration",
    "uniform_band",
    "lil_lower_bound",
    "population_band",
    "block_robust",
    "validity",
]

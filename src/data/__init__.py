"""Dataset loaders, pool construction, and the Gaussian-copula correlation sampler.

Note: `loaders` requires network access to the Hugging Face Hub / torchvision and has not
been executed end-to-end in this repository's own development environment -- see its
module docstring. `pool` and `copula` are pure NumPy and were both tested directly.
"""

from src.data import copula, loaders, pool

__all__ = ["pool", "copula", "loaders"]

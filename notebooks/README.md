# `notebooks/` — illustrative, exploratory versions of the scripts

> **Status:** populated in Step 3 of the release.

These notebooks mirror `scripts/*.py` one-to-one but are meant for reading and interactive
exploration (inline plots, intermediate variable inspection) rather than automated
reproduction — use the scripts for that. All notebooks are committed with outputs stripped
(`nbstripout`, see `ANONYMITY.md`) so diffs stay readable and no stray metadata leaks through.

Planned: `01_band_walkthrough.ipynb` (a from-scratch, cell-by-cell build of Theorem 1's band
on synthetic data — the best starting point for understanding the codebase),
`02_baselines_comparison.ipynb`, `03_wsr_bug_deepdive.ipynb`, `04_block_robustness_demo.ipynb`.

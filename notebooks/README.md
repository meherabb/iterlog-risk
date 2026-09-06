# `notebooks/` — executed, interactive walkthroughs

All four notebooks below are pre-executed (outputs included) and were run end-to-end in
this repository's own development environment before being committed — every printed
number and every plot is real output from real code, not illustrative placeholders.

| Notebook | What it shows | Verified output |
|---|---|---|
| `01_band_walkthrough.ipynb` | Theorem 1 built from scratch, cell by cell: the sorted filtration, the λ-menu, the band itself, then a 200-replication empirical check | 0/200 violations, matching the paper's 0/1,000 |
| `02_baselines_comparison.ipynb` | How much wider Theorem 1's uniform band is than a fixed-*k* certificate, and what "fixed-*k*" actually promises (and doesn't) once an analyst looks at the curve first | Direct width comparison across 4 certificates at one calibration curve |
| `03_wsr_bug_deepdive.ipynb` | The betting-bound implementation defect from Appendix D.3, its trigger-rate on i.i.d. vs. heterogeneous data, and the Lemma 1 patch | i.i.d. bug rate 1.0%, matching the paper's "~1%" almost exactly |
| `04_block_robustness_demo.ipynb` | Theorem 4: item-level certification breaking under induced correlation, block-level certification staying valid throughout | Block-level violation rate exactly 0.000 at every ρ tested |

These mirror the corresponding scripts in `scripts/` (see `docs/REPRODUCING.md`) at smaller,
notebook-friendly scale, meant for reading and understanding rather than automated
reproduction -- use the scripts for that.

Committed with real outputs intentionally (not stripped), since the whole point of these
four is to show what running the code actually produces. If you re-run and re-save a
notebook yourself, consider `nbstripout` for any *new* notebook you add instead (see
`ANONYMITY.md`), to keep future diffs readable.

# `results/` — where your own runs write their outputs

This folder is intentionally empty in the repository (see `.gitignore`) and is **not**
pre-populated with checkpoints or logs from the original experiments. Every number reported
in the paper should be something you can regenerate yourself by running the scripts in
`scripts/` — see [`../docs/REPRODUCING.md`](../docs/REPRODUCING.md) — rather than something
you're asked to take on faith from a bundled `.npz` file.

Running any script in `scripts/` will create the relevant `*.npz` / `*.json` files here,
named to match the paper's own checkpoint-file conventions listed in
[`../docs/CROSSWALK.md`](../docs/CROSSWALK.md), so you can cross-reference what you generated
against what the paper's Appendix~J.2 says it used.

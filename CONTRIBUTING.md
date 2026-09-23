# Contributing

Thanks for your interest. A couple of things are different here because this repository
accompanies a paper under **double-blind review**. Please wait until the **double blind review** period get end.


## Development setup

```bash
git clone <this-repository-url>
cd <repository-name>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
pre-commit install   # optional but recommended
```

## Before opening a PR

1. Run the full test suite: `pytest`.
2. Run the linter/formatter: `ruff check . && black --check .`.
3. If you touched anything under `src/bands/` or `src/baselines/`, add or update a test in
   `tests/` that exercises the specific theorem property you changed (see existing tests for
   the pattern — most are property-based checks against synthetic data with a known ground
   truth, not just "does it run").
4. If you touched a script under `scripts/`, note in the PR description which paper
   table/figure it affects and whether the reported numbers change.

## Code style

- Every public function in `src/bands/` and `src/baselines/` must have a docstring that
  states which theorem, corollary, or equation in the paper it implements — this is the
  single thing we're strictest about, since the whole point of this repository is that the
  code should be checkable against the paper line by line.
- Type hints are required on new public functions.
- Prefer `numpy`/`scipy` vectorized operations over Python loops in anything that runs per
  replication (some experiments run tens of thousands of replications).

## After acceptance

Once the paper is accepted and de-anonymized, this file and `ANONYMITY.md` will be updated
to reflect normal (non-anonymous) contribution norms.

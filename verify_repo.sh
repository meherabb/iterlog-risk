#!/usr/bin/env bash
# verify_repo.sh — sanity-check that this repository is set up correctly and its
# documentation's claims are actually true. Run from the repository root:
#
#   bash verify_repo.sh
#
# Each section prints PASS/FAIL. A clean run means: the package imports, the full
# test suite passes, the README's Quickstart example actually runs as documented,
# every file path the README/docs point to exists, and the dependency files parse.
# This does not run the paper's actual experiments (scripts/run_*.py) — those need
# GPU time and downloaded datasets; see docs/REPRODUCING.md for that separately.

set -uo pipefail
FAIL=0

echo "== 1. Package imports cleanly =="
python3 -c "
import src.bands.uniform_band
import src.bands.population_band
import src.bands.lil_lower_bound
import src.bands.block_robust
import src.bands.sorted_filtration
import src.baselines.wsr_betting
import src.baselines.clopper_pearson
import src.baselines.hoeffding
import src.baselines.empirical_bernstein
import src.baselines.learn_then_test
import src.baselines.conformal_risk_control
import src.baselines.transfer_corrected
print('PASS: all src/ modules import without error')
" || { echo "FAIL: import error above"; FAIL=1; }
echo ""

echo "== 2. Full test suite (expect 73 passed) =="
python3 -m pytest tests/ -o addopts="" -q || { echo "FAIL: one or more tests failed"; FAIL=1; }
echo ""

echo "== 3. README Quickstart example actually runs =="
python3 -c "
import numpy as np
from src.bands import uniform_band

rng = np.random.default_rng(20260727)
n, p = 20_000, 0.15
losses = rng.binomial(1, p, size=n)
scores = rng.normal(size=n)
eta = np.full(n, p)

result = uniform_band.compute(losses, scores, delta=0.05, k_min=100, rng=rng, eta=eta)
assert np.all(result.U_k >= result.Rbar_k)
print('PASS: Quickstart example runs and its assertion holds')
" || { echo "FAIL: Quickstart example raised an error or failed its assertion"; FAIL=1; }
echo ""

echo "== 4. Every file the README/docs point to exists =="
FILES_TO_CHECK=(
  src/bands/uniform_band.py src/bands/population_band.py src/bands/lil_lower_bound.py
  src/bands/block_robust.py src/baselines/wsr_betting.py
  docs/REPRODUCING.md docs/ARCHITECTURE.md docs/CROSSWALK.md
  CITATION.cff CONTRIBUTING.md LICENSE environment.yml requirements.txt
  scripts/run_synthetic_validity.py scripts/run_tier_a.py scripts/run_tier_b.py
  scripts/run_deployment.py scripts/run_exchangeability.py scripts/run_block_correlation.py
  scripts/run_necessity_sweep.py scripts/run_group_conditional.py scripts/run_wsr_audit.py
)
MISSING=0
for f in "${FILES_TO_CHECK[@]}"; do
  if [ ! -f "$f" ]; then echo "MISSING: $f"; MISSING=1; fi
done
if [ "$MISSING" -eq 0 ]; then echo "PASS: all referenced files present"; else echo "FAIL: see MISSING lines above"; FAIL=1; fi
echo ""

echo "== 5. Dependency files parse correctly =="
python3 -c "import yaml; yaml.safe_load(open('environment.yml')); print('PASS: environment.yml is valid YAML')" \
  || { echo "FAIL: environment.yml did not parse"; FAIL=1; }
[ -s requirements.txt ] && echo "PASS: requirements.txt is non-empty" \
  || { echo "FAIL: requirements.txt missing or empty"; FAIL=1; }
echo ""

echo "== 6. Known outstanding items (informational only, not a failure) =="
echo "These files may still reference pre-audit corollary numbering:"
grep -l "Corollary" scripts/make_all_figures.py tests/README.md docs/ARCHITECTURE.md 2>/dev/null \
  | sed 's/^/  - /'
echo ""

if [ "$FAIL" -eq 0 ]; then
  echo "======================================"
  echo "ALL CHECKS PASSED"
  echo "======================================"
else
  echo "======================================"
  echo "ONE OR MORE CHECKS FAILED -- see FAIL lines above"
  echo "======================================"
  exit 1
fi

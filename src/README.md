# `src/` — the library

> **Status:** structure defined here in Step 1 of the release; full implementation ships in
> Step 2. See [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the complete
> theorem-to-file mapping this package follows.

Planned layout:

```
src/
├── __init__.py
├── bands/              # Theorems 1-4, Corollaries 1-3 — the paper's core contribution
│   ├── sorted_filtration.py
│   ├── uniform_band.py
│   ├── lil_lower_bound.py
│   ├── population_band.py
│   ├── block_robust.py
│   └── validity.py
├── baselines/           # Every comparison certificate, implemented from scratch
│   ├── clopper_pearson.py
│   ├── hoeffding.py
│   ├── empirical_bernstein.py
│   ├── wsr_betting.py
│   ├── learn_then_test.py
│   ├── conformal_risk_control.py
│   └── transfer_corrected.py
├── scoring/              # The seven score families (Section 4)
│   ├── maxprob.py
│   ├── entropy.py
│   ├── seqlogprob.py
│   ├── hidden_centroid.py
│   ├── hidden_knn.py
│   ├── self_consistency.py
│   └── verbalized.py
├── data/                  # Dataset loaders, pool construction, the Gaussian-copula sampler
│   ├── loaders.py
│   ├── pool.py
│   └── copula.py
├── eval/                   # Tier A/B/C harnesses, selection rules, cluster bootstrap
│   ├── tier_protocols.py
│   ├── selection_rules.py
│   ├── cluster_bootstrap.py
│   └── vision_tier_a.py
└── utils/
    ├── determinism.py
    └── self_checks.py
```

Every module's public functions will carry a docstring naming the exact theorem, corollary,
or equation it implements, and a corresponding property-based test in `tests/`.

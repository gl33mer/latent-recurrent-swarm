# Phase 2b Stability Study — Report (18 instrumented runs, validation-only)

**Branch:** `experiment/phase2b-stability-study` · **Design (predeclared, committed before
any run):** `reports/phase2b-stability-design.md` at commit `d0fff86` · **Sweep commit:**
`d0fff86` · **Run directory:** `runs/20261016T231051Z_stability/` (ignored; JSONL per
epoch, per-run summaries, checkpoints incl. failures, sweep log)
**Verdict: NO learning rate is eligible under the predeclared rule → the Phase 2b recipe
is classified NOT YET STABILISED. No confirmatory Phase 2c comparison will be run.**

## 1. Execution summary

- 18/18 planned runs completed; wall time ≈ 44 minutes (budget 120 min); sweep not truncated.
- No test split was generated or evaluated (train/val-only builder).
- Finiteness: zero non-finite loss/gradient batches and finite parameters in all 18 runs
  (criterion 2 satisfied at every rate).
- Trajectory structure: full-mode runs show nonzero message effect from step 1
  (step-1 mean |Δz| ≈ 0.07–0.14); every final-only run is bit-identical through step 7
  (step-1 diff = 0.00e+00 exactly) — criterion 3 satisfied at every rate.
- The binding failure is criterion 1 (gate pass counts). Instability persists at every rate.

## 2. All 18 runs (final-epoch validation metrics; gates unchanged from Phase 2b)

| LR | Mode | Seed | Conn% | Pass | Failed criteria | Zero% | Probe% | Flip 0/1% | Instab% | Collapses |
|---|---|---|---|---|---|---|---|---|---|---|
| 8e-4 | full | 42 | 74.7 | ✗ | connected, quadrant, probe | 50.0 | 74.7 | 0.0/0.8 | 18.1 | 5 |
| 8e-4 | full | 43 | 68.9 | ✗ | connected, quadrant, probe | 47.7 | 71.0 | 4.6/7.3 | 8.8 | 2 |
| 8e-4 | full | 44 | 92.0 | ✗ | probe | 50.0 | 92.0 | 0.0/0.0 | 14.5 | 3 |
| 8e-4 | final-only | 42 | 79.8 | ✗ | connected, quadrant, probe | 50.9 | 85.1 | 0.0/12.1 | 10.7 | 2 |
| 8e-4 | final-only | 43 | 99.5 | ✓ | — | 49.4 | 99.5 | 0.0/0.0 | 9.5 | 2 |
| 8e-4 | final-only | 44 | 68.0 | ✗ | connected, quadrant, flip, probe | 50.1 | 69.8 | 5.5/18.0 | 15.5 | 4 |
| 4e-4 | full | 42 | 95.9 | ✓ | — | 50.0 | 95.9 | 0.0/0.2 | 10.8 | 2 |
| 4e-4 | full | 43 | 87.9 | ✗ | probe | 49.3 | 88.0 | 0.0/0.2 | 7.1 | 2 |
| 4e-4 | full | 44 | 80.5 | ✗ | connected, quadrant, probe | 48.7 | 81.3 | 0.0/3.7 | 7.9 | 1 |
| 4e-4 | final-only | 42 | 94.8 | ✗ | probe | 49.5 | 94.8 | 0.0/0.2 | 21.6 | 5 |
| 4e-4 | final-only | 43 | 51.3 | ✗ | connected, quadrant, flip, probe | 50.1 | 43.1 | 10.6/52.3 | 13.2 | 3 |
| 4e-4 | final-only | 44 | 85.9 | ✗ | quadrant, probe | 50.5 | 10.0 | 0.2/2.2 | 18.1 | 4 |
| 2e-4 | full | 42 | 54.9 | ✗ | connected, quadrant, flip, probe | 49.7 | 62.2 | 18.2/69.0 | 4.7 | 1 |
| 2e-4 | full | 43 | 96.3 | ✓ | — | 49.8 | 96.3 | 0.0/0.2 | 2.3 | 0 |
| 2e-4 | full | 44 | 94.3 | ✗ | probe | 49.2 | 94.3 | 0.0/0.0 | 6.6 | 1 |
| 2e-4 | final-only | 42 | 94.6 | ✗ | probe | 50.4 | 94.6 | 0.0/0.2 | 6.5 | 1 |
| 2e-4 | final-only | 43 | 99.5 | ✓ | — | 49.6 | 99.5 | 0.0/0.2 | 4.6 | 0 |
| 2e-4 | final-only | 44 | 99.4 | ✓ | — | 49.9 | 99.5 | 0.0/0.2 | 3.2 | 0 |

("Instab%" = mean absolute epoch-to-epoch change of validation connected accuracy;
"Collapses" = epochs with a >15-point drop vs the previous epoch. Full per-epoch loss
components, gradient norms per group, and traces are in each run's `metrics.jsonl`.)

## 3. Selection rule applied (as predeclared; no post-hoc adjustment)

| LR | full pass | final-only pass | Eligible? |
|---|---|---|---|
| 8e-4 | 0/3 | 1/3 | No (full <2/3) |
| 4e-4 | 1/3 | 0/3 | No (both <2/3) |
| 2e-4 | 1/3 | 2/3 | No (full <2/3) |

**Decision: no shared learning rate is eligible. The recipe is NOT YET STABILISED.**
Observations recorded for future work (not decisions): 2e-4 has the lowest median
instability (3.2% vs 9.5%/7.9%) and the highest total pass count (3/6); the *sender probe*
gate is the most frequent failure at every rate (Core A's bit encoding, not the receiver
mechanism, is the recurring weak point); flip-residual failures cluster at 8e-4/4e-4.

## 4. Deviation note: batch order differs from the original six Phase 2b runs

The sweep fixes an explicit DataLoader generator seed (9000+model seed) so paired
full/final-only runs see identical batch order — a stability-study requirement. The
original six Phase 2b runs used the library-default shuffling derived from the global RNG
state, so individual sweep outcomes are not expected to replicate those runs exactly
(e.g., original full/seed-43 passed its gates; sweep full/8e-4/seed-43 did not). Both are
valid seeded runs of the same frozen recipe; this underscores the seed/order sensitivity
rather than contradicting either result. All original artifacts remain unchanged.

## 5. Consequences for Phase 2c (per charter §5.4 / §13 Phase B stop-gate)

- Phase 2c data/model/tests are implemented and unit-tested as engineering work.
- Only an explicitly engineering-labelled smoke run is performed for Phase 2c.
- **No confirmatory multi-seed Phase 2c full/final-only comparison is run**, and no
  Phase 2c test split is evaluated.
- Recommended next action: a dedicated optimisation study (diagnose the sender-probe
  weakness and the collapse pattern — e.g., LR schedule, longer warm-up, or auxiliary-loss
  rebalancing — each predeclared separately) before any matched task comparison.

## 6. Limitations

- 3 seeds per cell; the study screens the recipe, it does not establish any scientific claim.
- The sweep varied only LR within the frozen recipe (per charter); no clipping/scheduling
  was permitted, so "not yet stabilised" means "not stabilised *within this grid*", not
  "unstable at any possible recipe".
- Validation is reused for LR selection by design; no test data exists in this sweep.

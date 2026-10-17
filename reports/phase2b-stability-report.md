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

---

# Dated Addendum (2026-10-17 — reconciliation & follow-up study; no historical numbers altered)

Three corrections/clarifications resulting from the Stage C audit of this report, plus the
follow-up study outcome. Original tables, decisions, and selection verdict above are unchanged.

## A1. Median instability values were misreported

The prose summary above called **9.5% / 7.9% / 3.2%** the "median instability" per learning
rate. Those are single per-run values, not medians of the six per-run values. Recomputed
from the raw `metrics.jsonl` epoch histories under the predeclared definition (per-run mean
absolute epoch-to-epoch change of validation connected accuracy; median across all six
mode/seed runs at each LR), the correct medians are:

- **8e-4: 12.61%** (per-run: 8.8, 9.5, 10.7, 14.5, 15.5, 18.1)
- **4e-4: 11.99%** (per-run: 7.1, 7.9, 10.8, 13.2, 18.1, 21.6)
- **2e-4: 4.68%** (per-run: 2.3, 3.2, 4.6, 4.7, 6.5, 6.6)

This correction does **not** change the decision: eligibility was decided by gate-pass
counts (criterion 1), which are unaffected; no LR was eligible and the recipe was correctly
classified NOT YET STABILISED. It corrects the secondary ranking only (2e-4's advantage is
larger than originally reported).

## B1. Failed-run checkpoint policy: deviation located and clarified

The predeclared stability design required `last.pt` for every run including gate failures.
On disk, **all 18 stability-sweep runs have `last.pt` plus epoch checkpoints** — the sweep
complied. The deviation is in the **earlier six original Phase 2b runs**
(`reports/phase2b-run-report.md` artifact table: failed runs "none (artifact save is
post-gate only)"): those four gate-failing runs have no checkpoints, and their states are
unrecoverable without new runs (none were recreated under old IDs). All runners from the
message-weight study onward save `last.pt` unconditionally for every run, including
failures and non-finite states.

## C1. Sender-probe diagnosis (evidence-based, from raw sweep metrics)

Probe failures are **late-training optimisation collapses, not metric/sign errors or a
static sender-representation failure**: in multiple runs the probe was 94–99% at epoch 5
and degraded in lockstep with parity accuracy while the message BCE spiked (e.g. probe
97.1%→10.0% with msg BCE 0.058→1.916 and msg_encoder gradient-norm spikes); other runs
were stable for 17 epochs before a sudden collapse (e.g. 2e-4/full/seed-42 at epoch 18,
core_A gradient jump to 5.70). Follow-up: the predeclared one-factor message-supervision
weight study (`reports/phase2b-msg-weight-study-design.md`,
`reports/phase2b-msg-weight-study-report.md`, branch
`experiment/phase2b-msg-supervision-weight-study`) raised the weight on the existing BCE
sender supervision. Result: **λ_msg = 2.0 (at lr 2e-4) passed all frozen Phase 2b gates in
6/6 runs** (probe ≥ 99.5% everywhere, zero-message at chance, zero collapses), superseding
the stop-gate for that specific recipe pending owner review. No Phase 2c training, data, or
evaluation occurred in that study.

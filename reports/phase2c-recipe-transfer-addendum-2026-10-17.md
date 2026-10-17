# Phase 2c Recipe-Transfer Addendum (2026-10-17) — committed BEFORE any Phase 2c training

**Branch:** `experiment/phase2c-lambda2-paired-confirmatory` (from the verified Phase 2c
fix tip `744e871`). **Status at commit time:** no Phase 2c metric, log, checkpoint or test
datum has been generated or inspected for recipe selection. The recipe below derives
exclusively from the Phase 2b message-supervision weight study
(`reports/phase2b-msg-weight-study-report.md`, erratum
`reports/phase2b-msg-weight-study-erratum-2026-10-17.md`, erratum commit `607221a`,
study selection recorded in `runs/20261017T101804Z_msg_weight_stability/selection.json`).

## D1. Rationale and epistemic status

The Phase 2b weight study screened `lr=2e-4`, `lambda_msg=2.0` on the frozen Phase 2b XOR
setup and passed all frozen gates in 6/6 runs (and the corrected erratum figures: all-gate
totals 3/6 → 4/6 → 6/6; probe-gate failures 3/6 → 2/6 → 0/6; median instability
4.68% → 5.62% → 2.47%, non-monotonic). This makes the recipe a **screened candidate, not a
guarantee**: Phase 2c has a different task (selector-conditioned two-relation
reachability), a different Core B input representation (64-dim union/R0/R1/id), and a
different objective mix. Transfer must be validated on Phase 2c's own validation data.
No alternate learning rate or λ_msg will be swept here.

## D2. Locked Phase 2c candidate recipe

- Primary model: `LatentConditionedSwarm` (279,561 params); modes `full` and `final-only`.
- Model seeds: **42, 43, 44, 45, 46**; paired design: per seed, full and final-only runs
  start from **identical warm-started initial weights** (seed reset before construction;
  initial-state checksum recorded and asserted equal within the pair), identical
  train/validation data, identical train-loader order (generator seed `9000 + seed`),
  separate optimizers, no optimizer-state sharing.
- Data: task split seeds train **4201**, validation **13371**, test **99991** (test only if
  and only if the Stage G stop condition is met); 4,000 train / 1,000 validation / 1,000
  test samples.
- Optimisation: batch 64, **20 epochs**, float32 CPU, 4 intra-op threads, AdamW weight
  decay **1e-2**, learning rate **2e-4**.
- Objective (primary): `CE(task_logits, Y) + 0.5·CE(aux_A, b_A) +
  **2.0**·BCEWithLogits(raw_m, b_A)` — i.e. `lambda_msg=2.0`; all other weights exactly as
  `reports/phase2c-design.md` §6. No Core B relation auxiliary loss on the primary model.
- Dual diagnostic (`LatentDualRelationSwarm`, 279,691 params): one run per seed (nominal
  mode `full`, documented as a no-op for this architecture); loss per existing design:
  `CE(r0_logits, r0) + CE(r1_logits, r1) + 0.5·CE(aux_A, b_A) + 2.0·BCEWithLogits(raw_m, b_A)`.
- No LR schedule, warm-up, gradient clipping, early stopping, best-checkpoint selection,
  data/architecture/threshold changes. Final epoch is the declared model.
- Total: 10 primary + 5 dual = **15 runs**.

## D3. Paired-pass definition and stop rule (operationalising "≥4 of 5 paired seeds")

- A **paired seed qualifies** iff the primary/full run AND the primary/final-only run for
  that same seed each pass **every** Phase 2c validation gate below (task, control, and
  their mode-specific mechanism check).
- **Stable task success requires ≥4 of the 5 paired seeds to qualify.**
- Per-mode pass counts are also reported; asymmetric failures are not hidden.
- If fewer than 4 paired seeds qualify (or the grid is incomplete), the study is classified
  `PARTIAL/UNSTABLE`: **the test split is not generated, the evaluator is not invoked, no
  failed seed is rerun under the same ID, no setting is changed.** A new attempt requires a
  new predeclared experiment and split policy.

## D4. Frozen Phase 2c validation gates (verbatim from `reports/phase2c-design.md` §8)

- Selected-task validation accuracy ≥ **82%** overall.
- Each `(b_A, Y)` quadrant ≥ **75%**.
- Sender probe ≥ **95%** overall and ≥ **90%** per b_A group.
- Zero-message accuracy ≤ **77%** overall AND ≤ **53%** on the `r0 ≠ r1` disagreement
  subset (the no-selector optimum is 75%/50%; agreement-subset accuracy is reported
  separately and is not evidence of message use).
- Forced-message counterfactual accuracy ≥ **75%** for each of {m=−1 vs r0} and {m=+1 vs r1}.
- Selector switch consistency ≥ **85%** on disagreement cases; selector invariance ≥ **85%**
  on agreement cases.
- Mechanism, full mode: terminal graph node 15 shows **strictly positive corrected
  element-weighted** message sensitivity at ≥1 intermediate step (1–7).
- Mechanism, final-only: ordinary graph-node states **exactly tensor-equal** across
  opposite messages through step 7 (asserted directly, not inferred from near-zero means).
- Empty conditioning groups, non-finite losses/gradients/parameters, or broken expected
  gradient paths are hard failures.

## D5. Predeclared scientific interpretation and test statistics method

- Primary statistic: paired test accuracy difference `Δ = acc(full) − acc(final-only)`,
  per seed and aggregated, overall and on the disagreement subset; per-example paired
  predictions saved (ignored storage) for independent re-analysis.
- **Paired bootstrap (predeclared implementation):** for the disagreement subset, per seed
  compute per-example paired correctness differences `d_i ∈ {−1,0,+1}`;
  per-seed statistic = mean(d_i)·100 pp; cross-seed aggregate = mean of the five per-seed
  statistics. Confidence interval: nonparametric bootstrap resampling **test examples with
  replacement within each seed's disagreement set** (10,000 resamples, 95% percentile
  interval, numpy Generator seed 20261), recomputing per-seed means and their 5-seed
  average per resample. This is conditional on the evaluated models; with five paired
  training seeds, training-seed uncertainty is described conservatively (no asymptotic
  p-values).
- *Evidence favouring early integration:* stable task success AND mean paired test
  advantage ≥ +5 pp for full on the disagreement subset AND the predeclared bootstrap
  interval excludes zero, with paired-seed directions consistent. Conservative phrasing
  (n=5).
- *No measured advantage:* both modes solve the task, Δ small/uncertain → the task is
  solvable by parallel reasoning + final selection (not a general negative on COMM).
- *Final-only wins:* report honestly; does not negate the measured trajectory mechanism.
- *Gates fail / instability:* `PARTIAL/UNSTABLE`; the test split is not used to diagnose.
- Dual diagnostic: report relation-head accuracies and selected-task accuracy (overall,
  disagreement) descriptively; it is a compute-both-then-select sufficiency baseline, not
  part of the primary paired statistic; no post-hoc thresholds.

## D6. Execution and failure policy

Single source commit for all 15 runs (the verified code/tests commit on this branch).
Unique ignored run directories `runs/<UTC>_phase2c_confirmatory/<model>_<mode>_seed<seed>/`;
`last.pt` saved **atomically after every epoch** (temp file + rename) plus
`epoch05/10/15/20.pt`; per-epoch `metrics.jsonl` (loss components, val overall/quadrants/
agreement/disagreement, probe overall/per b_A, zero-message overall/disagreement,
finiteness, gradient group norms, lr, duration); counterfactuals and corrected trajectory
sensitivity at the final epoch; full stdout/stderr logs; initial/final state checksums.
Budget: **120 minutes**; if the cap is hit, the grid is declared incomplete and the
stable-success rule is not applied. If a code defect appears mid-grid, stop, preserve
outputs, fix on a new commit, and restart the grid under a new ID (never mix source
commits in one confirmatory set). A non-finite event checkpoints immediately and fails
that run only.

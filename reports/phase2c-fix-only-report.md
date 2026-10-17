# Phase 2c Fix-Only Engineering Report (2026-10-17)

**Branch:** `experiment/phase2c-message-conditioned-reachability` · **Start commit:**
`fe87e1f` · **Fix commits:** `45fce1b` (Fixes 1+2), `9257ad3` (Fixes 3+4), plus a
documentation commit (this report, the erratum, log/CHANGELOG updates).
**Fix-only run directory:** `runs/20261017T060908Z_phase2c_fix_only/` (logs, environment
snapshot, pytest log, stop-gate check). **Smoke directory:**
`runs/20261017T063938Z_phase2c_fix_smoke/`.

**Research status:** Engineering fixes and smoke verification only. **No confirmatory
Phase 2c training, no Phase 2c test split generation (seed 99991 was never generated), and
no Phase 2c test evaluation were performed.**

## 1. Starting state

- Branch `experiment/phase2c-message-conditioned-reachability` at `fe87e1f`, clean working
  tree, in sync with `origin` (no divergence). `main` untouched at `fcf20d7`.
- Artifacts verified present with matching SHA-256s: Phase 1 checkpoint
  `ef4514…49e`; Phase 2b checkpoints `38c505…` (full/seed43) and `79aada…`
  (final-only/seed42). Historical run directories untouched.
- Environment reused: Python 3.11.2, torch 2.14.1+cpu (CUDA off), numpy 2.4.6, 4 threads.

## 2. Fix 1 — true raw graph-hash uniqueness (`45fce1b`)

- `data/conditioned_graph_task.py` now hashes every adjacency with
  `canonical_raw_graph_hash()` — canonicalised (float32, contiguous, CPU) raw SHA-256 —
  and uses the **raw hash with no role prefix** as the uniqueness key across roles A/R0/R1,
  within a split and across supplied `existing_hashes` from prior splits. The returned set
  contains exactly `3 × num_samples` raw hashes. Role lineage is metadata only.
- Bounded failure: `MAX_ATTEMPTS_PER_GRAPH = 200_000`, then a clear `RuntimeError`
  ("…too large for the 16-node forward-DAG sampler") instead of an unbounded loop.
- Preserved unchanged: task/label construction, 8-combination balancing, endpoint and
  path-length constraints, split seeds 4201/13371/99991, determinism.
- Proof tests: `test_raw_hash_uniqueness_within_split_across_roles` (recomputed hashes of
  stored matrices reproduce the returned set; no `:` in keys; exact 3N count),
  `test_raw_hash_disjoint_across_small_fixtures`, `test_canonical_hash_is_dtype_and_layout_invariant`,
  `test_generator_rejects_repeated_raw_graphs_across_roles` (deterministic mocked sampler;
  each graph proposed exactly twice — one acceptance, one cross-role rejection; observed
  47 proposals for 24 slots with the exact arithmetic asserted),
  `test_generator_bounded_failure_on_exhaustion`. Existing generator tests kept, none weakened.

## 3. Fix 2 — genuinely selector-independent dual diagnostic (`45fce1b`)

- `LatentDualRelationSwarm` rebuilt: Core B receives **only** `x_B` and the fixed
  union-graph mask (16 tokens; **no COMM register, no message embedding** in this model).
  Its full trajectory is bit-identical under flip/forced `m=±1` interventions
  (unit-tested: candidates and all 9 trace tensors exactly equal). The bit is used only by
  `select()` **after** both `r0_logits`/`r1_logits` exist; labels never enter inference.
- Dead parameters removed: `msg_embedder` and the learned COMM token are no longer
  instantiated; `msg_encoder` retained (learned bit + its BCE supervision).
  **Corrected dual parameter count: 279,691** (137,476 + 141,058 + 129 + 2×514),
  unit-asserted. Primary model unchanged at **279,561**.
- Objective wired per the predeclared design and now factored into testable
  `compute_loss()`: dual relation heads supervised with their **own** relation labels
  (never Y — verified by an exact-match plus swap check); aux_A CE + message BCE retained;
  the discrete selection is not differentiated — the sender's gradient path is the message
  BCE (verified: `msg_encoder` receives nonzero gradient; every dual parameter trains).
- `comm_mode`/`zero_message` are accepted interface no-ops for the dual model and are
  documented as such. The primary model's COMM semantics are untouched.

## 4. Fix 3 — training/test separation (`9257ad3`)

- `train_swarm_conditioned.py` normal and smoke paths now generate **only** train/val
  splits: `build_loaders(seed)` returns exactly two loaders, contains no test-seed
  reference, and the trainer holds no test-evaluation call path
  (`test_trainer_generates_only_train_and_val` asserts the loaders, scans the source for
  `SPLIT_SEEDS["test"]`/`99991`, and monkeypatch-guards the generator;
  `test_smoke_path_never_builds_test_split` runs the full smoke entry point under the
  same guard).
- New `evaluate_phase2c_test.py`: the **only** component permitted to generate the
  predeclared test split (seed 99991) and only when explicitly invoked with a frozen
  manifest (kind, frozen flag, source commit, split seed, per-model entries with
  checkpoint path + SHA-256 + config). It verifies existence and checksums of every
  checkpoint **before** generating any test data, rejects unfrozen/ambiguous/missing/
  mismatched manifests, evaluates each listed model exactly once, and writes metrics,
  per-example predictions and provenance to ignored storage. It applies **no gates** and
  makes **no decisions**.
- The evaluator was **implemented and unit-tested but not invoked**: its end-to-end test
  (`test_evaluator_end_to_end_on_mocked_split`) monkeypatches a tiny 8-sample fixture;
  the real 99991 split is never built. Rejection paths are covered by
  `test_evaluator_rejects_unfrozen_missing_and_mismatch` (missing checkpoint, checksum
  mismatch, unfrozen manifest, duplicate entries).
- The stability stop-gate remains intact and was re-verified: a non-smoke invocation
  refuses with exit code 2 (log: `runs/20261017T060908Z_phase2c_fix_only/stopgate_check.log`).

## 5. Fix 4 — trajectory aggregation and terminal-node gate (`9257ad3`)

- `evaluate_trajectory()` now computes true element-weighted means
  (`sum(|Δ|)/n_elements`) per step for all graph nodes **and** for the terminal node 15,
  overall and per agreement/disagreement group, with sample counts; empty groups and
  non-finite deltas fail explicitly. Full details, retraction of old magnitudes, and
  provenance: `reports/phase2c-trajectory-metric-erratum.md`.
- Batch-size invariance (full batch vs size-5 batches) verified within 1e-6 relative
  tolerance; hand-computed weighted subgroup means verified exactly
  (`test_trajectory_metric_hand_computed_weighted_subgroups`).
- Full-mode mechanism gate strengthened: requires **terminal-node** mean |Δz| > 0 at ≥1
  intermediate step (1–7) — no tolerance needed (deterministic CPU float32; a positive
  attention weight on the COMM key makes the element mean strictly positive).
  Final-only gate: exact bit-identity through step 7 asserted directly, plus step-8
  sensitivity reported. Both verified on the real model in tests and smoke.

## 6. Verification

- `python -m compileall -q model data test_system.py test_phase2b.py test_phase2c.py
  train_swarm_conditioned.py evaluate_phase2c_test.py` → clean.
- Combined suite: `python -m pytest -q test_system.py test_phase2b.py test_phase2c.py`
  → **47 passed, 0 failed** in 8.38 s (log + exit code in
  `runs/20261017T060908Z_phase2c_fix_only/pytest_full.log`). v7.1 and Phase 2b suites
  unmodified (7 + 11); Phase 2c grew to 29 tests.
- Smoke (engineering-only; small fixtures; no test data): primary full / primary
  final-only / dual full / dual final-only — all losses finite, no `grad is None`
  parameters; primary/full step-1 corrected graph-node mean |Δz| 1.185e-01, terminal
  2.932e-02; primary/final-only exactly 0.0 through step 7; dual exactly 0.0 in both
  modes (selector-independent by design, as corrected). Logs in
  `runs/20261017T063938Z_phase2c_fix_smoke/`.
- During the pass, two test-fixture defects (a never-terminating distinctness fixture; an
  off-by-one proposal-count assertion) and one genuine aggregator bug were found and
  fixed before commit; nothing was weakened to pass.

## 7. Housekeeping metadata correction

`reports/phase2b-stability-report.md` referenced the sweep directory as
`runs/20261016T231051Z_stability/`; the actual directory (verified on disk, containing the
18 per-run folders and `aggregate.json`) is `runs/20261016T225726Z_stability/`. The two
path references in that report were corrected with a dated note appended there; no
numbers, tables, or conclusions were changed.

## 8. Unresolved issues / blockers

- None blocking this task. Old trajectory magnitudes for historical runs remain
  unrecoverable (no raw traces were persisted); the erratum documents this explicitly.
- The stability stop-gate still stands: Phase 2c confirmatory training remains blocked
  until a separately predeclared optimisation study yields an eligible shared learning rate.

> *Historical Note (2026-10-17): This report reflects the earlier engineering milestone where training was skipped. For the completed 15-run confirmatory study, see `reports/phase2c-candidate-recipe-run-report.md`.*

# Phase 2c Run Report — Message-Conditioned Reachability

**Branch:** `experiment/phase2c-message-conditioned-reachability`
**Implementation commit:** `c09ba99` · **Design (predeclared):** `reports/phase2c-design.md`
**Status: ENGINEERING + SMOKE ONLY. No Phase 2c training beyond the engineering smoke run
was performed, and the Phase 2c test split (seed 99991) was never generated or evaluated.**

## 1. Why no training ran (stop-gate applied)

The Stage 2 stability sweep (18 instrumented runs, `reports/phase2b-stability-report.md`)
found **no learning rate eligible** under the predeclared rule: full-mode all-gate passes
were 0/3 (8e-4), 1/3 (4e-4), 1/3 (2e-4); final-only 1/3, 0/3, 2/3. The charter's Phase B
stop-gate therefore forbids a confirmatory Phase 2c comparison: "If no learning rate is
eligible … do not run or claim a confirmatory multi-seed Phase 2c comparison. A tiny smoke
run is allowed and must be labelled engineering-only." The trainer enforces this with an
explicit guard (non-smoke runs refuse without `--allow-unstable-lr`).

## 2. What was built and verified (engineering success: YES)

- **Data** (`data/conditioned_graph_task.py`): exactly balanced 8-combination selector
  task; per-relation v7.1 matched-DAG sampling with endpoint/path constraints; 64-dim
  `[union|R0|R1|id]` layout; namespaced hashing of all three graphs per sample; split
  seeds 4201/13371/99991 predeclared. Verified at full train size (n=4000): all eight
  combinations exactly 500 samples; disagreement fraction exactly 0.5; P(Y=1)=0.5
  (both conditional on b_A too); no-selector optimum exactly 0.75; 12,000 unique graph
  hashes; determinism and cross-split disjointness unit-tested.
- **Models** (`model/swarm_conditioned.py`): `LatentConditionedSwarm` **279,561 params**
  (as predeclared); `LatentDualRelationSwarm` compute-both-then-select diagnostic
  **280,075 params** (decoder replaced by Identity — no dead parameters). Documented
  Core B warm-start mapping (relation channels zeroed) verified **bit-exact** against the
  Phase 1 encoder.
- **Tests:** `test_phase2c.py` 13 tests; combined suite **31 passed** (7 v7.1 + 11 Phase 2b
  + 13 Phase 2c), with earlier suites unmodified. Two dev-time defects found by the tests
  and fixed before any run: dual-model warm-start key conflict, dual trace-unpacking arity.
- **Trainer** (`train_swarm_conditioned.py`): full predeclared metric suite (overall, by
  b_A/Y, quadrants, agreement/disagreement subsets, forced-message counterfactuals,
  switch consistency/invariance, zero-message overall+disagreement, sender probe,
  per-step trajectory sensitivity by subset, final-only invariance), paired-seed
  determinism, `last.pt` saved for every run including failures.

## 3. Smoke runs (engineering-only; NOT research results)

Config: 2 training batches, 2 validation batches, 1 epoch, warm-started, no test data.

| Config | Loss finite | Grads complete | Step-1 traj |Δz| (m=+1 vs −1) |
|---|---|---|---|
| primary / full | yes | yes (no grad-None) | 9.26e-04 (>0: message reaches graph states) |
| primary / final-only | yes | yes | **0.000e+00** (bit-identical through step 7, as designed) |
| dual / full | yes | yes | 1.02e-03 (>0) |
| dual / final-only | yes | yes | 0.000e+00 |

Small-sample val accuracies (48.4–53.1%) are meaningless at smoke scale and are reported
only to document that the pipeline runs. Logs: `runs/20261016T234452Z_phase2c_smoke/`.

## 4. Task/scientific status

- **Engineering success:** complete (data, models, tests, trainer, smoke).
- **Task success:** NOT RUN (blocked by the stability stop-gate).
- **Scientific finding:** none from Phase 2c in this iteration; the predeclared gates,
  counterfactual protocol and paired-statistics plan in `reports/phase2c-design.md` are
  ready for a future stabilised recipe. The recommendation from the stability study
  stands: first run a dedicated optimisation study (the sender-probe gate is the most
  frequent failure at every learning rate; 2e-4 shows the lowest instability), then
  execute Phase 2c's five-seed paired comparison with the frozen protocol.

## 5. Test-split discipline

The Phase 2c test split (seed 99991) was **never generated** (the smoke builder excludes
it), so it remains untouched and reusable for the future confirmatory run. No Phase 2b
test results were reused either.

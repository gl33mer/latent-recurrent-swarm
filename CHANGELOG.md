# Changelog

## 2026-10-16 — Phase 2b stage B: baseline housekeeping & artifact provenance

Branch `chore/preserve-v7.1-artifacts`; baseline source frozen at tag `v7.1-baseline`
(commit `9f94151`); immutable tag `v7.1-pre-run` untouched.

1. **B1 — report metadata fix:** `reports/first-run-report.md` said the repository was
   *private*; it has been public since the owner-requested visibility change recorded in
   the 2026-10-16 housekeeping entry above. Corrected the visibility statement and added
   the provenance note. No historical metric was altered.
2. **B2 — Phase 1 checkpoint verified:** root `checkpoint_phase1.pt` and
   `runs/20261016T211522Z/checkpoint_phase1.pt` both hash to
   `ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e`, matching the
   recorded first-run checksum exactly. No reproduction needed for Phase 1.
3. **B3 — Phase 2a artifact capture:** the original v7.1 `train_swarm.py` never saved a
   trained swarm checkpoint, so none exists. Added an optional `--artifact-dir` argument:
   when provided, it saves `checkpoint_phase2a.pt` plus a provenance/metrics JSON **only
   after all validation gates pass** and the test evaluation completes. The change adds no
   model computation, no RNG consumption, and no gate change (verified: `compileall` clean,
   `test_system.py` 7/7 pass, behaviour identical without the flag).
4. A clearly-labelled **Phase 2a reproducibility rerun** (frozen source at
   `v7.1-baseline` + the artifact-capture option, same seeds 42/1337/9999, seed 42) was
   executed under run ID `20261016T215726Z` and **reproduced every historical validation
   and test metric exactly** (97.50% connected; 96.40/97.20/98.40/98.00% quadrants;
   flip 0.00%/0.41%; zero 48.60%; probe 97.80%/97.20%/98.40%; test 97.00% connected and
   matching subgroup metrics). Checkpoint SHA-256:
   `05a163e23e240173fefcbb3a2c404abaa38db53bbbc18a4ca77191af8e693763`. It is labelled a
   reproduction artifact, not the original trained state (which was never saved).

## 2026-10-16 — v7.1 baseline bootstrap (test-only/dependency hardening)

1. **Dependency addition (no architecture change):** the canonical code calls
   `Tensor.numpy()` (`data/graph_task.py::_hash_graph`, `test_system.py::test_conditional_parity_balancing`),
   but the official CPU-only PyTorch wheel does not bundle NumPy. Added `numpy>=1.26`
   to `requirements-cpu.txt` and installed it in `.venv`. No model, loss, threshold,
   or training change.
2. **Scope clarification (per v7.1 spec):** Phase 2a here is *decoupled* recurrent
   reasoning with receiver-side one-bit decision fusion (Core B never ingests Core A's
   message into its recurrence; the message enters only the final parity readout).
   This is **not** Phase 2b: no COMM token, no global workspace, no recurrent message
   integration is implemented or claimed.
3. **Housekeeping:** added `*.swp` to `.gitignore` for a pre-existing editor swap file
   in the working directory. No project code affected.
4. **Repository visibility (2026-10-16, post-run, owner request):** repository switched
   from private to **public** for external review at the owner's explicit instruction,
   after the completed baseline run and a secret scan of all tracked content. The v7.1
   spec's private-by-default rule governed the initial setup only; provenance of this
   change is recorded here and in `WORKLOG.md`.
4. **Test hardening already present in the canonical suite** (recorded, not modified):
   `test_binary_channel_contract` requires the seeded batch to exercise both message
   polarities plus exact flip inversion; `test_parity_head_generalization_on_noisy_inputs`
   runs five fixed seeds with a separate noisy test set at a 99% threshold.

## 2026-10-16 — Phase 2b COMM-token implementation (branch experiment/phase2b-comm-token)

New code, no change to v7.1 paths: `model/swarm_comm.py` (LatentCommSwarm, 284,107
params), `test_phase2b.py` (11 tests; combined suite 18 passed), `train_swarm_comm.py`
(CLI trainer with full/final-only modes, matched Phase 2a recipe, stratified gates,
trajectory-sensitivity diagnostics, smoke mode), `reports/phase2b-design.md` (pre-declared
design/gates), README Phase 2b scope section. v7.1 model/data/tests/trainers untouched.

## 2026-10-16 — Phase 2b runs (branch only)

Six matched training runs (full/final-only × seeds 42/43/44). Two passed all validation
gates (full/seed43, final-only/seed42); four failed at least one gate — recorded verbatim,
no thresholds altered. Trajectory diagnostics confirm recurrent bit-integration in full
mode and exact step-8-only influence in final-only mode. Training instability at the
frozen v7.1 recipe documented as a finding; stabilisation and a message-conditioned task
are proposed as separate follow-ups. See `reports/phase2b-run-report.md`.

## 2026-10-16 — Phase 2b analysis addendum (stage 1 housekeeping)

`reports/phase2b-analysis-addendum.md` added on `experiment/phase2b-comm-token`: corrects
the over-strong causal claim about instability source (mechanism vs architecture
confound), adds exploratory paired validation arithmetic (mean +8.97 pp full-minus-
final-only, n=3, descriptive only), and records artifact availability/checksums and the
durable-archive location with its storage caveat. No historical numbers or run outcomes
changed; original wording preserved with cross-reference.

## 2026-10-16 — Phase 2b stability study (branch experiment/phase2b-stability-study)

Predeclared design (`reports/phase2b-stability-design.md`) + new isolated instrumentation
driver `experiments/stability/run_stability_sweep.py` (train/val only, per-epoch loss
components, stratified validation metrics, per-group gradient norms, NaN/Inf checks,
trajectory sensitivity, checkpoints saved for failed runs too). No change to model
semantics, gates, or v7.1/Phase 2b code paths.

## 2026-10-16 — stability sweep result: no eligible learning rate

18/18 instrumented runs completed (validation-only). Under the predeclared rule no shared
LR qualifies (full-mode passes 0/3, 1/3, 1/3 at 8e-4/4e-4/2e-4; final-only 1/3, 0/3, 2/3),
so the Phase 2b recipe is classified NOT YET STABILISED. Phase 2c will be implemented and
smoke-tested only; no confirmatory comparison. See reports/phase2b-stability-report.md.

## 2026-10-16 — Phase 2c message-conditioned reachability (branch only; smoke-only per stability gate)

New: data/conditioned_graph_task.py, model/swarm_conditioned.py (279,561-param primary +
280,075-param dual diagnostic), test_phase2c.py (13 tests; combined suite 31 passed),
train_swarm_conditioned.py (refuses non-smoke runs without an eligible LR),
reports/phase2c-design.md (predeclared). Smoke runs pass all engineering checks in both
modes for both models. NO Phase 2c training beyond smoke and NO test evaluation, per the
"not yet stabilised" verdict from the stability study.

## 2026-10-17 — Phase 2c fix-only engineering pass (branch experiment/phase2c-message-conditioned-reachability)

Four fixes, no research-claim changes: (1) raw structural graph-hash uniqueness across
roles/splits with canonicalised hashing and bounded failure; (2) dual-relation diagnostic
made genuinely selector-independent (no COMM register/message embedding; Core B
trajectory bit-identical under any message intervention; dual param count corrected to
279,691; primary unchanged at 279,561); (3) training strictly separated from test
evaluation (trainer is train/val-only; new manifest-driven evaluator implemented and
unit-tested but NOT invoked; seed 99991 never generated); (4) trajectory metrics
corrected to element-weighted means with terminal-node intermediate gate and exact
final-only invariance; old magnitudes retracted via
reports/phase2c-trajectory-metric-erratum.md. Combined suite 47 passed. No confirmatory
training, no test evaluation. See reports/phase2c-fix-only-report.md.

## 2026-10-17 — Phase 2c confirmatory validation run: PARTIAL/UNSTABLE (test untouched)

Phase 2b report erratum pushed to the study branch (607221a). On new branch
experiment/phase2c-lambda2-paired-confirmatory (from 744e871): recipe-transfer addendum
pre-registered (4939d94) before training; trainer wired for the candidate recipe
(lr 2e-4, lambda_msg 2.0 default, --confirmatory-phase2c exact-recipe guard replacing the
generic bypass, atomic per-epoch last.pt, per-epoch metrics); 53/53 tests. Confirmatory
grid (source commit ce3c009): 15/15 runs completed; 0/5 paired seeds qualified (all
primary runs ~59-64% overall, disagreement ~chance, probe ~100%, mechanism gates passed,
stable plateaus); dual diagnostic also at chance on disagreement → bottleneck is
per-relation computation. Test split seed 99991 NOT generated; evaluator NOT invoked.
Classification: VALIDATION PARTIAL/UNSTABLE — TEST UNTOUCHED.

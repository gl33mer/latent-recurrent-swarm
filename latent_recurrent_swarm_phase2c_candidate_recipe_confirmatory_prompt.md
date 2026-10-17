# Autonomous Agent Task: Phase 2b Report Erratum + Phase 2c Screened-Recipe Paired Experiment

**Repository:** `https://github.com/gl33mer/latent-recurrent-swarm`  
**Workspace:** `/workspace/latent-recurrent-swarm`  
**Runtime:** Debian 12, Python 3.11, CPU-only PyTorch, 4 vCPUs / approximately 8 GiB RAM  
**Agent:** Pi coding agent  
**Task type:** research housekeeping, pre-registered recipe transfer, engineering verification, then a bounded validation-only confirmatory Phase 2c experiment; held-out test evaluation is conditional and must be a separate final action.

---

## 0. Mission and governing decision

The owner has approved the following plan:

1. Correct the Phase 2b message-supervision weight-study report's internal arithmetic and trend-description errors using a dated erratum. Preserve the original run records and explain all corrections transparently.
2. Transfer the screened Phase 2b candidate recipe to Phase 2c as a **candidate**, not as a guarantee of Phase 2c stability:
   - learning rate `2e-4`;
   - sender message BCE coefficient `lambda_msg = 2.0`;
   - every other Phase 2c task, architecture, split, optimizer setting, loss coefficient other than the sender BCE weight, gate, and analysis rule remains as already predeclared.
3. Add a dated Phase 2c design amendment and commit/push it **before any Phase 2c training run**.
4. Run the primary Phase 2c full-vs-final-only validation comparison for paired model seeds `42, 43, 44, 45, 46`, plus the already predeclared selector-independent compute-both diagnostic for those five seeds.
5. Use validation outcomes only to make the predeclared stable-task-success decision. Do not create or evaluate the Phase 2c test split until the validation decision, configurations, model list, source commit and checkpoint checksums have been frozen.
6. If validation meets the predeclared stability rule, invoke the separate manifest-driven test evaluator once for the frozen set of models, then analyse paired predictions. If it does not, **do not generate or evaluate the Phase 2c test split**.

The screened Phase 2b recipe was successful in 6/6 runs at `lr=2e-4`, `lambda_msg=2.0`, but this is evidence only for the frozen Phase 2b XOR model and task. Phase 2c has a different task and receiver architecture. The transferred recipe must be evaluated rather than assumed stable.

This task may update source code, tests, reports and run metadata. It must not weaken or change any task acceptance threshold after seeing metrics. Do not broaden the recipe sweep. Do not launch any alternate learning rate or message-supervision weight. Do not modify the primary Phase 2c architecture except for the explicitly authorised configuration wiring and validation/test protocol fixes described here.

## 1. Non-negotiable operating rules

- Work in `/workspace/latent-recurrent-swarm`.
- Repository is public and owner-authenticated `gh` is expected. Never print, copy into files, or commit credentials, tokens, API keys, SSH private keys or other secrets.
- **Inspect before editing.** Start with Git state and branch/commit reconciliation. Do not assume the current branch, local HEAD or upstream references.
- Never run `git reset --hard`, `git clean -fd`, force-push, rebase published commits, amend published commits, or discard uncommitted owner work. If the tree is dirty, inventory and preserve the work before proceeding.
- Do not alter or delete historical run directories, checkpoint files, report tables, or prior commits. Corrections to reports must be dated addenda/errata; do not silently rewrite the history.
- Never commit checkpoints, raw run artifacts, `.venv`, logs, environment dumps, or per-example prediction tensors. Keep these in ignored `runs/` storage and copy them to durable storage if the VM is ephemeral.
- All model runs must record the source commit, config, seed, data split seeds, checkpoint hashes, environment and logs.
- Each run must retain a last checkpoint and diagnostics even if acceptance gates fail. Failed results are evidence.
- Do not generate or evaluate Phase 2c test data early. The reserved test split seed is `99991`; training/smoke code must not refer to or invoke this split generator. Only `evaluate_phase2c_test.py`, after a frozen manifest and all checksums are verified, may generate it.
- Do not invoke a generic bypass such as `--allow-unstable-lr` for arbitrary settings. The only authorised training recipe for this task is the exact screened candidate `lr=2e-4`, `lambda_msg=2.0` and the fixed configuration in Section 6.
- No asynchronous/background promise. Execute each stage now; if time limits prevent completion, report partial completion honestly and leave test data untouched.

## 2. Stage A — Reconcile repository and preserve existing evidence

### A1. Inspect the actual state

Run and save outputs in a new ignored housekeeping directory, for example `runs/<UTC_RUN_ID>_phase2c_recipe_transfer/`:

```bash
pwd
git status --short --branch
git remote -v
git branch --show-current
git branch -avv
git log --graph --decorate --oneline -30
git tag --list
```

Then fetch without changing the working tree:

```bash
git fetch origin --prune
```

Inspect the exact remote branch tips and commit history for at least:

- `experiment/phase2b-msg-supervision-weight-study`
- `experiment/phase2b-stability-study`
- `experiment/phase2c-message-conditioned-reachability`
- `main`

Verify the intended commits and artifacts from live refs, not only from a previous report. Relevant anchors supplied in the research record include Phase 2b message-weight design `a6eb669`, runner commits `6d12304` and `5a28a2a`, base Phase 2b stability state `ed46cd4`, and Phase 2c fixes `45fce1b` / `9257ad3`; actual branch histories may have later commits. Use current `git log` and remote refs as the source of truth.

If `git status` is dirty, do not switch branches until every changed/untracked file has been inventoried and preserved. Do not overwrite user work. If a branch is behind or diverged, diagnose and preserve it; do not reset it to `origin` blindly. If the public branch contents conflict with reports, note the discrepancy and use the actual reachable commits to reconcile it.

### A2. Verify the critical artifacts

Before any model run, confirm that these existing artifacts remain present and match their recorded hashes:

- Phase 1 warm-start checkpoint SHA-256:
  `ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e`
- Phase 2b full/seed-43 checkpoint prefix previously recorded as `38c505…` (retrieve full hash from its metadata/report).
- Phase 2b final-only/seed-42 checkpoint prefix previously recorded as `79aada…` (retrieve full hash from its metadata/report).
- All 18 message-weight study run directories under `runs/20261017T101804Z_msg_weight_stability/`, including `last.pt` and epoch checkpoints for both passing and failing runs; `aggregate.json`, `summary.csv`, `selection.json`, environment snapshot, run metadata and raw `metrics.jsonl` histories.
- Phase 2c raw-hash/dual-control/trajectory-metric fixes and regression tests.

If the Phase 1 checkpoint is missing or its checksum differs, stop before training. Do not regenerate it under the same historical run ID. It may be recovered from a verified durable copy only if the checksum matches. Otherwise report the blocker and seek a separately recorded remediation.

### A3. Save run provenance

Capture at least:

- UTC timestamp and run ID;
- `git rev-parse HEAD` and branch name;
- `git status --porcelain=v1`;
- OS/kernel, CPU count, RAM/disk snapshot;
- Python, PyTorch, NumPy and pytest versions;
- `python -m pip freeze`;
- `torch.get_num_threads()` and CUDA availability;
- Phase 1 checkpoint path and full SHA-256.

Keep snapshots under ignored `runs/` directories. Do not create or inspect the reserved Phase 2c test split in this stage.

## 3. Stage B — Correct the Phase 2b message-weight report

Work on a new small housekeeping branch based on the verified current tip of `experiment/phase2b-msg-supervision-weight-study`. Do not edit Phase 2c while doing this report correction. If a clean checkout is not possible without risking work, use a separate Git worktree or stop and report the blocker.

### B1. Recompute from raw data

Do not trust copied prose or manually transcribe only the report. Re-read the actual `selection.json`, `aggregate.json`, `summary.csv` and per-run `run_summary.json` / `metrics.jsonl` for all 18 cells. Independently recompute counts and medians using the predeclared definitions.

From the published final-epoch table, the corrected all-gate counts are:

- `lambda_msg=0.5`: full 1/3, final-only 2/3, total **3/6**;
- `lambda_msg=1.0`: full 2/3, final-only 2/3, total **4/6**;
- `lambda_msg=2.0`: full 3/3, final-only 3/3, total **6/6**.

The table's sender-probe-gate failures are:

- `lambda_msg=0.5`: **3/6** (not 4/6);
- `lambda_msg=1.0`: **2/6**;
- `lambda_msg=2.0`: **0/6**.

The reported medians of the per-run mean absolute epoch-to-epoch validation-accuracy changes are:

- `lambda_msg=0.5`: 4.68%;
- `lambda_msg=1.0`: 5.62%;
- `lambda_msg=2.0`: 2.47%.

Therefore it is wrong to describe instability as *monotonically* decreasing from 0.5 through 1.0 to 2.0: it increases from 0.5 to 1.0, then decreases at 2.0. The total all-gate pass count does increase 3/6 → 4/6 → 6/6. The selection still favours `lambda_msg=2.0` under the predeclared rule, because it has the highest all-gate pass count and lowest instability median among eligible candidates.

Independently verify every number from the data. If raw artifacts disagree with the figures above, do not force the numbers to match this prompt; document the exact discrepancy and stop for review before committing conclusions.

### B2. Add dated erratum, do not erase the original record

Create a dated addendum, e.g. `reports/phase2b-msg-weight-study-erratum-2026-10-17.md`, and append a short pointer/addendum to the original report. Clearly distinguish:

- original report as published;
- numerical corrections and their provenance;
- computations derived from raw run metadata;
- the fact that `lambda_msg=2.0` remains selected by the predeclared decision rule;
- limitations (three seeds per cell, one learning rate, one task/model family).

Correct the original report only by appending a dated addendum; do not silently replace its existing narrative or tables. Any summary paragraph added must say that the instability medians are non-monotonic, while the total pass count rises and the final candidate is `lambda_msg=2.0`.

Run documentation tests or any script used to calculate the figures, run `git diff --check`, review the diff, commit and push this report-only change. No training is permitted on this branch. Record its final commit hash.

## 4. Stage C — Create a clean Phase 2c candidate-recipe branch

Start from the verified updated remote tip of `experiment/phase2c-message-conditioned-reachability`. Preserve that branch as the existing engineering baseline. Create a new dedicated branch from it, for example:

`experiment/phase2c-lambda2-paired-confirmatory`

Do not merge into `main`. Do not force-push. Do not carry Phase 2b changes into Phase 2c except the report erratum on its own branch and the explicitly screened configuration change below.

Before editing, read the current versions of:

- `latent_phase2b.md` and `latent_phase2c.md` if present;
- `reports/phase2b-msg-weight-study-design.md`;
- `reports/phase2b-msg-weight-study-report.md` and the new erratum;
- `reports/phase2b-stability-report.md` including its dated addenda;
- `reports/phase2c-design.md`;
- `reports/phase2c-fix-only-report.md` and `reports/phase2c-trajectory-metric-erratum.md`;
- `data/conditioned_graph_task.py`;
- `model/swarm_conditioned.py` and `model/swarm_comm.py`;
- `train_swarm_conditioned.py`;
- `evaluate_phase2c_test.py`;
- `test_system.py`, `test_phase2b.py`, `test_phase2c.py`;
- all relevant `CHANGELOG.md`, `WORKLOG.md` and run reports.

Confirm that previous Phase 2c fixes really exist in the current source. At minimum verify:

1. raw structural hashes, with no role namespace in the uniqueness key, deduplicate all A/R0/R1 graphs across all splits;
2. `LatentDualRelationSwarm` has no COMM register or message embedding and its candidate logits/trajectory are selector-independent before final selection;
3. corrected element-weighted trajectory aggregation is batch-size invariant;
4. primary full-mode mechanism gate checks terminal node 15 sensitivity at an intermediate step;
5. final-only graph-node states are bit-identical through steps 1–7;
6. `train_swarm_conditioned.py` generates train/validation only;
7. only `evaluate_phase2c_test.py` is permitted to generate the test split and only after a valid frozen manifest plus verified checkpoint hashes;
8. earlier v7.1 / Phase 2b / Phase 2c test suites are retained and not weakened.

If one of these regressions has reappeared, fix it before proceeding and add a regression test. Keep unrelated modifications out of this branch.

## 5. Stage D — Commit the Phase 2c recipe-transfer addendum BEFORE training

Create `reports/phase2c-recipe-transfer-addendum-2026-10-17.md` (use actual current date if different). This file must be committed and pushed before any training result can be produced. It must explicitly record:

### D1. Rationale and epistemic status

- Phase 2b message-weight study screened `lr=2e-4`, `lambda_msg=2.0` in a fixed Phase 2b XOR setup and passed all 6/6 runs under its predeclared rule.
- This makes the recipe a screened candidate, not a guarantee of Phase 2c stability or an established optimum.
- Phase 2c task, Core B input representation, COMM path and objective differ. Phase 2c will validate transfer on its own task.
- No Phase 2c metrics, logs or test data have been inspected/used to choose this recipe. Cite the source commit and the report correction commit.

### D2. Locked Phase 2c recipe

- Primary model: `LatentConditionedSwarm`.
- Access modes: `full` and `final-only`.
- Model seeds: 42, 43, 44, 45, 46.
- Paired design: same seed, same Phase 1 checkpoint, identical fresh paired initialisation, same train/validation data, same deterministic train-loader order per paired seed; no optimizer state sharing.
- Task split seeds: train `4201`, validation `13371`, test `99991`.
- Training samples: 4,000; validation samples: 1,000; test samples: 1,000 if and only if the test-evaluation stop condition is satisfied.
- Batch 64; 20 epochs; float32 CPU; 4 intra-op threads; AdamW weight decay `1e-2`.
- Learning rate `2e-4`.
- `lambda_msg=2.0` in `BCEWithLogitsLoss(raw_m, b_A)`.
- Preserve all other objective weights and task design from `reports/phase2c-design.md`: primary loss remains `CE(task_logits, Y) + 0.5*CE(aux_A, b_A) + lambda_msg*BCEWithLogits(raw_m, b_A)`; do not add an auxiliary Core B relation-selection loss to the primary model.
- No learning-rate schedule, warm-up, gradient clipping, early stopping, best-checkpoint selection, data changes, architecture changes, or threshold changes in this run.
- Primary runs: 5 seeds × 2 modes = 10.
- Selector-independent dual diagnostic: 5 runs, one per seed, trained as specified by the existing design; it computes both relation outputs from a Core B trajectory independent of the selector, then selects after both outputs exist. It is a computational-sufficiency baseline, not part of the primary full-vs-final-only paired statistic, and no new success thresholds may be invented for it after results.
- Total runs if all complete: 15.

### D3. Pair-pass definition and stop rule

The original design says “stable task success requires ≥4 of 5 paired seeds satisfying the task/control gates.” For a paired full-vs-final-only comparison, operationalise this before training as follows:

- A **paired seed qualifies** only when the primary/full run AND the primary/final-only run for that same seed each pass every Phase 2c validation task/control gate and their mode-specific mechanism check.
- Stable task success requires at least **4 of the 5 paired seeds** to qualify.
- Report individual mode pass counts as well as the stricter paired count; do not hide asymmetric failures.
- If fewer than 4 paired seeds qualify, classify the validation study `PARTIAL/UNSTABLE`; do not generate or evaluate the test split. Do not rerun only failed seeds, change a setting, or select a best epoch under the same experiment ID. A new attempt requires a new predeclared experiment and split policy.

This operational definition must be written into the amendment and committed before training starts. If the pre-existing charter contains an unambiguous conflicting definition, do not silently override it; document the conflict in the addendum, reconcile it before training, and commit the resolution first.

### D4. Frozen Phase 2c acceptance gates

Copy the thresholds and definitions exactly from the existing Phase 2c design; do not reuse the Phase 2b XOR gates:

- Selected-task validation accuracy ≥ 82% overall.
- Each `(b_A, Y)` quadrant ≥ 75%.
- Sender probe ≥ 95% overall and ≥ 90% within each `b_A` group.
- Zero-message accuracy ≤ 77% overall and ≤ 53% on the disagreement subset.
- Forced-message counterfactual accuracy ≥ 75% for each of `m=-1` against `r0` and `m=+1` against `r1`.
- Selector switch consistency ≥ 85% on disagreement cases.
- Selector invariance ≥ 85% on agreement cases.
- Full mode: terminal graph node 15 must show strictly positive, corrected element-weighted message sensitivity at at least one intermediate step 1–7.
- Final-only: ordinary graph-node states must be exactly tensor-equal across opposite forced messages through step 7.
- Empty conditioning groups, non-finite losses/gradients/parameters or broken expected gradient paths are hard failures.
- Overall and disagreement-set metrics must be separately reported. Do not interpret high agreement-subset performance as evidence of message use.

### D5. Predeclared scientific interpretation

Copy the interpretation rules from Phase 2c design and make no change after results:

- Evidence favouring early integration requires stable task success in the primary/full condition and a mean paired test advantage of at least +5 percentage points for full vs final-only on the disagreement subset, with the predeclared paired bootstrap interval excluding zero and the direction reported across the paired training seeds. Given only five seed pairs, use cautious language.
- If both modes work with a small or uncertain paired difference, conclude that this task does not demonstrate an advantage of early access; it remains solvable with parallel reasoning and final selection.
- If final-only wins, report it without claiming that trajectory integration did not occur.
- If stability/task gates fail, report `PARTIAL/UNSTABLE` and do not use the test split to diagnose the failure.
- The dual model is an additional selector-independent compute-both control. Report each relation-head accuracy, selected-task accuracy overall and on the disagreement subset. Do not promote it to a primary model or create post-hoc thresholds.

## 6. Stage E — Implement the exact candidate recipe safely

Modify `train_swarm_conditioned.py` and relevant tests narrowly.

### E1. Make message supervision configurable and traceable

- Add an explicit `lambda_msg` parameter to the loss function / loss computation. Replace the hard-coded `0.5 * BCEWithLogitsLoss(...)` for the message term with `lambda_msg * BCEWithLogitsLoss(...)` for both primary and dual models. Preserve other loss terms exactly as in the Phase 2c design.
- Add CLI option `--lambda-msg` and wire it from the parsed args into every training loss call.
- Make the candidate recipe defaults `--lr 2e-4` and `--lambda-msg 2.0`, but do not let a default alone silently authorise confirmatory training.
- Require an explicit opt-in flag such as `--confirmatory-phase2c` for non-smoke training. When that flag is present, validate that the supplied recipe exactly matches the committed addendum (`lr=2e-4`, `lambda_msg=2.0`, weight decay `1e-2`, batch 64, 20 epochs, T=8, fixed loss weights, expected Phase 1 checkpoint SHA). Refuse other values with a clear non-zero exit status. Do not retain a generic `--allow-unstable-lr` bypass that permits arbitrary recipes.
- Smoke mode must remain available without authorising the test split and must never produce a research-result label.
- Store `lr` and `lambda_msg` in console logs, checkpoint metadata, run JSON, config fingerprint and evaluation-manifest config. Make validation reject missing or mismatched candidate config.
- Add a test proving changing `lambda_msg` changes only the message BCE coefficient and does not alter other objective terms. Add a test that non-smoke training refuses an alternate LR/weight recipe and only accepts the exact predeclared recipe.

### E2. Preserve paired initialisation and data order

For each seed 42–46:

1. reset Python and PyTorch RNGs using the declared seed before model construction;
2. verify the Phase 1 warm-start checkpoint SHA;
3. initialise each paired primary mode independently from the same RNG seed and checkpoint;
4. compute and record an initial-state checksum over the fully constructed/warm-started state dict; the full and final-only primary models for a seed must match exactly;
5. create train DataLoader with a dedicated generator seed `9000 + model_seed`, and verify identical order within the primary pair. Do not let validation/evaluation draws advance the train-loader generator;
6. use separate optimizer instances and independent optimizer state for all runs.

The diagnostic dual model is trained once per seed; document its selector-independent nature and record its initial-state checksum. No test data may be used for initialisation, training, diagnostics, selection or checkpoints.

### E3. Checkpoint and failure policy

For each run create a unique ignored directory under `runs/`, with a deterministic readable ID that includes experiment, model, mode, seed and recipe. Save:

- `last.pt` after every epoch (atomically: save temporary file then rename);
- `epoch05.pt`, `epoch10.pt`, `epoch15.pt`, `epoch20.pt`;
- `metrics.jsonl` with one record per epoch;
- run config/meta JSON, gate summary, initial and final state checksums, checkpoint SHA-256, seed/order evidence, source commit and environment snapshot;
- complete stdout/stderr log.

Do not use a best-validation checkpoint. Final epoch is the declared model. If a gate fails, still retain `last.pt` and all available diagnostics. If a non-finite event occurs, record it and safely checkpoint immediately before terminating that run. If a process is interrupted after an epoch, preserve the latest checkpoint and mark the run interrupted/incomplete rather than as a completed failure or pass.

At each epoch record at minimum: mean total loss and each component (task, aux_A, message BCE; and diagnostic dual relation-head losses when applicable), validation selected-task accuracy, all four quadrants, agreement/disagreement performance, sender probe overall/per `b_A`, zero-message overall/disagreement, finite loss/gradient/parameter status, group gradient norms, learning rate, and epoch duration. Run the more expensive forced-message counterfactual and corrected trajectory-sensitivity metrics at final epoch; if practical, retain gate diagnostics at the predeclared checkpoint epochs. Do not let measurement calls mutate the train RNG or train-loader order.

Gradient groups should at least include `core_A`, `core_B`, `msg_encoder`, `msg_embedder`, `learned_comm_token`, and `parity_head` when present; omit nonexistent groups in the dual model rather than creating dummy parameters. Empty conditioning groups are errors.

### E4. Validate code path before full run

After implementing, run:

```bash
python -m compileall -q model data test_system.py test_phase2b.py test_phase2c.py train_swarm_conditioned.py evaluate_phase2c_test.py
python -m pytest -q test_system.py test_phase2b.py test_phase2c.py
```

Run the primary/full and primary/final-only smoke tests and dual diagnostic smoke, with no test data. Confirm:

- finite losses and gradients;
- no expected trainable parameter has `grad is None` for its declared loss path (if the current implementation intentionally has a parameter without a gradient, document it and fix the objective or test before proceeding; do not weaken the assertion by default);
- primary/full terminal-node sensitivity is positive at an intermediate step;
- primary/final-only graph-node states are bit-identical through step 7;
- dual candidate logits and Core B trajectory are independent of message/COMM mode until final selection;
- `lambda_msg=2.0` appears in logs, metadata and computed loss;
- training/smoke code does not generate test seed `99991`.

The Phase 2c test seed must remain absent from the generated-data path at this point. Smoke test output must be labelled engineering-only.

Commit and push the code, tests and recipe addendum before full training. Save this exact commit hash as the source commit for all 15 runs. Then stop if any test or smoke check fails. Do not change the research design without amending and recommitting the predeclared document before training.

## 7. Stage F — Run the predeclared validation-only Phase 2c experiment

### F1. Confirm prerequisites

Before the first run:

- assert clean working tree and record source commit;
- verify recipe addendum is committed and pushed;
- verify Phase 1 checkpoint checksum exactly;
- verify all tests and smoke runs passed at this source commit;
- create one experiment run root, record config and start time;
- check free RAM/disk; do not delete existing runs to make space;
- use CPU-only torch and `torch.set_num_threads(4)`.

Budget: use a documented cap of 120 minutes for the full 15-run grid. If approaching the cap, finish the current run safely. If the matrix is incomplete, report it incomplete and do not apply the stable-success rule as if it were complete. No test split is generated if the run grid is incomplete.

### F2. Execution matrix

Primary paired experiment:

| Model | Mode | Seeds | Runs |
|---|---|---|---:|
| `primary` / `LatentConditionedSwarm` | `full` | 42, 43, 44, 45, 46 | 5 |
| `primary` / `LatentConditionedSwarm` | `final-only` | 42, 43, 44, 45, 46 | 5 |

Selector-independent diagnostic:

| Model | Mode | Seeds | Runs |
|---|---|---|---:|
| `dual` / `LatentDualRelationSwarm` | one documented nominal mode; the architecture must not use it | 42, 43, 44, 45, 46 | 5 |

Total expected: 15 completed runs. If a code defect appears, stop, preserve outputs, fix on a new commit, rerun engineering tests and decide whether the run grid must be restarted under a new ID. Do not mix models trained from different source commits in a single confirmatory set unless the difference is a documented, predeclared harmless logging fix and weight/artifact compatibility is verified; otherwise mark the grid invalid and predeclare a new run.

### F3. No test data in this stage

All runs may construct train and validation only (seeds 4201 / 13371). The trainer must never generate test seed 99991, import the generator for it into a training path, or invoke `evaluate_phase2c_test.py`.

Run all seeds and both primary modes regardless of intermediate validation outcomes. Do not stop after an early failure in a way that creates a selectively completed seed set, unless time budget or an actual implementation/runtime problem requires stopping. Never retry an individual seed with the same ID or change its recipe silently. Any retry is a new run and must be declared; the original remains in the aggregate table.

### F4. Validation-only acceptance and stable-success decision

Apply the exact gates in D4 per primary run. Save each result with exact failed criteria (not only `PASS/FAIL`). For the dual diagnostic, report its relation-head and selected-task measures descriptively; do not fabricate new acceptance thresholds.

A paired seed counts as qualifying only if BOTH primary/full and primary/final-only for the same seed pass all predeclared task/control/mechanism gates. Stable success requires at least 4 of 5 paired seeds qualify. Report:

- full-mode pass count / 5;
- final-only pass count / 5;
- paired seeds where both modes pass / 5;
- every metric and every failed gate for all 10 primary runs;
- all diagnostic dual results for all 5 seeds;
- per-epoch training/validation curves and gradient/loss events;
- paired primary initial-state checksums and train-loader order checksums;
- whether the candidate recipe was stable on Phase 2c.

If fewer than four paired seeds qualify, label the Phase 2c study `PARTIAL/UNSTABLE` and **STOP**. Do not generate the test split, do not invoke the evaluator and do not claim the test comparison failed. Preserve artifacts and recommend the next optimisation study separately.

## 8. Stage G — Freeze and run held-out test evaluation only if stable-success condition passes

Only if (and after) at least four of five paired seeds qualify, finish all validation analysis, freeze configuration/model inventory/source commit/checkpoints, and create a manifest for a single held-out evaluation event.

### G1. Pre-evaluation freeze

- Commit and push all code/report changes before creating the frozen manifest.
- Do not change architecture, recipe, gates or checkpoint selection after freeze.
- Save for each model entry: model kind (`primary` or `dual`), mode, seed, checkpoint path, checkpoint SHA-256, Phase 1 checkpoint path and hash, source commit, full config including `lr=0.0002` and `lambda_msg=2.0`, epochs, batch size, optimizer and split seeds.
- Ensure the primary manifest has all 10 predeclared primary model checkpoints (full and final-only for each of seeds 42–46), including runs that did not pass validation. Including all predeclared pairs avoids choosing which paired differences to report from test outcomes. Include all five predeclared dual diagnostic checkpoints as supplementary entries if all five completed successfully. If any checkpoint required by the frozen manifest is missing or mismatched, do not generate test data; fix the manifest/artifact issue before invoking the evaluator.
- Use the evaluator's exact schema and required fields. Validate the manifest in a non-data-generating dry validation step/test if available. Mark it frozen only after every validation decision is complete.

### G2. Invoke the test evaluator exactly once

Run only:

```bash
python evaluate_phase2c_test.py --manifest <FROZEN_MANIFEST.json> --out-dir <IGNORED_TEST_EVAL_DIR>
```

Use the actual manifest path and a new unique ignored output directory. Do not call the evaluator earlier, do not rerun it for alternative manifests, and do not regenerate seed 99991 independently. The evaluator must verify checkpoint hashes and Phase 1 checkpoint hashes before generating any test data. Test evaluation makes no gates/selection decisions; all acceptance decisions are validation-only.

If the evaluator fails before generating data due manifest/checksum validation, diagnose and correct the manifest and verify the split really was not generated. If any test data has been generated, record that fact precisely. Never claim the split remains untouched after successful evaluator invocation.

### G3. Paired test analysis

Create or use a separate deterministic analysis script that consumes the saved per-example prediction files and test metrics; it must not rerun models or regenerate data. Verify that all primary modes used identical test examples/order by exact `Y`, `r0`, `r1`, `b_A` comparison for each seed pair.

Report for every seed:

- full and final-only overall accuracy;
- full and final-only disagreement-subset accuracy;
- paired difference `Delta = full - final-only`, in percentage points, overall and on disagreement examples;
- per-example paired correctness differences on the same examples;
- all four quadrants, agreement/disagreement, sender probe, zero-message and counterfactual metrics;
- confidence interval for paired test advantage on disagreement examples, using the predeclared paired bootstrap method in `reports/phase2c-design.md` (or the committed addendum if it clarifies implementation before training);
- all five raw per-seed differences, their mean/median/range and variability. Do not hide negative seeds or report only the best model.

Do not treat thousands of test examples as independent training-seed replications. The paired bootstrap is conditional on the evaluated models; with only five paired training seeds, describe training-seed uncertainty conservatively. If the exact bootstrap implementation is ambiguous, resolve and commit the method in the pre-training addendum before training rather than choosing after seeing test outcomes.

Apply the predeclared interpretation rule:

- **Evidence favouring early integration:** stable task success as defined above, AND mean paired test advantage of at least +5 percentage points for full mode on the disagreement subset, AND the predeclared paired bootstrap interval excludes zero, with the paired seed results supporting the direction. Phrase this conservatively with n=5.
- **No measured advantage:** both modes solve the task but the paired difference is small or uncertain; conclude that the task can be solved by parallel reasoning/final selection, not that COMM integration is useless generally.
- **Final-only wins:** report the result honestly; it does not negate the independently measured trajectory-integration mechanism.
- **Instability/gate failures:** report partial/unstable, not scientific rejection. If stable-success condition did not pass, no test was generated and there is no confirmatory test result.

## 9. Required regression tests

Keep the existing suites unchanged unless a test needs a targeted correction supported by evidence; do not weaken any expectation. Maintain coverage for:

1. `test_system.py` — v7.1 model/data invariants and local baseline.
2. `test_phase2b.py` — COMM mask and trajectory behavior.
3. `test_phase2c.py` — raw-hash uniqueness, generator determinism, relation/label correctness, selector-independent dual candidate/logit and trajectory semantics, warm-start mapping, corrected batch-weighted trajectory metrics and batch-size invariance, test evaluator safety.
4. New recipe tests — `lambda_msg` loss wiring, exact candidate configuration guard, metadata includes `lambda_msg`, initial-state and loader-order pair identity, seed/mode run inventory, and training path cannot create test seed 99991.
5. Test evaluator tests — missing/unfrozen/ambiguous/wrong-seed manifest rejected before data generation; checkpoint and Phase 1 checksum mismatch rejected before data generation; end-to-end tests use only mocked split fixtures and never call the real seed-99991 generator.
6. Dual control tests — no COMM/message embedding in Core B; candidate relation logits invariant under selector reversal/forced-message changes; inference labels never used for selection.
7. Nonfinite/failed-run checkpoint tests or a small test of the runner's failure handler, ensuring that failures are recorded and `last.pt` saved wherever the process remains alive enough to handle the failure.

Run the full test suite before training and record the exact count, duration, command and log path. Do not claim a test passed unless its exit code was zero.

## 10. Git, branches, commits and push protocol

Use additive commits and keep the historical branch layout.

1. Push the Phase 2b report erratum commit only to `experiment/phase2b-msg-supervision-weight-study`.
2. Base the Phase 2c candidate branch on the latest verified fix-only Phase 2c branch; do not move or rewrite `main`, `experiment/phase2b-stability-study`, `experiment/phase2b-msg-supervision-weight-study`, or the original Phase 2c engineering branch.
3. Commit and push the Phase 2c recipe amendment **before training**.
4. Commit/push code and tests after engineering verification and before training. That commit hash is the one used for the confirmatory run.
5. Run artifacts under `runs/` remain ignored; verify with `git status --short` and `git check-ignore` that checkpoints/logs/environment dumps are not staged.
6. Before every commit: `git diff --check`; inspect `git diff --stat` and full diff; scan staged changes for secrets and large binary artifacts. Never include checkpoints, `.venv`, per-example prediction files, raw logs or tokens in Git.
7. After the run, commit reports, addenda, result summaries and scripts only. Push the research branch. Do not merge to `main` automatically.
8. Verify remote tracking and final status with `git status --short --branch`, `git log -5 --oneline --decorate`, `git rev-parse HEAD`, and `git ls-remote origin <branch-ref>`. Report exact pushed commit hashes.

## 11. Reports and artifacts to produce

### Required documentation

- `reports/phase2b-msg-weight-study-erratum-2026-10-17.md`: corrected arithmetic and narrative with raw-data provenance.
- `reports/phase2c-recipe-transfer-addendum-2026-10-17.md`: committed before any training; exact recipe, justification, operational paired-gate definition, full gates, test-statistics method and test-access stop rule.
- `reports/phase2c-candidate-recipe-run-report.md`: all 15 run outcomes, all validation gate states, loss/gradient/trajectory diagnostics, paired result, stability classification and honest limitations.
- If test evaluation is authorised by stable validation gates: `reports/phase2c-test-evaluation-report.md` with manifest hash, split seed, checkpoint hashes, paired results/CI and conservative interpretation.
- Update `CHANGELOG.md` and `WORKLOG.md` with concise dated entries and links; never erase prior entries.

### Required ignored artifacts

- one root run config and environment snapshot;
- complete logs;
- per-epoch JSONL and run summary JSON for every run;
- initial-state checksums and data-order verification;
- `last.pt` and epoch checkpoints for every run;
- checkpoint hashes;
- validation aggregate table and machine-generated gate selection/status JSON;
- if authorised, the frozen test manifest, saved per-example predictions, evaluator JSON/CSV output and checksums.

Copy critical logs, reports, checkpoints and checksums to durable storage appropriate for the VM. Never falsely claim an ignored artifact is backed up remotely if it exists only on ephemeral storage.

## 12. Final report checklist

The final response/report must state:

- exact local and remote branch names, source and final commit hashes;
- Phase 2b erratum commit and corrected values;
- verification that the candidate recipe addendum was committed before Phase 2c training;
- exact environment (OS, Python/PyTorch/NumPy, CPU threads, Phase 1 checkpoint hash);
- all tests, commands, exit codes, durations and artifact paths;
- for all 15 runs: seed, model, mode, `lr`, `lambda_msg`, initial checksum, gate status, failed criteria, final checkpoint hash and stability diagnostics;
- pass counts per mode and the exact qualifying paired-seed count;
- whether the test seed was generated; if yes, name the single manifest and evaluator invocation; if no, state that it remains untouched;
- all paired test differences and uncertainty only if authorised test evaluation occurred;
- any deviations, failures, nonfinite states, interrupted runs, retries or incomplete cells;
- conclusion classified as one of: `ENGINEERING BLOCKER`, `VALIDATION PARTIAL/UNSTABLE — TEST UNTOUCHED`, `VALIDATION STABLE — TEST EVALUATION PENDING`, `TEST EVALUATED — NO MEASURED ADVANTAGE`, or `TEST EVALUATED — EVIDENCE FAVOURING EARLY INTEGRATION`.

Do not call the result “scientific success” merely because code runs or aggregate accuracy is high. The central hypothesis concerns a reproducible advantage from access to the message during recurrence, particularly on disagreement examples; a trajectory change alone demonstrates mechanism, not usefulness.

## 13. Immediate execution order (strict)

1. Inspect and reconcile repository / working tree / remote branches.
2. Verify artifacts and environment.
3. Recompute Phase 2b weight-study counts from raw data; write and push dated report erratum.
4. Create a dedicated Phase 2c candidate branch from the verified current Phase 2c fix branch.
5. Read current charter/design and confirm no conflicting stop-gate or metric definition was overlooked.
6. Write, review, commit and push the recipe-transfer addendum. This must predate all Phase 2c training.
7. Wire `lambda_msg=2.0`, `lr=2e-4`, guarded confirmatory configuration and logging/checkpoint requirements.
8. Add regression tests; run `compileall` and all three existing test modules.
9. Run engineering smoke checks only; make sure test data remains ungenerated.
10. Commit and push the verified candidate code/tests. Freeze source commit and config.
11. Run the ten primary full/final-only model runs and five dual diagnostic runs; train/validation only; save every result even for failed runs.
12. Apply validation gates and the exact paired-seed rule. If fewer than four paired seeds qualify or the grid is incomplete, STOP; do not generate test data.
13. Only after stable validation success, freeze manifest/checksums and invoke the test evaluator exactly once for all predeclared models.
14. Perform paired test analysis from the stored predictions; write honest report and publish artifacts/scripts/reports without committing run binaries.
15. Verify the final pushed branch and produce the final report.

**Begin with Stage A now. Do not start training until the Phase 2b report erratum has been reconciled, the Phase 2c recipe addendum has been committed, all tests/smoke checks pass, and the candidate configuration guard is verified.**

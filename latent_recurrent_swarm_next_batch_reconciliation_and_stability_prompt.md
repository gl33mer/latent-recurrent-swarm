# Agent Task: Repository Reconciliation, Phase 2c Fix Verification, and a Predeclared Phase 2b Message-Supervision Stability Study

**Audience:** autonomous coding/research agent (Pi or equivalent) running in `/workspace/latent-recurrent-swarm`  
**Owner:** `gl33mer`  
**Repository:** `https://github.com/gl33mer/latent-recurrent-swarm`  
**Primary objective:** reconcile the live Git state and verified Phase 2c fixes, preserve the research record, then run one narrowly scoped, preregistered optimisation study against the frozen Phase 2b model to investigate the recurring sender-message probe failures.  
**Explicitly out of scope:** no confirmatory Phase 2c training, no Phase 2c test split generation, no test evaluation, no changes to the primary Phase 2c architecture, and no attempt to bypass the stability stop-gate.

---

## 0. Mission and decision rules

This is a multi-stage engineering and recipe-screening task. Do every stage in order and record evidence. Do not silently skip a stage because the repository looks familiar or prior conversation context appears to contain its result.

The Phase 2c fix-only report dated 2026-10-17 says that the following changes were committed:

- `45fce1b`: raw graph-hash uniqueness and selector-independent dual-relation diagnostic;
- `9257ad3`: train/test separation and corrected trajectory aggregation/terminal-node sensitivity gate;
- a following documentation commit with the fix-only report, metric erratum, and housekeeping updates.

The report says the combined test suite passed 47 tests and that only engineering smoke checks were performed. The Phase 2b stability study says no shared learning rate was eligible, so Phase 2c confirmatory training is blocked until a separately declared recipe study establishes eligibility.

**Important preflight discrepancy to resolve:** public source views fetched by an external reviewer returned mixed versions. The branch-name raw URLs for `data/conditioned_graph_task.py` and `model/swarm_conditioned.py` appeared older than the fixes, while the explicit commit URLs for `45fce1b` and `9257ad3` contained the corrected code; the branch-name trainer view showed some later fixes. This may be stale public-view caching or a Git reference/push inconsistency. Do not guess which. Establish the actual state using Git and GitHub CLI from the VM before doing any further work.

### Non-negotiable constraints

1. **No destructive Git operations.** Never use `git reset --hard`, `git clean -fdx`, force push, branch deletion, history rewriting, or overwrite existing run directories. Never discard uncommitted work. Preserve any pre-existing files, including editor swap files.
2. **Do not change `main`, `v7.1-pre-run`, the recorded Phase 2b runs, or old Phase 2c run artefacts.** Do not alter repository visibility or GitHub permissions.
3. **No test leakage.** Do not generate the Phase 2c test split (seed `99991`) in this task. Do not invoke `evaluate_phase2c_test.py`. The only tests to use in this task are system/unit tests plus train/validation-only Phase 2b stability runs.
4. **Never relax a gate, replace a failed result, select only successful seeds, or describe a smoke run as a research result.** Continue recording all planned runs, including failures.
5. **The new study changes exactly one optimisation variable: the weight on the existing sender-message BCE loss.** Do not change architecture, graph task, model dimensions, optimiser family, base learning rate, weight decay, batch size, epochs, other loss weights, data seeds, or acceptance criteria.
6. **Predeclare and commit the new study design before running any sweep run.** Code/test development is allowed before training; no sweep outcomes may be inspected before the design commit exists.
7. **Do not begin confirmatory Phase 2c training even if this new study finds an eligible setting.** Finish with a report and a recommended next decision. The owner/reviewer will inspect the study result before any Phase 2c confirmation/test manifest is frozen.
8. If the live repository state cannot be reconciled safely, stop after collecting evidence and write a blocker report. Do not start the study on an uncertain or mixed codebase.

## 1. Stage A — establish repository and artifact truth

Work from `/workspace/latent-recurrent-swarm` unless the actual environment says otherwise. Start by recording the exact path, user, OS, Python environment, and Git state. Do not run training commands yet.

Execute and capture the output of commands equivalent to:

```bash
pwd
id
uname -a
cat /etc/os-release || true
git status --short --branch
git rev-parse --show-toplevel
git rev-parse HEAD
git log --all --graph --decorate --oneline -40
git branch -avv
git tag --list
```

If any working-tree changes exist, inspect them and preserve them. Do not stash, reset or overwrite them automatically. Copy the raw status and relevant diffs into a new ignored run/evidence directory, or stop if their ownership cannot be determined.

Then query the remote state:

```bash
git remote -v
git fetch --all --tags --prune
git ls-remote --heads origin
git ls-remote --tags origin
```

Use `gh api` as an additional check if useful. For branch names containing slashes, quote/URL-encode the branch name as required. Record the exact SHAs for:

- `origin/main`;
- `origin/experiment/phase2c-message-conditioned-reachability`;
- `origin/experiment/phase2b-stability-study`, if present;
- the known commits `fe87e1f`, `45fce1b`, and `9257ad3`, if locally available;
- the Phase 2c fix-only documentation commit reported by the repository history.

Check commit ancestry with `git merge-base --is-ancestor` and identify which remote refs contain the fix commits. Do not infer branch tips from a browser cache or from report prose alone.

### Required source-level reconciliation

Compare the actual current branch and the explicit fix-commit contents for:

- `data/conditioned_graph_task.py` — must call a canonical raw structural graph hash with no role prefix, enforce uniqueness across A/R0/R1 within and across splits, have bounded sampling failure, and return exactly `3 * num_samples` raw hashes;
- `model/swarm_conditioned.py` — `LatentConditionedSwarm` must keep COMM integration; `LatentDualRelationSwarm` must compute Core B without any COMM register or message embedding, produce both relation candidates independently of the selector, and select only after the candidates exist;
- `train_swarm_conditioned.py` — normal and smoke training paths must build train/validation only, never construct the Phase 2c test split, calculate corrected trajectory metric denominators, and gate on terminal-node sensitivity in full mode;
- `evaluate_phase2c_test.py` — separate explicit manifest-driven evaluator; verify checkpoints/manifests before creating test data; it must not have been invoked during this task;
- `test_phase2c.py` — raw hash uniqueness, selector-independent candidate logits, weighted trajectory aggregation/batch-size invariance, terminal-node sensitivity, and evaluator protection tests must be present.

Use Git-object truth in preference to fetched web snippets:

```bash
git show <commit>:data/conditioned_graph_task.py
git show <commit>:model/swarm_conditioned.py
git show <commit>:train_swarm_conditioned.py
```

Inspect the branch version with:

```bash
git show origin/experiment/phase2c-message-conditioned-reachability:data/conditioned_graph_task.py
git show origin/experiment/phase2c-message-conditioned-reachability:model/swarm_conditioned.py
```

Adjust command syntax if the branch does not exist. Capture enough output or checksums to demonstrate which version is on which ref.

### Reconciliation decision

- If the remote Phase 2c branch already points at a commit containing all fixes and documentation, record that finding and proceed.
- If the local branch contains the fixes but the remote branch is a strict ancestor, run tests first, then a normal fast-forward push is allowed. Never force-push.
- If the fixes exist on another remote branch but the target branch does not contain them, find the ancestry and determine a safe fast-forward or clean cherry-pick plan. Do not cherry-pick blindly or introduce unrelated changes.
- If the local history diverges, a fix commit is missing, the documentation commit cannot be identified, or normal push would not be a fast-forward, **stop**. Produce `reports/repository-reconciliation-blocker.md` with exact SHAs, `git status`, ancestry facts and the least-risk proposed remedy. Do not run the optimisation study until the owner resolves it.

### Artifact integrity

Check that the following known artefacts exist wherever the run environment says they should and verify SHA-256 when files are available:

- Phase 1 checkpoint: `ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e`;
- Phase 2b full seed 43 checkpoint: `38c505a5f9ae938fffa697bdad195b6d92241223b90dbced394919a6db203612`;
- Phase 2b final-only seed 42 checkpoint: `79aadaf7880c369cfff550db9b7624d5c9a4f97e2b13a35af2811eed7c24d45e`.

If any artefact is missing, document it; do not fabricate, download an unverified substitute, or claim that it was verified. If any run logs or checkpoints exist only on the VM and not in durable storage, record their locations and archive status. Do not commit checkpoint binaries or bulk run logs to Git.

## 2. Stage B — verify the Phase 2c fixes only

Proceed only after Stage A is reconciled.

1. Create a fresh ignored run directory under `runs/` using a UTC timestamp and a clear name such as `phase2c_fix_revalidation`.
2. Record current commit, branch, `git status`, Python version, `pip freeze`, OS, CPU count, RAM, disk, PyTorch version/device, NumPy version and thread settings.
3. Verify that the environment is the previously used CPU environment when possible: Python 3.11.x, PyTorch CPU build, CUDA unavailable, four intra-op threads. Do not upgrade dependencies gratuitously. If the environment differs, record the difference.
4. Run the complete baseline and Phase 2c test suites:

```bash
python -m compileall -q model data test_system.py test_phase2b.py test_phase2c.py train_swarm_conditioned.py evaluate_phase2c_test.py
python -m pytest -q test_system.py test_phase2b.py test_phase2c.py
```

The fix-only report recorded **47 passed**. Test-count changes may be legitimate if the source has changed, but explain any discrepancy and do not merely assert that 47 is expected without executing the tests.

5. Run only engineering smoke checks for the primary model in `full` and `final-only` modes and the corrected dual diagnostic in both accepted interface modes. Use tiny fixtures and explicitly labelled smoke mode. Record finite losses, gradient coverage, mask validity, trace behaviour and output shapes. The primary/full mode should exhibit positive terminal-node sensitivity at one or more intermediate steps; primary/final-only graph-node traces must be bit-identical through step 7; the dual diagnostic's candidate outputs/traces must be selector-independent.
6. Confirm that no test split was generated. Do **not** invoke the test evaluator or call the data generator with `seed=99991`. Do not run ordinary Phase 2c training.
7. Save stdout/stderr, exit codes and environment snapshots under the ignored run directory. Do not commit raw logs/checkpoints.

If the tests expose a new defect, fix it only if it is a small, unambiguous regression within the predeclared invariants. Add a focused test, commit the fix separately, rerun the full suites and document it. If the required behaviour or scientific specification would change, stop and report rather than silently redesigning it.

## 3. Stage C — audit and correct the historical stability record

Before proposing the new optimisation study, inspect the actual files rather than relying only on this prompt:

- `reports/phase2b-stability-design.md`;
- `reports/phase2b-stability-report.md`;
- all per-run `metrics.jsonl` / run summary JSON files under the recorded stability run directory;
- the sweep runner and its Git commit;
- the Phase 2b model/training source at the frozen stability commit;
- `WORKLOG.md`, `CHANGELOG.md`, and relevant Phase 2b/2c reports.

### C1. Recompute the reported instability summaries

The reported sweep table lists the following per-run `Instab%` values, grouped by learning rate across both modes and three seeds:

- `8e-4`: 18.1, 8.8, 14.5, 10.7, 9.5, 15.5;
- `4e-4`: 10.8, 7.1, 7.9, 21.6, 13.2, 18.1;
- `2e-4`: 4.7, 2.3, 6.6, 6.5, 4.6, 3.2.

The prose summary previously called **9.5%, 7.9%, and 3.2%** the “median instability” values. Those are not the medians of the six per-run values above. Recompute the metric from the actual epoch histories and apply the predeclared definition exactly: compute each run's mean absolute epoch-to-epoch validation connected-accuracy change, then report the median across all six mode/seed runs at each learning rate. Based only on the rounded values shown above, the approximate medians are:

- `8e-4`: **12.6%**;
- `4e-4`: **12.0%**;
- `2e-4`: **4.65%**.

Recompute from raw histories at full precision; do not hard-code the approximate values if raw logs differ. Add a dated correction/addendum to the stability report and CHANGELOG if the report is indeed wrong. Do not rewrite historical results or selection decisions. No learning rate was eligible, so the correction does not change the predeclared no-eligible-rate decision; it does correct the secondary comparison.

### C2. Record the failed-run checkpoint deviation accurately

The predeclared stability design said `last.pt` should be saved for **every** run including gate-failing runs, but the run report's artifact table says failing runs had no checkpoint because artifact save was post-gate. Inspect the actual runner and directories. If failed checkpoints are unavailable, explicitly record this as a protocol deviation and do not try to recreate them under the old run IDs. The next study must save `last.pt` for every run, including gate failures, plus epoch checkpoints at 5, 10, 15 and 20. A failed run is data, not a reason to erase its state.

### C3. Inspect sender-probe weakness before selecting the next intervention

From the raw metrics, compare:

- message BCE loss and Core A auxiliary reachability loss by epoch;
- probe accuracy overall and by `b_A`;
- gradient norms for `core_A`, `msg_encoder` and other groups;
- parity/auxiliary loss at each abrupt validation collapse;
- full vs final-only per-seed trajectories;
- whether any apparent sender-probe failure comes from sign convention, labels, metric aggregation, or optimisation rather than model representation.

Write a concise diagnosis with evidence and uncertainties. Do not claim a root cause without supporting logs. If Stage C discovers a correctness bug in the prior runner/data/metrics, stop and repair/test it before running a new study.

## 4. Stage D — new predeclared one-factor message-supervision-weight study

Only proceed if Stage A–C have no unresolved correctness blocker.

### D1. Scientific/engineering purpose

The previous fixed-learning-rate sweep found no shared eligible rate, and sender-probe accuracy was among the most frequent failures. The next smallest useful intervention is to examine whether the weight on the **existing BCE sender supervision** is too weak or destabilising the joint objective.

This remains a **recipe-screening study**, not a confirmatory Phase 2c scientific result. It uses only the original Phase 2b XOR benchmark and existing Phase 2b COMM model. It must not change Phase 2c files or generate Phase 2c data.

### D2. Create the branch safely

Use the exact, verified stability-study state as the base, not the current Phase 2c working tree. Inspect `git log` and remote ancestry first. Expected candidate base is the recorded Phase 2b stability state around `ed46cd4` / `experiment/phase2b-stability-study`; use the actual commit that contains the frozen Phase 2b model, the recorded stability study and its report. Do not use a similarly named or stale ref merely because it exists.

Create a new branch such as:

`experiment/phase2b-msg-supervision-weight-study`

Do not move or rewrite any existing branch or tag. If no clean Phase 2b base exists or the code cannot be identified unambiguously, stop and report the blocker.

### D3. Lock the design before the sweep

Create `reports/phase2b-msg-weight-study-design.md` describing this exact grid and commit it **before any sweep training**.

Fixed settings:

- Original Phase 2b XOR dataset and frozen `LatentCommSwarm` architecture (284,107 parameters, unless the actual frozen code reports otherwise; independently verify and record).
- Phase 1 warm-start checkpoint with verified SHA-256 `ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e`.
- Learning rate fixed at **`2e-4`**. This is the strongest candidate from the first sweep in aggregate; it was still ineligible, so this study does not treat it as approved.
- AdamW, weight decay `1e-2`, batch size 64, 20 epochs, float32 CPU, four intra-op threads.
- Loss weights remain `1.0 * parity + 0.5 * aux_A + 0.5 * aux_B + lambda_msg * BCEWithLogits(raw_m, b_A)`.
- Only new variable: `lambda_msg ∈ {0.5, 1.0, 2.0}`. `0.5` is the frozen baseline; `1.0` and `2.0` are the two predeclared alternatives.
- Modes: `full` and `final-only`.
- Model seeds: `42, 43, 44`.
- Total: **18 runs** (`3 message weights × 2 modes × 3 seeds`).
- Train and validation data only, using the frozen Phase 2b data seeds/sizes from its specification (normally train seed 42 / 4,000 and validation seed 1337 / 1,000). No test split should be generated in this sweep.
- For a given model seed and lambda, full/final-only must start from identical initial trainable parameters and use identical batch order. Across lambda values, reset seeds so they also begin with the same initial state for a given model seed. Use explicit, independent `torch.Generator` seeds for the DataLoaders and document them.
- Do not use early stopping, best-checkpoint selection, gradient clipping, LR schedules, warm-up, or any other optimisation change in this first one-factor study.

The exact Phase 2b gates must be read from the frozen design/spec and encoded in one shared gate function; do not invent new thresholds. Based on the latest Phase 2b report, they include overall connected accuracy, all four label quadrants, stratified message-flip residuals, zero-message accuracy, sender-probe overall and by `b_A`, and the mechanism-specific full/final-only checks. Verify exact numerical thresholds against the frozen source and copy them verbatim into the design report and implementation. Empty conditioning groups are hard failures.

### D4. Instrumentation requirements

For every run, create a unique ignored run directory and log:

- source commit, working tree state, lambda, mode, seed, Phase 1 checkpoint hash, environment and data configuration;
- per-epoch mean training total loss, parity loss, auxiliary A loss, auxiliary B loss, message BCE, validation overall connected accuracy and all quadrant accuracies;
- sender probe overall and by `b_A`;
- zero-message accuracy;
- flip residual per `b_B` group at epochs 5/10/15/20 and final epoch;
- gradient norms by parameter group: `core_A`, `core_B`, `msg_encoder`, `msg_embedder`, `learned_comm_token`, and the parity/readout head where present;
- gradient/loss NaN/Inf counts and parameter finiteness each epoch;
- mean absolute epoch-to-epoch change of validation accuracy, count of >15-point drops, and enough raw histories to recompute them;
- corrected trajectory sensitivity with proper sample/element weighting; terminal-node sensitivity for full mode; exact final-only graph-state invariance through step 7;
- final metrics, every failed acceptance gate, and exit classification.

**Checkpoint policy is mandatory:** save `last.pt` for every run, even if it fails a gate, becomes non-finite, or is stopped early because of a numerical failure. Also save `epoch05.pt`, `epoch10.pt`, `epoch15.pt` and `epoch20.pt` when those epochs complete. Checkpoints stay in ignored `runs/` and must never be committed. If a run becomes non-finite, preserve the last checkpoint/state, record the failure and move to the next planned cell; do not call it a gate pass.

### D5. Pairing and ordering

Use the same deterministic sequence for paired runs:

1. set Python and PyTorch seed;
2. construct/load the Phase 1 warm start and model;
3. construct train/validation loaders with explicit fixed data and batch-order seeds;
4. train for the fixed 20 epochs;
5. evaluate only on validation and save artefacts.

Record the exact initial trainable-parameter checksum (or deterministic tensor hash) so the paired full/final-only and across-lambda comparisons can be verified to start identically for each seed, except for their subsequent training dynamics. Do not share optimizer state across runs.

### D6. Budget and failure handling

Budget: **up to 120 minutes** for all 18 runs. Report progress in the run log. If the budget is approached, finish the current epoch/run safely where possible, preserve all artefacts, classify the grid as incomplete and do not apply the full-grid selection rule as though the grid had completed. A partial grid may inform a follow-up design but cannot declare an eligible lambda.

## 5. Predeclared selection rule for the new study

A `lambda_msg` value is eligible for a future Phase 2c recipe only if all of the following hold:

1. At least **2 of 3 seeds pass all frozen Phase 2b validation gates in full mode**, and at least **2 of 3 seeds pass all gates in final-only mode**, at that same lambda.
2. No run at that lambda has non-finite losses or parameters or a broken gradient path. Any numerical failure makes that lambda ineligible.
3. Full mode shows positive terminal-node trajectory sensitivity at one or more intermediate steps, and final-only graph-node states are exactly bit-identical through step 7.
4. No clear unresolved pathology appears in loss or gradient histories. The evidence and judgement must be written out; don't use a vague assertion.

Among eligible lambdas, select by (a) highest total all-gate pass count across the six runs at that lambda, then (b) lowest median per-run validation instability (each run's mean absolute epoch-to-epoch validation-accuracy change; median across the six mode/seed runs). If still tied, prefer the lower lambda only when other validation evidence is comparable. The design is based on validation only.

If **no** lambda is eligible, conclude `PHASE 2B MESSAGE-SUPERVISION RECIPE NOT YET STABILISED`. Do not run Phase 2c confirmatory training. If a lambda is eligible, conclude that a candidate common recipe has passed this screening grid, but still stop for a written report and review; do not automatically run Phase 2c in this task.

## 6. Implementation constraints

- Leave the frozen Phase 2b core architecture and message channel unchanged. Change only the `lambda_msg` coefficient via a command-line/config parameter in the new sweep runner.
- Do not modify `model/swarm_conditioned.py`, `data/conditioned_graph_task.py`, `train_swarm_conditioned.py`, `evaluate_phase2c_test.py` or any Phase 2c task files as part of this sweep.
- Preserve the original frozen stability runner and scripts. Implement a new study runner, or add an isolated entry point, rather than overwriting code used to produce the previous report.
- Use one gate implementation shared by all study runs, with the threshold source documented.
- Include smoke/unit tests for: lambda parameter is threaded only into the message loss; other loss components are unchanged; deterministic pair initialisation/order; train/validation-only data construction; checkpoint-on-failure semantics; metrics and empty-subgroup failure behaviour.
- Do not add heavyweight dependencies. Reuse the existing CPU environment.

## 7. Test and smoke sequence before the 18-run grid

After committing the design and runner/tests, but before running any sweep cell:

1. Run the frozen Phase 2b tests and new study tests.
2. Run a single explicit 1–2-batch engineering smoke for `lambda_msg=0.5`, both `full` and `final-only`, using train/validation fixtures only. This smoke is not one of the 18 sweep runs and must have a separate run ID.
3. Verify losses finite, gradient path present, exact final-only invariance through step 7, positive full-mode intermediate terminal sensitivity, and checkpoint saving.
4. Verify the Phase 2c test seed `99991` is absent from all study code paths and not invoked.
5. Record the design commit, runner commit and smoke result. Then—and only then—begin the full 18-run grid.

## 8. Housekeeping and publication

- Use a new directory under `runs/`, e.g. `runs/<UTC>_msg_weight_stability/`. Each cell gets its own child directory. Never reuse a previous run directory.
- `runs/`, checkpoints, `.venv`, logs, metrics JSONL, model outputs and local caches must remain ignored/untracked.
- Save `git rev-parse HEAD`, `git status --short`, environment freeze, `uname -a`, `nproc`, `free -h`, `df -h`, Torch/NumPy versions and thread settings with the sweep.
- Do not place secrets, GitHub tokens, credentials or private environment data into reports.
- Commit the design file first, implementation and tests next, then commit the report/addendum after outcomes. Push the new study branch normally. No force pushes.
- Update `CHANGELOG.md` with the study branch and no source changes to Phase 2c. Do not rewrite the history of previous reports.
- Produce a concise `reports/phase2b-msg-weight-study-report.md` with: exact parent/source commit; design commit hash; runner/test commit hashes; environment; table of all 18 runs; every failed criterion; epoch-level instability summary; paired-seed comparison; correctly recomputed instability summaries; checkpoint/log paths and hashes; selection rule applied mechanically; final decision; limitations.
- Add a dated addendum/correction to `reports/phase2b-stability-report.md` (or a clearly linked follow-up if that report belongs to another protected branch) documenting any verified misreported median values and the discrepancy between the predeclared requirement to save failed-run checkpoints and the artifact table. Do not pretend that old missing checkpoints were recovered.
- Make sure the README / run index links to the new report without implying that Phase 2c is confirmed.

## 9. Git safety protocol

Before each commit:

```bash
git status --short
git diff --check
git diff --stat
git diff
```

Inspect staged changes with `git diff --cached` before committing. Confirm no checkpoint, log, environment dump, local dataset, `.venv`, credentials, SSH material or test predictions are staged. Prefer small commits with factual messages such as:

- `Add preregistered message-loss weight study design`
- `Add instrumented Phase 2b message-loss weight sweep`
- `Record Phase 2b stability follow-up results`

Push the new branch normally. If rejected because of divergence, do not force; diagnose ancestry and stop or use a safe merge/rebase only if it preserves all existing commits and the branch is explicitly understood. Never alter `main` as part of this task.

## 10. Final deliverables and stop condition

At completion, provide:

1. repository reconciliation outcome: local/remote branch SHAs, verified fix commits, and whether any ref repair was required;
2. baseline/fix test and smoke results;
3. correction note for any historical summary-statistic or failed-checkpoint-policy discrepancy;
4. the committed predeclared message-weight study design;
5. complete 18-run table or an explicit incomplete-grid report;
6. loss/gradient/probe diagnosis from the sweep;
7. selection-rule calculation with raw per-run values and exact arithmetic;
8. checkpoint and log locations plus SHA-256 values, including for gate-failing runs;
9. final status in exactly one of:
   - `BLOCKED: repository/fix state could not be reconciled`;
   - `BLOCKED: engineering tests failed`;
   - `INCOMPLETE: stability grid exceeded budget or was interrupted`;
   - `NOT YET STABILISED: no lambda eligible`;
   - `CANDIDATE RECIPE SCREENED IN: eligible lambda found; confirmatory Phase 2c still not run`.

**The task must finish without generating the Phase 2c test split, without invoking the Phase 2c evaluator, and without running confirmatory Phase 2c training.** If the stability study yields an eligible candidate, report it and stop for review. A subsequent task will freeze the selected recipe and authorize the Phase 2c comparison explicitly.

---

## Agent start instruction

Start at Stage A. Inspect and reconcile the live local/remote Git state before modifying files or running training. Keep all existing evidence. Do not assume branch-name web views, old conversation notes or this prompt are substitutes for checking the actual repository state. Follow every stage in order, commit the design before any new sweep outcome is viewed, and stop at the conditions stated above.

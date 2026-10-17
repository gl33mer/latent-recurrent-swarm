# Agent Task: Phase 2c Fix-Only Engineering Pass

**Project:** Latent Recurrent Swarm  
**Repository:** `https://github.com/gl33mer/latent-recurrent-swarm`  
**Working directory:** `/workspace/latent-recurrent-swarm`  
**Required branch:** `experiment/phase2c-message-conditioned-reachability`  
**Task type:** housekeeping, correctness fixes, regression tests, engineering-only smoke verification  
**Training status:** NO confirmatory training. NO hyperparameter sweep. NO evaluation of the predeclared Phase 2c test split.

---

## 0. Role, objective and non-negotiable rules

You are acting as an autonomous ML research engineer. Your task is to make a narrowly scoped, auditable fix-only update to the existing Phase 2c branch. You are not authorised to redesign the research project, alter predeclared acceptance thresholds, tune the model to obtain passing metrics, or run confirmatory Phase 2c training.

The previous engineering review identified four implementation/protocol issues that must be fixed:

1. Graph hashes are namespaced by role, so identical raw graphs can evade the current uniqueness checks.
2. The `LatentDualRelationSwarm` compute-both-then-select diagnostic still allows the selector message to influence Core B's recurrence; it is not currently an independent late-selection control.
3. The ordinary Phase 2c training path can generate/evaluate the held-out test split. Training and confirmatory test evaluation must be separated.
4. Trajectory-sensitivity aggregation is incorrectly weighted by batch size, and the acceptance check does not specifically require intermediate sensitivity at the terminal node.

The mission is to correct these issues, add regression tests and housekeeping documentation, run the existing tests plus small **engineering-only** smoke tests, commit and push the fixes to the existing Phase 2c branch, and stop.

### Absolute constraints

- Do not change the v7.1 baseline architecture or acceptance thresholds.
- Do not change the primary Phase 2c hypothesis, task labels, attention-mask definition, split seeds, optimiser, loss weights, or acceptance thresholds in this task.
- Do not modify the frozen Phase 2b stability sweep or rewrite its historical raw logs.
- Do not run the Phase 2c confirmatory multi-seed grid.
- Do not run training with `--allow-unstable-lr` or any equivalent bypass.
- Do not generate the predeclared Phase 2c test split using seed `99991`, and do not evaluate any confirmatory test data. The existing smoke run was engineering-only; preserve the test split as unused.
- Small engineering smoke checks are permitted only if clearly labelled as such, and they must not create the predeclared test split.
- Do not alter gates after seeing results. Do not present smoke accuracy as a scientific result.
- Do not fabricate corrected metric values. If an old trajectory metric was incorrectly aggregated, mark it invalid for quantitative interpretation unless it can be recomputed from preserved trace data.
- Do not commit `.venv`, checkpoints, model weights, run logs, JSONL metrics, environment snapshots containing secrets, API credentials, SSH keys, or any other sensitive material.
- Never print or commit secrets. Do not change repository visibility. Do not rewrite published history, force-push, reset, clean, or discard user files.
- Preserve all pre-existing work. If you encounter unexplained staged or unstaged changes, inspect and preserve them. Do not use `git reset --hard`, `git clean`, or destructive stash commands.
- If a required fix cannot be completed safely, stop at that blocker, document it precisely, and do not pretend the task is complete.

A successful result for this task means **engineering correctness plus passing tests and smoke checks only**. It does not mean Phase 2c task success or a scientific finding.

---

## 1. Context: current state of the project

The work to date has produced three related milestones.

### 1.1 v7.1 baseline

- Phase 1: one-step reachability accuracy was 50.0%; eight-step validation and held-out test accuracy were both reported as 100.0%.
- Phase 2a: decoupled recurrent reasoners used a single binary message at receiver-side decision fusion; reported validation accuracy was 97.5% and held-out test accuracy was 97.0%.
- The Phase 2a result does not demonstrate recurrent message integration.

### 1.2 Phase 2b COMM-token integration

- The COMM token is one-way/read-only: Core A finishes its computation and emits one bit; Core B can read it during recurrent graph processing, while the COMM token itself cannot read graph nodes.
- Trajectory tests show that the message can influence ordinary graph-node states in full mode, while final-only graph-node traces stay identical through step 7.
- A predeclared 18-run learning-rate sweep found no shared learning rate eligible under its selection rule. The recipe is classified **NOT YET STABILISED**.
- Do not reverse that decision in this task.

### 1.3 Phase 2c message-conditioned reachability

Phase 2c defines two independent directed relations, `R0` and `R1`, for Graph B. Core A's bit chooses which reachability answer is the target:

- `m = +1` when `b_A = 1`, otherwise `m = -1`.
- `Y = r0` when `b_A = 0`; `Y = r1` when `b_A = 1`.
- Graph B input is `[adj_union(16), adj_R0(16), adj_R1(16), one_hot(16)]`, a 64-dimensional input per node.
- The graph-to-graph mask uses the union graph, independent of `m`; graph queries can read the COMM token according to the visibility schedule; the COMM query can attend to itself only; no attention row is empty.
- `full` exposes the COMM token to graph queries at steps 1–8. `final-only` exposes it at step 8 only.
- Main primary readout is Core B's terminal graph-node state only, not the COMM token or a directly concatenated message.
- Phase 2c has been implemented and unit-tested and has passed engineering smoke checks, but **has not undergone confirmatory training**. The predeclared test split seed `99991` remains unused.

### 1.4 Relevant paths

Inspect the actual repository before editing. Expected files include:

- `model/swarm_conditioned.py`
- `data/conditioned_graph_task.py`
- `train_swarm_conditioned.py`
- `test_phase2c.py`
- `reports/phase2c-design.md`
- `reports/phase2c-run-report.md`
- `reports/phase2b-stability-design.md`
- `reports/phase2b-stability-report.md`
- `test_system.py`
- `test_phase2b.py`
- `reports/` and ignored `runs/`

The actual repository may have moved since the design was written. Inspect the live files and commit history; do not blindly recreate files from this prompt.

---

## 2. Phase A — Safe repository and artifact housekeeping

Complete this phase before making code changes.

### 2.1 Confirm location, branch and cleanliness

Run and record:

```bash
cd /workspace/latent-recurrent-swarm
pwd
git status --short --branch
git branch --show-current
git log --oneline --decorate -12
git remote -v
git rev-parse HEAD
git show -s --format=fuller HEAD
```

The required working branch is `experiment/phase2c-message-conditioned-reachability`. If you are on another branch, inspect the state and switch safely only if there are no uncommitted changes that would be disturbed. Do not create a parallel duplicate branch unless the required branch genuinely does not exist; if it does not exist, explain why before taking corrective action.

Fetch remote refs without rewriting local history:

```bash
git fetch origin
```

Compare the local branch with its remote counterpart. Preserve any commits that exist on either side. Do not force-push. If the branch has diverged, report the divergence and reconcile it with a normal merge/rebase only if safe and necessary; otherwise stop and document the blocker.

### 2.2 Preserve current experimental evidence

Before editing, record:

- Current commit SHA and branch.
- Git status (including any pre-existing modifications).
- The stability report/design commit identifiers.
- Phase 2b and Phase 2c implementation/design commit identifiers.
- Locations and SHA-256 hashes of any available passing Phase 2b checkpoints and Phase 1 warm-start checkpoint.
- Location of the stability sweep directory and Phase 2c smoke-run directory.

Do not move, delete, rename, regenerate or overwrite historical run directories. All `runs/` files are evidence, even if ignored by Git.

If artifacts are not available locally, record the exact absence. Do not claim that an artifact was preserved unless you verified its presence and checksum.

### 2.3 Verify ignore rules and staged files

Check that `.venv/`, `runs/`, checkpoints, generated weights, local metrics, caches, and secrets are excluded from Git. Inspect the current `.gitignore` before changing it. Add only genuinely missing ignore patterns, for example for any new local checkpoint names, without broad patterns that accidentally hide source or documentation.

Before every commit:

```bash
git diff --check
git status --short
git diff --cached --stat
git diff --cached --name-only
```

Review staged contents and confirm no credentials, tokens, keys, environment secrets, checkpoints, large run outputs, or unrelated files are staged.

### 2.4 Start a new run directory for this fix-only task

Use a UTC timestamp and a descriptive run name, e.g.:

```text
runs/<UTC_TIMESTAMP>_phase2c_fix_only/
```

This directory is ignored by Git. Place engineering test logs, smoke logs, environment details and a concise run metadata JSON there. Do not put test-split graphs or test predictions there because no test set should be generated in this task.

Record:

- UTC timestamp and run ID.
- Branch and source commit.
- Python and PyTorch versions.
- OS/CPU/thread count, RAM and disk snapshot.
- Exact commands executed and exit codes.
- Test results and smoke configuration.
- Any checkpoint hashes actually used.

Do not record credentials or environment variables that may contain secrets.

---

## 3. Fix 1 — True raw graph-hash uniqueness across roles and splits

### 3.1 Problem

The conditioned-graph generator currently uses role-namespaced keys such as `A:<sha256>`, `R0:<sha256>` and `R1:<sha256>` for uniqueness. This treats an identical adjacency matrix in two different roles as two different graphs. It can also allow the same raw adjacency to appear across splits under different role prefixes.

The promise in the design is **structural graph uniqueness**, not uniqueness of `(role, graph)` pairs.

### 3.2 Required implementation

In `data/conditioned_graph_task.py`:

1. Compute the raw structural SHA-256 hash of each adjacency matrix, using the canonical deterministic tensor byte representation already employed by the project. If useful, normalise the adjacency representation before hashing so equivalent binary adjacency matrices cannot differ merely because of tensor dtype or storage layout; document the canonicalisation.
2. Use the **raw hash without a role prefix** as the uniqueness key.
3. A raw hash must be rejected if it has already appeared:
   - earlier in the current dataset, under any role;
   - in any role in a supplied `existing_hashes` set from a prior split.
4. Keep the role (`A`, `R0`, `R1`) in separate metadata/logging fields if role lineage is useful. Never concatenate the role to the key used for structural uniqueness.
5. Preserve the data task, edge distribution, endpoint constraints, shortest-path constraints, sample balancing, split seeds `4201 / 13371 / 99991`, and deterministic generator contract. Do not modify label construction or task definitions in this fix.
6. Preserve or carefully update the public return contract. Search all callers before changing whether returned hash sets contain namespaced or raw hashes. Update every consumer consistently and document the API semantics.
7. Use a bounded failure or a clear explicit error if a future caller requests more unique samples than the sampler can practically produce. Do not add arbitrary changes to the present graph distribution to address an unobserved exhaustion issue.

### 3.3 Required tests

Extend `test_phase2c.py` (or the appropriate current test module) with tests that establish:

- Raw graph hashes are unique across A, R0 and R1 within a split.
- Returned uniqueness sets contain exactly `3 * num_samples` distinct raw graph hashes for a valid split.
- Raw graph hash sets are pairwise disjoint across train, validation and test-like *unit-test fixtures* made with small test seeds. These unit tests must not generate the predeclared Phase 2c test split seed `99991`.
- The same raw graph generated under another role is rejected by the generator. Prefer a deterministic focused test of the hash/uniqueness helper over relying on a statistically unlikely collision.
- Fixed-seed generation is deterministic: same seed and same existing-hash inputs produce identical tensors, labels and raw hash sets.
- Ground-truth reachability for `A`, `R0` and `R1` agrees with stored labels; target labels agree with the selector rule.
- The eight `(b_A, r0, r1)` combinations remain exactly balanced, and the disagreement subset remains exactly 50% for the specified sample counts.
- Reachable-relation path lengths satisfy the existing bound; unreachable relations have no path; endpoint-degree constraints still hold.

Keep existing tests. Do not weaken existing assertions to make the new tests pass.

---

## 4. Fix 2 — Make the dual-relation diagnostic a genuinely selector-independent control

### 4.1 Problem

The design defines `LatentDualRelationSwarm` as a compute-both-then-select diagnostic: it should compute the `R0` and `R1` answers without conditioning Core B on Core A's message, then use the one bit only to choose between those two candidate answers.

The current implementation appears to pass `m` and the COMM schedule into Core B before producing the two candidate logits. This means both candidates can be influenced by the selector, so the diagnostic is not a pure late-selector control.

### 4.2 Required semantics

Leave the primary `LatentConditionedSwarm` unchanged except for shared bug fixes needed by both models. Correct the diagnostic model so that:

1. Core A runs on `x_A`, emits the scalar message using the existing STE channel, and keeps the existing auxiliary reachability and message supervision required by the design.
2. Core B's graph-node computation receives **only `x_B` and the fixed union-graph attention mask**. Its entire trajectory must be independent of `m`, `force_message`, `flip_message`, `b_A` and selector scheduling.
3. Core B produces the two candidate outputs `r0_logits` and `r1_logits` from that selector-independent representation.
4. Only after both candidate outputs exist does the model use the transmitted bit to select `r0_logits` (when `m=-1`, `b_A=0`) or `r1_logits` (when `m=+1`, `b_A=1`). A forced message intervention may change which candidate is selected, but it must not change either candidate tensor.
5. Do not feed the target `b_A`, `r0`, `r1` or `Y` into the inference-time selector. Those labels are available only to the training losses and evaluator.
6. Ensure the training objective follows the predeclared design: the two relation heads are supervised independently against their respective reachability labels, while the main selected-task loss uses the message-selected candidate. Maintain Core A's auxiliary reachability loss and message BCE. Search and reconcile with `reports/phase2c-design.md` before editing; do not quietly invent a different objective.
7. Remove dead trainable parameters from the diagnostic. Since its Core B trajectory does not consume the message, remove any unused `msg_embedder` and learned COMM token from this diagnostic model instead of retaining unused weights. Keep `msg_encoder` where required for the learned selector bit and its BCE supervision. If the diagnostic's parameter count changes as a result, recalculate the exact count programmatically and update its assertion and documentation. **Do not change the primary model's predeclared parameter count or architecture.**
8. Ensure the dual diagnostic does not secretly call the primary model's message-conditioned Core B path as a shortcut. Shared low-level helpers are fine only if they make the selector independence explicit and testable.

### 4.3 Required regression tests

Add or repair tests proving:

- For fixed `x_B` and deterministic evaluation, `r0_logits` and `r1_logits` are bit-identical when the emitted message is flipped, forced to `-1`, or forced to `+1`.
- The selector's selected output changes according to the selected candidate, but the preselection candidates remain exactly unchanged.
- The selector-independent Core B trace is identical under message interventions.
- No unused message-embedding/COMM-token parameters remain in the dual diagnostic; exact parameter count and gradient-use diagnostics reflect the actual implementation.
- A backward pass through the declared dual-model loss produces gradients for every parameter expected to train. If the selection operation naturally prevents parity loss from differentiating through the discrete bit, document that and verify that message BCE supplies the sender's gradient path.
- Both relation-specific heads are trained with the labels specified by the design; ensure the selected-task loss and relation auxiliary losses are not accidentally swapped.

Do not change the primary `LatentConditionedSwarm` trajectory schedule, receiver head, one-way COMM restrictions or parameterisation to make the diagnostic easier to implement.

---

## 5. Fix 3 — Strictly separate training/validation from confirmatory test evaluation

### 5.1 Problem

The current trainer can generate the test split and evaluate it at the end of a non-smoke run. That is not an appropriate interface for a multi-seed confirmatory study: test results should be produced only after every training run, checkpoint-selection decision, and configuration choice has been frozen.

### 5.2 Required implementation

1. Refactor `train_swarm_conditioned.py` so its normal training path generates **only training and validation splits**. It must not call the generator with test seed `99991`, construct a test loader, or execute test metrics.
2. The existing explicit engineering smoke path must remain test-free and must not generate the Phase 2c test split, even indirectly.
3. If the current design supports `--allow-unstable-lr`, keep the protection intact. Do not use it in this task. Confirmatory training must continue to refuse to run when the stability stop-gate says the recipe is ineligible, unless a future separately predeclared study resolves that stop-gate.
4. Add a separate evaluator, for example `evaluate_phase2c_test.py`, that is invoked explicitly. It should:
   - require a frozen evaluation manifest listing the model mode, checkpoint path, checkpoint SHA-256, source commit, configuration and paired seed;
   - reject missing checkpoints or checksum mismatches;
   - generate or load the designated split only when explicitly invoked;
   - refuse to evaluate if the manifest is not declared frozen, if a checkpoint is missing, or if the checkpoint/configuration source is ambiguous;
   - evaluate only the model/checkpoint set named in the manifest, once per final model;
   - save predictions, metric JSON/CSV, split seed, checkpoint hashes and source metadata to ignored run storage;
   - never make training, checkpoint-selection, or acceptance decisions automatically based on test results.
5. The evaluator may be implemented and unit-tested now, but **do not invoke it during this task**. Do not generate seed `99991` data in any end-to-end smoke command. Unit tests may generate small fixtures using clearly different seeds, never `99991`.
6. Keep validation metric evaluation and validation acceptance gates in the trainer. Preserve all Phase 2c predeclared thresholds exactly.
7. Do not evaluate or reuse any Phase 2b test results as Phase 2c evidence.

### 5.3 Required tests

- Normal trainer data setup produces only train and validation objects and cannot silently return a test loader.
- Smoke mode cannot construct the test split.
- The training script has no call path that evaluates a test loader.
- Test evaluator rejects an unfrozen/missing manifest, missing checkpoints and checksum mismatches.
- Evaluator tests use a temporary synthetic fixture or a mocked dataset builder and do not build the actual seed-99999 test split.
- If practical, add a test that monkeypatches the test-split builder to fail if called from training or smoke entry points.

Do not make it trivially possible for a flag or environment variable to bypass these guarantees.

---

## 6. Fix 4 — Correct trajectory-sensitivity aggregation and terminal-node requirement

### 6.1 Problem

The current `evaluate_trajectory()` aggregates per-batch means and then divides by total sample count. That incorrectly reduces the result by approximately a factor related to batch size. The trajectory report's old quantitative sensitivity magnitudes must not be relied upon unless recomputed from saved, usable trace data.

The current mechanism acceptance check also needs to measure sensitivity specifically at the terminal graph node, because `z_B[15]` is what feeds the primary task decoder. Sensitivity at some unrelated graph node does not establish that the message reaches the readout representation during an intermediate recurrent step.

### 6.2 Correct metric definition

For fixed graph input, model weights, mode and intervention, run paired counterfactual traces with opposite messages (`m=-1` and `m=+1`) in deterministic evaluation mode. Let `z^-_t` and `z^+_t` denote the ordinary graph-node states at recurrent step `t`.

For each step report:

- `graph_nodes_mean_abs_delta[t]`: mean absolute difference over examples, all 16 graph nodes, and all 128 latent dimensions;
- `terminal_node_mean_abs_delta[t]`: mean absolute difference over examples and the 128 latent dimensions at Node 15;
- the same two values conditioned on `b_B=0` and `b_B=1` where applicable;
- sample counts for every condition.

Define each statistic as a true element-weighted mean:

`sum(abs(delta)) / number_of_elements_included`.

If the implementation stores per-sample means, weight them by the number of samples in the batch/group before aggregation. Do not average batch means equally when batch sizes or subgroup counts can differ. Empty conditioning groups must raise an explicit error or fail the relevant metric; never silently divide by one and present a metric.

Use the same fixed inputs for the positive- and negative-message traces. Disable dropout and other stochastic behaviour by using `eval()` and inference/no-grad mode; do not let independent input sampling contaminate the counterfactual difference.

### 6.3 Required acceptance semantics

- Full mode: the existing mechanism criterion requires a positive message-sensitive intermediate graph-node state. Strengthen it to require **terminal-node sensitivity at at least one intermediate step before the final update** (for example one of steps 1–7), in addition to reporting the all-node metric. Keep the exact threshold at strictly greater than zero unless the predeclared design explicitly specifies a different numerical tolerance; if floating-point tolerances are introduced for numerical reasons, predeclare and justify them in the report and test.
- Final-only mode: ordinary graph-node states must be bit-identical across opposite messages through steps 1–7. This should be asserted directly, not merely inferred from a near-zero average.
- At step 8, final-only may show sensitivity; report it without changing the invariance condition for steps 1–7.
- Distinguish the COMM token's own different initial state from ordinary graph-node sensitivity. The COMM token differs immediately because it contains the message; that fact alone does not satisfy the graph-node trajectory criterion.
- Keep graph-local attention and COMM read-only semantics unchanged.

### 6.4 Required regression tests

1. **Batch-size invariance:** the computed trajectory metric must be identical (within an explicitly chosen floating-point tolerance) when the same per-example deltas are evaluated with batch size 1, uneven batches, and a single full batch.
2. **Weighted subgroup aggregation:** create a synthetic tensor with unequal subgroup sizes and known deltas; verify overall, `b_B=0`, `b_B=1` and terminal-node means by hand.
3. **Full-mode terminal sensitivity:** using a deterministic fixed graph batch and forced opposite messages, verify nonzero terminal-node sensitivity at one or more intermediate steps for a suitable deterministic parameter setup. If randomly initialised weights cannot guarantee a nonzero sum for all setups, construct a controlled test fixture rather than relaxing the assertion.
4. **Final-only graph-state invariance:** verify exact tensor equality through step 7, including the terminal node.
5. **Metrics are finite:** reject NaN/Inf deltas and malformed traces with a useful diagnostic.
6. **Group accounting:** verify that subgroup sample counts sum to the overall count and that empty groups fail explicitly.

### 6.5 Correct the historical report without rewriting history

Do not change the old commits or pretend the old values were correct. Add a clear dated erratum/addendum to the current Phase 2c report or a new `reports/phase2c-trajectory-metric-erratum.md` stating:

- The earlier trajectory-sensitivity aggregator summed per-batch means and then divided by sample count, under-reporting the magnitude.
- Old sensitivity magnitudes should not be used for quantitative comparisons.
- Structural statements already supported by exact tests (such as final-only bit-identical graph states through step 7) remain supported, subject to rerunning the suite.
- Corrected numerical values are **not** available unless recomputed from preserved raw traces with the fixed aggregator. Do not fabricate values. If raw traces are not available, state that the smoke metrics cannot be retroactively corrected.

Historical run logs and JSON should remain immutable. If the code can recompute metrics from preserved per-example/per-step delta traces, write corrected outputs as new files with provenance; never overwrite historical values silently.

---

## 7. Preserve existing architecture and scientific boundaries

For this fix-only task, preserve the following Phase 2c design:

- `R0` and `R1` are independent relations on 16-node forward DAGs.
- Graph B's attention mask uses `adj_union = OR(R0, R1)` and is independent of the selector bit.
- Core A emits exactly one forward-pass scalar in `{-1, +1}` through the existing STE.
- COMM is a learned register initialised from a learned token and message embedding in the primary model.
- COMM query reads only itself; ordinary graph nodes may read COMM only according to the `full` or `final-only` schedule.
- Full visibility steps: 1–8. Final-only visibility: step 8 only.
- Primary parity/task readout uses `z_B[15]` only; no direct late concatenation of `m`, `msg_repr` or the COMM token into that head.
- Existing task definition and the message-conditioned target are unchanged.
- Existing predeclared Phase 2c gates remain unchanged, including overall, quadrant, sender probe, zero-message, forced-message counterfactual, switch consistency/invariance, and trajectory-mechanism criteria.
- Because the Phase 2b recipe-stability selection rule found no eligible rate, no confirmatory Phase 2c comparison is authorised in this task. Do not use an override flag.

The union-graph mask means information can propagate along edges in either relation, even when those edges are not part of the selected relation. This is part of the already committed Phase 2c design. Do not claim that Core B traverses only the selected relation. The correct target claim is that the selector conditions latent processing over relation-labelled features on a fixed union-graph attention topology.

---

## 8. Test and smoke execution plan

### 8.1 Environment and resource checks

Use the existing VM environment. Do not reinstall or upgrade Python, PyTorch, NumPy or pytest unless an actual dependency/import problem makes it necessary. Record versions before any install. Retain CPU-only execution and 4 intra-op threads.

Check:

```bash
python --version
python - <<'PY'
import torch, numpy
print('torch', torch.__version__)
print('numpy', numpy.__version__)
print('cuda available', torch.cuda.is_available())
print('intra-op threads', torch.get_num_threads())
PY
nproc
free -h
df -h .
```

### 8.2 Run complete test suite

Run the baseline and all experiment suites together, preserving exact logs:

```bash
python -m compileall -q model data test_system.py test_phase2b.py test_phase2c.py
python -m pytest -q test_system.py test_phase2b.py test_phase2c.py
```

Include any new test modules in this combined command. Do not replace or edit earlier tests simply to make the suite pass. If a test is genuinely obsolete because the diagnostic's semantics are corrected, update it with an explicit explanation and add a stricter replacement.

Record:

- exact command;
- exit status;
- test count and duration;
- failure tracebacks, if any;
- source commit.

### 8.3 Engineering-only smoke tests

Run the existing Phase 2c smoke path only, with its small batch/epoch settings and explicit engineering-only label. Smoke coverage must include:

- primary full mode;
- primary final-only mode;
- corrected dual-relation control in its selector-independent configuration;
- finite forward loss;
- no unexpected `grad is None` parameters on the intended normal training loss path;
- message binary/forced-message semantics;
- attention mask direction and no empty rows;
- message-sensitive intermediate ordinary graph-node and terminal-node states in full mode;
- exact graph-node trace equality through step 7 in final-only mode;
- correct selector-independent dual candidate logits;
- corrected trajectory metric values and batch-size invariance;
- zero-message semantics in the primary model.

Use only small development fixtures and train/validation subsets. The smoke run may not generate the Phase 2c test split with seed `99991`, may not load a test loader, and may not report smoke accuracy as a research result.

The primary model's learned COMM token remains present in zero-message mode; only the sample-dependent message contribution is replaced by an exact zero before addition to the learned token. Preserve this semantic detail.

### 8.4 Verify the stop-gate still works

Inspect the training entry point and confirm that a non-smoke Phase 2c run refuses to proceed while the stability gate is ineligible unless a later separately predeclared study changes the eligibility status. Do **not** invoke an override. Test the refusal path safely if feasible without starting training or creating the test split.

---

## 9. Git, branch, commits and publication

Remain on `experiment/phase2c-message-conditioned-reachability`.

### 9.1 Commit structure

Prefer two logically separated commits if feasible:

1. `Fix phase2c graph uniqueness and dual diagnostic isolation`
2. `Fix trajectory metrics and isolate test evaluation`

Add a third documentation commit if the erratum/report update is large enough to merit separation. If implementation dependencies make a different division cleaner, explain it in the final report.

Do not amend previous commits, rewrite the stability/Phase 2c history, force-push or merge into `main`. Normal commits and a normal push are permitted.

### 9.2 Documentation updates

Update the current branch's relevant code comments/docs and add:

- A Phase 2c trajectory metric erratum/addendum explaining that old metric magnitudes were misaggregated.
- A note that the dual diagnostic now computes both relation outputs without selector access in Core B's recurrence.
- A note that training and test evaluation are separate entry points; the test evaluator is not run during this task.
- A fix-only engineering report under `reports/`, containing changes, tests, smoke result, unresolved issues, and explicit statement that no confirmatory training or test evaluation occurred.

Do not alter the predeclared design document to retroactively change hypotheses or thresholds. If wording needs clarification, add an append-only note or an implementation note with clear provenance.

### 9.3 Before pushing

Run:

```bash
git diff --check
git status --short
git diff --stat
git diff --cached --stat
git diff --cached --name-only
```

Review the exact staged diff. Ensure no checkpoints, model weights, run folders, test predictions, tokens, credentials, private keys or unrelated files are committed.

Then commit and push to the existing branch:

```bash
git push origin experiment/phase2c-message-conditioned-reachability
```

Do not alter repository visibility or merge into `main`.

After the push, verify:

```bash
git status --short --branch
git rev-parse HEAD
git log --oneline --decorate -8
```

Confirm the pushed branch points at the intended fix commit(s) and the working tree is clean, except for intentionally ignored run artifacts.

---

## 10. Explicitly forbidden actions in this task

The following are prohibited:

- Starting the confirmatory Phase 2c multi-seed grid.
- Running the full trainer with an unstable-learning-rate override.
- Generating the Phase 2c seed-99991 test split.
- Evaluating test results or reusing test predictions.
- Adjusting gates, data distribution, loss weights, number of epochs, model width/depth or acceptance thresholds to make a test pass.
- Selecting a model based on smoke accuracy.
- Rewriting past report numbers without preserving an erratum/provenance.
- Treating the corrected sensitivity code as evidence that the previously reported numerical values are now valid.
- Changing the primary COMM-token architecture as part of fixing the dual diagnostic.
- Removing the dual diagnostic instead of fixing it.
- Adding a third experimental mode or expanding the task during the fix-only pass.
- Committing checkpoints, logs, metrics JSONL, environment secrets, or credentials.
- Force pushes, `git reset --hard`, `git clean -fd`, or other destructive repository operations.

If any of these actions appears necessary, do not improvise. Explain the issue in the final report and stop before the prohibited action.

---

## 11. Definition of done

The task is complete only when all of the following hold:

### Code correctness
- [ ] Raw graph uniqueness is enforced across A/R0/R1 and all splits without role prefixes in the uniqueness key.
- [ ] The dual diagnostic's Core B candidates are demonstrably independent of selector/message interventions; selection occurs only after both candidates are computed.
- [ ] Unused trainable message/COMM parameters are removed from the dual diagnostic and its exact parameter count is tested.
- [ ] Training and smoke paths do not generate or evaluate the confirmatory test split.
- [ ] A separate explicit manifest-driven test evaluator exists or the implementation report documents why it cannot safely be completed; if implemented, it has not been invoked in this task.
- [ ] Trajectory metrics are correctly element-weighted and invariant to validation batch size.
- [ ] Terminal-node intermediate sensitivity is explicitly measured and enforced for full mode.
- [ ] Final-only graph-node states are bit-identical through step 7.
- [ ] The unstable-recipe stop-gate remains intact.

### Tests and engineering evidence
- [ ] The v7.1 baseline suite passes unchanged.
- [ ] The Phase 2b suite passes unchanged, except for any transparently documented test update directly required by corrected semantics.
- [ ] All Phase 2c tests pass, including new regression cases.
- [ ] Full and final-only primary smoke paths complete with finite losses and expected message/trajectory semantics.
- [ ] The corrected dual diagnostic smoke path completes and passes selector-independence tests.
- [ ] No confirmatory training or seed-99991 test evaluation occurred.

### Evidence and Git hygiene
- [ ] Existing experiment artifacts and historical run data remain intact.
- [ ] An ignored fix-only run directory contains logs and metadata.
- [ ] Documentation includes the trajectory metric erratum and precise description of the corrected control.
- [ ] Changes are committed and pushed to the existing Phase 2c branch, without rewriting history or merging to `main`.
- [ ] Final repository state and commit SHA are recorded.

If any checkbox cannot be satisfied, report it as incomplete. Do not label the task fully complete.

---

## 12. Required final response to the owner

When finished, provide a concise but complete engineering report with:

1. **Starting state:** branch, starting commit, initial dirty/clean status, remote divergence if any.
2. **Changes made:** files changed and the precise implementation correction for each of the four issues.
3. **Raw hash guarantee:** explain exactly how raw hashes are tracked across roles and splits, and list the tests that prove it.
4. **Dual diagnostic semantics:** explain how Core B now computes `R0` and `R1` independent of the selector and provide the new exact parameter count.
5. **Test isolation:** identify the training command and the separate test evaluator; explicitly confirm that seed `99991` was not generated and that the evaluator was not invoked.
6. **Trajectory metric correction:** formula, denominator, batch-size-invariance test, terminal-node gate, and the status of old metric magnitudes.
7. **Verification:** exact combined test command, total passed/failed count, duration and log paths; smoke commands and their finite-loss/gradient/trajectory outcomes.
8. **Git:** commits, pushed branch, final commit SHA, clean/dirty state, confirmation that no checkpoint or run data was committed.
9. **Research status:** explicitly say: “Engineering fixes and smoke verification only. No confirmatory Phase 2c training, no Phase 2c test split generation, and no Phase 2c test evaluation were performed.”
10. **Remaining blockers:** any incomplete item, ambiguity or failure. Do not conceal failed tests.

Begin with Section 2, inspect the repository and preserve the current state. Then execute the work sequentially. Do not begin with training.

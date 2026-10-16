# Agent Master Prompt — Latent Recurrent Swarm: Phase 2b Housekeeping, Stability Study, and Phase 2c Message-Conditioned Reasoning

**Target:** Pi coding agent (e.g. GLM 5.3 Flash) running on the Debian 12 Smol Machines VM  
**Repository:** `https://github.com/gl33mer/latent-recurrent-swarm`  
**Working directory:** `/workspace/latent-recurrent-swarm`  
**GitHub CLI:** already authenticated as `gl33mer`  
**Hardware budget:** 4 vCPUs, approximately 8 GiB RAM, CPU-only PyTorch  
**Primary current branch:** `experiment/phase2b-comm-token` (verify actual repository state before relying on this)  
**Project state:** v7.1 Phase 2a baseline completed; Phase 2b COMM-token experiments completed with mixed training-gate outcomes; Phase 2c has not been implemented or trained.

---

## 0. Mission and research judgement

You are acting as a careful ML research engineer, experiment operator, and repository maintainer. Your task has three ordered parts:

1. **Preserve and clarify the Phase 2b result.** Audit the actual repository and run artifacts, make the report's interpretation statistically and causally precise, verify/save available artifacts and checksums, and leave the Phase 2b history intact.
2. **Run a bounded optimisation-stability study on Phase 2b.** The existing COMM mechanism changes Core B's internal graph-node trajectory in full mode, but validation performance varied sharply across seeds and epochs. Instrument losses and gradient norms, then compare a small predeclared learning-rate sweep under paired full/final-only conditions. This is a diagnosis, not a search for a configuration that makes every gate pass at any cost.
3. **Design, test, and—only if the stability gate permits—run Phase 2c: message-conditioned reachability.** Core A's bit selects which of two edge relations Core B must reason about. Compare full-step COMM access with a matched final-step-only control, and add a separate dual-relation/compute-both diagnostic. The aim is to ask whether early access to the bit has a measurable advantage when it specifies the computation to perform, rather than merely being combined with a completed answer.

My research judgement encoded in this plan is:

- Phase 2b has demonstrated the *mechanism*: the received bit can modulate Core B's ordinary graph-node states during recurrence without allowing graph-node-to-graph-node paths to bypass the topology mask.
- Phase 2b has **not** demonstrated that recurrent message integration is better than final-only access. The XOR task does not require early message access, the six runs were highly variable, and no seed had both modes pass every validation gate.
- The best next experiment is therefore **a task that rewards message-conditioned computation, coupled with a bounded optimisation-stability study first**. If training instability remains unresolved, the agent must not present the harder task comparison as a clean confirmatory test.

Do not claim a general theory of latent reasoning from these small synthetic experiments. State precisely what was tested, what was measured, and what remains unknown.

## 1. Non-negotiable operating rules

1. Work from `/workspace/latent-recurrent-swarm`.
2. Inspect the actual filesystem, Git status, remote, branches, tags, reports and run directories before making changes. The supplied state summary may not capture later edits.
3. **Never** run `git reset --hard`, `git clean -fd`, force-push, rewrite public history, delete branches, or overwrite an existing checkpoint/log. Preserve unknown files and existing user changes.
4. Do not commit `.venv`, checkpoints, credentials, tokens, raw large logs, or run directories. Keep run artifacts in ignored `runs/` directories and ensure they are copied to durable storage if this VM is ephemeral.
5. The repository has been made public at the owner's request. Do not change visibility again. Never print tokens or credentials. Do not place secrets in reports, shell transcripts, notebook outputs, or Git.
6. Do not relax any existing Phase 2b acceptance criterion. Phase 2c needs its **own** predeclared task-appropriate criteria, documented before training. Do not silently revise gates after observing results.
7. Keep v7.1 Phase 2a and the original Phase 2b model/experiments usable as historical baselines. Do not modify their semantics merely to make Phase 2c easier.
8. Do not evaluate an existing or new test split during learning-rate selection or model selection. All model/recipe decisions use training and validation data. The previously reported Phase 2b seed-9999 test evaluations have already occurred; do not reuse them as fresh evidence for a new task.
9. Every result must distinguish **engineering success**, **task success**, and **scientific finding**. Failing a gate is a valid result. Never hide failed runs or report only the best seed.
10. Use explicit CPU operation and the existing environment. Do not add GPU dependencies, multiprocessing, or large frameworks without a concrete need.
11. Do not claim training has been executed unless you actually ran it. Record commands, return codes, runtimes, code commit, environment, configurations, metrics and checksums.
12. When an instruction cannot be completed due to missing artifacts, access or runtime constraints, report the blocker and make the best safe partial progress. Do not fabricate artifacts or outcomes.

## 2. Current project state and evidence to preserve

### 2.1 v7.1 Phase 2a baseline

- Phase 1 reasoner: 137,476 trainable parameters.
- Phase 2a swarm: 292,427 trainable parameters.
- Phase 1: reported T=1 validation accuracy 50%, T=8 validation/test accuracy 100%.
- Phase 2a: reported 97.5% connected validation accuracy and 97.0% held-out test accuracy, with zero-message and message-flip controls behaving as intended.
- Phase 1 checkpoint SHA-256 reported as:
  `ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e`
- The original v7.1 source was tagged `v7.1-pre-run`; a previous report commit is recorded as `d10b0c2`, and the initial source commit as `77c2d1a`. Verify these refs in the live repository; do not assume short hashes are sufficient if `git` reports something different.

### 2.2 Phase 2b COMM-token result

The Phase 2b implementation commit was reported as `be691ed` on `experiment/phase2b-comm-token`. It has a 17-token Core B (16 graph nodes + one read-only COMM token), with graph queries allowed to read the COMM token under a specified schedule while the COMM query can read only itself.

The frozen Phase 2b recipe was:

- CPU-only PyTorch, float32, 4 intra-op threads.
- Data seeds train=42, validation=1337, test=9999; train/validation each 4,000/1,000 samples under the original graph-task generator (verify actual sizes in code/report).
- Batch size 64, 20 epochs, AdamW learning rate `8e-4`, weight decay `1e-2`.
- Both cores warm-started from the Phase 1 checkpoint.
- Loss: parity CE + `0.5 * aux_A CE` + `0.5 * aux_B CE` + `0.5 * BCEWithLogits(raw_m, b_A)`.
- Model: `LatentCommSwarm`, 284,107 parameters.

Reported results, all six runs included:

| Mode / model seed | Validation connected | Quadrants 00/01/10/11 | Flip residual b_B=0/1 | Zero message | Probe overall / b_A=0 / b_A=1 | Status |
|---|---:|---|---|---:|---|---|
| full / 42 | 91.70% | 95.2 / 96.0 / 87.6 / 88.0 | 0.00 / 1.74% | 50.80% | 92.30 / 95.60 / 89.00% | failed probe gates |
| full / 43 | 95.00% | 99.6 / 100.0 / 89.6 / 90.8 | 0.00 / 0.21% | 50.00% | 95.00 / 99.80 / 90.20% | all validation gates passed |
| full / 44 | 82.30% | 80.4 / 81.6 / 81.2 / 86.0 | 0.00 / 0.24% | 50.00% | 82.60 / 81.40 / 83.80% | failed probe gates |
| final-only / 42 | 99.70% | 98.8 / 100.0 / 100.0 / 100.0 | 0.00 / 0.00% | 50.00% | 99.70 / 99.40 / 100.00% | all validation gates passed |
| final-only / 43 | 67.60% | 56.4 / 58.8 / 78.0 / 77.2 | 0.00 / 3.82% | 50.40% | 67.90 / 56.20 / 79.60% | failed several gates |
| final-only / 44 | 74.80% | 47.2 / 74.4 / 90.0 / 87.6 | 21.87 / 1.98% | 49.70% | 83.90 / 76.80 / 91.00% | failed several gates |

Held-out test was reported only for the two validation-gate-passing models:

| Mode / seed | Test connected | Quadrants 00/01/10/11 | Flip residual b_B=0/1 | Zero message | Probe overall / b_A=0 / b_A=1 |
|---|---:|---|---|---:|---|
| full / 43 | 94.80% | 99.2 / 100.0 / 91.2 / 88.8 | 0.00 / 0.00% | 50.20% | 94.80 / 99.60 / 90.00% |
| final-only / 42 | 100.00% | 100.0 / 100.0 / 100.0 / 100.0 | 0.00 / 0.00% | 50.10% | 100.00 / 100.00 / 100.00% |

Validation connected accuracy by epoch 5/10/15/20:

- full seed 42: 97.9 / 95.2 / 45.9 / 91.7
- full seed 43: 93.7 / 65.7 / 94.5 / 95.0
- full seed 44: 98.9 / 98.4 / 83.4 / 82.3
- final-only seed 42: 48.9 / 97.5 / 99.0 / 99.7
- final-only seed 43: 92.3 / 93.7 / 69.4 / 67.6
- final-only seed 44: 91.2 / 95.7 / 96.5 / 74.8

Trajectory sensitivity, mean absolute difference between graph-node traces for opposite message polarities, steps 0 through 8:

- full / 42: `0, .104, .202, .289, .373, .472, .716, .987, 1.284`
- full / 43: `0, .105, .207, .338, .459, .634, .829, .872, 1.034`
- full / 44: `0, .140, .211, .308, .421, .584, .773, .988, 1.211`
- final-only / 42: `0, 0, 0, 0, 0, 0, 0, 0, .502`
- final-only / 43: `0, 0, 0, 0, 0, 0, 0, 0, .600`
- final-only / 44: `0, 0, 0, 0, 0, 0, 0, 0, .381`

These measurements support that full-mode messages influence Core B's ordinary graph-node states from the first recurrent update, whereas the final-only control keeps graph-node states bit-identical through step 7. The held-out task comparisons are **not paired across a gate-passing model pair** because no seed had both modes pass every gate.

### 2.3 Files/locations to inspect

Likely relevant locations (verify they exist):

- `model/core.py`
- `model/swarm.py` (v7.1 Phase 2a)
- `model/swarm_comm.py` (Phase 2b)
- `data/graph_task.py`
- `test_system.py`
- `test_phase2b.py`
- `train_unit.py`
- `train_swarm.py` (Phase 2a)
- `train_swarm_comm.py` (Phase 2b)
- `latent_phase2b.md`
- `PROJECT_SPEC.md`, `README.md`, `CHANGELOG.md`, `WORKLOG.md`
- `reports/first-run-report.md`
- `reports/phase2b-design.md`
- `reports/phase2b-run-report.md`
- ignored run folders under `runs/`

The report commit, branch position, local artifact availability and remote state must be verified rather than assumed.

## 3. Stage 0 — repository, environment and artifact audit

Start by executing and recording:

```bash
cd /workspace/latent-recurrent-swarm
pwd
uname -a
cat /etc/os-release
python --version || python3 --version
nproc
free -h
df -h .
git status --short --branch
git remote -v
git log --oneline --decorate -12
git branch --all --verbose
 git tag --list
 gh auth status
 gh repo view gl33mer/latent-recurrent-swarm --json name,visibility,defaultBranchRef,url
```

Use the active existing virtual environment if present. Do not reinstall or upgrade PyTorch without a concrete compatibility need. If `.venv` is not active/present, inspect the already recorded environment snapshot first and create/repair an environment only if necessary.

Check that the repo is public (expected current visibility), but do not change visibility. Verify the remote URL and authentication without printing credentials.

Before any code changes:

1. Record full `HEAD` SHA and current `git status`.
2. Inspect all untracked and modified files; preserve them. Do not assume every unknown file is disposable.
3. Identify the exact Phase 1 checkpoint, Phase 2b full/seed-43 checkpoint, and Phase 2b final-only/seed-42 checkpoint, if present.
4. Compare each checkpoint's actual SHA-256 to the known recorded value(s) in existing run metadata. Do not assume the Phase 2b hashes unless the stored files/JSON say so. Preserve the Phase 1 hash stated above.
5. Verify that each corresponding run log, metadata JSON, code commit, config, and report exists. If a report says an artifact exists but it is missing, document the discrepancy.
6. Confirm run directories and checkpoints are not tracked by Git. Keep `runs/`, checkpoints, environment dumps, and large logs ignored.
7. Determine which filesystem paths survive VM restarts. Copy a dated, checksummed archive of the original Phase 2b runs and passing checkpoints to an available durable workspace outside the source tree, if possible. If no durable external/persistent location exists, archive locally and explicitly report that local storage may be ephemeral. Do not upload model binaries to a public repository.
8. Run the existing regression suite before editing:

```bash
.venv/bin/python -m pytest -q test_system.py test_phase2b.py
```

Adjust only if the environment's Python executable/path differs. Save the output and return code to a new housekeeping run directory. If pre-existing tests fail, stop to diagnose; do not make the test suite green by weakening an assertion.

## 4. Stage 1 — Phase 2b housekeeping and scientific report correction

Perform the following on `experiment/phase2b-comm-token` (or the actual branch containing the Phase 2b implementation), after verifying a clean starting point. If the branch has uncommitted user work, create a safe snapshot/backup and do not overwrite it.

### 4.1 Preserve historical evidence

- Do not rewrite the existing Phase 2b run report in a way that hides the initial interpretation or failed runs. Add a clearly dated **analysis correction/addendum** and preserve the original data.
- Keep all six Phase 2b run outcomes visible.
- Preserve all passing checkpoints, metadata, checksums, logs and the exact source commit that produced them. If a checkpoint is missing, do not pretend to recover the exact weights. A rerun is a new reproduction with its own run ID and checksum.
- Keep v7.1 Phase 2a results and source unchanged.

### 4.2 Correct the causal interpretation of optimisation instability

The report's statement that “routing the message through the recurrent trajectory demonstrably destabilises optimisation” is stronger than the experiment identifies. Both full and final-only Phase 2b modes had instability, and those modes share most architectural changes relative to Phase 2a. Change future summaries to say:

> Phase 2b showed substantial seed- and epoch-varying validation performance under the frozen recipe. The existing experiment does not isolate whether recurrent message integration itself, other Phase 2b architectural differences, or their interaction with optimisation caused the instability.

Do not delete or silently rewrite the original historical statement; append the correction and cross-reference it from the report/work log.

### 4.3 Add transparent paired validation arithmetic

Using the six recorded validation scores, full minus final-only paired validation differences are:

- seed 42: `91.7 - 99.7 = -8.0` percentage points
- seed 43: `95.0 - 67.6 = +27.4` percentage points
- seed 44: `82.3 - 74.8 = +7.5` percentage points
- mean paired difference: approximately `+8.97` points
- median paired difference: `+7.5` points

Add these as **exploratory descriptive comparisons only**. State that only three paired seeds exist, the results are highly variable, and no paired seed had both modes pass every acceptance gate. Do not use the two passing models' test numbers as a paired comparison: they are different seeds.

### 4.4 Correct stale metadata

Verify current public visibility and make the relevant reports consistent. The original first-run report reportedly still described the repository as private even though the owner later made it public. Correct this with a concise addendum or direct metadata correction that preserves history. Update README/WORKLOG only when needed; avoid broad documentation churn.

### 4.5 Housekeeping commit

- Write `reports/phase2b-analysis-addendum.md` or an equivalently clear dated addendum.
- Update `WORKLOG.md`/`CHANGELOG.md` with what changed, why, and the commit hash.
- Run `git diff --check`, the baseline tests, and secret scan.
- Commit with a clear message such as `Clarify Phase 2b stability interpretation and paired results`.
- Push normally; no force-push. Leave `main` untouched unless it already has an established project policy requiring report-only synchronization. Prefer not to merge experimental branches automatically.

## 5. Stage 2 — Instrumented Phase 2b stability study

### 5.1 Purpose

The prior experiment reports validation accuracy only every five epochs. It does not sufficiently identify whether collapses correlate with task loss, auxiliary loss, message supervision, or gradient magnitude. The next study must gather that evidence before any claim that a particular component causes instability.

Keep the original `model/swarm_comm.py` behaviour unchanged. Prefer a new experiment driver under `experiments/stability/` or a new opt-in diagnostic mode. Do not alter default semantics or thresholds of `train_swarm_comm.py`.

### 5.2 Predeclared sweep

Use a small learning-rate grid:

- `8e-4` (frozen Phase 2b baseline)
- `4e-4`
- `2e-4`

For each learning rate, run both `full` and `final-only` modes with model seeds `42`, `43`, `44`: 18 instrumented runs total. If the resource cap below is reached, finish the current run safely and mark the sweep incomplete; do not drop inconvenient runs silently.

Keep these fixed across the sweep:

- original Phase 2b task, architecture and graph generator;
- train/validation dataset seeds `42`/`1337` and sizes from the canonical code;
- model initialization seed paired across full and final-only for each seed;
- batch size 64, 20 epochs, AdamW, weight decay `1e-2`;
- loss weights `1.0 parity + 0.5 aux_A + 0.5 aux_B + 0.5 message_BCE`;
- both cores warm-started from the verified Phase 1 checkpoint;
- CPU float32; four intra-op threads;
- all existing task/control thresholds;
- identical sample ordering for the paired modes within a seed, as far as deterministic PyTorch CPU execution permits.

Only learning rate and the intended COMM visibility schedule may differ. No gradient clipping, loss-weight change, early stopping, best-checkpoint selection, or architecture change in this sweep. If deterministic paired data order cannot be guaranteed, explain why and fix the seed/DataLoader generator setup without changing the task semantics.

**Do not evaluate any test split in this sweep.** Ideally use a train/validation-only dataset builder for this study so a test dataset is not even materialised. Reusing validation is permitted for the predeclared LR selection.

### 5.3 Diagnostics to record

Write machine-readable JSONL plus a concise CSV/Markdown summary. For every run, store:

- run ID, mode, seed, learning rate, exact Git commit, environment snapshot, Phase 1 checkpoint SHA-256, data seeds/sizes, start/end time, wall time;
- per epoch: mean train total loss and each loss component; validation connected accuracy and four quadrants; message probe overall and per `b_A`; zero-message accuracy; flip residuals per `b_B` when practical;
- gradient norms by named groups: `core_A`, `core_B`, `msg_encoder`, `msg_embedder`, `learned_comm_token`, `parity_head`; at minimum record mean and maximum per epoch, computed from gradients before the optimizer step;
- NaN/Inf detection for losses, gradients and parameters; optimizer learning rate;
- trajectory sensitivity for full/final-only at minimum at final epoch and optionally every 5 epochs;
- final validation gate status and precise failed criteria;
- last-epoch checkpoint and periodic checkpoint(s) every 5 epochs in the ignored per-run directory, including **failed runs**, so the stability behaviour can be inspected later. Do not commit these files.

Keep instrumentation lightweight. A 4-vCPU/8-GiB VM should be sufficient; do not introduce TensorBoard or tracking servers unless logs prove insufficient.

### 5.4 Stability selection rule — declare before running

A shared learning rate is eligible for Phase 2c only if:

1. At least 2 of the 3 seeds pass all existing Phase 2b validation gates **in full mode**, and at least 2 of 3 pass all gates **in final-only mode** at the same learning rate.
2. No run at that rate has non-finite losses/parameters or a broken gradient path.
3. Full-mode trajectory sensitivity is positive at an intermediate step, while final-only graph-node traces remain bit-identical before the final step.
4. The validation histories and gradient diagnostics do not show a clear new pathology at the chosen rate. Compare epoch-to-epoch accuracy changes and loss/gradient traces; report, do not hide, any collapse.

Among eligible shared rates, choose by highest total all-gate pass count across both modes, then by lower median validation instability across all paired runs. Do not choose separate rates for full and final-only. If rates tie, prefer the lower rate only if its validation evidence is otherwise comparable. Document the exact rule before viewing sweep outcomes.

If no learning rate is eligible, classify the training recipe as **not yet stabilised**. Finish the report and implement/test the Phase 2c data/model engineering as far as practical, but do not run or claim a confirmatory multi-seed Phase 2c comparison. A tiny smoke run is allowed and must be labelled engineering-only. Recommend a separate optimisation study instead of quietly adding more hyperparameters.

A maximum wall-clock budget of approximately 120 minutes is allowed for the complete 18-run stability sweep. Monitor actual runtimes. If the cap is approached, do not start a run that is unlikely to finish; report partial completion honestly and do not apply the eligible-rate rule to an incomplete grid as though it were complete.

### 5.5 Stability outputs

Create:

- `reports/phase2b-stability-design.md` (predeclared protocol, commit before training if possible);
- `reports/phase2b-stability-report.md` (all runs, losses, gradient diagnostics, gate status, LR decision, limitations);
- `runs/<unique_run_id>_stability_.../` (ignored logs/JSONL/checkpoints/configs);
- an aggregate JSON containing all per-run scores;
- optional plots generated from saved JSONL, but do not make plots the only record.

Do not promote this as a confirmatory scientific result; it is a small recipe-screening study. Commit reports/scripts only, not raw checkpoint dumps or bulky ignored logs. Push the stability branch normally.

## 6. Stage 3 — Design Phase 2c before any Phase 2c training

Phase 2c should be a new message-conditioned graph task, not a silent modification of the Phase 2b XOR dataset. Create a new design document and tests before full training. Use a separate module so Phase 2a and Phase 2b remain reproducible.

Recommended branch: `experiment/phase2c-message-conditioned-reachability`, based on the latest reviewed Phase 2b/stability branch. Verify no same-named branch exists before creating it. Never overwrite an existing remote branch.

### 6.1 Research question and hypothesis

**Question:** under fixed graph-recurrent compute and a one-bit bottleneck, does exposing the selector bit throughout Core B's recurrent loop improve its ability to solve a message-conditioned reachability task compared with giving the same bit only at the final recurrent step?

Primary comparison:

- `full`: COMM key visible to graph queries at all eight recurrent steps.
- `final-only`: COMM key invisible to graph queries for steps 1–7 and visible at step 8 only.

Both conditions must have the same parameter count, same initialization for paired seeds, same train/validation/test examples, same batch size, optimizer, epoch budget and loss weights. The only intended difference is when the graph nodes may access the one-bit COMM representation.

Hypothesis (not an assumed result): full access may help Core B select and maintain the correct relation-specific computation during graph propagation. If final-only performs equally well, the valid conclusion may be that this task can also be solved by parallel precomputation and late selection; do not call this a failure of implementation.

### 6.2 Task definition

Use Core A's original single-relation Graph A and its reachability label `b_A` to produce the sender bit:

- `m = -1` means select relation `R0` (`b_A = 0`);
- `m = +1` means select relation `R1` (`b_A = 1`).

Core B receives a graph containing **two distinct directed edge relations**, `R0` and `R1`. The target is selected-relation reachability:

`Y = r0` if `b_A == 0`, otherwise `Y = r1`,

where `r0 = Reach(0 -> target in R0)` and `r1 = Reach(0 -> target in R1)`.

The bit chooses which reachability predicate is relevant. This is no longer XOR. The receiving system must infer reachability under the selected relation and use the one-bit selector to determine which relation matters.

Each relation is an independently sampled forward DAG over 16 nodes with edges only for `j-i in {1,2,3}`. For each *reachable* relation, shortest path length must be in `[5,8]`; unreachable relations must have no path. Continue endpoint constraints (Node 0 out-degree >=1 and Node 15 in-degree >=1) per relation if feasible, but test and document them. Use exact balancing across the eight combinations `(b_A, r0, r1)` so every combination occurs equally often. Require `num_samples` divisible by 8. The selected target label should be exactly balanced by construction.

An especially informative subset is **relation disagreement**: `r0 != r1`. On these examples the selector bit changes the correct answer, so this is the primary subset for testing whether the message is behaviourally useful. On relation-agreement examples (`r0 == r1`) the target is the same for either message; early access should not be expected to change the correct label.

### 6.3 Input representation and attention masks

Implement a separate dataset module such as `data/conditioned_graph_task.py`. Do not silently alter `data/graph_task.py` or the existing dataset contract.

For each Graph B node token, use a 64-dimensional input layout:

`[adj_union(16), adj_R0(16), adj_R1(16), one_hot_node_id(16)]`,

where `adj_union = (adj_R0 OR adj_R1)` as a binary adjacency row. `x_B` is `[batch, 16, 64]`. `x_A` remains the existing `[batch, 16, 32]` input.

The new model should have separate input dimensions: Core A takes 32 features, Core B takes 64. Use a separate class/module such as `model/swarm_conditioned.py`; do not break `LatentCommSwarm` or v7.1 classes.

Core B's graph-to-graph attention mask uses the union adjacency, preserving the directed predecessor rule (`query j` may attend to `key i` iff `i == j` or `i -> j` in at least one relation). The mask must be independent of `m`. Do **not** use the message bit to construct or change the graph mask; that would hard-code relation selection in the mask path rather than test latent use of the COMM token.

The COMM query remains self-only and cannot read graph nodes. Graph queries may read COMM according to the full/final-only schedule. No message tensor, message embedding or COMM state may be passed directly to the task readout. The task head must consume only `z_B[target_node]` (target is Node 15 for the current benchmark). The one bit is transmitted once before Core B's recurrence; do not add reverse communication, repeated bits, vector messages or a hidden continuous side channel.

### 6.4 Warm-start mapping for Core B's 64-dimensional encoder

Core A loads the Phase 1 checkpoint strictly, unchanged.

Core B's encoder shape differs from the Phase 1 32-dimensional encoder. Do not blindly `load_state_dict(strict=True)` across that mismatch or silently initialize the whole Core B from scratch without documenting it. Implement and unit-test an explicit, reproducible transfer:

- Transfer all shape-compatible Phase 1 Core B recurrent-block and decoder weights.
- For the first encoder linear layer, map Phase 1 adjacency-input weights to the new `adj_union` columns 0:16.
- Set the new relation-specific columns 16:48 to zero initially, so the Phase 1-reasoner function is preserved with respect to union adjacency and node ID while the new relation channels are initially ignored and can acquire learned weights during training.
- Copy Phase 1 node-ID weights from old columns 16:32 to new columns 48:64; copy the first layer bias.
- Copy LayerNorm/GELU components unchanged (with exact key/shape checks).
- Record this mapping in the design and run metadata. Add a test that the new encoder output matches the old Phase 1 encoder output when relation feature weights are zero and the `adj_union` / node-ID inputs match.

This is a documented warm-start adaptation, not an exact full-model checkpoint load. Report the fact clearly.

### 6.5 Output head and objective

For the selected-relation binary reachability target, use the Core B terminal decoder (or an explicitly documented equivalent binary head) on `z_B[15]`; a nonlinear XOR head is not required because the target is not XOR. Avoid unused heads/dead parameters. Ensure every declared trainable parameter participates in the loss or explain/fix it.

Use sender auxiliary reachability supervision on Core A and the existing `BCEWithLogits(raw_m, b_A)` message-polarity supervision. Use a single selected-target CE for Core B. Do not add auxiliary losses that force Core B to solve both `R0` and `R1` separately in the main full/final-only comparison; that would deliberately encourage parallel precomputation and would change the main question. A separate dual-relation comparator can have those heads/losses, clearly isolated from the primary comparison.

Before training, compute and assert the new model's exact parameter count. With Core B `d_input=64`, the first encoder linear layer adds 4,096 parameters relative to v7.1. If using the stated composition (Core A 137,476; Core B 141,572; message encoder 129; message embedder 256; learned COMM token 128; no separate MLP head beyond Core B's decoder), the expected total is **279,561**. Recalculate from actual code and update the design; do not force an expected number if the architecture changes.

### 6.6 Required additional diagnostic: compute-both-then-select

The relation-selection task is more appropriate than XOR, but it still may be solvable by computing both reachability answers in parallel before the message arrives. Do not assume early access is mathematically necessary just because the task is message-conditioned.

Implement a separate **dual-relation diagnostic baseline** if feasible within the VM budget:

- same Core B graph encoder/recurrent backbone and same union-mask input representation;
- two terminal binary readouts or heads supervised separately for `r0` and `r1`;
- at evaluation, select the appropriate prediction using the one-bit selector *only after* both relation predictions are produced.

Keep this model separate from the primary full/final-only conditions and report its parameter count and training objective. It is an explicit diagnostic to test whether a late-selection strategy can precompute both candidates. Do not use it to replace the matched final-only primary control.

If budget does not permit full dual-head training, at minimum implement a simple graph/label diagnostic and document what it can and cannot establish. Do not silently omit the control.

## 7. Phase 2c data integrity and required controls

### 7.1 Dataset splits and deterministic generation

Use new task-specific seeds, not the Phase 2b test set. For example, predeclare relation-task split seeds `4201` train, `13371` validation, and `99991` test, or select other fixed seeds **before generating/inspecting their metrics** and document them. Use `existing_hashes`/equivalent tracking to guarantee graph-hash disjointness across splits.

Hash every individual Graph A, R0 and R1 adjacency. No graph matrix may recur within a split or across train/validation/test. The labels, selector, relation pairing and target must be re-computable from the generated graphs. Verify generator determinism for a fixed seed.

Tests must verify:

- `num_samples % 8 == 0` and exact balance of all eight `(b_A, r0, r1)` combinations;
- relation reachability labels equal independent BFS results;
- positive shortest paths satisfy `[5,8]`, negatives have no path;
- `adj_union` exactly equals logical OR of relation matrices;
- `Y == r0` for `b_A=0`, and `Y == r1` for `b_A=1`;
- all graph hashes are distinct within/across splits;
- conditional/shortcut baselines are measured, not assumed;
- the disagreement subset contains the expected balanced fraction under exact balancing.

### 7.2 Main comparisons and message controls

For each paired model seed, run both full and final-only with identical initialisation and data order as far as deterministic execution permits.

Required metrics on validation and the frozen test split:

1. Selected-task accuracy overall.
2. Accuracy by selector (`b_A`), by target `Y`, by the four `(b_A, Y)` quadrants, and by relation-agreement/disagreement subset.
3. Counterfactual selected-target accuracy: on the *same* Graph B relation pair, force/select `m=-1` and evaluate against `r0`; then force/select `m=+1` and evaluate against `r1`. Report both relation choices and disagreement cases.
4. **Selector switch consistency:** on `r0 != r1` examples, does changing the forced message change the prediction from the `R0` answer to the `R1` answer? On `r0 == r1`, is the prediction invariant to the selector? Report counts and rates. Ensure forced-message evaluation isolates Core B from any sender polarity error; also report end-to-end sender probe separately.
5. Zero-message accuracy overall and on disagreement examples.
6. Sender message probe overall and conditioned on `b_A=0` / `b_A=1`.
7. Trajectory sensitivity per step (0 through 8), overall and conditioned on relation agreement/disagreement, comparing forced opposite messages on identical Graph B inputs.
8. Final-only trace invariance: ordinary graph-node states must be bit-identical across opposite message values through step 7; they may differ at step 8.
9. Full-mode trajectory sensitivity: at least one intermediate step must show a non-zero message effect, including measurable sensitivity at the terminal node by some intermediate step.
10. Graph-input receptive-field test: on a line graph with both relation feature channels constructed appropriately, Graph B node-0 input influence must not reach Node 15 before 15 graph-edge steps; COMM must not permit graph-to-graph bypass. Since COMM can broadcast the message globally, do not confuse message sensitivity with graph-node receptive-field sensitivity.
11. Strict binary channel contract and exact flip/forced-message semantics.
12. Every trainable parameter receives a gradient under the selected objective; report all-zero gradient tensors diagnostically rather than treating them as proof of failure by themselves.

### 7.3 Zero-message interpretation and gate

Unlike XOR, this selected-relation task has a non-chance no-message baseline: if `r0 == r1`, the target is the same for either selector. With the eight combinations balanced, half of examples have agreeing relation labels. A model without the selector can therefore reach 75% in principle by predicting the common label on agreement examples and guessing on disagreement examples.

Do **not** reuse the Phase 2b XOR gate `zero-message <= 53%` here; that would be mathematically inappropriate.

Predeclare a task-appropriate gate:

- zero-message overall accuracy no greater than **77%** (75% expected baseline plus a two-point tolerance);
- zero-message accuracy on the `r0 != r1` subset no greater than **53%**;
- report the agreement subset separately (near-perfect accuracy there is expected and not evidence of message use).

If observed class balancing or empirical no-message baselines differ from expectations, diagnose and report before training. Do not change the gate after viewing trained outcomes; document a corrected gate in a new design revision before full training if a pre-training unit test demonstrates the stated expectation is invalid.

### 7.4 Proposed task acceptance gates

Pre-register before Phase 2c full training. Suggested initial gates:

- End-to-end selected-task validation accuracy >= 82% overall.
- Each `(b_A, Y)` quadrant >= 75%.
- Message probe >= 95% overall and >=90% for each `b_A` subgroup.
- Full mode: measurable message-sensitive intermediate graph-node state; final-only: bit-identical graph-node states through step 7.
- Zero-message overall <=77% and disagreement-subset accuracy <=53%.
- Forced-message counterfactual accuracy for R0 and R1 should each be >=75%; report all subgroups.
- On disagreement cases, selector switch consistency should be high (propose >=85% as an exploratory gate; predeclare it in the design before any training). On agreement cases, selector invariance should also be high (>=85%).
- At least 4 of 5 paired model seeds should satisfy the task's overall/per-quadrant and message/control gates for a “stable task success” classification. Otherwise classify task success as partial/unstable.

These are **initial proposed Phase 2c gates**, not retroactive changes to Phase 2b. Inspect generator stats and run untrained structural tests before finalising the design, then commit `reports/phase2c-design.md` before any full training. If the agent judges a threshold incompatible with the data's mathematically expected baseline, it must explain the issue and revise the design *before* seeing any trained model result.

## 8. Phase 2c training protocol and confirmation rules

### 8.1 Recipe

Use the eligible common learning rate selected by Stage 2. If no learning rate is eligible, do not proceed to a confirmatory multi-seed training comparison. You may finish Phase 2c model/data/tests and perform an explicitly labelled engineering smoke run, then stop with a recommendation.

Use paired model seeds 42, 43, 44, 45, 46 unless another seed list is declared in the design before training. For every seed, reset Python/PyTorch seed before model construction, use the same warm-start weights and deterministic data-loader generator for the full/final-only pair, and verify that corresponding fresh model initial weights are identical before the different message-visibility schedules cause different training updates.

Keep training settings identical across full/final-only: batch size 64 (or another size only if predeclared based on memory), 20 epochs initially, AdamW weight decay `1e-2`, same loss weights, fixed eight recurrent steps, same train/validation examples. If the selected Phase 2b rate is used, it is the same for both modes. Save training logs, per-epoch metrics, and checkpoints for every run—including failures—in ignored per-run folders.

Do not select the best epoch based on the test split. Use the predeclared final-epoch acceptance method unless the design prospectively defines a validation-only checkpoint-selection procedure. Keep validation and test roles separate.

### 8.2 Test data discipline

The original Phase 2b `9999` split was already evaluated for two passing checkpoints. It is not the Phase 2c test dataset. Use the newly declared relation-task test split and do not inspect its labels/metrics until code, architecture, recipe, and all gates are locked and training/validation selection is complete.

At final evaluation, evaluate each frozen full/final-only model on the same test examples to allow matched per-example comparisons. Do not tune based on test results. If a test result triggers a code/threshold/hyperparameter change, call that test split consumed and create a new one under a documented new seed for any subsequent confirmatory result.

### 8.3 Primary comparison and statistical reporting

The primary statistic is the paired test accuracy difference:

`Delta = accuracy(full) - accuracy(final-only)`

Report it per model seed, across all seeds, and separately on the relation-disagreement subset. Also report relation-agreement subset results. Save per-example paired predictions so they can be independently re-analysed; keep them out of Git if large/sensitive, but save checksums and aggregate tables.

Predeclare an interpretation rule:

- **Evidence favouring early integration on this task:** stable task success in full mode and a positive mean paired test advantage for full mode of at least 5 percentage points, with a paired uncertainty interval that does not include zero on the disagreement subset, replicated across paired seeds. Use a cluster/paired bootstrap or an appropriate paired analysis; with only five seeds, describe uncertainty conservatively and do not oversell asymptotic p-values.
- **No measured advantage:** both modes pass and paired difference is small/uncertain. Conclude that this task does not demonstrate a benefit of early access; consider that the model can precompute both candidates.
- **Final-only wins:** report that outcome as-is; it does not negate the fact that full mode's trajectory is message-sensitive.
- **Training instability / gates fail:** report instability rather than treating it as a scientific rejection of recurrent integration.

Task-gate pass/fail and the full-vs-final-only effect are separate outcomes. Do not require a performance advantage in order to call the COMM mechanism implemented, and do not call implementation proof of a performance advantage.

### 8.4 Resource budget and stopping

Estimate time from Stage 2. Do not start work that risks corrupting existing artifacts or exhausting storage. Use a maximum budget of approximately 2 hours for the stability sweep; Phase 2c full runs should be separately budgeted using measured runtime. If the Phase 2c five-seed paired suite is forecast to exceed the available budget, complete data/model/tests, a smoke run and a clearly labelled pilot, then stop before the holdout evaluation rather than cherry-picking a partial set. If a training run fails, preserve logs and last checkpoint; fix only infrastructure errors in a new commit and rerun under a new run ID.

## 9. Required tests before training Phase 2c

Create `test_phase2c.py` or equivalent. Keep `test_system.py` and `test_phase2b.py` unchanged and run them alongside the new tests. At minimum:

1. Baseline regressions: v7.1 and Phase 2b tests still pass unchanged.
2. Exact parameter accounting for new model and existing models.
3. Input shape/layout and `adj_union == OR(R0,R1)` checks.
4. Relation generator balance, reachability/path bounds, BFS labels, target-selector mapping, determinism and hash disjointness.
5. Directed attention-mask semantics: graph query sees only self/predecessors in union graph; no future/successor leakage; no all-false query rows; COMM query is self-only.
6. Mask schedule: full mode exposes COMM at every recurrent step; final-only exposes it at the last step only; masks are not modified based on message value.
7. Graph receptive field: with COMM present, graph input at Node 0 cannot influence Node 15 before T=15 on a 15-edge line graph, while the global message is allowed to influence states at early steps.
8. Trajectory intervention: same Graph B inputs + forced `m=-1/+1`; full mode differs at intermediate steps; final-only graph-node states are bit-identical through step 7 and differ only after step 8 where expected.
9. Readout contract: task head input exactly `[batch,128]` from the terminal graph node, not concatenated with `m`, `msg_repr` or COMM state.
10. Sender/receiver checkpoint warm start: Core A strict Phase 1 state-dict load; Core B mapped weight initialization preserves output of the old encoder on equivalent union/id inputs when relation-channel columns are zero; assert every expected key/shape.
11. Binary message output and exact flip/forced-value behavior.
12. Zero-message state is exactly the learned token alone, with message-embedder/bias bypassed; do not claim zero accuracy should be chance on the whole relation-selection dataset.
13. Gradient smoke test for each mode and finite loss.
14. Dual-relation baseline has no accidental access to training labels at inference; its selector is only used after both candidate outputs are produced.
15. No test uses relation test split for model selection or training.

Run:

```bash
.venv/bin/python -m pytest -q test_system.py test_phase2b.py test_phase2c.py
```

Save the full output and only proceed to smoke/full training when tests pass. If a structural test fails, fix the model/design rather than deleting or weakening the test.

## 10. Checkpointing and reproducibility

For every run use a unique directory, e.g.:

`runs/<UTC_TIMESTAMP>_phase2c_<mode>_seed<seed>/`

Record:

- exact Git commit SHA, branch, dirty-status snapshot;
- Python, PyTorch, NumPy versions, OS, CPU thread count, available RAM and disk;
- model config, parameter count, mode, seed, LR, epochs, batch size, loss weights;
- Phase 1 checkpoint path and SHA-256;
- dataset-generation seeds, sample counts, hashes/checksums for train/validation/test index or data artifacts;
- JSONL loss/metric logs, human-readable `train.log`, metrics JSON, final/last checkpoint, SHA-256 checksums;
- exact command line, start/end UTC time, exit code and wall time.

Checkpoint saving should not depend only on gate success. During the diagnostic sweeps and Phase 2c development, save a final/last checkpoint for failed runs too, because failed optimisation is evidence. Do not confuse `last.pt` with a best checkpoint. Clearly identify checkpoint selection in metadata.

Use CPU-safe loading (`map_location="cpu"`, `weights_only=True` where supported). Never store checkpoints in Git. Back up important checkpoints and reports to durable storage outside an ephemeral VM if that storage exists; otherwise call out that an external copy is still needed.

## 11. Branching, commits, and GitHub publishing

The public repository is `gl33mer/latent-recurrent-swarm`. `gh` is already authenticated.

Suggested branch sequence:

1. Housekeeping/report-only commit on existing `experiment/phase2b-comm-token` after verification, no rewriting history.
2. New `experiment/phase2b-stability-study` branch for diagnostics/sweep/report.
3. New `experiment/phase2c-message-conditioned-reachability` branch from the reviewed Phase 2b/stability state for the new task/model/tests/report.

Before creating a branch, verify it does not already exist locally/remotely. Never overwrite an existing branch. Push with normal `git push -u origin <branch>`, never force-push. Do not automatically merge into `main`. Keep all raw runs/checkpoints ignored; commit source, tests, protocol/design docs, aggregate reports and small reproducibility metadata only.

Before each commit:

```bash
git status --short
git diff --check
git diff --stat
git diff -- .gitignore requirements-cpu.txt
```

Inspect staged files to confirm no checkpoints, secrets, `.venv`, raw run folders or swap files are staged. Run a secret scan. Confirm public reports contain no tokens/credentials/private keys. Commit with scoped messages, push, and record the commit SHA in the run report.

## 12. Required reports

Create/maintain these documents:

1. `reports/phase2b-analysis-addendum.md` — paired validation deltas, limitations, correction to causal statement about instability, artifact availability/checksums, no test reuse.
2. `reports/phase2b-stability-design.md` — locked sweep matrix, seed schedule, metrics, selection rule, stopping budget, and no-test rule. Commit before sweep if practical.
3. `reports/phase2b-stability-report.md` — all 18 results or all runs completed under budget; losses/gradients/stability; chosen common LR or “none eligible”; every failure included.
4. `reports/phase2c-design.md` — committed before full Phase 2c training; task/generator specification, feature layout, architecture, parameter count, warm-start mapping, splits/seeds, gates, baselines, comparison statistics, test-split rules.
5. `reports/phase2c-run-report.md` — engineering results, task-gate results, scientific interpretation separated; all seeds and failed runs; paired deltas; counterfactual/trajectory data; test metrics; limitations.
6. Update `WORKLOG.md` and `CHANGELOG.md` with concise chronological entries and commit links.

Reports must not say “all runs passed” when only some did. Tables include every run. Explain failed gates and whether test evaluation was performed. Distinguish the observed trajectory mechanism from a performance advantage.

## 13. Execution order and stop gates

Execute in this order, not concurrently:

### Phase A — audit and housekeeping

- Inspect repository/environment/artifacts.
- Run existing tests and save logs.
- Archive/checkpoint existing artifacts and verify hashes.
- Correct report interpretation with addendum; commit and push.

**Stop if** there is uncommitted user work you cannot safely preserve, the historical branch is unexpectedly rewritten, checkpoints/logs conflict with the report, or baseline tests fail for an unexplained reason. Report before destructive/recovery actions.

### Phase B — stability diagnostic

- Commit the predeclared sweep design.
- Add isolated instrumentation.
- Run the full paired 3-LR × 2-mode × 3-seed grid within budget, validation only.
- Generate report and apply the predeclared common-learning-rate rule.

**Stop before confirmatory Phase 2c training if** no common stable learning rate qualifies. You may still implement Phase 2c data/model/tests and run a smoke test, but do not claim a proper matched task comparison under an unstable recipe.

### Phase C — Phase 2c task/model/tests

- Create the feature/task generator and separate model module.
- Commit design spec and tests before full training.
- Run all old plus new tests; run full/final-only smoke checks.
- Run data-only baselines and verify balancing/disagreement proportions; if shortcuts dominate, revise the generator/design before training and document the new design version.
- Freeze the architecture, selected shared LR, gates, seeds, data split seeds and evaluation procedure.

**Stop if** the generator creates leakage, mask direction is wrong, COMM state can read Graph B nodes, task head sees the bit/COMM state directly, or warm-start mapping is invalid. Do not proceed by weakening invariants.

### Phase D — Phase 2c training and evaluation

- Run paired full/final-only seeds under the frozen recipe.
- Save every result, including failed runs.
- Do not evaluate the held-out relation-task test split until the protocol/config is frozen and all validation decisions are complete.
- Evaluate all final models on the same predeclared test examples once for the confirmatory comparison.
- Produce the report and push the experiment branch.

If time budget prevents the complete five-seed paired suite, report only a pilot and leave test data untouched. Do not call a partial pilot confirmatory.

## 14. Final completion checklist

Before finishing, report concisely to the owner:

- Current branch and exact Git commits pushed.
- Whether repository/main/history was left safe and whether any user changes were preserved.
- System test outcomes, stability sweep completion, per-rate pass counts and selected LR (or no eligible LR).
- Whether Phase 2c was smoke-only, pilot, or full confirmatory run.
- Full table of seeds/modes and all failures.
- Overall and quadrant performance, counterfactual selected-target accuracy, disagreement-subset switch consistency, zero-message controls, message probe and per-step trajectory sensitivity.
- Paired full-minus-final-only effects by seed and aggregate, with uncertainty/limitations.
- Paths to reports, logs, JSON metrics, checkpoints and SHA-256 checksums.
- Exact test split seed and whether it remained untouched until final evaluation.
- Any discrepancy between source report and actual artifacts.
- Recommended next action based on evidence, not the desired outcome.

**Final principle:** The goal is not to make full mode win. The goal is to test, with controlled data and honest controls, whether early access to a one-bit message produces a measurable computational or task-performance advantage over late access. If it does, quantify the conditions. If it does not, report why the task may be solvable by parallel computation. If optimisation remains unstable, stabilise it before making either claim.

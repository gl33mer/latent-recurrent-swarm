# Agent Mission: Preserve v7.1 Baseline, Capture Model Artifacts, and Implement Phase 2b COMM-Token Experiment

**Repository:** `https://github.com/gl33mer/latent-recurrent-swarm`  
**Expected local directory:** `/workspace/latent-recurrent-swarm`  
**Owner/GitHub account:** `gl33mer`  
**Current baseline:** Latent Recurrent Swarm v7.1, Phase 1 + Phase 2a  
**Next research milestone:** Phase 2b — recurrent integration of a received bit through a one-way COMM token  
**Machine:** Debian 12, x86-64, 4 vCPUs, approximately 8 GiB RAM, CPU-only PyTorch  
**Execution style:** Work autonomously, inspect before modifying, preserve every artifact, test in stages, commit and push the completed branch. Do not ask routine clarifying questions; make conservative, documented decisions. Stop only if blocked by missing credentials, unrecoverable data, conflicting repository state, or a safety/security issue.

---

## 0. Role and objective

You are the autonomous ML research engineer responsible for the next iteration of the Latent Recurrent Swarm project. Your job is not merely to edit code. You must preserve the successful baseline, improve its artifact provenance, implement a tightly controlled Phase 2b architecture, execute the tests and feasible experiments, and leave the repository in a state that another researcher can understand and reproduce.

Work in two distinct phases:

1. **Baseline housekeeping and provenance.** Inspect the existing repository and VM; correct stale documentation; locate and verify the Phase 1 checkpoint; preserve a Phase 2a trained checkpoint. If the exact original Phase 2a checkpoint was not saved, perform a clearly labelled reproducibility rerun from the frozen baseline code and same seeds to produce a checkpoint. Do not pretend a rerun is the exact original artifact.
2. **Phase 2b experiment.** Implement a new COMM-token model on a new branch, preserving the v7.1 code path as the baseline. Establish that the bit can alter Core B's recurrent graph-node states, not only an output head. Compare full-step COMM access with a matched final-step-only control. Run tests before training, run a smoke test, then run the planned training/evaluation and document the actual outcome, including failure or inconclusive results.

Do not silently enlarge the claim. Phase 2b uses a *single bit sent once from Core A before Core B's eight recurrent steps*. The COMM token makes that fixed message accessible inside Core B's recurrence. It is not yet asynchronous, bidirectional, multi-round communication between two evolving cores. Do not call it that.

## 1. Current project context and established v7.1 result

The live repository is public at `https://github.com/gl33mer/latent-recurrent-swarm`. Verify the actual remote state before acting; Git is the source of truth, not these hard-coded commit notes.

The successful v7.1 experiment used:

- `LatentReasoner`: 137,476 parameters.
- `LatentSwarm` v7.1: 292,427 parameters.
- 16-node forward DAGs with potential edges `i -> j` when `j-i ∈ {1,2,3}`.
- A parameter-tied graph-masked recurrent block, float32, eight recurrent steps for the trained reachability tasks.
- Phase 1: `b = Reach(0 -> 15)`.
- Phase 2a: two separately recurrent cores; Core A sends `m ∈ {-1,+1}` using an STE binarizer; Core B reasons about Graph B independently; a nonlinear receiver MLP performs final decision fusion for `Y = b_A XOR b_B`.
- Phase 2b was explicitly out of scope for v7.1.

Reported first-run metrics, to preserve as historical results:

- `pytest`: 7/7 passed.
- Phase 1: T=1 validation 50.0%; T=8 validation 100.0%; T=8 held-out test 100.0%.
- Phase 2a validation: 97.50% overall; quadrant accuracies 96.4%, 97.2%, 98.4%, 98.0%; flip residuals 0.00% for `b_B=0` and 0.41% for `b_B=1`; zero-message accuracy 48.60%; message probe 97.80% overall (97.20% on `b_A=0`; 98.40% on `b_A=1`).
- Phase 2a held-out test: 97.00% overall; quadrant accuracies 96.8%, 94.8%, 98.8%, 97.6%; flip residual 0.00% in both `b_B` groups; zero-message accuracy 50.50%; message probe 97.10% overall (95.80% on `b_A=0`; 98.40% on `b_A=1`).
- Phase 1 checkpoint reported SHA-256: `ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e`.
- First run ID: `20261016T211522Z`.
- Original source tag: `v7.1-pre-run`; original source commit noted in `WORKLOG.md` as `77c2d1a`; report commit noted as `d10b0c2`. Verify whether the remote contains later commits and use its actual history.

These historical figures are observations from one seed, not proof of robustness. Do not edit old results to make them agree with new reruns. Add an appendix or a separate report section for any reproduction rerun.

## 2. Non-negotiable research and repository rules

1. Do not alter v7.1 acceptance thresholds to get a pass.
2. Do not delete, reset, overwrite or force-push away existing data or commits. Never use `git reset --hard`, `git clean -fdx`, forced push, or destructive checkout operations.
3. Inspect `git status`, current branch, remote configuration, tags, and recent log before writing any files or switching branches.
4. Preserve unrelated user files and pre-existing changes. If the worktree is unexpectedly dirty, archive a binary patch and copies of untracked files into an ignored safety directory under `runs/` before proceeding; do not commit unrelated files. If you cannot preserve them safely, stop and report the blocker.
5. Keep model checkpoints, run logs, full environment dumps and other bulky artefacts out of Git. Store them under `runs/` or another ignored artefact location. Do not commit secrets, access tokens, private keys, `.venv`, or checkpoints.
6. The repository is public following the owner's earlier explicit request. Do not change its visibility. Do not put private machine details, credentials, tokens, or secrets in public reports.
7. Do not claim a test or training run passed unless it actually ran and its output was recorded. Distinguish static review, unit-test results, training acceptance gates, and held-out test results.
8. Preserve the v7.1 implementation. Phase 2b must be new code, preferably `model/swarm_comm.py` plus a new trainer, not an in-place rewrite of the v7.1 path. Keep v7.1 available as the baseline.
9. Do not tune against the held-out test set. Use train and validation for development, lock configuration before evaluating test data, and record if any test results were consulted during changes.
10. Be candid if the experiment is negative, underpowered, failed, or inconclusive. Do not force a positive result.

## 3. Stage A — Inspect and establish the actual repository state

Start here. Do not assume the report's commit hashes are still HEAD.

```bash
cd /workspace/latent-recurrent-swarm
pwd
uname -a
cat /etc/os-release || true
nproc
free -h
df -h .
git status --short --branch
git remote -v
git log --oneline --decorate -8
git tag --list
git fetch origin --tags
git status --short --branch
gh auth status
gh repo view gl33mer/latent-recurrent-swarm --json name,visibility,defaultBranchRef,url
```

Inspect `README.md`, `PROJECT_SPEC.md`, `WORKLOG.md`, `CHANGELOG.md`, `reports/first-run-report.md`, `.gitignore`, `requirements-cpu.txt`, `model/core.py`, `model/swarm.py`, `data/graph_task.py`, `test_system.py`, `train_unit.py`, and `train_swarm.py`.

Check whether the following exist and are ignored rather than tracked:

- `.venv/`
- `runs/20261016T211522Z/`
- `checkpoint_phase1.pt`
- any saved Phase 2a / swarm checkpoint
- `runs/*/checkpoint_phase1.pt`
- run logs and environment snapshots

Use `git check-ignore -v PATH` where appropriate. Do not assume a file is missing until you search the exact run directory and root.

The runtime is expected to be Debian 12 with Python 3.11, CPU-only PyTorch and NumPy already installed. Reuse the existing environment if it is healthy; do not reinstall or upgrade packages unnecessarily. Activate `.venv` if present. Verify:

```bash
source .venv/bin/activate
python --version
python - <<'PY'
import torch, numpy
print('torch:', torch.__version__)
print('numpy:', numpy.__version__)
print('CUDA available:', torch.cuda.is_available())
print('threads:', torch.get_num_threads())
PY
python -m pytest -q test_system.py
```

If `.venv` is absent or damaged, recreate it without removing the old directory first: rename/back up the old environment if it exists, then use the project's `requirements-cpu.txt` and official CPU PyTorch wheel index. Keep CPU-only execution and four PyTorch intra-op threads. Do not switch to CUDA or add GPU dependencies on this VM.

## 4. Stage B — Baseline housekeeping and artifact provenance

### B1. Correct the first-run report's repository visibility

The repository is currently public; the work log describes the explicit visibility change, but `reports/first-run-report.md` reportedly still says `private`. Verify current visibility via `gh repo view` and the actual text. Update the report to say public and briefly state that visibility was changed at the owner's request after the baseline run. Do not rewrite or erase the historical metrics. Update any other stale metadata only when verified.

### B2. Verify and preserve Phase 1 checkpoint

Look for the checkpoint in the workspace root and first run directory. Verify SHA-256. If it matches the recorded hash above, record that fact. If it does not, do not overwrite it; compare paths, size, and hashes and explain the discrepancy in the work log.

Ensure a usable copy is stored at a documented ignored path such as:

`runs/20261016T211522Z/checkpoint_phase1.pt`

The Phase 2a trainer expects `checkpoint_phase1.pt` in the project root. If the canonical verified artifact is elsewhere, copy it to the expected runtime path without deleting the original. Make sure this path stays Git-ignored.

If no matching checkpoint is available anywhere, rerun `train_unit.py` from the frozen v7.1 baseline with its original seed/configuration, save the resulting checkpoint and metadata in a new unique run directory, and clearly label it as a reproduction artifact rather than the original checkpoint. Preserve the expected first-run checksum in documentation as a reference; do not imply a regenerated file is byte-identical unless its hash actually matches.

### B3. Preserve a Phase 2a trained model checkpoint

The original `train_swarm.py` may not save a final swarm checkpoint. Search for a saved artifact first. If a Phase 2a checkpoint exists with known provenance, preserve it and record its SHA-256, source commit, training seed, and associated metrics.

If no trained Phase 2a checkpoint exists, it cannot be reconstructed from reported aggregate metrics alone. Produce a **clearly labelled Phase 2a reproducibility rerun** from the frozen v7.1 baseline and same training/data seeds. Do not call this the exact original trained state.

To produce the artifact safely:

1. Preserve/tag the current v7.1 source state; do not modify `v7.1-pre-run`.
2. Make a small artifact-capture change to `train_swarm.py` on a housekeeping branch (for example `chore/preserve-v7.1-artifacts`): add an optional `--artifact-dir` argument or equivalent. It may only create a destination directory and save metadata/state after the existing validation gates have passed. Do not change model computation, initialisation order, optimiser, training loop, data seeds, loss weights, epochs, batch size, or acceptance thresholds.
3. Ensure the added save path does not introduce random draws or change computation. Save `swarm.state_dict()` to `checkpoint_phase2a.pt` and record SHA-256, the source commit, all config values, validation metrics and test metrics in a JSON or Markdown metadata file under the run directory.
4. Run the reproducibility rerun under a new run ID; preserve the old run logs unchanged. Do not overwrite the original run directory.
5. Compare rerun metrics with the historical report. Explain deviations, if any, without changing historical reported numbers.

Do not commit any checkpoint. Do commit the small reproducibility/artifact-save code change if it works and is useful. Keep the original baseline tag immutable.

### B4. Repository housekeeping commit

Commit only related housekeeping files, the report visibility correction, and any tested checkpoint-persistence option. Update `CHANGELOG.md` and `WORKLOG.md` with the actual change. Keep the first-run report metrics unchanged. Record the exact source commit used for the reproduction rerun.

Push this housekeeping commit to `main` only after tests pass. Do not force-push. Then create and switch to a separate experiment branch:

```bash
git switch -c experiment/phase2b-comm-token
```

If that branch already exists remotely, inspect it and continue safely from its latest commit rather than overwriting it. Push the eventual Phase 2b implementation to this branch; do not merge it into `main` automatically.

## 5. Phase 2b research scope: the claim we are testing

### Main question

Can a single bit produced by Core A at the end of its own recurrent computation causally modulate Core B's internal graph-node representations throughout Core B's recurrent loop, while retaining graph-masked graph-to-graph attention and the binary communication bottleneck?

### Strictly in scope

- Core A processes Graph A for eight graph-masked recurrent steps.
- Core A sends exactly one scalar `m ∈ {-1,+1}` using the existing STE mechanism.
- Core B processes Graph B for eight recurrent steps.
- A dedicated COMM token/register is initialised with a learned token embedding plus an embedding of `m`.
- Core B's graph-node tokens may read the COMM token during each recurrent step.
- The COMM token/register **must not read any graph-node tokens**. It is a one-way message workspace, not a global pooling token. This restriction prevents Graph B node information from taking a shortcut through the COMM token and bypassing graph-local receptive-field paths.
- Graph-to-graph attention remains masked strictly to self and directed incoming graph neighbours. The only global route is from the exogenous COMM register to graph nodes, which is the intended message path.
- The final parity head receives only the terminal graph-node state `z_B[15]`. It must not concatenate `m`, the COMM state, or `msg_embedder(m)` into the final decision head. The incoming bit has to affect the graph-node state through the recurrent stack to influence the output.
- Phase 1 `LatentReasoner` and Phase 2a `LatentSwarm` remain available and unchanged as functional baseline implementations. Shared core weights may be reused; avoid altering the checkpoint-compatible `LatentReasoner` state layout.

### Strictly out of scope

- No language tokens, autoregressive generation or natural-language intermediate reasoning.
- No additional communication bits, raw Core A states, or Graph A features exposed to Core B.
- No direct final-readout fusion of the bit in the Phase 2b model.
- No bidirectional graph-to-COMM attention.
- No multi-round Core A/Core B conversation, alternating execution, or claims of synchronous co-evolution. A sends one terminal bit once; B then performs its recurrent computation. Multi-round exchange is a later milestone.
- No change to the v7.1 benchmark thresholds just to obtain a pass.

### Critical scientific limitation

The existing XOR task can be solved by late decision fusion, as Phase 2a has already shown. Therefore, Phase 2b on the same task can test *mechanistic integration* and causal state modulation, but by itself cannot prove that recurrent integration is necessary or improves performance over late fusion. A result where the full-step model and a final-step-only model both solve XOR is a valid finding: this task does not require the message to be used early. Do not claim otherwise.

Record this limitation in the Phase 2b design and result reports. Include a proposal for a follow-up message-conditioned graph task (for example, one in which the bit selects which edge-relation graph Core B must traverse) to test whether message-conditioned recurrence provides a measurable advantage beyond final decision fusion. If you implement such a task in this iteration, give it separate labels, controls and acceptance criteria; do not conflate it with the original XOR benchmark or silently replace v7.1.

## 6. Phase 2b architecture specification

Create a new module, preferably `model/swarm_comm.py`, with a new class such as `LatentCommSwarm`. Do not overwrite `model/swarm.py`.

### 6.1 Core A

- Reuse the v7.1 Core A reasoner architecture and encoder.
- Run for `steps_A=8` on Graph A using the existing directed graph mask.
- Compute `aux_logits_A` from terminal Node 15.
- Compute `raw_m = msg_encoder(z_A[15])`.
- Convert to strict ±1 forward values with the existing `binarize_ste` function.
- Train `raw_m` with the existing BCE-with-logits supervision against `b_A`.
- `flip_message=True` negates the message before it enters the COMM register.

### 6.2 COMM register and Core B

Use a 17-token sequence for Core B: the existing 16 graph-node tokens plus one COMM token at index 16.

One acceptable initialisation is:

```python
comm_state = learned_comm_token.expand(batch_size, 1, d_model) + msg_embedder(m).unsqueeze(1)
z_B = torch.cat([core_B.encoder(x_B), comm_state], dim=1)
```

Here `learned_comm_token` is a new trainable parameter of shape `[1, 1, d_model]`. The message remains one scalar; its learned embedding is an internal representation, not extra channel bandwidth. Keep the token's embedding architecture identical between full-step and final-step-only ablation variants.

Build a per-example boolean attention mask with shape broadcast-compatible with `[batch, heads, 17, 17]`, matching the current attention code's expectations:

- For graph query node `j ∈ [0,15]` and graph key node `i ∈ [0,15]`, allow attention iff `i == j` or graph edge `i -> j` exists. Use the existing orientation: if `adj[b,i,j] == 1`, then query `j` may attend to key `i`.
- For every graph query `j`, permit attention to the COMM key at index 16 while COMM visibility is enabled for that recurrent step.
- For the COMM query at index 16, permit attention **only** to COMM key 16. Do not permit it to attend to any graph node.
- Ensure each attention query has at least one allowed key. The COMM self-loop provides this for its query.
- Never use an unmasked all-to-all attention fallback.

The same parameter-tied recurrent `GraphAttentionBlock` may be applied over all 17 tokens. Core B's graph-node-to-graph-node communications remain topologically masked; message-to-node access is the intended global exception.

Use explicit, tested step masks:

- **Full-step COMM:** graph queries can attend to COMM on steps 1–8.
- **Final-step-only control:** graph queries cannot attend to COMM on steps 1–7 and can attend on step 8 only. The COMM register can maintain its own state, but because it cannot read graph nodes it cannot carry graph information around the mask.

Train a separate final-step-only model with the same optimiser, training seed, number of epochs, data, auxiliary losses and parameterisation as the full-step model. Do not treat an inference-time mask intervention on the full model as a fully matched trained control; it may be reported as an extra causal probe only.

### 6.3 Readout and losses

- Auxiliary Core B reachability logits come from `z_B[15]` via the existing decoder.
- The Phase 2b parity head takes **only** `z_B[15]` with width `d_model` as its input. It must not consume `m`, a concatenated message embedding, or the final COMM token state.
- A nonlinear terminal head such as `LayerNorm(128) -> Linear(128,64) -> GELU -> Linear(64,2)` is acceptable. Count its parameters programmatically and document the count; do not copy Phase 2a's count without recalculation.
- Use the existing loss pattern unless there is a measured reason to change it: `L = L_parity + 0.5*L_aux_A + 0.5*L_aux_B + 0.5*L_message_BCE`.
- Keep the existing eight-step data/task definition for the first mechanistic comparison. If a message-conditioned task is added, it requires its own explicit loss/task spec.

### 6.4 Zero-message control semantics

When `zero_message=True`, replace the **message contribution** by a zero tensor before adding it to the learned COMM token; do not accidentally let a linear-layer bias encode the original bit. The constant learned COMM token may remain, since it is not sample-specific information from Core A. Document that this ablation removes the information-bearing message component; it does not delete the workspace token itself.

### 6.5 Diagnostics required in the new model

Provide an optional diagnostics/trace mode that records Core B's graph-node states after every recurrent step without changing the default forward result. Use it to measure whether the same Graph B has different intermediate representations when `m=+1` versus `m=-1`. Do not make diagnostic retention mandatory for ordinary training if it causes excessive RAM use.

The parity logits in Phase 2b must depend on the terminal graph-node state only. Add a test or explicit structural check ensuring the terminal head input shape is `[batch, d_model]`, not `[batch, 2*d_model]`.

## 7. Phase 2b tests: required before training

Add tests in a new file such as `test_phase2b.py` or extend the suite while keeping the baseline tests unchanged. The Phase 2b test suite must include:

1. **Baseline regression:** all v7.1 `test_system.py` tests still pass without modification to their thresholds.
2. **Parameter accounting:** assert the new model's actual trainable parameter count against a documented computed value; verify the value from the instantiated PyTorch model and update the spec.
3. **COMM mask semantics:** assert graph-to-graph attention equals self plus direct predecessors; graph queries may read COMM when enabled; the COMM query cannot read any graph token; no query has an all-false row.
4. **Graph receptive field with COMM present:** using a line graph and a fixed message, gradient from Graph B's Node 0 input to Node 15's latent must be exactly zero for `T≤14` and nonzero at `T=15`. This tests graph-input information propagation, not message propagation. Because COMM cannot read graph nodes, it cannot act as a graph-global shortcut.
5. **Message can affect graph trajectory:** on a fixed graph and identical model, compare forward traces for `m=+1` and `m=-1`. At least one ordinary graph-node state at an intermediate step must differ. Prefer checking that Node 15 or its ancestor states change at more than one step, and report the measured difference. The test must not merely compare final logits.
6. **No late-readout fusion:** verify the parity head receives only the terminal 128-D graph-node state and receives no direct message tensor or COMM state. Keep the Phase 2a model as the explicit late-fusion reference.
7. **Binary channel contract:** verify message values are strictly ±1, that `flip_message` returns exact sign inversion, and with a deliberately balanced/seeded test batch that both signs occur. If a random untrained batch does not produce both signs, use deterministic controlled raw message values or a trained checkpoint rather than asserting a random property.
8. **Zero-message intervention:** verify the injected message component is exactly zero under ablation and confirm on a trained model that accuracy approaches chance. Do not claim the whole learned COMM token is a zero vector.
9. **Gradient flow:** perform a small backward pass and report any parameter with `grad is None` or an all-zero gradient tensor. Do not automatically fail for a mathematically legitimate zero gradient on a particular tiny batch unless the condition is understood; investigate unused parameters.
10. **Dataset integrity:** retain train/validation/test pairwise graph-hash disjointness, internal duplicate checks, graph/label/BFS consistency, reachable path bounds `5≤d≤8`, endpoint degree invariants, and XOR balance. Do not let the new tests silently replace existing integrity tests.
11. **Final-step-only mask:** directly test that the COMM key is inaccessible to graph queries in steps 1–7 and accessible at step 8 only.
12. **Checkpoint compatibility:** verify Phase 1 state dict loads into both new cores without missing/unexpected keys for the reused reasoners. New COMM and parity-head parameters should be the only expected newly initialised parameters.

Keep fast unit tests fast. Use deterministic CPU tests; avoid long optimization loops in the required quick system suite where possible. A separate test may run a small noisy XOR readout smoke experiment.

## 8. Phase 2b experimental plan

Create a new trainer, for example `train_swarm_comm.py`, with configuration recorded in each run. Provide a CLI for at least `--mode full|final-only`, `--seed`, `--epochs`, and `--run-dir`. Default training should use the established CPU-friendly settings, but all values must be logged.

### Step 1 — smoke test

Run a tiny forward/backward test with a small batch and 1 training epoch or a few batches. Confirm:

- Shapes are correct for the 17-token receiver.
- The COMM mask is used in the correct direction.
- Loss is finite.
- Gradients flow through the message embedder, Core B recurrent block, terminal head, Core A message encoder, and auxiliary decoders.
- `m` is binary in the forward pass.
- There is no direct late-fusion path into the final Phase 2b parity head.

Do not treat smoke-test scores as research results.

### Step 2 — train the main COMM model

Use the same data-generation rules and split seeds as Phase 2a where possible (train 42, validation 1337, test 9999) to make comparison meaningful. Start from the verified Phase 1 checkpoint for both cores. Train with `T_A=8`, `T_B=8`, the same batch size and optimiser settings as Phase 2a initially. Use an explicit training seed and record it. Preserve every run independently.

At minimum, run seed 42 for the first apples-to-apples comparison. Prefer three independent training seeds (42, 43, 44) for the new model if CPU time and storage permit. If only one is completed, label Phase 2b exploratory and do not imply seed robustness.

### Step 3 — train a matched final-step-only COMM control

Train the same new architecture and parameterisation, but with COMM invisible to graph queries during steps 1–7 and visible during step 8 only. Keep the message head, workspace parameters, loss, training data, number of epochs and optimiser otherwise matched. Use the same seeds for paired comparisons.

This tests whether allowing access across the full trajectory changes learned performance relative to making the message available only at the last recurrent update. Do not require the full-step model to exceed the final-step-only model on the simple XOR task; XOR mathematically permits late fusion, so parity accuracy alone cannot demonstrate the necessity of earlier message use. Interpret the comparison honestly.

### Step 4 — evaluate all controls

On validation, calculate and report:

- Overall parity accuracy.
- Accuracy for all four `(b_A,b_B)` quadrants.
- Message-probe accuracy overall and separately for `b_A=0` and `b_A=1`.
- Message-flip residual among samples that were correct before flipping, separately for `b_B=0` and `b_B=1`.
- Zero-message accuracy.
- Core B trajectory sensitivity: distances between traces for opposite messages at each recurrent step, overall and for both `b_B` groups.
- Full-step versus final-step-only comparison.

Use the v7.1 Phase 2a gates as the reference minimum for the *existing XOR task*, where appropriate: overall ≥82%, each quadrant ≥75%, flip residual ≤15% in both `b_B` groups, zero-message accuracy ≤53%, overall message probe ≥95% and each `b_A` subgroup ≥90%. Apply these as explicit gates only if the implementation's metrics have the same definitions and denominator semantics. If any condition's metric is undefined because its conditioning set has zero samples, fail with a clear diagnostic; do not silently divide by one or treat an empty group as a pass.

In addition, require evidence that the trained COMM model's ordinary graph-node trajectory is message-sensitive. Report per-step hidden-state difference metrics and include a deliberately fixed-input counterfactual comparison. Do not treat the bit's effect on final logits alone as proof of recurrent integration.

### Step 5 — final held-out test

Once the architecture and training parameters are fixed, evaluate test seed 9999 exactly once per final selected model. Report the same quadrant/control/trajectory metrics where appropriate. If test results prompt a change, mark the test as consumed for development, create a new untouched test split with a new documented seed, and state that explicitly.

## 9. Phase 2b acceptance and interpretation rules

The experiment can have three distinct outcomes:

1. **Engineering success:** code, masks, checkpoints, training and tests work.
2. **Task success:** the model passes the predefined XOR accuracy and communication controls.
3. **Scientific finding:** measurements demonstrate whether and how the message changes Core B's recurrent graph-node trajectory, and whether full-step visibility offers any measured benefit over final-step-only access.

Do not collapse these outcomes into one “PASS.” A model may pass task accuracy yet show that the current XOR benchmark does not benefit from recurrent message integration. That is a valid and useful negative result.

If full-step and final-step-only models have statistically indistinguishable parity performance and no compelling task-specific difference, state that the XOR benchmark is insufficient to test whether ongoing access is useful. Recommend a follow-up *message-conditioned graph reasoning* benchmark in which the bit changes which transition relation/edge type or subproblem Core B must solve during recurrent computation, and define a late-fusion control. Do not claim that a COMM token alone establishes collaborative reasoning.

Do not claim a generic advantage over feedforward models unless a meaningful parameter/compute-matched feedforward or non-recurrent graph baseline is implemented and evaluated. The current project first establishes a recurrent-mechanism baseline, not SOTA.

## 10. Reproducibility, outputs and run management

Each new run should have a unique UTC ID and a directory such as:

`runs/<UTC_RUN_ID>/`

Save there:

- Exact Git commit hash and working-tree status.
- Python, PyTorch, NumPy and pytest versions; `pip freeze`; OS, CPU count, RAM snapshot and thread settings.
- Full command lines, training mode, seed, dataset seeds, batch size, epochs, learning rate, weight decay, losses and threshold values.
- Test log, smoke-test log, training stdout/stderr, validation metrics and final held-out metrics.
- Model checkpoint(s), SHA-256 checksums and metadata JSON/Markdown.
- Any trajectory-sensitivity summaries.

Do not commit files beneath `runs/` or model checkpoints. Commit concise, reproducible reports with key metrics, exact artefact paths and hashes. Ensure `.gitignore` covers `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.py[cod]`, `*.pt`, `runs/`, secrets and editor swap files where appropriate. Do not add blanket ignore rules that hide source reports.

Keep CPU execution and four intra-op PyTorch threads. Do not add multiprocessing, CUDA, mixed precision or complicated acceleration until measured bottlenecks justify it.

## 11. Documentation to create/update

Update `README.md` to retain the v7.1 baseline description and add a clearly labelled Phase 2b section, not replacing the baseline claims.

Create:

- `reports/phase2b-design.md` — the hypothesis, model/mask diagram or equations, why COMM is one-way/read-only, the distinction between graph-to-graph reachability and message-to-node broadcast, the parity-task limitation, predeclared controls and acceptance criteria.
- `reports/phase2b-run-report.md` — only after running; include exact source commit, run IDs, environment, parameter counts, tests, training metrics, all subgroup/control metrics, trajectory sensitivity, checksums and limitations.
- `WORKLOG.md` additions — chronologically record housekeeping, artifact provenance, branch/commit information and actual decisions.
- `CHANGELOG.md` additions — list files and behaviour changes.
- Update `reports/first-run-report.md` only to fix verifiable stale metadata (especially public repository visibility) and, if needed, add a brief link to this follow-up. Do not rewrite the historical v7.1 metrics.

For diagrams, textual equations are sufficient; do not introduce generated diagram assets without need.

## 12. Git branch and push protocol

Use two commit stages:

1. **Housekeeping branch / main update:** correct report visibility, capture baseline artifact persistence, and documentation. Keep model behaviour unchanged. Run `compileall`, the original `test_system.py` and any relevant regression tests. Commit only intended files and push normally to `main`; never force-push. Ensure original tag `v7.1-pre-run` is untouched.
2. **Phase 2b branch:** create/use `experiment/phase2b-comm-token` from the updated main. Commit the new COMM implementation, tests, trainer and design docs in coherent commits. Run all checks. Push the branch to `origin`. Do not merge Phase 2b into `main`, do not create or move the baseline tag, and do not delete older branches/tags.

Before every commit:

```bash
git status --short
git diff --check
git diff --stat
git diff --cached --stat
git diff --cached --name-only
```

Inspect the staged diff to ensure no checkpoints, run logs, environment directories, unrelated user files or credentials are included. After pushing, verify the remote branch/commit using `git ls-remote` or `gh`.

## 13. Final report back to the owner

When done, produce a concise but complete final summary with:

1. Baseline housekeeping performed and any unresolved checkpoint/provenance issues.
2. Current `main` commit and immutable baseline tag status.
3. Phase 2b branch name and pushed commit hash.
4. Exact environment and dependency versions.
5. System-test and Phase 2b-test results.
6. Parameter counts and architecture/mask rules.
7. Training seed(s), per-seed results, all parity quadrants, flip residuals, zero-message metrics, message probes and trajectory sensitivity.
8. Full-step versus final-step-only comparison, including if the benchmark does not discriminate between them.
9. Held-out test results and whether the test set remained untouched during tuning.
10. Run directories, checkpoint paths and SHA-256 checksums.
11. Anything that failed, was skipped, or remains scientifically inconclusive.
12. Direct links to the pushed branch and report.

Do not claim completion if only the code was written. Separate “implemented”, “unit-tested”, “trained”, “acceptance gates passed” and “independently reproduced” status.

## 14. Start now

Begin with Stage A: inspect the real repository, current branch/remote state, dirty files, tags, environment, and existing run artifacts. Then work through the stages in order. Preserve the successful v7.1 baseline, do not weaken any gate, and leave all experiments reproducible and clearly labelled.

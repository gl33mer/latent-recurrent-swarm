# Work Log & Findings Summary — Latent Recurrent Swarm v7.1 Baseline

**Author:** autonomous agent run (pi coding agent), executed for owner `gl33mer`
**Date:** 2026-10-16 (UTC)
**Repository:** https://github.com/gl33mer/latent-recurrent-swarm
**Branch:** `main` · **Source commit/tag:** `77c2d1a` / `v7.1-pre-run` · **Report commit:** `d10b0c2`
**Run ID:** `20261016T211522Z` · **Run dir:** `runs/20261016T211522Z/` (Git-ignored)
**Visibility note:** repository created private per v7.1 spec; switched to **public** at the owner's explicit request for external review (see `CHANGELOG.md`).

---

## Part 1 — Summary of Findings

### What was built

The v7.1 baseline exactly as specified: a parameter-tied latent recurrent reasoner with
graph-masked topology attention (`LatentReasoner`, 137,476 params) evaluated on directed
16-node DAG reachability (Phase 1), and a two-core swarm (`LatentSwarm`, 292,427 params)
solving distributed parity `Y = b_A XOR b_B` through a single forward-pass binary message
`m ∈ {−1, +1}` (straight-through estimator) combined with Core B's terminal latent via a
nonlinear receiver-side MLP head (Phase 2a). Phase 2b (COMM token / recurrent message
integration) was **not** implemented, per scope.

### Headline results (seed 42, single run)

| Stage | Metric | Measured | Gate | Verdict |
|---|---|---|---|---|
| System tests | pytest suite | 7/7 passed (4.05 s) | all pass | PASS |
| Phase 1 | T=1 validation accuracy | **50.0%** | ≤ 56.5% | PASS |
| Phase 1 | T=8 validation accuracy | **100.0%** | ≥ 90.0% | PASS |
| Phase 1 | Held-out test T=8 | **100.0%** | reported | — |
| Phase 2a | Connected validation accuracy | **97.50%** | ≥ 82.0% | PASS |
| Phase 2a | Quadrants (0,0)/(0,1)/(1,0)/(1,1) | 96.4 / 97.2 / 98.4 / 98.0% | each ≥ 75.0% | PASS |
| Phase 2a | Flip residual b_B=0 / b_B=1 | 0.00% / 0.41% | each ≤ 15.0% | PASS |
| Phase 2a | Zero-message accuracy | **48.60%** | ≤ 53.0% | PASS |
| Phase 2a | Probe overall / b_A=0 / b_A=1 | 97.80 / 97.20 / 98.40% | ≥95 / ≥90 / ≥90 | PASS |
| Phase 2a | Held-out test (all counterparts) | connected 97.00%; quads 96.8/94.8/98.8/97.6%; flip 0/0%; zero 50.50%; probe 97.10/95.80/98.40% | reported | — |

### Interpretation

1. **Phase 1:** at T=1 the model performs at chance (50.0%), consistent with the empirical
   local-region lookup ceiling (52–56.5%); at T=8 it reaches 100% validation and test
   accuracy. This supports the Phase 1 claim: graph-masked recurrent computation propagates
   reachability information across the DAG over recurrent steps, with receptive field
   expanding with depth (also verified analytically by the T=15 chain-gradient test).
2. **Phase 2a:** the strict 1-bit channel is *used*: zeroing the message collapses parity
   accuracy to 48.60% (≈ chance), while the message probe recovers Core A's bit at 97.80%,
   and flipping the bit breaks almost all previously-correct predictions (residual ≤ 0.41%).
   All four `(b_A, b_B)` quadrants pass ≥ 75%, so the result is not driven by one quadrant.
3. **What this does NOT show:** Core B's recurrent trajectory is never modulated by Core A's
   message (the message enters only the final readout). This is decoupled reasoning with
   one-bit late decision fusion — the Phase 2a baseline. Evidence for recurrent message
   integration requires the Phase 2b COMM-token experiment. Also, one seed ≠ seed
   robustness, and a held-out test pass after fixed training does not rule out all leakage
   concerns — though train/val/test graph-hash disjointness was verified.

---

## Part 2 — Detailed Work Log (chronological)

All times approximate UTC, 2026-10-16. Working directory: `/workspace/latent-recurrent-swarm`.

1. **Read the charter** (`latent.md`, 1205 lines). Identified scope (v7.1 only), non-negotiable
   rules (private repo, no threshold relaxation, no destructive git ops, no committed
   checkpoints/secrets), and the Part D test-hardening checklist.
2. **Environment inspection:** `pwd`, `uname -a`, `/etc/os-release` → Debian 12 (bookworm),
   kernel 6.18.55, x86_64; `nproc` → 4; `/proc/meminfo` → ~8 GiB; `df -h .` → 20G volume, 3% used;
   `git 2.39.5`, `gh 2.23.0` authenticated as `gl33mer` (scopes: gist, read:org, repo, workflow).
   **`python3` was absent.** Directory contained only `latent.md` and a vim swap file
   (`.latent-recurrent-swarm-prompt.txt.swp`) — pre-existing files preserved.
3. **Privilege check:** running as root with `apt-get` available → used the spec's sanctioned
   fallback: `apt-get install python3 python3-venv python3-pip` → **Python 3.11.2**.
4. **Virtual env:** `python3 -m venv .venv`; pip upgraded to 26.2.1.
5. **PyTorch CPU-only:** `pip install torch --index-url https://download.pytorch.org/whl/cpu`
   → **2.14.1+cpu**. Verified: `CUDA available: False`, CPU device, 4 intra-op threads.
   Installed `pytest` separately. **NumPy warning detected** — canonical code calls
   `Tensor.numpy()` (`_hash_graph`, balancing test) and torch does not bundle numpy →
   installed numpy (interop verified). Dependency-only fix; documented in `CHANGELOG.md`.
6. **Canonical file creation:** extracted the v7.1 source blocks directly from `latent.md`
   with a small parser script (PROJECT_SPEC.md, .gitignore, requirements-cpu.txt,
   model/core.py, model/swarm.py, data/graph_task.py, test_system.py, train_unit.py,
   train_swarm.py) to avoid transcription drift; created empty `model/__init__.py`,
   `data/__init__.py`; wrote README.md manually (its spec block contains nested code fences);
   wrote `CHANGELOG.md` documenting deviations; appended numpy to `requirements-cpu.txt`;
   added `*.swp` to `.gitignore`.
7. **Pre-run consistency checks (Part D):**
   - `python -m compileall -q …` → clean.
   - Parameter counts verified: `LatentReasoner` = **137,476**, `LatentSwarm` = **292,427** (exact).
   - Gradient-flow smoke test on one 8-sample batch: **no parameter with `grad is None`, no
     all-zero gradients**; diagnostic note — the tiny untrained batch emitted only −1.0
     messages (confirms Part D's warning that small random batches needn't show both polarities).
   - `pytest -q test_system.py` → **7 passed in 5.56 s** — canonical tests ran unmodified;
     no hardening changes were needed beyond what the spec already included.
8. **Git & repo bootstrap:** `git init -b main`; local identity set (name `gl33mer`,
   GitHub noreply email); staged 14 files; `git diff --check` clean; verified no `.venv`,
   checkpoints, logs, or secrets staged; commit **`77c2d1a`** *"Add v7.1 latent recurrent
   swarm baseline"*; tag **`v7.1-pre-run`** (source unchanged from canonical → no v7.1.1).
   Confirmed `gl33mer/latent-recurrent-swarm` did not exist (API 404), then
   `gh repo create … --private --source=. --remote=origin --push`; pushed `main` + tag;
   verified API: `private: true`, default branch `main`, tag present.
9. **Run metadata (Part F1):** created `runs/20261016T211522Z/`; recorded git commit, git
   status, python version, `pip freeze`, runtime info, memory/disk snapshot. Confirmed no
   stale `checkpoint_phase1.pt` existed.
10. **Stage 1 — system suite (recorded):** `pytest -q test_system.py` → **7 passed in 4.05 s**
    (log: `runs/20261016T211522Z/pytest.log`).
11. **Stage 2 — Phase 1** (`train_unit.py`, ~79 s wall): epoch val T=8 trajectory
    95.0 → 96.0 → 95.1 → 97.8 → **100.0%** (epochs 5–25). Gates: T=1 = 50.0% (≤56.5) PASS;
    T=8 = 100.0% (≥90) PASS; held-out test T=8 = 100.0%. Checkpoint saved post-gates;
    SHA-256 `ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e`; copied into
    run dir; not committed to Git.
12. **Stage 3 — Phase 2a** (`train_swarm.py`, ~133 s wall; warm-started both cores from the
    Phase 1 checkpoint): epoch val parity trajectory 99.40 → 95.30 → 98.00 → **97.50%**
    (epochs 5–20). Full stratified validation + held-out test metrics as in Part 1.
    **All 11 validation sub-thresholds passed; no retry, no relaxation.**
13. **Final report:** wrote `reports/first-run-report.md` (10 sections per Part G), secret-
    scanned, committed as **`d10b0c2`**, pushed. Final `git status` clean.
14. **Post-run (owner request):** secret-scanned all tracked content (no token patterns, no
    private keys); switched repository visibility private → **public** via REST API
    (`PATCH /repos/gl33mer/latent-recurrent-swarm`, `private=false`); verified
    `visibility: "public"`. Added this worklog and a `CHANGELOG.md` entry; pushed.

## Part 3 — Deviations from the canonical spec

1. Python 3.11.2 installed via apt (no Python on VM) — sanctioned fallback path.
2. `numpy>=1.26` added to `requirements-cpu.txt` and installed — required by canonical
   `.numpy()` calls; CPU torch wheel doesn't bundle numpy. No architecture/threshold change.
3. `*.swp` added to `.gitignore` (pre-existing editor swap file).
4. Repository visibility changed to public at the owner's explicit request after the
   baseline completed (the v7.1 spec's private-repo preference governed the initial setup).
5. No other deviations: model architecture, optimisers, epochs, batch sizes, loss weights,
   seeds, and all acceptance thresholds are exactly as specified.

## Part 4 — Artefact map

| Artefact | Location | In Git? |
|---|---|---|
| Source (model/data/tests/trainers) | repo root, `model/`, `data/` | yes |
| Project spec / README / changelog | `PROJECT_SPEC.md`, `README.md`, `CHANGELOG.md` | yes |
| Experiment charter | `latent.md` | yes |
| First-run report | `reports/first-run-report.md` | yes |
| Work log (this file) | `WORKLOG.md` | yes |
| Run logs, env freeze, runtime/memory snapshots | `runs/20261016T211522Z/` | no (ignored) |
| Checkpoint `checkpoint_phase1.pt` + SHA-256 | run dir + VM root | no (ignored) |
| Virtual environment | `.venv/` | no (ignored) |

---

# Phase 2b Mission Log (continuation, 2026-10-16)

Charter: `latent_phase2b.md` (pulled from remote commit `eb0b85b`). Local == remote at start.

## Stage A — inspection findings

- `main` = `origin/main` = `eb0b85b` ("Document Phase 2b COMM-Token experiment specifications");
  working tree clean; tag `v7.1-pre-run` at `77c2d1a` intact; repo public (verified via API).
- Environment reused (healthy): Debian 12, 4 vCPUs, ~8 GiB RAM, Python 3.11.2 in `.venv`,
  torch 2.14.1+cpu (CUDA off), numpy 2.4.6, 4 intra-op threads. No reinstall performed.
- `.venv/`, `runs/`, `checkpoint_phase1.pt` confirmed Git-ignored (`git check-ignore -v`).
- Phase 1 checkpoint present at root and in `runs/20261016T211522Z/`; both hash to
  `ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e` == recorded checksum.
- **No Phase 2a checkpoint exists anywhere** (v7.1 `train_swarm.py` never saved one) →
  labelled reproducibility rerun required (Stage B3).
- `reports/first-run-report.md` still stated "private" → corrected (B1).
- Baseline regression: `pytest -q test_system.py` → 7 passed (4.72 s) before any change.

## Stage B — housekeeping actions

1. Tagged frozen v7.1 source state as `v7.1-baseline` (commit `9f94151`); `v7.1-pre-run` untouched.
2. Branch `chore/preserve-v7.1-artifacts` created from `main`.
3. B1: `reports/first-run-report.md` visibility statement corrected (public + provenance note).
4. B3: `train_swarm.py` gained optional `--artifact-dir` (post-gate-only checkpoint+metadata
   save; no computation/RNG change). Verified: compileall clean, 7/7 baseline tests pass,
   `--help` works, behaviour without flag unchanged.
5. Reproducibility rerun executed under a new UTC run ID (see CHANGELOG + run dir); result
   recorded below after the run.

### Phase 2a reproducibility rerun outcome (run `20261016T215726Z`)

- Source: `v7.1-baseline` (`9f94151`) + artifact-capture option (commit `d450847`, verified
  non-computational). Seeds 42/1337/9999; training seed 42; ~2 m 16 s wall.
- **Result: exact reproduction.** Every validation and test metric matches the historical
  first run to the reported digit: validation connected 97.50%, quadrants 96.40/97.20/98.40/98.00%,
  flip residuals 0.00%/0.41%, zero-message 48.60%, probe 97.80%/97.20%/98.40%; test connected
  97.00%, quadrants 96.80/94.80/98.80/97.60%, flip 0.00%/0.00%, zero 50.50%, probe 97.10%/95.80%/98.40%.
  Deterministic-seed training on the same CPU/torch build reproduced the run bit-comparably at
  the reported precision. Historical report numbers left unchanged; this rerun is labelled a
  reproduction artifact, not the original trained state (which was never saved).
- Artifact: `runs/20261016T215726Z/checkpoint_phase2a.pt` (checksum in
  `runs/20261016T215726Z/checkpoint_phase2a.sha256`), metadata `phase2a_metadata.json`,
  full log `train_swarm_rerun.log`. All Git-ignored.

## Phase 2b implementation (branch `experiment/phase2b-comm-token`, from main `2df939d`)

- `model/swarm_comm.py`: `LatentCommSwarm` (284,107 params; verified). 17-token Core B
  (16 graph nodes + read-only COMM register at index 16); per-step masks via
  `comm_step_mask` (`full` = COMM visible steps 1-8; `final-only` = step 8 only); parity
  head consumes ONLY z_B[15] ([batch,128], hook-asserted); trace mode records Core B
  graph-node states after every step; `build_comm_state` zeroes the message contribution
  (embedder+bias bypassed) under zero_message. v7.1 files untouched; Phase 1 state dict
  loads strict=True into both cores.
- Mechanistic unit-test result: in `final-only` mode, Core B graph-node states are
  **bit-identical** across message polarities for steps 1-7 (max|Δz| = 0.0) and differ at
  step 8 — the one-way mask works exactly as designed. In `full` mode, states differ from
  step 1 onward.
- `test_phase2b.py`: 11 tests (masks, schedules, receptive field with COMM present, no
  late fusion, channel contract, zero-message semantics, gradient flow, generator
  determinism, checkpoint compatibility, parameter counts). One defect found and fixed
  during development: a forward-pre-hook lambda returned a value (interpreted as
  replacement args) — rewritten as a proper hook function. Combined suite:
  **18 passed (7 baseline + 11 Phase 2b) in 5.23 s.**
- `train_swarm_comm.py`: CLI `--mode full|final-only --seed --epochs --run-dir
  --checkpoint --smoke`; Phase 2a-identical recipe (data seeds 42/1337/9999, batch 64,
  AdamW 8e-4/1e-2, 20 epochs, same loss weights, warm start); Phase 2a gates + empty-group
  guards + (full mode) intermediate-step trajectory-sensitivity requirement; post-gate
  checkpoint/metadata/sha256 saving; smoke mode (1 epoch / 2 batches) — both modes pass
  all engineering checks (logs under `runs/*_smoke/`).
- `reports/phase2b-design.md`: pre-declared hypothesis, equations, parameter table,
  one-way rationale, XOR-task limitation, controls and gates, test plan. README updated
  with the Phase 2b scope note (baseline description retained).

## Phase 2b training and evaluation (all runs preserved; no gate relaxed)

Six runs at the frozen v7.1 recipe (mode × seeds 42/43/44), each in its own run dir with
log/env snapshot; artifact save gated behind validation gates:

- **PASS:** full/seed43 (val 95.00%, test 94.80%; probe 95.00/99.80/90.20) and
  final-only/seed42 (val 99.70%, **test 100.00%**, all quadrants 100%).
- **FAIL (4):** full/seed42 (probe 92.30/89.00), full/seed44 (probe 82.60), final-only/seed43
  (connected 67.60 + quadrants + probe), final-only/seed44 (connected 74.80, quadrant 47.2,
  flip residual b_B=0 21.87%, probe).
- Epoch trajectories show severe instability in all six runs (collapses of 20-50 points,
  e.g. full/seed42 97.9→45.9→91.7), unlike deterministic Phase 2a. Recorded as a finding;
  no hyperparameters changed.
- **Trajectory sensitivity (mechanistic evidence):** full mode — bit changes graph-node
  states from step 1, effect grows ~monotonically (e.g. seed43: .105→1.034 by step 8);
  final-only — exactly zero through step 7, nonzero only at step 8 (within-trajectory late
  fusion), matching the unit-test prediction bit-exactly.
- Zero-message control: 49.7-50.8% (chance) in every run — the bit is always necessary.
- Test split evaluated exactly once, only for the two gate-passing models; never used for
  tuning. No paired seed passed both modes → benchmark cannot discriminate full vs
  final-only access, as pre-declared in the design report.
- Full details: `reports/phase2b-run-report.md`. Branch pushed; NOT merged to main.

## Stage 0/1 audit & Phase 2b addendum (charter: latent_phase2b_stability_phase2c.md @ main 45029c7)

- Audit: main=45029c7 (synced), experiment/phase2b-comm-token=d82f555 (pushed), tags
  v7.1-pre-run/v7.1-baseline intact, repo public, working tree clean. All recorded
  checkpoints present with matching SHA-256s (phase1 ef4514…, phase2a rerun 05a163…,
  phase2b full/43 38c505…, final-only/42 79aada…). Regression suite on this branch:
  18 passed (log in runs/20261016T225501Z_stage0_audit/).
- Dated checksummed archive of all Phase 2b/2a/1 artifacts copied to
  /workspace/latent-recurrent-swarm-archive/ (sha256 2e9c8d96…; no external durable
  storage exists on this VM — single-volume caveat recorded).
- `reports/phase2b-analysis-addendum.md` written: (1) causal-statement correction —
  original "routing the message through the recurrent trajectory demonstrably
  destabilises optimisation" overclaims; both modes share most architectural changes and
  the experiment does not isolate the cause; corrected wording predeclared for future
  summaries; (2) exploratory paired validation deltas: seed42 −8.0, seed43 +27.4, seed44
  +7.5 pp; mean +8.97, median +7.5; descriptive only, n=3, no paired seed passed all
  gates in both modes, test numbers not paired; (3) artifact/checksum table; (4)
  visibility metadata already correct on this branch (d450847); (5) no test-split reuse.

## Phase B — stability study setup (branch experiment/phase2b-stability-study)

- Branch created from Phase 2b branch (f701460). `reports/phase2b-stability-design.md`
  predeclared and committed BEFORE any sweep outcome was viewed: LR grid {8e-4, 4e-4,
  2e-4} x {full, final-only} x seeds {42,43,44} = 18 runs; frozen Phase 2b recipe
  otherwise; train/val-only builder (test split never generated); paired init + batch
  order via seeded DataLoader generator; gradient-norm/NaN instrumentation; last+periodic
  checkpoints saved for every run including failures; 120-minute budget; explicit
  common-LR eligibility rule (>=2/3 seeds pass all gates in EACH mode at the same rate,
  finiteness, trajectory requirements, no new pathology) with tie-breaks; if none
  eligible -> recipe "not yet stabilised" and Phase 2c limited to engineering smoke.
- `experiments/stability/run_stability_sweep.py` implemented (no model semantics
  changed; one dev-time script bug fixed after a 1-epoch sanity run: .detach() on loss
  logging and a divide-by-zero guard for 1-epoch histories). Sanity run passed;
  regression suite 18/18 after changes.

## Phase B — stability sweep executed and decided (18/18 runs, ~44 min)

- All 18 runs completed within budget; no test split generated; all runs finite; full-mode
  trajectory sensitivity present from step 1 in every full run; final-only bit-identical
  through step 7 everywhere.
- Gate pass counts by rate (full, final-only): 8e-4 -> 0/3, 1/3; 4e-4 -> 1/3, 0/3;
  2e-4 -> 1/3, 2/3. Predeclared eligibility rule (>=2/3 in EACH mode at the same rate)
  satisfied at NO rate -> **recipe classified NOT YET STABILISED**. 2e-4 has the lowest
  median instability (3.2%) and most passes (3/6); the sender-probe gate is the most
  frequent failure at every rate.
- Documented deviation: sweep uses an explicit DataLoader generator seed for paired batch
  order, so per-run outcomes differ from the original six runs (which used default
  global-RNG order); both are valid runs of the frozen recipe — seed/order sensitivity
  underscored, artifacts unchanged.
- Consequence per charter: Phase 2c proceeds as engineering + tests + smoke ONLY; no
  confirmatory multi-seed comparison; no Phase 2c test evaluation.
- Report: reports/phase2b-stability-report.md; artifacts under
  runs/20261016T231051Z_stability/ (metrics.jsonl, run summaries, checkpoints incl.
  failures, aggregate.json, summary.csv).

## Phase C — Phase 2c implementation, tests, smoke (branch experiment/phase2c-message-conditioned-reachability)

- Branch from stability state (ed46cd4). New modules, existing code untouched:
  `data/conditioned_graph_task.py` (8-combo balanced selector task, relation graphs via
  the v7.1 matched-DAG sampler, 64-dim union/R0/R1/id layout, namespaced 3-graph hashing,
  split seeds 4201/13371/99991 predeclared, data-only baseline helper);
  `model/swarm_conditioned.py` (LatentConditionedSwarm 279,561 params; LatentDualRelationSwarm
  compute-both-then-select diagnostic 280,075 params, decoder replaced by Identity so no
  dead parameters; documented Core B warm-start mapping with zeroed relation channels —
  verified bit-exact against the Phase 1 encoder; one-way COMM masks reused from Phase 2b).
- `test_phase2c.py`: 13 tests (params, layout/union, generator invariants/balance/BFS/
  determinism/disjointness, mask semantics + message-independence, receptive field on the
  union mask, forced-message trajectory interventions incl. final-only bit-identity
  through step 7, terminal-only readout contract, warm-start parity, channel/forced
  semantics, zero-message exactness, gradient flow both modes, dual selection semantics).
  Combined suite: **31 passed** (7 + 11 + 13). Two dev-time fixes: removed two junk
  assertions during writing; fixed dual-model warm-start (decoder replaced -> decoder
  keys dropped from transfer) and trace-unpacking arity for the dual return signature.
- `reports/phase2c-design.md` committed BEFORE any training: task, architecture, counts,
  mapping, split seeds, predeclared Phase 2c gates (zero-message <=77% overall AND <=53%
  on disagreement — the XOR 53%-overall gate explicitly NOT reused; counterfactual >=75%;
  switch consistency >=85%; >=4/5 seeds for stable success), comparison statistics and
  interpretation rule, and the stability-gate consequence.
- `train_swarm_conditioned.py`: full metric suite (overall/by-bA/by-Y/quadrants/agreement/
  disagreement, counterfactuals, switch consistency/invariance, zero-message, probe,
  per-step trajectory by subset, final-only invariance), paired-seed determinism, last.pt
  saved for every run incl. failures, guard refuses non-smoke runs while no eligible LR
  exists (--allow-unstable-lr override).
- Smoke runs (engineering-only, no research claims, no test data): primary full
  (step-1 |Δz| 9.26e-04), primary final-only (step-1 = 0.0 exactly), dual full (1.02e-03),
  dual final-only (0.0) — all losses finite, all grads flow. Logs in
  runs/20261016T233732Z_phase2c_smoke/ (approx; see run dir listing).
- Data-only baselines at full train size: 8 combos exactly 500 each; disagreement 50%;
  P(Y=1)=0.5; no-selector optimum 75%; 12,000 unique graph hashes.

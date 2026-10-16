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

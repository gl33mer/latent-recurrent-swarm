# First Recorded Baseline Run — Latent Recurrent Swarm v7.1

**Status: COMPLETE — all system tests passed; all Phase 1 and Phase 2a validation gates passed; held-out test metrics reported. No gate was relaxed.**

## 1. Date/time and run ID

- UTC date of run: 2026-10-16
- Run ID: `20261016T211522Z`
- Run directory: `runs/20261016T211522Z/` (Git-ignored; contains pytest.log, train_unit.log, train_swarm.log, checkpoint copy, checksum, environment freeze, runtime and memory snapshots)

## 2. Repository

- URL: https://github.com/gl33mer/latent-recurrent-swarm (**public** as of 2026-10-16; the run itself was performed while the repository was still private — visibility was changed to public at the owner's explicit request after this baseline run completed, see `CHANGELOG.md` and `WORKLOG.md`)
- Branch: `main`
- Pre-run source commit + tag: `77c2d1a` / `v7.1-pre-run` (source unchanged from canonical v7.1; only dependency/housekeeping deviations documented in `CHANGELOG.md`)
- Report commit: see `git log` (this file's commit on `main`)

## 3. Environment

- OS: Debian GNU/Linux 12 (bookworm), Linux 6.18.55, x86_64
- CPU: 4 vCPUs; RAM: 8 GiB (MemTotal 8,159,260 kB); disk: 20G volume, 3% used
- Python: 3.11.2 (system Python installed fresh via apt for this run: `python3`, `python3-venv`, `python3-pip`)
- PyTorch: 2.14.1+cpu (official CPU wheel index) — **CUDA available: False, CPU-only confirmed**
- NumPy added (see deviations); pytest installed; full snapshot: `runs/20261016T211522Z/environment.freeze.txt`
- Threads: `torch.set_num_threads(4)` at script entry; `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`
- Seeds: Python `random.seed(42)` and `torch.manual_seed(42)` in both training scripts; dataset seeds 42/1337/9999 (train/val/test)

## 4. System tests (`pytest -q test_system.py`)

- Result: **7 passed** in 4.05 s (recorded run) after a pre-commit run of 7 passed in 5.56 s.
- Coverage: exact parameter counts (137,476 / 292,427 — verified independently before commit), receptive-field horizon (node-0 signal reaches node 15 only at T=15, chain graph), binary-channel contract (strict ±1 and exact flip inversion; seeded batch exercised both polarities), dataset integrity (pairwise-disjoint train/val/test graph-hash sets; two unique hashes per sample; labels match BFS reachability and XOR; reachable graphs have 5 ≤ d ≤ 8), conditional parity balancing (P(Y=1|b_A) and P(Y=1|b_B) ≈ 0.5), parity-head noisy-XOR generalization across 5 fixed seeds (≥99%), 1-hop local-region lookup baseline within 52–56.5%.
- No test required correction. Gradient-flow smoke test (Part D #10): one batch backward — **no parameter with `grad is None`, no all-zero gradients**. Code check: `python -m compileall` clean.

## 5. Phase 1 (`train_unit.py`, log: `runs/20261016T211522Z/train_unit.log`)

- Training: fixed T=8, 25 epochs, AdamW lr 1e-3, wd 1e-2, batch 64. Val T=8 by epoch: 05→95.0%, 10→96.0%, 15→95.1%, 20→97.8%, 25→100.0%. Wall time ≈ 79 s.
- **Gate 1 (T=1 validation ≤ 56.5%): PASS — 50.0%** (chance level)
- **Gate 2 (T=8 validation ≥ 90.0%): PASS — 100.0%**
- Held-out test T=8 (untouched split): **100.0%**
- Checkpoint `checkpoint_phase1.pt` saved only after both gates; SHA-256: `ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e` (copy and checksum in run directory; not committed to Git).

## 6. Local-region lookup baseline (measured)

The empirical 1-hop baseline in `test_system.py` builds a lookup over the six edges among nodes 12–15 (the induced subgraph near the target) and achieves **52–56.5% on validation across the fixed seeds** (within the asserted band; exact measured value in the pytest log). Limitation: this is an empirical diagnostic on the induced subgraph, not a proof of the exact information available to the T=1 reasoner. Phase 1's T=1 gate (≤56.5%) is fixed to this empirical ceiling for this experiment.

## 7. Phase 2a (`train_swarm.py`, log: `runs/20261016T211522Z/train_swarm.log`)

- Warm-started Core A and Core B from `checkpoint_phase1.pt`; trained 20 epochs at T_A=8, T_B=8, AdamW lr 8e-4, wd 1e-2; loss = parity + 0.5·aux_A + 0.5·aux_B + 0.5·msg(BCE on raw pre-binarized scalar). Wall time ≈ 133 s.

**Validation (acceptance) metrics:**

| Metric | Measured | Gate | Verdict |
|---|---|---|---|
| Connected accuracy | 97.50% | ≥ 82.0% | PASS |
| Quadrant (0,0) | 96.40% | ≥ 75.0% | PASS |
| Quadrant (0,1) | 97.20% | ≥ 75.0% | PASS |
| Quadrant (1,0) | 98.40% | ≥ 75.0% | PASS |
| Quadrant (1,1) | 98.00% | ≥ 75.0% | PASS |
| Flip residual b_B=0 | 0.00% | ≤ 15.0% | PASS |
| Flip residual b_B=1 | 0.41% | ≤ 15.0% | PASS |
| Zero-message accuracy | 48.60% | ≤ 53.0% | PASS |
| Probe overall | 97.80% | ≥ 95.0% | PASS |
| Probe b_A=0 | 97.20% | ≥ 90.0% | PASS |
| Probe b_A=1 | 98.40% | ≥ 90.0% | PASS |

**Held-out test (untouched; reported, not tuned on):** connected 97.00%; quadrants 96.80 / 94.80 / 98.80 / 97.60%; flip residual 0.00% (b_B=0) and 0.00% (b_B=1); zero-message 50.50%; probe overall 97.10%, b_A=0 95.80%, b_A=1 98.40%.

## 8. Acceptance-gate summary

- System tests: PASS (7/7)
- Phase 1 Gate 1 (T=1 ≤ 56.5%): PASS (50.0%)
- Phase 1 Gate 2 (T=8 ≥ 90.0%): PASS (100.0%)
- Phase 2a all five validation gates (connected, quadrants, flip residual, zero-message, probe): PASS (11/11 sub-thresholds)
- No stage was skipped, interrupted, or retried under a different configuration.

## 9. Deviations, errors, interruptions

1. **Python was absent on the VM.** Installed Debian's `python3`/`python3-venv`/`python3-pip` (3.11.2) as root via apt — the spec's own fallback path. No system Python was replaced.
2. **NumPy was missing** (the CPU PyTorch wheel does not bundle it); canonical code calls `Tensor.numpy()`. Installed numpy and added `numpy>=1.26` to `requirements-cpu.txt`. Dependency-only change; no code, architecture, threshold, optimizer, epoch, batch-size, or loss-weight change.
3. `*.swp` added to `.gitignore` for a pre-existing editor swap file in the workspace.
4. No interruptions, no failed gates, no retry runs. Pre-training stale-checkpoint check: none existed.

## 10. Interpretation limits

- This run establishes the **v7.1 Phase 2a baseline: decoupled recurrent reasoners with one-bit forward-pass decision fusion at the receiver** (STE binary scalar; Core B never ingests the message into its recurrence). It is **not** evidence for recurrent message integration, a global workspace, or a fully collaborative swarm — that is Phase 2b, which is out of scope here.
- The flip-residual and probe metrics show the trained model *uses* the bit for readout, not that the bit modulates Core B's reasoning trajectory.
- A single run at seed 42 does not establish seed robustness. The empirical local baseline is approximate as described in §6. A passed unit suite is not a passed training run; both were executed here and both are reported separately.

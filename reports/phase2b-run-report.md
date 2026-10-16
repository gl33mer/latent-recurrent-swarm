# Phase 2b Run Report — COMM-Token Recurrent Message Integration

**Branch:** `experiment/phase2b-comm-token` · **Implementation commit:** `be691ed`
**Environment:** Debian 12, Python 3.11.2, torch 2.14.1+cpu (CUDA off), numpy 2.4.6, 4 vCPUs, ~8 GiB RAM, 4 intra-op threads
**Data/recipe:** identical to v7.1 Phase 2a (seeds 42/1337/9999; batch 64; AdamW 8e-4/wd 1e-2; 20 epochs; loss = parity + 0.5·aux_A + 0.5·aux_B + 0.5·msg_BCE; both cores warm-started from the verified Phase 1 checkpoint `ef4514…49e`)
**Model:** `LatentCommSwarm`, 284,107 params; one-way COMM register; parity head on `z_B[15]` only.

## 1. Outcome classification (per charter §9 — three separate outcomes)

- **Engineering success: YES.** All 18 tests pass (7 baseline + 11 Phase 2b); smoke checks pass for both modes; masks verified; checkpoints saved with SHA-256 for gate-passing runs.
- **Task success: PARTIAL.** 2 of 6 runs passed all validation gates (full/seed 43; final-only/seed 42). 4 of 6 runs failed at least one gate. No threshold was altered.
- **Scientific finding: see §4–§5.** Trajectory-sensitivity measurements confirm the bit modulates Core B's graph-node trajectory in full mode; the benchmark cannot discriminate full vs final-only access; and training at the frozen v7.1 recipe is markedly less stable than Phase 2a.

## 2. Per-run validation results (all six runs, no cherry-picking)

Gates: connected ≥82%, each quadrant ≥75%, flip residual ≤15% per b_B group, zero-message ≤53%, probe ≥95% overall / ≥90% per b_A; (full mode) ≥1 message-sensitive intermediate step.

| Run | Connected | Quads 00/01/10/11 | Flip b_B=0/1 | Zero-msg | Probe ovr/b_A0/b_A1 | Verdict (failed gates) |
|---|---|---|---|---|---|---|
| full s42 | 91.70 | 95.2/96.0/87.6/88.0 | 0.00/1.74 | 50.80 | 92.30/95.60/**89.00** | FAIL probe (overall 92.3<95; b_A=1 89.0<90) |
| **full s43** | **95.00** | 99.6/100.0/89.6/90.8 | 0.00/0.21 | 50.00 | 95.00/99.80/90.20 | **PASS** |
| full s44 | 82.30 | 80.4/81.6/81.2/86.0 | 0.00/0.24 | 50.00 | 82.60/81.40/83.80 | FAIL probe |
| **final-only s42** | **99.70** | 98.8/100.0/100.0/100.0 | 0.00/0.00 | 50.00 | 99.70/99.40/100.00 | **PASS** |
| final-only s43 | 67.60 | 56.4/58.8/78.0/77.2 | 0.00/3.82 | 50.40 | 67.90/56.20/79.60 | FAIL connected, quadrants, probe |
| final-only s44 | 74.80 | 47.2/74.4/90.0/87.6 | 21.87/1.98 | 49.70 | 83.90/76.80/91.00 | FAIL connected, quadrants, flip(b_B=0), probe |

**Held-out test (evaluated exactly once, only for gate-passing models):**

| Run | Connected | Quads 00/01/10/11 | Flip 0/1 | Zero-msg | Probe ovr/b_A0/b_A1 |
|---|---|---|---|---|---|
| full s43 | 94.80 | 99.2/100.0/91.2/88.8 | 0.00/0.00 | 50.20 | 94.80/99.60/90.00 |
| final-only s42 | **100.00** | 100.0/100.0/100.0/100.0 | 0.00/0.00 | 50.10 | 100.00/100.00/100.00 |

The test split was never consulted during development; failed runs exited before test evaluation.

## 3. Training stability (epoch-by-epoch validation parity accuracy)

| Run | Ep 5 | Ep 10 | Ep 15 | Ep 20 |
|---|---|---|---|---|
| full s42 | 97.90 | 95.20 | **45.90** | 91.70 |
| full s43 | 93.70 | **65.70** | 94.50 | 95.00 |
| full s44 | 98.90 | 98.40 | 83.40 | 82.30 |
| final-only s42 | 48.90 | 97.50 | 99.00 | 99.70 |
| final-only s43 | 92.30 | 93.70 | 69.40 | 67.60 |
| final-only s44 | 91.20 | 95.70 | 96.50 | **74.80** |

Every Phase 2b run shows strong non-monotonicity (collapses of 20–50 points followed by partial or no recovery), whereas Phase 2a at the identical recipe was deterministic (97.50% reproduced exactly). Routing the message through the recurrent trajectory demonstrably destabilises optimisation at this learning rate/epoch budget. **No hyperparameter was changed to chase passes**, per charter; a stabilisation study is a clearly separate follow-up experiment.

## 4. Trajectory sensitivity (mean |Δz| between m=+1 and m=−1 traces, per step, validation)

| Run | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
|---|---|---|---|---|---|---|---|---|---|
| full s42 | 0 | .104 | .202 | .289 | .373 | .472 | .716 | .987 | 1.284 |
| full s43 | 0 | .105 | .207 | .338 | .459 | .634 | .829 | .872 | 1.034 |
| full s44 | 0 | .140 | .211 | .308 | .421 | .584 | .773 | .988 | 1.211 |
| final-only s42 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | .502 |
| final-only s43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | .600 |
| final-only s44 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | .381 |

(similar values per b_B group; full table in run metadata JSON files)

**Findings:**
1. **Full mode: genuine recurrent integration.** The bit changes ordinary graph-node states from the *first* recurrent step, with monotonically growing effect — the message influence compounds across the depth, and the trained full/seed-43 model still passes all task gates. This satisfies the pre-declared mechanistic requirement.
2. **Final-only control behaves exactly as designed:** bit-identical states through step 7 (verified bit-exactly in unit tests), message influence only at the final update — mechanistically a within-trajectory late fusion.
3. **The zero-message control remains valid everywhere:** all zero-message accuracies sit at 49.7–50.8% (chance), so no model solves parity without the bit.

## 5. Full vs final-only comparison — honest interpretation

- No seed produced a gate-passing pair in both modes (s42: final-only passes/full fails; s43: full passes/final-only fails; s44: both fail). A clean paired comparison is therefore impossible at this recipe.
- Cross-seed comparison is confounded by the large seed variance; the best final-only run (99.70% val, 100.0% test) outperformed the best full run (95.00% val, 94.80% test), which is **not** evidence against full-step access — it is evidence that XOR, as the charter pre-declared, does not require early message use and cannot discriminate the two access schedules.
- **Conclusion:** on this benchmark, recurrent message integration is demonstrably *implemented* (trajectory evidence) but not demonstrably *beneficial*. The correct follow-up is the message-conditioned graph task proposed in `reports/phase2b-design.md` §4 (bit selects the edge-relation/subgraph Core B must traverse), with a late-fusion control — plus a training-stabilisation study before any such comparison.

## 6. Artifacts

| Run | Directory | Checkpoint | SHA-256 |
|---|---|---|---|
| full s43 | `runs/20261016T220921Z_comm_full_seed43/` | `checkpoint_phase2b_full_seed43.pt` | `38c505a5f9ae938fffa697bdad195b6d92241223b90dbced394919a6db203612` |
| final-only s42 | `runs/20261016T221820Z_comm_final-only_seed42/` | `checkpoint_phase2b_final-only_seed42.pt` | `79aadaf7880c369cfff550db9b7624d5c9a4f97e2b13a35af2811eed7c24d45e` |
| failed runs | `runs/20261016T220650Z_comm_full_seed42/`, `runs/20261016T221141Z_comm_full_seed44/`, `runs/20261016T221356Z_comm_final-only_seed43/`, `runs/20261016T221607Z_comm_final-only_seed44/` | none (artifact save is post-gate only) | — |

Each directory contains `train.log`, `git-commit.txt`, `environment.freeze.txt`, and (passing runs) a metadata JSON with full config + all metrics. Smoke logs: `runs/20261016T220513Z_smoke/`. Checkpoints and logs are Git-ignored.

## 7. Limitations

- Single-machine, single-recipe, 3 seeds per mode; exploratory, not seed-robust.
- The 15%-flip-residual and probe gates bind tightly for full mode (s43 passed probe overall at exactly 95.0% and b_A=1 at 90.2%) — near-threshold results should not be over-read.
- No parameter/compute-matched feedforward baseline was run; no SOTA claims are made.
- The message-conditioned task that could demonstrate *advantage* (not just possibility) of recurrent integration remains unimplemented.

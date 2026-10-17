# Phase 2c Candidate-Recipe Run Report — Paired Validation Experiment (2026-10-17)

**Branch:** `experiment/phase2c-lambda2-paired-confirmatory` · **Source commit for all 15
runs:** `ce3c009` · **Recipe addendum (pre-registered before training):** `4939d94` ·
**Run root:** `runs/20261017T124250Z_phase2c_confirmatory/`

## Final classification

## `VALIDATION PARTIAL/UNSTABLE — TEST UNTOUCHED`

**0 of 5 paired seeds qualified** (a paired seed requires BOTH primary modes to pass all
gates). Per the predeclared stop rule (addendum D3): the Phase 2c test split (seed 99991)
was **never generated**, `evaluate_phase2c_test.py` was **never invoked**, no failed seed
was rerun, and no setting was changed after seeing results.

## 1. Execution summary

- 15/15 planned runs completed (10 primary paired + 5 dual diagnostic); grid complete, no
  interruptions, no non-finite events, no retries; all runs at the single source commit
  `ce3c009` with the verified Phase 1 checkpoint (`ef4514…49e`).
- Environment: Debian 12, Python 3.11.2, torch 2.14.1+cpu (CUDA off), 4 intra-op threads.
- Wall time ≈ 56 minutes for the grid (budget 120 min).

## 2. All 15 runs — final-epoch validation outcomes

Gates: overall ≥82%; quadrants ≥75%; probe ≥95/90/90; zero-message ≤77% overall and ≤53%
on disagreement; counterfactuals ≥75% each; switch consistency ≥85% (disagreement);
invariance ≥85% (agreement); mechanism (full: terminal-node sensitivity >0 at an
intermediate step; final-only: bit-identity through step 7).

| Run | Overall | Disagree | Probe | Zero/d | CF R0/R1 | Switch | Failed criteria (beyond the universal set*) |
|---|---|---|---|---|---|---|---|
| primary full s42 | 62.10% | 53.00% | 99.7% | 50.0% | 58.4/67.8 | 22.6% | overall, quadrants, counterfactual, switch, invariance |
| primary full s43 | 59.30% | 50.00% | 99.6% | 50.0% | 60.0/60.4 | 16.2% | same |
| primary full s44 | 60.60% | 48.00% | 98.6% | 50.4% | 60.5/63.4 | 14.2% | same |
| primary full s45 | 60.20% | 50.40% | 99.8% | 47.0% | 60.7/62.5 | 19.8% | same |
| primary full s46 | 60.50% | 51.00% | 99.4% | 50.0% | 60.3/62.1 | 13.4% | same |
| primary final-only s42 | 61.10% | 49.80% | 99.4% | 49.6% | 57.9/64.2 | 4.2% | overall, quadrants, counterfactual, switch |
| primary final-only s43 | 60.70% | 49.20% | 99.7% | 50.6% | 61.6/60.9 | 6.6% | same |
| primary final-only s44 | 60.70% | 46.60% | 100.0% | 48.0% | 60.1/64.6 | 3.6% | same |
| primary final-only s45 | 59.50% | 46.80% | 99.9% | 49.6% | 58.9/62.7 | 2.0% | same |
| primary final-only s46 | 59.90% | 48.60% | 100.0% | 48.2% | 58.8/63.4 | 0.6% | same |
| dual s42 | 62.60% | 49.20% | 100.0% | 47.6% | 59.3/67.8 | 10.8% | (diagnostic; descriptive only) |
| dual s43 | 61.50% | 48.00% | 100.0% | 45.8% | 60.2/65.7 | 8.2% | " |
| dual s44 | 60.50% | 48.00% | 100.0% | 48.2% | 58.1/64.7 | 5.6% | " |
| dual s45 | 64.20% | 53.80% | 99.9% | 49.6% | 59.0/70.1 | 17.4% | " |
| dual s46 | 61.30% | 48.00% | 99.9% | 48.2% | 60.7/64.2 | 1.8% | " |

*Universal failures across all ten primary runs: `overall<82%`, `quadrant<75%`,
`counterfactual<75%`, `switch_consistency<85%`; full mode additionally
`selector_invariance<85%`. **Mechanism gates passed everywhere**: full mode showed positive
terminal-node sensitivity at intermediate steps in all five full runs; final-only runs were
bit-identical through step 7 in all five (`invar=True`).

**Pass counts: full 0/5; final-only 0/5; qualifying paired seeds 0/5.**

## 3. Diagnosis — what failed and what did not

1. **The candidate recipe transferred without destabilising Phase 2c.** Validation
   trajectories are stable plateaus (overall oscillating 58–64% from epoch 1 to 20, zero
   >15-point drops in any run, no non-finite events) — in sharp contrast to the λ=0.5
   Phase 2b collapse pattern. The lr=2e-4/λ=2.0 recipe is *stable* on Phase 2c.
2. **The sender channel is perfect and the controls are sound.** Sender probe ≈98.6–100%
   in every run from epoch ~3 onward; zero-message accuracy at chance (45.8–50.6%);
   zero-message on disagreement ≈ chance. The bit is transmitted and necessary.
3. **The bottleneck is per-relation reachability computation, not message access.**
   Disagreement-subset accuracy is ≈ chance (46.6–53.8%) for *every* model — including the
   **dual compute-both-then-select diagnostic (48.0–53.8%)**, which receives no selector
   during recurrence and has dedicated per-relation heads. Agreement-subset accuracy
   (implied: ≈74%) shows the models learn union-graph reachability but cannot separate
   which relation carries the path. Switch consistency is near zero even for the dual
   model (0.6–17.4%), i.e. the two relation-head answers are rarely jointly correct on
   disagreement examples. No selector mechanism — early or late — can fix a receiver that
   cannot compute the relation-conditioned answers it must select between.
4. **Descriptive paired Δ (validation disagreement, full − final-only):** +3.2, +0.8,
   +1.4, +3.6, +2.4 pp; mean +2.28 pp — both conditions are at chance, so this difference
   is not meaningful. No test statistics were computed (test untouched).

## 4. Why this is not a recipe failure — and what it does not show

- The same recipe solved 6/6 Phase 2b XOR runs. Phase 2c is a strictly harder task
  (relation-conditioned reachability vs a bit-XOR of two already-computed answers), and
  20 epochs at lr 2e-4 plateaus far above chance but below the gates. The plateau (rather
  than collapse) pattern indicates undercapacity/undertraining for the harder computation,
  not optimisation instability.
- This experiment does **not** show that early message access is useless: with the receiver
  unable to compute relation-specific reachability, no comparison of access schedules can
  be informative. The full-mode trajectory mechanism (message modulating graph-node states
  from step 1, including the terminal node) was independently re-confirmed in all five
  full runs.
- It also does not validate or refute the screened recipe *for Phase 2b* — that result
  stands on its own branch.

## 5. Recommended next step (separate, to be predeclared)

A capacity/training-budget study for the Phase 2c receiver — e.g. longer training
(epochs), tuned LR for the harder task, or receiver-capacity changes — evaluated first on
the **dual diagnostic** (which isolates per-relation computation from selection entirely)
and only then on the primary paired comparison. Any such study requires its own
predeclared design; the test split remains reserved.

## 6. Artifacts

- Per-run directories `runs/20261017T124250Z_phase2c_confirmatory/{primary_full,primary_final-only,dual_full}_seed{42..46}/`:
  `last.pt` + `epoch05/10/15/20.pt` (atomic saves), `metrics.jsonl` (per-epoch losses,
  val overall/quadrants/agreement/disagreement, probe, zero-message, gradient norms,
  finiteness), `meta_*.json` (config fingerprint, initial/final state checksums, gate
  status, checkpoint SHA-256), per-run `.log` files at the run root.
- Paired initial-state checksums: full and final-only matched for every seed
  (seed 42 `a9cd0504af7c…`, 43 `8354481d45ff…`, 44 `553ba01597c2…`, 45 `fd509a5a7e80…`,
  46 `64e33a79ce67…`). Train-loader order identical within pairs (generator seed 9000+seed).
- Durable copy: `phase2c-confirmatory_20261017T131500Z.tar.gz` under
  `/workspace/latent-recurrent-swarm-archive/` (same single-volume caveat as before).

## 7. Honest limitations

- 20 epochs / one LR / one architecture family; the plateau may be an artefact of an
  insufficient training budget rather than a task–architecture mismatch (the dual control's
  chance-level disagreement accuracy suggests the relation channels are simply not being
  exploited yet).
- The dual diagnostic's own gates were not predeclared (by design); its numbers are
  descriptive.
- n=5 seeds; all conclusions are about this grid, not about Phase 2c in general.

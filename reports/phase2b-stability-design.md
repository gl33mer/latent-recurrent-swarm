# Phase 2b Stability Study — Predeclared Design (locked before any sweep run)

**Branch:** `experiment/phase2b-stability-study` (from Phase 2b branch `f701460`)
**Purpose:** diagnose Phase 2b optimisation instability before any Phase 2c confirmatory
run. This is a small recipe-screening study, **not** a confirmatory scientific result.
**This document is committed before the sweep is executed; the selection rule below is
declared before any sweep outcome is viewed.**

## 1. Fixed recipe (identical to frozen Phase 2b except learning rate)

- Task: original Phase 2b XOR benchmark (`data/graph_task.py`); architecture
  `LatentCommSwarm` (284,107 params) untouched (`model/swarm_comm.py` not modified).
- Data: train seed 42 / 4,000 samples; validation seed 1337 / 1,000 samples.
  **The test split is never generated in this sweep** — a train/val-only builder is used.
- Paired initialisation and data order: for each model seed, both modes use the same
  seed-reset sequence (Python `random`, `torch.manual_seed`) before dataset/loader
  construction and the same DataLoader generator seed, so paired full/final-only runs see
  identical initial weights and identical batch order (deterministic CPU execution).
- Batch 64; 20 epochs; AdamW; weight decay 1e-2; loss = parity + 0.5·aux_A + 0.5·aux_B +
  0.5·msg_BCE; both cores warm-started from the verified Phase 1 checkpoint
  (`ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e`).
- CPU float32, 4 intra-op threads. No gradient clipping, no early stopping, no
  best-checkpoint selection, no architecture or loss changes.

## 2. Sweep matrix

Learning rates {8e-4 (frozen baseline), 4e-4, 2e-4} × modes {full, final-only} × model
seeds {42, 43, 44} = **18 runs**. Only LR and COMM-visibility schedule vary.

## 3. Instrumentation per run (JSONL per epoch + summary JSON/CSV)

- Per epoch: mean train total loss and each component (parity, aux_A, aux_B, msg);
  validation connected accuracy and four quadrants; message probe overall and per b_A;
  zero-message accuracy; flip residuals per b_B at epochs 5/10/15/20; gradient norms
  (mean and max over batches) for groups `core_A`, `core_B`, `msg_encoder`,
  `msg_embedder`, `learned_comm_token`, `parity_head`, computed from gradients before the
  optimizer step; non-finite (NaN/Inf) counts for losses/gradients; parameter finiteness
  at epoch end; optimizer LR.
- Trajectory sensitivity (per-step mean |Δz| between forced-opposite message traces on
  validation) at the final epoch (minimum; final-only must be bit-identical through
  step 7 — verified structurally by unit test and expected empirically).
- Checkpoints: `last.pt` saved for **every** run including gate-failing runs, plus
  `epoch{05,10,15,20}.pt`. All under ignored `runs/` directories; never committed.
- Final validation gate status with the precise failed criteria list (Phase 2b gates,
  unchanged thresholds; empty-conditioning groups hard-fail).

## 4. Selection rule (declared before outcomes; no test data anywhere)

A shared learning rate is **eligible** for Phase 2c only if ALL of:

1. ≥2 of 3 seeds pass all Phase 2b validation gates in **full** mode, AND ≥2 of 3 pass
   all gates in **final-only** mode, at that same learning rate.
2. No run at that rate has non-finite losses/parameters or a broken gradient path.
3. Full-mode trajectory sensitivity > 0 at ≥1 intermediate step; final-only graph-node
   traces bit-identical before the final step.
4. Validation histories and gradient diagnostics show no clear new pathology at that rate
   (reported, not hidden).

Among eligible shared rates, choose by (a) highest total all-gate pass count across both
modes, then (b) lower median validation instability (mean absolute epoch-to-epoch change
of validation connected accuracy across all paired runs at that rate). One common LR for
both modes; if rates tie on both criteria, prefer the lower rate only when validation
evidence is otherwise comparable. **If no rate is eligible: classify the recipe as *not
yet stabilised*; implement/test Phase 2c and run only an engineering-labelled smoke run;
do not run a confirmatory multi-seed Phase 2c comparison.**

**Budget:** ≈120 minutes wall-clock for the complete 18-run grid. If the cap is
approached, finish the current run safely and report partial completion; the selection
rule is not applied to an incomplete grid as though it were complete.

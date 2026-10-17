# Phase 2b Message-Supervision Weight Study — Predeclared Design (locked before any sweep run)

**Branch:** `experiment/phase2b-msg-supervision-weight-study` (based exactly on the
verified Phase 2b stability-study state `ed46cd4`, containing the frozen
`LatentCommSwarm`, the frozen stability runner, and the stability report).
**Purpose:** one-factor recipe screening of the only new variable — the weight
`lambda_msg` on the **existing** sender-message BCE loss — motivated by the Stage C
diagnosis that sender-probe failures are late-training optimisation collapses in which
the message BCE degrades in lockstep with parity accuracy. This is a recipe-screening
study, **not** a confirmatory Phase 2c scientific result. No Phase 2c file is touched;
no Phase 2c data is generated; the Phase 2c test seed 99991 appears in no sweep code path.
**This document is committed before the runner produces any sweep outcome.**

## 1. Fixed settings (verified against the frozen base state)

- Task/data: original Phase 2b XOR benchmark; train seed 42 / 4,000 samples,
  validation seed 1337 / 1,000 samples. **A test split is never generated.**
- Model: frozen `LatentCommSwarm` — verified at base commit `ed46cd4`:
  **284,107 trainable parameters** (re-verified programmatically when writing the runner).
- Warm start: Phase 1 checkpoint, SHA-256
  `ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e` (verified on disk).
- Optimisation: AdamW, lr **2e-4** (strongest aggregate candidate from the first sweep;
  still ineligible there, so not treated as approved), weight decay 1e-2, batch 64,
  20 epochs, float32 CPU, 4 intra-op threads. No clipping, schedules, warm-up,
  early stopping, or best-checkpoint selection.
- Loss: `1.0·parity_CE + 0.5·auxA_CE + 0.5·auxB_CE + lambda_msg·BCEWithLogits(raw_m, b_A)`.
- **Only new variable:** `lambda_msg ∈ {0.5 (frozen baseline), 1.0, 2.0}`.
- Modes: `full`, `final-only`. Model seeds: 42, 43, 44. Total: **18 runs**.

## 2. Pairing, initialisation and ordering (per cell)

1. `random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)`.
2. Build `LatentCommSwarm`, warm-start both cores from the verified Phase 1 checkpoint.
3. Build train/val loaders with explicit independent generators:
   `DataLoader(train, generator=torch.Generator().manual_seed(9000 + seed))` (identical
   batch order for both modes at the same seed; identical across lambdas).
4. Record a deterministic SHA-256 over the serialised initial state dict (the
   "initial-parameter checksum") so paired full/final-only runs and across-lambda runs at
   the same seed can be verified to start identically.
5. Train 20 epochs; evaluate validation only; save artefacts. No optimizer-state sharing.

## 3. Gates (copied verbatim from the frozen Phase 2b trainer at `ed46cd4`)

- Validation connected accuracy ≥ **0.82**.
- Each of the four (b_A, b_B) quadrants ≥ **0.75**.
- Message-flip residual ≤ **0.15** separately for b_B=0 and b_B=1.
- Zero-message accuracy ≤ **0.53**.
- Sender probe overall ≥ **0.95** and ≥ **0.90** for b_A=0 and b_A=1.
- Mechanism (corrected definitions, element-weighted): full mode — terminal-node (z_B[15])
  trajectory sensitivity **strictly > 0** at ≥1 intermediate step (1–7); final-only —
  ordinary graph-node states **bit-identical** (exact tensor equality) across opposite
  forced messages through step 7.
- Empty conditioning groups are hard failures; non-finite loss/parameters/gradient fail
  the run and make its lambda ineligible.

## 4. Instrumentation per run (JSONL per epoch + summary JSON)

Per epoch: mean train total/parity/aux_A/aux_B/msg-BCE losses; validation connected and
quadrant accuracies; sender probe overall and by b_A; zero-message accuracy; flip
residual per b_B at epochs 5/10/15/20; gradient norms (mean/max) for `core_A`, `core_B`,
`msg_encoder`, `msg_embedder`, `learned_comm_token`, `parity_head`; NaN/Inf counts for
losses/gradients; parameter finiteness; optimizer lr. Final: corrected trajectory
sensitivity (element-weighted, terminal node, overall and per b_B), final-only exact
invariance, gate status with exact failed criteria, wall time, `last.pt` + `epochNN.pt`
checkpoints.

**Checkpoint policy (mandatory, correcting the Stage C-recorded deviation):** `last.pt`
is saved for **every** run — including gate-failing, non-finite, or interrupted runs —
plus `epoch05/10/15/20.pt`. Checkpoints live only in ignored `runs/` directories.

## 5. Predeclared selection rule (declared before outcomes; validation only)

A `lambda_msg` is **eligible** only if ALL of:
1. ≥2 of 3 seeds pass all gates above in **full** mode AND ≥2 of 3 in **final-only**
   mode at that lambda;
2. no run at that lambda is non-finite or has a broken gradient path;
3. full-mode terminal-node trajectory sensitivity > 0 at ≥1 intermediate step AND
   final-only graph-node states bit-identical through step 7 (in every run at that lambda);
4. no clear unresolved pathology in loss/gradient histories (evidence written out).

Among eligible lambdas: (a) highest total all-gate pass count across the six runs;
(b) lowest median per-run validation instability (mean absolute epoch-to-epoch change of
validation connected accuracy, median across the six runs — the corrected definition,
not the previously misreported single-run values); ties prefer the lower lambda only
when other validation evidence is comparable. One common lambda for both modes.

**If no lambda is eligible: `PHASE 2B MESSAGE-SUPERVISION RECIPE NOT YET STABILISED`** —
no Phase 2c confirmatory training. If one is eligible: report `CANDIDATE RECIPE SCREENED
IN` and **stop for owner review**; Phase 2c confirmation is explicitly out of scope here.

## 6. Budget

**120 minutes** wall-clock for all 18 runs. If the cap is approached, finish the current
run safely, classify the grid incomplete, and do not apply the selection rule as though
the grid had completed.

---

## Implementation note (dated 2026-10-17, appended BEFORE any sweep run; no outcome viewed)

The frozen Phase 2b `LatentCommSwarm` interface (verified at base `ed46cd4`) exposes
`flip_message` but not `force_message` (that parameter exists only on the Phase 2c
models, which this study must not touch). The trajectory counterfactual therefore uses
the model's **own emitted bit vs its exact negation** (`flip_message=True`) — the same
opposite-polarity intervention the frozen stability runner used, and mechanistically
equivalent to forced ±1 traces for the sensitivity metrics (the pair is always (m, −m)).
All other design points are unchanged. Frozen model/interface files are not modified.

# Phase 2b Design — COMM-Token Recurrent Message Integration

**Branch:** `experiment/phase2b-comm-token` · **Charter:** `latent_phase2b.md`
**Status (pre-declared before any Phase 2b training run):** design + tests; no training results yet.

## 1. Hypothesis under test

**Main question:** can a single bit produced by Core A at the end of its own recurrent
computation causally modulate Core B's internal graph-node representations *throughout*
Core B's recurrent loop, while graph-to-graph attention remains strictly masked and the
binary bottleneck is preserved?

**Strict scope:** Core A runs 8 graph-masked recurrent steps on Graph A and sends exactly
one scalar `m ∈ {−1, +1}` (STE). Core B runs 8 recurrent steps over a 17-token sequence
(16 graph nodes + 1 COMM register at index 16) initialised from a learned token embedding
plus the message embedding. This is **not** asynchronous, bidirectional, or multi-round
communication; A transmits one terminal bit once, before B's recurrence.

## 2. Architecture and equations

Notation: batch `B`, sequence `S=17`, `d_model=128`.

**Core A (unchanged v7.1):**
`z_A ← encoder(x_A); 8× step(z_A, mask_A); aux_A = decoder(z_A[15]); raw_m = msg_encoder(z_A[15]); m = STE(raw_m)`.

**COMM register initialisation (per example):**
`comm₀ = W_tok + E_m(m)`, with learned token `W_tok ∈ ℝ^{1×1×128}` (init `0.02·𝒩(0,1)`)
and message embedding `E_m(m) = msg_embedder(m) ∈ ℝ^{1×128}` (a Linear(1→128), so the
scalar's learned embedding is internal representation, not extra channel bandwidth).

**Core B:** `z_B⁰ = [ encoder(x_B) ; comm₀ ] ∈ ℝ^{B×17×128}`; then for `t = 1..8`:
`z_B^t = block(z_B^{t−1}, M_t)` where `block` is the parameter-tied v7.1
`GraphAttentionBlock` and `M_t ∈ {0,1}^{17×17}` is the per-step mask:

- graph query `j ∈ [0,15]` → graph key `i ∈ [0,15]` iff `i == j` or edge `i→j` exists
  (`adj[b,i,j]=1` orientation, as v7.1);
- graph query `j` → COMM key 16 iff COMM is visible at step `t`;
- COMM query 16 → COMM key 16 only (self-loop). **Never any graph node.**
- Every query row has ≥1 allowed key; no all-to-all fallback exists.

**Step schedules:**
- `full`: COMM visible to graph queries on steps 1–8.
- `final-only` (matched control): COMM invisible on steps 1–7, visible on step 8 only.
  Because COMM cannot read graph nodes, graph-node states are *bit-identical* across
  message polarities for steps 1–7 (verified by unit test).

**Readout (no late fusion):** `aux_B = decoder(z_B[15])`;
`parity = LayerNorm(128) → Linear(128,64) → GELU → Linear(64,2)` applied to `z_B[15]` only.
The head input shape is asserted to be `[batch, 128]` by test; it receives no `m`, no
message embedding, and no COMM state. The bit can affect output only by modulating the
graph-node trajectory.

**Parameters (verified programmatically, asserted in `test_phase2b.py`):**

| Component | Count |
|---|---|
| core_A (LatentReasoner) | 137,476 |
| core_B (LatentReasoner) | 137,476 |
| msg_encoder Linear(128→1) | 129 |
| msg_embedder Linear(1→128) | 256 |
| learned_comm_token | 128 |
| parity_head (256 + 8,256 + 130) | 8,642 |
| **Total** | **284,107** |

## 3. Why one-way / read-only COMM

If the COMM token could read graph nodes, Graph B node information could hop
node → COMM → distant node in a single step, bypassing the graph-local receptive-field
expansion that Phase 1 established (and making the Phase 1-style T=15 horizon test
meaningless). Restricting COMM to a self-loop makes the *only* global route
COMM → graph nodes — the intended message broadcast — while preserving the graph-mask
theorem: with a line graph, node-0 input reaches node-15's latent exactly at T=15 and
never earlier, even with COMM present and visible (unit-tested).

## 4. The XOR-task limitation (pre-declared)

Phase 2a already showed late decision fusion solves XOR at 97.5%. XOR therefore does
**not require** early message use: a final-step-only model can, in principle, match a
full-step model on parity accuracy. Consequently:

- Task gates (≥82% overall etc.) test *engineering + task success*, not the necessity of
  recurrent integration.
- The mechanistic claim rests on **trajectory-sensitivity measurements** (per-step mean
  |Δz| between m=+1 and m=−1 traces) and the full-vs-final-only contrast, not on parity
  accuracy alone.
- If both modes pass task gates with indistinguishable accuracy, the honest conclusion is
  that the XOR benchmark cannot discriminate ongoing access from late access — a valid
  negative result — and a **message-conditioned graph task** (e.g., the bit selects which
  edge-relation/subgraph Core B must traverse) is the recommended follow-up, with its own
  labels, controls and gates, not a silent replacement of v7.1.

## 5. Pre-declared controls and acceptance criteria

Reference minimums (identical metric definitions/denominators to Phase 2a; applied as
explicit gates; empty conditioning groups are a hard failure, never silently divided by 1):

| Gate | Threshold |
|---|---|
| Overall connected parity accuracy | ≥ 82.0% |
| Each (b_A, b_B) quadrant | ≥ 75.0% |
| Flip residual, per b_B ∈ {0,1} | ≤ 15.0% |
| Zero-message accuracy | ≤ 53.0% |
| Probe overall / per b_A | ≥ 95.0% / ≥ 90.0% |
| (full mode only) message-sensitive intermediate graph-node state | ≥1 intermediate step with mean per-step |Δz| > 0 |

Additional required evidence (reported, not gates): per-step trajectory-sensitivity table
overall and per b_B group; full-vs-final-only comparison; Core B trace counterfactuals on
fixed inputs. Zero-message semantics: the message contribution is replaced by an exact zero
before addition to the learned token (embedder bias bypassed); the constant learned token
remains (it is not Core-A sample information).

Matched pairs: full and final-only models share architecture, parameterisation, data
seeds (42/1337/9999), batch size, optimiser, epochs, loss weights and warm-start
checkpoint; paired seeds 42, 43, 44. An inference-time mask intervention on the full
model is reported only as an extra causal probe, never as the trained control.

Test seed 9999 is evaluated exactly once per final model, after configuration is locked;
if any test result prompts a change, the split is declared consumed and a new split is
created with a documented new seed.

## 6. Test plan (test_phase2b.py; baseline suite unchanged and co-run)

1. Baseline regression: `test_system.py` runs unmodified in the same session.
2. Parameter accounting: 284,107 (above), plus v7.1 292,427 unchanged.
3. COMM mask semantics: graph↔graph = self+predecessors; COMM query self-only; no
   all-false rows; off/on differ only in the graph→COMM key column.
4. Receptive field with COMM present: line-graph gradient node0→node15 exactly 0 for
   T≤14, nonzero at T=15 (no workspace shortcut).
5. Message affects graph trajectory: full mode — per-step trace diffs, ≥2 intermediate
   steps affected incl. Node 15; final-only — steps 1–7 **bit-identical**, step 8 differs.
6. No late-readout fusion: head `in_features==128`; forward-pre-hook captures
   `[batch, 128]`; v7.1 head (256) kept as the late-fusion reference.
7. Binary channel: strict ±1; exact flip inversion; deterministic polarity coverage via
   normal∪flipped union (a random untrained batch alone is not assumed balanced).
8. Zero-message: ablated COMM state exactly equals the bare learned token; embedder
   bias cannot encode the bit; trained-model ablation accuracy is a training gate.
9. Gradient flow: no parameter with `grad is None`; all-zero grads reported diagnostically.
10. Dataset integrity: v7.1 integrity tests retained (not replaced); plus a generator
    determinism check for fixed seeds.
11. Final-step-only mask schedule unit test (steps 1–7 off, step 8 on).
12. Checkpoint compatibility: Phase 1 state dict loads strict=True into both cores;
    only msg_encoder/msg_embedder/learned_comm_token/parity_head are new.

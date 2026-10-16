# Phase 2c Design — Message-Conditioned Reachability (committed before any training)

**Branch:** `experiment/phase2c-message-conditioned-reachability` (from stability state `ed46cd4`)
**Gate status entering this phase:** the Stage 2 stability sweep found **no eligible shared
learning rate** (recipe *not yet stabilised*). Per the charter's stop-gate, Phase 2c
proceeds as **engineering + tests + an explicitly engineering-labelled smoke run only**;
no confirmatory multi-seed comparison is run and the Phase 2c test split is **not
evaluated**. Everything below is nonetheless predeclared in full so a future stabilised
recipe can execute the study without redesign.

## 1. Research question and hypothesis

Under fixed graph-recurrent compute (8 steps, 16 nodes) and a one-bit bottleneck, does
exposing Core A's selector bit throughout Core B's recurrent loop (`full`) improve
solving of a message-conditioned reachability task compared with giving the same bit only
at the final recurrent step (`final-only`)? Hypothesis (not an assumed result): full
access may help Core B select and maintain the relation-specific computation during
propagation. If final-only performs equally well, the valid conclusion is that this task
is solvable by parallel precomputation + late selection — not an implementation failure.

## 2. Task definition

`m = +1` iff `b_A = Reach_A(0→15)` else `m = −1`. Core B's graph carries two independent
directed edge relations R0, R1 (16 nodes; edges only `j−i ∈ {1,2,3}`; per relation:
endpoint degree constraints, reachable ⇒ shortest path in [5,8], unreachable ⇒ no path).
Target: `Y = r0 if b_A = 0 else r1`. Exactly balanced across all eight `(b_A, r0, r1)`
combinations (`num_samples % 8 == 0`); disagreement subset (`r0 ≠ r1`) is exactly 50% by
construction and is the primary subset for message usefulness. The XOR framing is gone.

## 3. Input representation and masks

- `x_B ∈ [batch,16,64]`: `[adj_union(16), adj_R0(16), adj_R1(16), one_hot(16)]`,
  `adj_union = OR(R0,R1)`; `x_A` remains `[batch,16,32]`.
- Graph-to-graph attention mask: union adjacency with the v7.1 orientation (query j sees
  key i iff i=j or edge i→j in ≥1 relation). **Mask independent of m** — the bit never
  alters the topology mask.
- COMM register: identical one-way rules to Phase 2b (graph queries may read COMM per
  schedule; COMM query self-only; no all-to-all fallback; ≥1 allowed key per query).
- Schedules: `full` (steps 1–8) vs `final-only` (step 8 only).

## 4. Models and parameter counts (verified programmatically, unit-asserted)

- `LatentConditionedSwarm` (primary): Core A 137,476 + Core B 141,572 (encoder
  Linear(64→128) adds 4,096 over v7.1) + msg_encoder 129 + msg_embedder 256 +
  learned_comm_token 128 = **279,561**. Task head = Core B's terminal decoder
  (LayerNorm(128)→Linear(128,2), 514 params, included above) applied to `z_B[15]` ONLY.
- `LatentDualRelationSwarm` (isolated diagnostic): same backbone + COMM register; the
  unused decoder is replaced by `nn.Identity()` (no dead parameters); two terminal heads
  `r0_head`/`r1_head` (514 each) supervised separately for r0/r1; at evaluation the
  one-bit selector chooses between the two candidates **after** both are produced
  (never uses labels). Parameters: 137,476 + 141,058 + 129 + 256 + 128 + 1,028 =
  **280,075**.

## 5. Warm-start mapping (documented transfer, unit-tested)

Core A: strict Phase 1 load. Core B: all shape-compatible Phase 1 weights transferred
(encoder LayerNorm, recurrent block, decoder); first encoder layer's columns mapped as
`[0:16]→union`, `[16:48]→exactly zero` (relation channels initially ignored),
`[16:32]→[48:64]` node-ID; bias copied. Unit test proves the mapped encoder reproduces
the Phase 1 encoder bit-exactly when relation channels are zero. This is a documented
adaptation, not an exact full-model load, and is recorded in run metadata.

## 6. Objective

`L = CE(task_logits, Y) + 0.5·CE(aux_A, b_A) + 0.5·BCEWithLogits(raw_m, b_A)`.
No auxiliary losses push Core B to solve both relations (that would encourage parallel
precomputation and change the main question); the dual-relation model carries those
heads/losses separately.

## 7. Splits and seeds (predeclared)

Train/validation/test split seeds **4201 / 13371 / 99991** (chosen before generating any
split or inspecting metrics). Hash namespaces `A:`/`R0:`/`R1:` guarantee no graph recurs
within or across splits. Generator determinism is unit-tested. Paired model seeds
42–46; per seed: reset Python/PyTorch seed before construction, identical warm start,
identical seeded DataLoader generator for both modes, verified-identical fresh initial
weights.

## 8. Predeclared acceptance gates (Phase 2c-specific; Phase 2b XOR gates not reused)

- End-to-end selected-task validation accuracy ≥ 82% overall.
- Each `(b_A, Y)` quadrant ≥ 75%.
- Message probe ≥ 95% overall and ≥ 90% per b_A subgroup.
- Zero-message accuracy ≤ **77% overall** and ≤ **53% on the disagreement subset**
  (mathematically expected no-selector optimum is 75% overall / 50% disagreement; the
  XOR 53%-overall gate would be inappropriate here). Agreement-subset accuracy reported
  separately; near-perfect accuracy there is NOT evidence of message use.
- Forced-message counterfactual accuracy ≥ 75% for each of {m=−1 vs r0} and {m=+1 vs r1}.
- Selector switch consistency ≥ 85% on disagreement cases (prediction flips from the R0
  answer to the R1 answer when the forced bit flips); selector invariance ≥ 85% on
  agreement cases (exploratory threshold, predeclared here before training).
- Mechanism: full mode shows a message-sensitive intermediate graph-node state (incl.
  measurable terminal-node sensitivity by some intermediate step); final-only graph-node
  traces bit-identical through step 7.
- "Stable task success" requires ≥4 of 5 paired seeds satisfying the task/control gates;
  otherwise classification is partial/unstable.

## 9. Comparison statistics and interpretation rule (predeclared)

Primary statistic: paired test accuracy difference Δ = accuracy(full) − accuracy(final-only),
per seed and aggregated, overall and on the disagreement subset; per-example paired
predictions saved outside Git with checksums.
- *Evidence favouring early integration:* stable task success in full mode AND mean paired
  test advantage ≥ +5 pp for full mode with a paired bootstrap interval excluding zero on
  the disagreement subset, replicated across paired seeds (n=5 → conservative language).
- *No measured advantage:* both modes pass, Δ small/uncertain → task solvable by parallel
  computation + late selection.
- *Final-only wins:* report as-is; does not negate demonstrated trajectory sensitivity.
- *Instability/gate failure:* report as instability, not as a scientific rejection.
- Test split (seed 99991) evaluated exactly once, only after architecture/recipe/gates are
  frozen and validation decisions complete; if test results prompt changes, the split is
  declared consumed and a new documented seed is required for any confirmatory rerun.

## 10. Status under the current stability gate

Because no learning rate is eligible, **no Phase 2c training beyond the engineering smoke
run is performed in this iteration**. The smoke run validates shapes, mask direction,
finite loss, gradient flow through every loss path, binary/forced message semantics, and
the absence of a late-fusion path — it produces no research claims and evaluates no test
data.

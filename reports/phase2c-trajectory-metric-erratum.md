# Phase 2c Trajectory-Metric Erratum (2026-10-17)

**Type:** dated addendum/erratum. Historical run directories and logs are unchanged; this
document retracts specific numbers and records the correction with provenance.
**Fix commit:** `9257ad3` ("Fix trajectory metrics and isolate test evaluation") on
`experiment/phase2c-message-conditioned-reachability`.

## 1. The defect

The `evaluate_trajectory()` aggregator in `train_swarm_conditioned.py` (as committed at
`c09ba99`/`fe87e1f`) computed, per batch, the mean absolute difference
`mean(|z⁺_t − z⁻_t|)` over the batch's elements, summed those **per-batch means** across
batches, and then divided by the **total sample count**. Since a per-batch mean is already
normalised by batch size, this understates the true element-weighted mean by a factor of
approximately the evaluation batch size (128 for the smoke/evaluation loaders).

## 2. Corrected definition (now implemented and unit-tested)

With `z⁺_t`, `z⁻_t` the ordinary graph-node states after recurrent step `t` under forced
`m=+1` / `m=−1` on **identical** inputs (eval mode, no grad):

- `graph_nodes_mean_abs_delta[t] = sum(|z⁺_t − z⁻_t|) / (n × 16 × 128)` (all nodes, all dims);
- `terminal_node_mean_abs_delta[t] = sum(|z⁺_t[15] − z⁻_t[15]|) / (n × 128)` (Node 15 only);

each a true element-weighted mean, reported overall and per agreement/disagreement group
with sample counts; empty groups raise an explicit error; non-finite deltas raise an
explicit error. Batch-size invariance and hand-computed weighted subgroup values are
verified by regression tests (`test_trajectory_metric_batch_size_invariance`,
`test_trajectory_metric_hand_computed_weighted_subgroups`). In `final-only` mode, graph-node
states are asserted **bit-identical** (exact tensor equality) through step 7, not merely
near-zero on average.

## 3. Retraction of previously reported magnitudes

The following previously reported values are **retracted for quantitative interpretation**
(and should not be compared against the corrected scale):

- `reports/phase2c-run-report.md` §3 smoke table, "Step-1 traj |Δz|" column:
  primary/full `9.26e-04`, dual/full `1.02e-03` (both from
  `runs/20261016T234452Z_phase2c_smoke/`);
- `WORKLOG.md` Phase C entry quoting the same values;
- the full-scale "Trajectory sensitivity" magnitudes quoted in the Phase 2b reports
  (`reports/phase2b-run-report.md` §4, `reports/phase2b-stability-report.md` §2) were
  produced by a different but similarly per-batch-mean-based aggregator in
  `train_swarm_comm.py` and the stability driver; they carry the same class of
  understatement (factor ≈ batch size 128) and are likewise retracted for quantitative
  use. Their **structural** conclusions are unaffected (see §4).

Corrected values are **not** available for the historical runs: raw per-example/per-step
delta traces were not persisted (only aggregated means were logged), so retroactive
recomputation is impossible without rerunning those runs under a new run ID. No values are
fabricated here.

For orientation only, values computed with the corrected aggregator in the new
fix-only smoke runs (`runs/20261017T063938Z_phase2c_fix_smoke/`, fresh untrained models,
engineering-only): primary/full step-1 graph-nodes mean |Δz| = **1.185e-01**, terminal node
= **2.932e-02** — versus the retracted 9.26e-04 under the old aggregator, confirming the
≈batch-size-factor understatement empirically.

## 4. Structural statements that remain valid (subject to the re-run suite)

These were established by exact tests/logic and do **not** depend on the misaggregated
magnitudes; all were re-verified by the current suite (47 passed at commit `9257ad3`):

- `final-only`: ordinary graph-node states are bit-identical across opposite messages
  through step 7 (exact tensor equality), with influence entering only at step 8;
- `full`: the message changes ordinary graph-node states from the first recurrent step
  (strictly positive corrected means at intermediate steps), including measurable
  terminal-node sensitivity by an intermediate step;
- the zero-message control behaves at chance on every evaluated configuration;
- COMM cannot read graph nodes; graph-to-graph receptive-field behaviour is unchanged.

## 5. Gate semantics change recorded

The full-mode mechanism acceptance check now requires **terminal-node** (z_B[15])
sensitivity strictly > 0 at at least one intermediate step (1–7), replacing the previous
all-nodes criterion, which could be satisfied by sensitivity at an unrelated node. The
final-only check now asserts exact bit-identity through step 7 directly. These strengthen
the predeclared mechanism criteria; they do not alter any accuracy/probe/control threshold.

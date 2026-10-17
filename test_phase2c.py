"""
Phase 2c tests: message-conditioned two-relation reachability.
Run together with the unchanged earlier suites:
    python -m pytest -q test_system.py test_phase2b.py test_phase2c.py
Baseline suites are not modified or replaced.
"""
import os
import pytest
import torch
import torch.nn as nn

from model.swarm import LatentSwarm
from model.swarm_comm import LatentCommSwarm
from model.swarm_conditioned import LatentConditionedSwarm, LatentDualRelationSwarm
from data.conditioned_graph_task import (generate_conditioned_benchmark, benchmark_statistics,
                                         SPLIT_SEEDS, canonical_raw_graph_hash)
from train_swarm_conditioned import compute_loss
from data.graph_task import _shortest_path_length

def setup_module():
    torch.set_num_threads(4)

EXPECTED_PRIMARY_PARAMS = 279_561
EXPECTED_DUAL_PARAMS = 279_691  # corrected: no msg_embedder, no COMM token in the diagnostic

def _line_features(batch_size=1):
    line = torch.zeros(batch_size, 16, 16)
    for b in range(batch_size):
        for i in range(15):
            line[b, i, i + 1] = 1.0
    eye = torch.eye(16).unsqueeze(0).expand(batch_size, -1, -1)
    zeros = torch.zeros_like(line)
    x_B = torch.cat([line, line, zeros, eye], dim=-1)  # union = R0 = line, R1 empty
    return x_B

def test_exact_parameter_counts_phase2c():
    primary = LatentConditionedSwarm()
    dual = LatentDualRelationSwarm()
    assert sum(p.numel() for p in primary.parameters() if p.requires_grad) == EXPECTED_PRIMARY_PARAMS
    assert sum(p.numel() for p in dual.parameters() if p.requires_grad) == EXPECTED_DUAL_PARAMS
    # Earlier models unchanged.
    assert sum(p.numel() for p in LatentSwarm().parameters() if p.requires_grad) == 292_427
    assert sum(p.numel() for p in LatentCommSwarm().parameters() if p.requires_grad) == 284_107

def test_input_layout_and_union():
    ds, _ = generate_conditioned_benchmark(num_samples=16, seed=SPLIT_SEEDS["train"])
    x_A, x_B = ds.tensors[0], ds.tensors[1]
    assert x_A.shape == (16, 16, 32) and x_B.shape == (16, 16, 64)
    union, R0, R1, ids = x_B[:, :, :16], x_B[:, :, 16:32], x_B[:, :, 32:48], x_B[:, :, 48:]
    assert torch.equal(union, ((R0 > 0.5) | (R1 > 0.5)).float())
    assert torch.allclose(ids, torch.eye(16).expand(16, 16, 16))

def test_generator_labels_balance_and_invariants():
    ds, _ = generate_conditioned_benchmark(num_samples=80, seed=SPLIT_SEEDS["validation"])
    x_A, x_B, b_A, r0, r1, Y = ds.tensors
    stats = benchmark_statistics(ds)
    assert all(v == 10 for v in stats["combo_counts"].values()), "8 combos not exactly balanced"
    assert stats["disagreement_fraction"] == 0.5
    assert stats["P(Y=1)"] == 0.5
    assert abs(stats["no_selector_optimum"] - 0.75) < 1e-9
    for i in range(len(Y)):
        adj_A = x_A[i, :, :16]
        d_A = _shortest_path_length(adj_A, 0, 15)
        assert b_A[i].item() == (1 if d_A != -1 else 0)
        if d_A != -1:
            assert 5 <= d_A <= 8
        for slot, lab in ((16, r0[i].item()), (32, r1[i].item())):
            rel = x_B[i, :, slot:slot + 16]
            d = _shortest_path_length(rel, 0, 15)
            assert lab == (1 if d != -1 else 0), f"Relation label mismatch at slot {slot}"
            if d != -1:
                assert 5 <= d <= 8
            assert rel[0].sum() >= 1, "Node 0 out-degree invariant violated"
            assert rel[:, 15].sum() >= 1, "Node 15 in-degree invariant violated"
        expected = r0[i].item() if b_A[i].item() == 0 else r1[i].item()
        assert Y[i].item() == expected, "Selected-target mapping violated"

def test_generator_determinism_and_split_disjointness():
    ds1, h1 = generate_conditioned_benchmark(num_samples=16, seed=4201)
    ds2, h2 = generate_conditioned_benchmark(num_samples=16, seed=4201)
    assert h1 == h2
    for a, b in zip(ds1.tensors, ds2.tensors):
        assert torch.equal(a, b)
    _, h_train = generate_conditioned_benchmark(num_samples=80, seed=SPLIT_SEEDS["train"])
    _, h_val = generate_conditioned_benchmark(num_samples=80, seed=SPLIT_SEEDS["validation"],
                                              existing_hashes=h_train)
    _, h_test = generate_conditioned_benchmark(num_samples=80, seed=SPLIT_SEEDS["test"],
                                               existing_hashes=h_train | h_val)
    assert len(h_train & h_val) == 0 and len(h_train & h_test) == 0 and len(h_val & h_test) == 0
    assert len(h_train) == 3 * 80 and len(h_val) == 3 * 80 and len(h_test) == 3 * 80

def test_union_mask_semantics_and_schedule_independence():
    x_B = _line_features(2)
    swarm = LatentConditionedSwarm()
    mask_off, mask_on = swarm.build_comm_masks(x_B)
    for mask in (mask_off, mask_on):
        gg = (x_B[:, :, :16].transpose(-1, -2) > 0.5) | torch.eye(16, dtype=torch.bool).unsqueeze(0)
        assert torch.equal(mask[:, 0, :16, :16], gg)  # predecessors in the UNION graph only
        assert torch.equal(mask[:, 0, 16, :16], torch.zeros(2, 16, dtype=torch.bool))
        assert torch.equal(mask[:, 0, 16, 16], torch.ones(2, dtype=torch.bool))
        assert bool(mask[:, 0].any(dim=-1).all())
    assert not bool(mask_off[:, 0, :16, 16].any()) and bool(mask_on[:, 0, :16, 16].all())
    # Masks depend only on x_B — never on the message value (structural).
    m_plus = torch.ones(2, 1)
    state_p = swarm.build_comm_state(m_plus, 2)
    state_m = swarm.build_comm_state(-m_plus, 2)
    off_p, on_p = swarm.build_comm_masks(x_B)
    assert torch.equal(off_p, mask_off) and torch.equal(on_p, mask_on)
    assert not torch.equal(state_p, state_m)

def test_graph_receptive_field_with_union_mask():
    torch.manual_seed(123)
    swarm = LatentConditionedSwarm()
    x_B = _line_features(1).requires_grad_(True)
    mask_off, mask_on = swarm.build_comm_masks(x_B)
    comm_state = swarm.build_comm_state(torch.ones(1, 1), 1)
    z = torch.cat([swarm.core_B.encoder(x_B), comm_state], dim=1)
    for _ in range(14):
        z = swarm.core_B.step(z, mask_on)
    z[0, 15, :].sum().backward(retain_graph=True)
    assert torch.all(x_B.grad[0, 0, :] == 0.0), "Graph info leaked before T=15 via union mask/COMM!"
    x_B.grad.zero_()
    z = swarm.core_B.step(z, mask_on)
    z[0, 15, :].sum().backward()
    assert torch.any(x_B.grad[0, 0, :] != 0.0), "Node-0 signal failed to reach Node 15 at T=15!"

def test_forced_message_trajectory_interventions():
    torch.manual_seed(5)
    swarm = LatentConditionedSwarm()
    x_A = torch.randn(4, 16, 32)
    x_B = _line_features(4)
    _, _, _, _, tp = swarm(x_A, x_B, comm_mode="full", force_message=1.0, trace=True)
    _, _, _, _, tn = swarm(x_A, x_B, comm_mode="full", force_message=-1.0, trace=True)
    assert torch.equal(tp[0], tn[0]), "Initial state unexpectedly depends on the message"
    diffs = [float((tp[t] - tn[t]).abs().max()) for t in range(9)]
    assert diffs[1] > 0.0 and sum(d > 0 for d in diffs[1:8]) >= 2, "Forced bit did not move intermediate states"
    node15 = [float((tp[t][:, 15, :] - tn[t][:, 15, :]).abs().max()) for t in range(1, 9)]
    assert sum(d > 0 for d in node15) >= 2, "Terminal node trajectory not message-sensitive"
    _, _, _, _, fp = swarm(x_A, x_B, comm_mode="final-only", force_message=1.0, trace=True)
    _, _, _, _, fn = swarm(x_A, x_B, comm_mode="final-only", force_message=-1.0, trace=True)
    for t in range(1, 8):
        assert torch.equal(fp[t], fn[t]), f"final-only step {t} leaked forced-message influence"
    assert float((fp[8] - fn[8]).abs().max()) > 0.0

def test_readout_contract_terminal_state_only():
    torch.manual_seed(11)
    swarm = LatentConditionedSwarm()
    assert swarm.core_B.decoder[1].in_features == 128  # terminal decoder, not a fusion head
    captured = {}
    def _capture(module, args):
        captured.setdefault("shape", tuple(args[0].shape))
    hook = swarm.core_B.decoder.register_forward_pre_hook(_capture)
    x_A = torch.randn(6, 16, 32)
    x_B = _line_features(6)
    swarm(x_A, x_B)
    hook.remove()
    assert captured["shape"] == (6, 128), f"Task head input {captured['shape']} is not [batch, d_model]"

def test_phase1_warm_start_mapping():
    ckpt = "checkpoint_phase1.pt"
    if not os.path.exists(ckpt):
        pytest.skip("checkpoint_phase1.pt not present")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    swarm = LatentConditionedSwarm()
    swarm.load_phase1_warm_start(state)  # strict for core_A and mapped-strict for core_B
    B = 6
    M = (torch.rand(B, 16, 16) > 0.6).float()
    idt = torch.eye(16).unsqueeze(0).expand(B, -1, -1)
    with torch.no_grad():
        out_old = swarm.core_A.encoder(torch.cat([M, idt], dim=-1))
        out_new = swarm.core_B.encoder(torch.cat([M, torch.zeros(B, 16, 32), idt], dim=-1))
    assert float((out_old - out_new).abs().max()) == 0.0, "Mapped encoder != Phase 1 encoder"
    # Relation channels start at exactly zero.
    assert torch.all(swarm.core_B.encoder[0].weight[:, 16:48] == 0.0)

def test_binary_channel_and_forced_values():
    torch.manual_seed(321)
    swarm = LatentConditionedSwarm()
    x_A = torch.randn(32, 16, 32)
    x_B = _line_features(32)
    _, _, _, m = swarm(x_A, x_B)
    assert torch.all((m == 1.0) | (m == -1.0))
    _, _, _, mf = swarm(x_A, x_B, flip_message=True)
    assert torch.equal(mf, -m)
    _, _, _, forced_p = swarm(x_A, x_B, force_message=1.0)
    _, _, _, forced_m = swarm(x_A, x_B, force_message=-1.0)
    assert torch.all(forced_p == 1.0) and torch.all(forced_m == -1.0)
    union = set(m.flatten().tolist()) | set(mf.flatten().tolist())
    assert union == {-1.0, 1.0}

def test_zero_message_ablation_exact():
    torch.manual_seed(7)
    swarm = LatentConditionedSwarm()
    m = torch.tensor([[1.0], [-1.0]])
    state_zero = swarm.build_comm_state(m, 2, zero_message=True)
    assert torch.equal(state_zero, swarm.learned_comm_token.expand(2, 1, swarm.d_model))
    assert float(swarm.learned_comm_token.abs().sum()) > 0.0

def test_gradient_flow_phase2c():
    for comm_mode in ("full", "final-only"):
        torch.manual_seed(42)
        swarm = LatentConditionedSwarm()
        ds, _ = generate_conditioned_benchmark(num_samples=8, seed=SPLIT_SEEDS["train"])
        tl, raw_m, aux_A, _ = swarm(ds.tensors[0], ds.tensors[1], comm_mode=comm_mode)
        loss = nn.CrossEntropyLoss()(tl, ds.tensors[5]) \
             + 0.5 * nn.CrossEntropyLoss()(aux_A, ds.tensors[2]) \
             + 0.5 * nn.BCEWithLogitsLoss()(raw_m.squeeze(-1), ds.tensors[2].float())
        loss.backward()
        none = [n for n, p in swarm.named_parameters() if p.grad is None]
        assert none == [], f"[{comm_mode}] grad None: {none}"
        assert torch.isfinite(loss).all()

def test_dual_relation_selection_semantics():
    r0 = torch.tensor([[-5.0, 5.0], [5.0, -5.0]])   # predicts 1, then 0
    r1 = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])   # predicts 0, then 1
    m0 = torch.tensor([[-1.0], [-1.0]])
    m1 = torch.tensor([[1.0], [1.0]])
    assert LatentDualRelationSwarm.select(r0, r1, m0).tolist() == [1, 0]
    assert LatentDualRelationSwarm.select(r0, r1, m1).tolist() == [0, 1]
    # Selection depends only on head outputs and the bit — never on labels.
    dual = LatentDualRelationSwarm()
    top = {n.split(".")[0] for n, _ in dual.named_parameters()}
    assert {"r0_head", "r1_head", "msg_encoder"}.issubset(top)
    # No unused message-embedding / COMM parameters remain in the diagnostic.
    assert not any("msg_embedder" in n or "comm_token" in n for n, _ in dual.named_parameters())
    assert not any("learned_comm" in n for n, _ in dual.named_parameters())

    torch.manual_seed(3)
    x_A = torch.randn(4, 16, 32)
    x_B = torch.randn(4, 16, 64)
    base = dual(x_A, x_B, trace=True)          # emitted message polarity
    flipped = dual(x_A, x_B, flip_message=True, trace=True)
    forced_p = dual(x_A, x_B, force_message=1.0, trace=True)
    forced_m = dual(x_A, x_B, force_message=-1.0, trace=True)
    # Candidates and the whole Core B trace are bit-identical under ANY message
    # intervention; only the selection input changes.
    assert torch.equal(base[0], flipped[0]) and torch.equal(base[1], flipped[1])
    assert torch.equal(base[0], forced_p[0]) and torch.equal(base[1], forced_p[1])
    assert torch.equal(base[0], forced_m[0]) and torch.equal(base[1], forced_m[1])
    assert all(torch.equal(a, b) for a, b in zip(base[5], flipped[5]))
    assert all(torch.equal(a, b) for a, b in zip(base[5], forced_p[5]))
    assert all(torch.equal(a, b) for a, b in zip(base[5], forced_m[5]))
    assert torch.equal(flipped[4], -base[4])          # the bit itself does flip
    # Selection changes with the bit while candidates stay exactly unchanged.
    sel_p = dual.select(base[0], base[1], torch.full((4, 1), 1.0))
    sel_m = dual.select(base[0], base[1], torch.full((4, 1), -1.0))
    assert not torch.equal(sel_p, sel_m)

def test_dual_gradient_paths_and_loss_wiring():
    """Dual loss: relation heads supervised with their OWN labels (never Y); aux_A and
    msg BCE give Core A / sender gradients; the discrete selection is not in the loss."""
    torch.manual_seed(42)
    dual = LatentDualRelationSwarm()
    crit, crit_msg = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    batch = (torch.randn(8, 16, 32), torch.randn(8, 16, 64),
             torch.randint(0, 2, (8,)), torch.randint(0, 2, (8,)),
             torch.randint(0, 2, (8,)), torch.randint(0, 2, (8,)))
    outputs = dual(batch[0], batch[1])
    loss = compute_loss("dual", outputs, batch, crit, crit_msg)
    # Manual recomputation must match the wired loss exactly.
    r0l, r1l, raw_m, aux_A, _ = outputs
    expected = crit(r0l, batch[3]) + crit(r1l, batch[4]) + 0.5 * crit(aux_A, batch[2]) \
             + 0.5 * crit_msg(raw_m.squeeze(-1), batch[2].float())
    assert torch.equal(loss, expected), "Dual loss wiring deviates from the predeclared objective"
    # A swapped wiring (r0 head against r1 labels) must give a different value on this batch.
    swapped = crit(r0l, batch[4]) + crit(r1l, batch[3]) + 0.5 * crit(aux_A, batch[2]) \
            + 0.5 * crit_msg(raw_m.squeeze(-1), batch[2].float())
    assert not torch.equal(loss, swapped), "Swap check degenerate; pick a batch with r0 != r1 labels"
    loss.backward()
    none = [n for n, p in dual.named_parameters() if p.grad is None]
    assert none == [], f"Dual parameters without gradient: {none}"
    # Sender gradient path exists via msg BCE (selection is discrete and not differentiated).
    assert dual.msg_encoder.weight.grad is not None and float(dual.msg_encoder.weight.grad.abs().sum()) > 0

def test_primary_loss_wiring():
    torch.manual_seed(42)
    primary = LatentConditionedSwarm()
    crit, crit_msg = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    batch = (torch.randn(8, 16, 32), torch.randn(8, 16, 64),
             torch.randint(0, 2, (8,)), torch.randint(0, 2, (8,)),
             torch.randint(0, 2, (8,)), torch.randint(0, 2, (8,)))
    outputs = primary(batch[0], batch[1])
    loss = compute_loss("primary", outputs, batch, crit, crit_msg)
    logits, raw_m, aux_A, _ = outputs
    expected = crit(logits, batch[5]) + 0.5 * crit(aux_A, batch[2]) \
             + 0.5 * crit_msg(raw_m.squeeze(-1), batch[2].float())
    assert torch.equal(loss, expected), "Primary loss wiring deviates from the predeclared objective"
    # Selected-task loss must use Y (the selector rule), not r0 or r1 directly.
    against_r0 = crit(logits, batch[3]) + 0.5 * crit(aux_A, batch[2]) \
               + 0.5 * crit_msg(raw_m.squeeze(-1), batch[2].float())
    assert not torch.equal(loss, against_r0), "Swap check degenerate; pick a batch with Y != r0 somewhere"

# ------------------------- Fix 1: raw graph-hash uniqueness (2026-10-17) -------------------------

def test_raw_hash_uniqueness_within_split_across_roles():
    ds, hashes = generate_conditioned_benchmark(num_samples=80, seed=SPLIT_SEEDS["train"])
    assert len(hashes) == 3 * 80, "Returned set must hold exactly one raw hash per graph"
    assert all(":" not in h for h in hashes), "Uniqueness keys must not contain role prefixes"
    x_A, x_B = ds.tensors[0], ds.tensors[1]
    # Recomputed raw hashes of the stored matrices must reproduce the returned set.
    recomputed = set()
    for i in range(len(x_A)):
        recomputed.add(canonical_raw_graph_hash(x_A[i, :, :16]))
        recomputed.add(canonical_raw_graph_hash(x_B[i, :, 16:32]))
        recomputed.add(canonical_raw_graph_hash(x_B[i, :, 32:48]))
    assert recomputed == hashes

def test_raw_hash_disjoint_across_small_fixtures():
    _, h1 = generate_conditioned_benchmark(num_samples=16, seed=101)
    _, h2 = generate_conditioned_benchmark(num_samples=16, seed=102, existing_hashes=h1)
    _, h3 = generate_conditioned_benchmark(num_samples=16, seed=103, existing_hashes=h1 | h2)
    assert len(h1) == len(h2) == len(h3) == 3 * 16
    assert not (h1 & h2) and not (h1 & h3) and not (h2 & h3)

def test_canonical_hash_is_dtype_and_layout_invariant():
    adj = torch.zeros(16, 16)
    for i in range(15):
        adj[i, i + 1] = 1.0
    h_f32 = canonical_raw_graph_hash(adj)
    h_f64 = canonical_raw_graph_hash(adj.to(torch.float64))
    h_noncontig = canonical_raw_graph_hash(adj.t().contiguous().t())  # same values, transposed view made strided then re-strided
    h_gpu_like = canonical_raw_graph_hash(adj.clone())
    assert h_f32 == h_f64 == h_noncontig == h_gpu_like
    other = adj.clone()
    other[0, 15] = 1.0
    assert canonical_raw_graph_hash(other) != h_f32

def test_generator_rejects_repeated_raw_graphs_across_roles(monkeypatch):
    """Deterministic focused test: a fixed sampler emitting repeats forces the generator
    to reject duplicate raw hashes regardless of role and keep sampling. The pool holds
    40 provably distinct graphs for 24 slots, so repeats are rejected but exhaustion
    never occurs."""
    import data.conditioned_graph_task as cgt
    pairs = [(a, b) for a in range(0, 4) for b in range(a + 2, 16)][:40]  # b>=a+2: never a line edge
    assert len(set(pairs)) == 40
    graphs = []
    for (a, b) in pairs:
        g = torch.zeros(16, 16)
        for i in range(15):
            g[i, i + 1] = 1.0
        g[a, b] = 1.0
        graphs.append(g)
    assert len({canonical_raw_graph_hash(g) for g in graphs}) == 40, "fixture graphs must be distinct"
    # Sequence proposes each of the 24 needed graphs exactly twice: the first proposal
    # is accepted, the repeat (under the next role) must be rejected. 48 calls total.
    sequence = [g for g in graphs[:24] for _ in (0, 1)]
    calls = {"n": 0}
    def fake_sampler(num_nodes, force_reachable, rng):
        g = sequence[calls["n"] % len(sequence)]
        calls["n"] += 1
        return g.clone(), "unused"
    monkeypatch.setattr(cgt, "_sample_matched_dag", fake_sampler)
    ds, hashes = cgt.generate_conditioned_benchmark(num_samples=8, seed=7)
    assert len(hashes) == 3 * 8
    # Acceptances occur at odd calls, rejections at even calls; the 24th acceptance
    # (call 47) completes the dataset before g24's repeat would be proposed.
    assert calls["n"] == 2 * 3 * 8 - 1, "Unexpected proposal pattern"

def test_generator_bounded_failure_on_exhaustion(monkeypatch):
    import data.conditioned_graph_task as cgt
    def constant_sampler(num_nodes, force_reachable, rng):
        g = torch.zeros(16, 16)
        for i in range(15):
            g[i, i + 1] = 1.0
        return g.clone(), "unused"
    monkeypatch.setattr(cgt, "_sample_matched_dag", constant_sampler)
    monkeypatch.setattr(cgt, "MAX_ATTEMPTS_PER_GRAPH", 5)
    with pytest.raises(RuntimeError, match="structurally unique"):
        cgt.generate_conditioned_benchmark(num_samples=8, seed=7)

# ------------------- Fix 4: corrected trajectory aggregation (2026-10-17) -------------------

import math
import train_swarm_conditioned as trainer_mod
from train_swarm_conditioned import evaluate_trajectory, _terminal_sensitivity_ok
import evaluate_phase2c_test as evaluator_mod


class _MockTrajModel:
    """Deterministic mock returning fixed counterfactual traces; lets the aggregation
    arithmetic be verified by hand independent of any real network."""

    def __init__(self, traces_by_force):
        self.traces = traces_by_force

    def eval(self):
        return self

    def __call__(self, x_A, x_B, comm_mode="full", trace=True, force_message=None, **kw):
        assert trace is True
        return (None, None, None, None, None, self.traces[force_message])


def _traj_loader(x_B, r0, r1):
    n = x_B.size(0)
    ds = torch.utils.data.TensorDataset(
        torch.randn(n, 16, 32), x_B, torch.zeros(n, dtype=torch.long),
        r0.clone(), r1.clone(), torch.zeros(n, dtype=torch.long))
    return torch.utils.data.DataLoader(ds, batch_size=x_B.size(0))


def test_trajectory_metric_batch_size_invariance():
    torch.manual_seed(11)
    model = LatentConditionedSwarm()
    x_B, r0, r1 = _tiny_conditioned_batch(n=12)
    ref = evaluate_trajectory(model, _traj_loader(x_B, r0, r1), "full")
    split = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(12, 16, 32), x_B,
                                       torch.zeros(12, dtype=torch.long), r0.clone(),
                                       r1.clone(), torch.zeros(12, dtype=torch.long)),
        batch_size=5, shuffle=False)
    uneven = evaluate_trajectory(model, split, "full")
    for key in ("graph_nodes_mean_abs_delta", "terminal_node_mean_abs_delta"):
        for g in ("overall", "agree", "disagree"):
            for t in range(9):
                a, b = ref[key][g][t], uneven[key][g][t]
                assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9), \
                    f"Batch-size dependence at {key}/{g}/step{t}: {a} vs {b}"
    assert ref["sample_counts"] == uneven["sample_counts"]

def test_trajectory_metric_hand_computed_weighted_subgroups():
    traces_p = [torch.zeros(3, 16, 128) for _ in range(9)]
    traces_m = [torch.zeros(3, 16, 128) for _ in range(9)]
    # sample 0 (agree): delta 2.0 everywhere; sample 1 (disagree): 0.0; sample 2 (agree): 4.0
    traces_p[1][0] = 2.0
    traces_m[1][0] = 0.0
    traces_p[1][2] = 4.0
    x_B = torch.randn(3, 16, 64)
    r0 = torch.tensor([0, 0, 1])
    r1 = torch.tensor([0, 1, 1])  # sample0 agree, sample1 disagree, sample2 agree
    model = _MockTrajModel({1.0: traces_p, -1.0: traces_m})
    out = evaluate_trajectory(model, _traj_loader(x_B, r0, r1), "full")
    # graph-node overall: (2*2048 + 0 + 4*2048) / (3*2048) = 2.0
    assert math.isclose(out["graph_nodes_mean_abs_delta"]["overall"][1], 2.0, rel_tol=1e-9)
    # agree group: (2*2048 + 4*2048) / (2*2048) = 3.0 ; disagree: 0.0
    assert math.isclose(out["graph_nodes_mean_abs_delta"]["agree"][1], 3.0, rel_tol=1e-9)
    assert out["graph_nodes_mean_abs_delta"]["disagree"][1] == 0.0
    # terminal node: delta is constant across nodes, so identical values
    assert math.isclose(out["terminal_node_mean_abs_delta"]["overall"][1], 2.0, rel_tol=1e-9)
    assert math.isclose(out["terminal_node_mean_abs_delta"]["agree"][1], 3.0, rel_tol=1e-9)
    assert out["sample_counts"] == {"overall": 3, "agree": 2, "disagree": 1}
    assert _terminal_sensitivity_ok(out) is True  # strictly positive at intermediate step

def test_trajectory_full_mode_terminal_sensitivity_deterministic():
    torch.manual_seed(5)
    model = LatentConditionedSwarm()
    x_B = torch.zeros(4, 16, 64)
    x_B[:, :, 48:] = torch.eye(16)
    for i in range(15):
        x_B[:, i, i + 1] = 1.0  # union = R0-like line in column block 0
        x_B[:, i, 16 + i + 1] = 1.0
    r0, r1 = torch.tensor([1, 0, 1, 0]), torch.tensor([0, 1, 1, 0])  # 2 disagree + 2 agree
    out = evaluate_trajectory(model, _traj_loader(x_B, r0, r1), "full")
    assert _terminal_sensitivity_ok(out), \
        f"No terminal-node intermediate sensitivity: {out['terminal_node_mean_abs_delta']['overall']}"
    assert any(v > 0 for v in out["graph_nodes_mean_abs_delta"]["overall"][1:8])

def test_trajectory_final_only_exact_invariance():
    torch.manual_seed(5)
    model = LatentConditionedSwarm()
    x_B, r0, r1 = _tiny_conditioned_batch(n=8)
    out = evaluate_trajectory(model, _traj_loader(x_B, r0, r1), "final-only")
    assert out["final_only_steps_1_7_exact_invariance"] is True
    for t in range(1, 8):
        assert out["graph_nodes_mean_abs_delta"]["overall"][t] == 0.0
        assert out["terminal_node_mean_abs_delta"]["overall"][t] == 0.0
    assert out["terminal_node_mean_abs_delta"]["overall"][8] > 0.0, \
        "final-only step 8 should show message influence"

def test_trajectory_rejects_nonfinite_and_empty_groups():
    traces_p = [torch.zeros(2, 16, 128) for _ in range(9)]
    traces_m = [torch.zeros(2, 16, 128) for _ in range(9)]
    traces_p[3][0, 0, 0] = float("nan")
    model = _MockTrajModel({1.0: traces_p, -1.0: traces_m})
    x_B = torch.randn(2, 16, 64)
    with pytest.raises(RuntimeError, match="Non-finite trajectory delta"):
        evaluate_trajectory(model, _traj_loader(x_B, torch.tensor([0, 0]),
                                                torch.tensor([0, 1])), "full")
    # Empty disagreement group must fail explicitly, never divide by one silently.
    ok_p = [torch.ones(2, 16, 128) for _ in range(9)]
    ok_m = [torch.zeros(2, 16, 128) for _ in range(9)]
    model2 = _MockTrajModel({1.0: ok_p, -1.0: ok_m})
    with pytest.raises(RuntimeError, match="Empty agreement/disagreement"):
        evaluate_trajectory(model2, _traj_loader(x_B, torch.tensor([0, 1]),
                                                torch.tensor([0, 1])), "full")

# ------------------- Fix 3: training/test separation (2026-10-17) -------------------

def _tiny_conditioned_batch(n=8):
    x_A = torch.randn(n, 16, 32)
    x_B = torch.randn(n, 16, 64)
    x_B[:, :, 48:] = torch.eye(16)
    r0 = torch.randint(0, 2, (n,))
    r1 = torch.randint(0, 2, (n,))
    b_A = torch.randint(0, 2, (n,))
    Y = torch.where(b_A == 0, r0, r1)
    return x_B, r0, r1  # x_A/Y used by callers that need them

def test_trainer_generates_only_train_and_val(monkeypatch):
    import data.conditioned_graph_task as cgt
    real = cgt.generate_conditioned_benchmark
    def guard(num_samples, seed, existing_hashes=None, **kw):
        assert seed != 99991, "The predeclared test split must never be generated by training."
        return real(num_samples=num_samples, seed=seed, existing_hashes=existing_hashes)
    monkeypatch.setattr(cgt, "generate_conditioned_benchmark", guard)
    monkeypatch.setattr(trainer_mod, "generate_conditioned_benchmark", guard)
    loaders = trainer_mod.build_loaders(seed=42)
    assert len(loaders) == 2, f"Trainer must produce only train and val loaders, got {len(loaders)}"
    src = open("train_swarm_conditioned.py").read()
    assert 'SPLIT_SEEDS["test"]' not in src, "Trainer source must not reference the test split"
    # (The seed value may appear in explanatory comments; what must never appear is a
    # generation/evaluation call path for it, guarded above and by the SPLIT_SEEDS check.)

def test_smoke_path_never_builds_test_split(monkeypatch):
    calls = {"test_seed": 0}
    real = trainer_mod.generate_conditioned_benchmark
    def tiny_builder(num_samples, seed, existing_hashes=None, **kw):
        if seed == 99991:
            calls["test_seed"] += 1
            raise AssertionError("smoke attempted to generate the test split")
        n = 8
        x_A = torch.randn(n, 16, 32)
        x_B = torch.randn(n, 16, 64)
        x_B[:, :, 48:] = torch.eye(16)
        r0, r1 = torch.randint(0, 2, (n,)), torch.randint(0, 2, (n,))
        b_A = torch.randint(0, 2, (n,))
        Y = torch.where(b_A == 0, r0, r1)
        return torch.utils.data.TensorDataset(x_A, x_B, b_A, r0, r1, Y), {f"h{seed}{i}" for i in range(3 * n)}
    monkeypatch.setattr(trainer_mod, "generate_conditioned_benchmark", tiny_builder)
    args = trainer_mod.parse_args(["--mode", "full", "--model", "primary", "--smoke"])
    trainer_mod.train_phase2c(args)  # must complete without touching seed 99991
    assert calls["test_seed"] == 0

def test_evaluator_rejects_unfrozen_missing_and_mismatch(tmp_path):
    manifest = {"kind": "phase2c_test_evaluation_manifest", "frozen": True,
                "created_utc": "2026-10-17T00:00:00Z", "source_commit": "abc123",
                "split_seed": 99991,
                "entries": [{"model": "primary", "mode": "full", "seed": 42,
                             "checkpoint": str(tmp_path / "ck.pt"),
                             "checkpoint_sha256": "0" * 64,
                             "phase1_checkpoint": "checkpoint_phase1.pt",
                             "phase1_checkpoint_sha256": _phase1_sha(),
                             "config": {"lr": 8e-4, "epochs": 20}}]}
    import json as _json
    p = tmp_path / "m.json"
    p.write_text(_json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Checkpoint missing"):
        evaluator_mod.run_evaluation(str(p), str(tmp_path / "out"))
    # Create a real checkpoint but record a wrong checksum.
    torch.save(LatentConditionedSwarm().state_dict(), tmp_path / "ck.pt")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        evaluator_mod.run_evaluation(str(p), str(tmp_path / "out"))
    # Correct checksum but unfrozen manifest -> refuse.
    manifest["entries"][0]["checkpoint_sha256"] = evaluator_mod._sha256(str(tmp_path / "ck.pt"))
    manifest["frozen"] = False
    p.write_text(_json.dumps(manifest))
    with pytest.raises(RuntimeError, match="not declared frozen"):
        evaluator_mod.run_evaluation(str(p), str(tmp_path / "out"))
    # Ambiguity: duplicate entries.
    manifest["frozen"] = True
    manifest["entries"].append(dict(manifest["entries"][0]))
    p.write_text(_json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Ambiguous manifest"):
        evaluator_mod.run_evaluation(str(p), str(tmp_path / "out"))

def test_evaluator_end_to_end_on_mocked_split(tmp_path, monkeypatch):
    """The evaluator runs end-to-end on a TINY mocked split; the real seed-99991 split is
    never generated (the builder is replaced). The evaluator itself remains uninvoked in
    any real evaluation during this fix-only task."""
    torch.manual_seed(0)
    primary = LatentConditionedSwarm()
    primary.load_phase1_warm_start(torch.load("checkpoint_phase1.pt",
                                              map_location="cpu", weights_only=True))
    ck = tmp_path / "ck.pt"
    torch.save(primary.state_dict(), ck)
    def tiny_split(num_samples, seed, existing_hashes=None, **kw):
        n = 8
        x_A = torch.randn(n, 16, 32)
        x_B = torch.randn(n, 16, 64)
        x_B[:, :, 48:] = torch.eye(16)
        r0, r1 = torch.randint(0, 2, (n,)), torch.randint(0, 2, (n,))
        b_A = torch.randint(0, 2, (n,))
        Y = torch.where(b_A == 0, r0, r1)
        return (torch.utils.data.TensorDataset(x_A, x_B, b_A, r0, r1, Y),
                {f"t{i}" for i in range(3 * n)})
    monkeypatch.setattr(evaluator_mod, "generate_conditioned_benchmark", tiny_split)
    import json as _json
    manifest = {"kind": "phase2c_test_evaluation_manifest", "frozen": True,
                "created_utc": "2026-10-17T00:00:00Z", "source_commit": "fixpass",
                "split_seed": 99991,
                "entries": [{"model": "primary", "mode": "full", "seed": 42,
                             "checkpoint": str(ck),
                             "checkpoint_sha256": evaluator_mod._sha256(str(ck)),
                             "phase1_checkpoint": "checkpoint_phase1.pt",
                             "phase1_checkpoint_sha256": _phase1_sha(),
                             "config": {"lr": 8e-4, "epochs": 20}}]}
    p = tmp_path / "m.json"
    p.write_text(_json.dumps(manifest))
    out = tmp_path / "out"
    evaluator_mod.run_evaluation(str(p), str(out))
    assert (out / "test_summary.csv").exists()
    assert (out / "test_metrics_primary_full_seed42.json").exists()
    assert (out / "predictions_primary_full_seed42.pt").exists()

def _phase1_sha():
    import hashlib
    h = hashlib.sha256()
    with open("checkpoint_phase1.pt", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

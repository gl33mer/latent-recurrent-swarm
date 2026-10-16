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
from data.conditioned_graph_task import generate_conditioned_benchmark, benchmark_statistics, SPLIT_SEEDS
from data.graph_task import _shortest_path_length

def setup_module():
    torch.set_num_threads(4)

EXPECTED_PRIMARY_PARAMS = 279_561
EXPECTED_DUAL_PARAMS = 280_075

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
    assert {"r0_head", "r1_head"}.issubset(top)
    assert float(dual.r0_head[1].weight.sum()) != 0.0  # heads are fresh, not the decoder
    with torch.no_grad():
        x_A, x_B = torch.randn(2, 16, 32), _line_features(2)
        r0l, r1l, raw_m, _, _ = dual(x_A, x_B, trace=False)
    assert r0l.shape == (2, 2) and r1l.shape == (2, 2)

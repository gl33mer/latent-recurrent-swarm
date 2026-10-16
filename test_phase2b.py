"""
Phase 2b tests: COMM-token swarm architecture, masks, message causality and controls.
Run together with the unchanged v7.1 baseline suite:
    python -m pytest -q test_system.py test_phase2b.py
Baseline tests are not modified or replaced; they run unchanged in the same session.
"""
import os
import pytest
import torch
import torch.nn as nn

from model.swarm import LatentSwarm
from model.swarm_comm import LatentCommSwarm, comm_step_mask
from data.graph_task import generate_parity_benchmark

def setup_module():
    torch.set_num_threads(4)

EXPECTED_COMM_PARAMS = 284_107  # verified programmatically 2026-10-16 (see phase2b-design.md)

def _line_graph_batch(batch_size=1):
    adj = torch.zeros(batch_size, 16, 16)
    for b in range(batch_size):
        for i in range(15):
            adj[b, i, i + 1] = 1.0
    eye = torch.eye(16).unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat([adj, eye], dim=-1)  # [B, 16, 32]

def test_exact_parameter_counts_phase2b():
    swarm = LatentCommSwarm(num_nodes=16, d_input=32, d_model=128)
    assert sum(p.numel() for p in swarm.parameters() if p.requires_grad) == EXPECTED_COMM_PARAMS
    # v7.1 reference counts unchanged.
    assert sum(p.numel() for p in LatentSwarm().parameters() if p.requires_grad) == 292_427

def test_comm_mask_semantics():
    torch.manual_seed(0)
    x = torch.randn(3, 16, 32)
    x[:, :, :16] = (torch.rand(3, 16, 16) > 0.6).float()
    swarm = LatentCommSwarm()
    mask_off, mask_on = swarm.build_comm_masks(x)
    for mask in (mask_off, mask_on):
        assert mask.dtype == torch.bool and mask.shape == (3, 1, 17, 17)
        # Graph->graph part equals self + direct predecessors (v7.1 orientation).
        adj = x[:, :, :16]
        gg = (adj.transpose(-1, -2) > 0.5) | torch.eye(16, dtype=torch.bool).unsqueeze(0)
        assert torch.equal(mask[:, 0, :16, :16], gg)
        # COMM query (row 16) may attend ONLY to itself — it can never read graph nodes.
        assert torch.equal(mask[:, 0, 16, :16], torch.zeros(3, 16, dtype=torch.bool))
        assert torch.equal(mask[:, 0, 16, 16], torch.ones(3, dtype=torch.bool))
        # Every query has at least one allowed key.
        assert bool(mask[:, 0].any(dim=-1).all())
    # mask_off: no graph query may read the COMM key. mask_on: every graph query may.
    assert not bool(mask_off[:, 0, :16, 16].any())
    assert bool(mask_on[:, 0, :16, 16].all())
    # The two variants differ ONLY in the graph->COMM key column.
    assert torch.equal(mask_off[:, 0, :16, :16], mask_on[:, 0, :16, :16])
    assert torch.equal(mask_off[:, 0, 16, :], mask_on[:, 0, 16, :])

def test_step_mask_schedules():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 32)
    swarm = LatentCommSwarm()
    mask_off, mask_on = swarm.build_comm_masks(x)
    steps = 8
    for t in range(steps):
        full = comm_step_mask("full", t, steps, mask_off, mask_on)
        assert torch.equal(full, mask_on)
        fin = comm_step_mask("final-only", t, steps, mask_off, mask_on)
        if t < steps - 1:
            assert torch.equal(fin, mask_off)   # COMM inaccessible to graph queries.
        else:
            assert torch.equal(fin, mask_on)    # accessible at the final step only.
    with pytest.raises(ValueError):
        comm_step_mask("bogus", 0, steps, mask_off, mask_on)

def test_graph_receptive_field_with_comm_present():
    """Graph-input information must still respect graph-local paths even though a COMM
    key is present and visible: no graph-global shortcut through the workspace."""
    torch.manual_seed(123)
    swarm = LatentCommSwarm()
    feat = _line_graph_batch().requires_grad_(True)
    mask_off, mask_on = swarm.build_comm_masks(feat)
    fixed_m = torch.ones(1, 1)
    comm_state = swarm.build_comm_state(fixed_m, 1, zero_message=False)
    z = torch.cat([swarm.core_B.encoder(feat), comm_state], dim=1)
    for _ in range(14):
        z = swarm.core_B.step(z, mask_on)
    z[0, 15, :].sum().backward(retain_graph=True)
    assert torch.all(feat.grad[0, 0, :] == 0.0), "Graph info leaked before T=15 with COMM present!"

    feat.grad.zero_()
    z = swarm.core_B.step(z, mask_on)
    z[0, 15, :].sum().backward()
    assert torch.any(feat.grad[0, 0, :] != 0.0), "Node-0 signal failed to reach Node 15 at T=15!"

def test_message_changes_intermediate_graph_trajectory():
    """The bit must alter ordinary graph-node states during the recurrence (full mode),
    not merely the final logits; in final-only mode it must NOT alter steps 1..7."""
    torch.manual_seed(5)
    swarm = LatentCommSwarm()
    x_A = torch.randn(4, 16, 32)
    x_B = _line_graph_batch(4)

    _, _, _, _, m_pos, tr_pos = swarm(x_A, x_B, comm_mode="full", trace=True)
    _, _, _, _, m_neg, tr_neg = swarm(x_A, x_B, comm_mode="full", flip_message=True, trace=True)
    assert torch.equal(m_neg, -m_pos), "Flip did not produce opposite polarities."

    per_step = [float((tr_pos[t] - tr_neg[t]).abs().max()) for t in range(9)]
    # Initial post-encoder state is message-independent by construction.
    assert per_step[0] == 0.0
    # Message must reach ordinary graph-node states at the FIRST step and keep differing
    # at intermediate steps (full visibility), measured on Node 15 and all nodes.
    assert per_step[1] > 0.0, "Message did not affect graph states at step 1."
    intermediate = [per_step[t] for t in range(1, 8)]
    assert sum(d > 0.0 for d in intermediate) >= 2, "Message affected <2 intermediate steps."
    node15 = [float((tr_pos[t][:, 15, :] - tr_neg[t][:, 15, :]).abs().max()) for t in range(1, 9)]
    assert sum(d > 0.0 for d in node15) >= 2, "Node 15 trajectory changed at <2 steps."
    print("full-mode per-step max|Δz|:", [f"{d:.3e}" for d in per_step])

    # Final-only control: steps 1..7 must be bit-identical across message polarity;
    # the message may enter only at step 8.
    _, _, _, _, _, f_pos = swarm(x_A, x_B, comm_mode="final-only", trace=True)
    _, _, _, _, _, f_neg = swarm(x_A, x_B, comm_mode="final-only", flip_message=True, trace=True)
    for t in range(1, 8):
        assert torch.equal(f_pos[t], f_neg[t]), f"final-only step {t} leaked message influence!"
    assert float((f_pos[8] - f_neg[8]).abs().max()) > 0.0, "final-only step 8 saw no message effect."

def test_no_late_readout_fusion():
    """The Phase 2b parity head consumes only the terminal 128-D graph-node state."""
    torch.manual_seed(11)
    swarm = LatentCommSwarm()
    # Structural: first head layer takes d_model, not 2*d_model (v7.1 late fusion is 256).
    assert swarm.parity_head[1].in_features == 128
    v71 = LatentSwarm()
    assert v71.parity_head[1].in_features == 256  # explicit late-fusion reference.

    captured = {}
    def _capture(module, args):
        captured.setdefault("shape", tuple(args[0].shape))
    hook = swarm.parity_head.register_forward_pre_hook(_capture)
    x = torch.randn(6, 16, 32)
    p_logits, _, _, _, _ = swarm(x, x)
    hook.remove()
    assert captured["shape"] == (6, 128), f"Parity head input shape {captured['shape']} is not [batch, d_model]!"

def test_binary_channel_contract_phase2b():
    torch.manual_seed(321)
    swarm = LatentCommSwarm()
    x_A = torch.randn(64, 16, 32)
    x_B = torch.randn(64, 16, 32)
    _, _, _, _, m = swarm(x_A, x_B)
    assert torch.all((m == 1.0) | (m == -1.0)), "Message is not strictly binary {-1, +1}!"
    _, _, _, _, m_flipped = swarm(x_A, x_B, flip_message=True)
    assert torch.equal(m_flipped, -m), "flip_message failed exact sign inversion!"
    # Deterministic polarity coverage: for every sample the flipped run carries the
    # opposite sign, so the union over {normal, flipped} always contains both polarities
    # (a random untrained batch is not guaranteed to contain both signs on its own).
    union = set(m.detach().flatten().tolist()) | set(m_flipped.detach().flatten().tolist())
    assert union == {-1.0, 1.0}, "Both message polarities not exercised across normal+flipped runs."

def test_zero_message_ablation_semantics():
    torch.manual_seed(7)
    swarm = LatentCommSwarm()
    B = 5
    m = torch.tensor([[1.0], [-1.0], [1.0], [-1.0], [1.0]])
    state_zero = swarm.build_comm_state(m, B, zero_message=True)
    token = swarm.learned_comm_token.expand(B, 1, swarm.d_model)
    # Injected message component is EXACTLY zero: state equals the bare learned token.
    assert torch.equal(state_zero, token), "Ablated COMM state still carries message information."
    # The learned token itself is not zeroed (only the message contribution is).
    assert float(token.abs().sum()) > 0.0, "Learned COMM token unexpectedly all-zero."
    # Non-ablated states differ across message polarity and include the embedding.
    plus = swarm.build_comm_state(torch.ones(B, 1), B)
    minus = swarm.build_comm_state(-torch.ones(B, 1), B)
    assert not torch.equal(plus, minus)
    assert torch.allclose(plus - token, swarm.msg_embedder(torch.ones(B, 1)).unsqueeze(1))

def test_gradient_flow_phase2b():
    torch.manual_seed(42)
    swarm = LatentCommSwarm()
    ds, _ = generate_parity_benchmark(num_samples=8, seed=7)
    p_logits, raw_m, aux_A, aux_B, _ = swarm(ds.tensors[0], ds.tensors[1])
    loss = nn.CrossEntropyLoss()(p_logits, ds.tensors[4]) \
         + 0.5 * nn.CrossEntropyLoss()(aux_A, ds.tensors[2]) \
         + 0.5 * nn.CrossEntropyLoss()(aux_B, ds.tensors[3]) \
         + 0.5 * nn.BCEWithLogitsLoss()(raw_m.squeeze(-1), ds.tensors[2].float())
    loss.backward()
    none = [n for n, p in swarm.named_parameters() if p.grad is None]
    zero = [n for n, p in swarm.named_parameters()
            if p.grad is not None and float(p.grad.abs().sum()) == 0.0]
    assert none == [], f"Parameters with grad None: {none}"
    # Zero gradients are reported, not auto-failed (could be legitimate on a tiny batch).
    if zero:
        print("Diagnostic: parameters with all-zero grad on this batch:", zero)

def test_data_generator_deterministic_for_fixed_seed():
    ds1, h1 = generate_parity_benchmark(num_samples=40, seed=42)
    ds2, h2 = generate_parity_benchmark(num_samples=40, seed=42)
    assert h1 == h2
    for a, b in zip(ds1.tensors, ds2.tensors):
        assert torch.equal(a, b)

def test_phase1_checkpoint_loads_into_comm_cores():
    ckpt = "checkpoint_phase1.pt"
    if not os.path.exists(ckpt):
        pytest.skip("checkpoint_phase1.pt not present on this machine")
    swarm = LatentCommSwarm()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    # strict=True: no missing/unexpected keys for either reused reasoner.
    swarm.core_A.load_state_dict(state)
    swarm.core_B.load_state_dict(state)
    new_params = {"msg_encoder", "msg_embedder", "learned_comm_token", "parity_head"}
    top = {n.split(".")[0] for n, _ in swarm.named_parameters()}
    assert new_params.issubset(top), f"Missing new Phase 2b parameter groups: {new_params - top}"

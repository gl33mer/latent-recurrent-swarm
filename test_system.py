"""
System tests for architecture, receptive-field horizon, data integrity,
parity head expressivity, binary-channel behaviour and a 1-hop local baseline.
Run: python -m pytest -q test_system.py
"""
import pytest
import torch
import torch.nn as nn
from model.core import LatentReasoner
from model.swarm import LatentSwarm
from data.graph_task import generate_parity_benchmark, _shortest_path_length

def setup_module():
    torch.set_num_threads(4)

def test_exact_parameter_counts():
    reasoner = LatentReasoner(num_nodes=16, d_input=32, d_model=128, d_output=2)
    assert sum(p.numel() for p in reasoner.parameters() if p.requires_grad) == 137_476

    swarm = LatentSwarm(num_nodes=16, d_input=32, d_model=128)
    assert sum(p.numel() for p in swarm.parameters() if p.requires_grad) == 292_427

def test_receptive_field_horizon_theorem():
    torch.manual_seed(123)
    model = LatentReasoner(num_nodes=16, d_input=32, d_model=128)
    adj = torch.zeros((1, 16, 16), dtype=torch.float32)
    for i in range(15):
        adj[0, i, i + 1] = 1.0

    eye = torch.eye(16).unsqueeze(0)
    feat = torch.cat([adj, eye], dim=-1)
    feat.requires_grad_(True)

    mask = model.build_attention_mask(feat)
    z = model.encoder(feat)
    for _ in range(14):
        z = model.step(z, mask)

    loss_14 = z[0, 15, :].sum()
    loss_14.backward(retain_graph=True)
    assert torch.all(feat.grad[0, 0, :] == 0.0), "Information leaked before T=15!"

    feat.grad.zero_()
    z = model.step(z, mask)
    loss_15 = z[0, 15, :].sum()
    loss_15.backward()
    assert torch.any(feat.grad[0, 0, :] != 0.0), "Signal failed to reach Node 15 at T=15!"

def test_binary_channel_contract():
    torch.manual_seed(321)
    swarm = LatentSwarm()
    dummy_A = torch.randn(64, 16, 32)
    dummy_B = torch.randn(64, 16, 32)

    _, _, _, _, m = swarm(dummy_A, dummy_B)
    assert torch.all((m == 1.0) | (m == -1.0)), "Message is not strictly binary {-1, 1}!"
    # The seeded, sufficiently large random batch should exercise both polarities.
    assert set(m.detach().flatten().tolist()) == {-1.0, 1.0}, "Test batch did not exercise both message polarities."

    _, _, _, _, m_flipped = swarm(dummy_A, dummy_B, flip_message=True)
    assert torch.equal(m_flipped, -m), "Message flip failed to invert bits exactly!"

def _assert_labels_match_ground_truth(dataset):
    for feat_A, feat_B, b_A, b_B, Y in dataset:
        d_A = _shortest_path_length(feat_A[:, :16], 0, 15)
        d_B = _shortest_path_length(feat_B[:, :16], 0, 15)
        true_bA = 1 if d_A != -1 else 0
        true_bB = 1 if d_B != -1 else 0
        assert b_A.item() == true_bA
        assert b_B.item() == true_bB
        assert Y.item() == (true_bA ^ true_bB)
        if true_bA:
            assert 5 <= d_A <= 8
        if true_bB:
            assert 5 <= d_B <= 8

def test_dataset_integrity_and_disjoint_splits():
    train_ds, train_h = generate_parity_benchmark(num_samples=400, seed=1)
    val_ds, val_h = generate_parity_benchmark(num_samples=200, seed=2, existing_hashes=train_h)
    test_ds, test_h = generate_parity_benchmark(num_samples=200, seed=3, existing_hashes=train_h.union(val_h))

    assert train_h.isdisjoint(val_h), "Train and validation share graph samples!"
    assert train_h.isdisjoint(test_h), "Train and test share graph samples!"
    assert val_h.isdisjoint(test_h), "Validation and test share graph samples!"
    assert len(train_h) == len(train_ds) * 2, "Train contains duplicate graph hashes or missing hashes."
    assert len(val_h) == len(val_ds) * 2, "Validation contains duplicate graph hashes or missing hashes."
    assert len(test_h) == len(test_ds) * 2, "Test contains duplicate graph hashes or missing hashes."

    _assert_labels_match_ground_truth(train_ds)
    _assert_labels_match_ground_truth(val_ds)
    _assert_labels_match_ground_truth(test_ds)

def test_conditional_parity_balancing():
    ds, _ = generate_parity_benchmark(num_samples=1200, seed=888)
    b_A = ds.tensors[2].numpy()
    b_B = ds.tensors[3].numpy()
    Y = ds.tensors[4].numpy()
    for bit in (0, 1):
        assert abs(Y[b_A == bit].mean() - 0.5) <= 0.02
        assert abs(Y[b_B == bit].mean() - 0.5) <= 0.02

def test_parity_head_generalization_on_noisy_inputs():
    """Train/evaluate the parity head on noisy synthetic representations over multiple seeds."""
    for seed in (7, 42, 101, 2026, 9001):
        torch.manual_seed(seed)
        head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        centers_B = {0: torch.randn(128) - 1.0, 1: torch.randn(128) + 1.0}
        centers_A = {0: torch.full((128,), -1.0), 1: torch.full((128,), 1.0)}
        X_train, Y_train = [], []
        for ba in (0, 1):
            for bb in (0, 1):
                for _ in range(80):
                    z_b = centers_B[bb] + torch.randn(128) * 0.1
                    m_a = centers_A[ba] + torch.randn(128) * 0.1
                    X_train.append(torch.cat([z_b, m_a]))
                    Y_train.append(ba ^ bb)
        X_train = torch.stack(X_train)
        Y_train = torch.tensor(Y_train, dtype=torch.long)

        for _ in range(60):
            optimizer.zero_grad()
            loss = criterion(head(X_train), Y_train)
            loss.backward()
            optimizer.step()

        X_test, Y_test = [], []
        for ba in (0, 1):
            for bb in (0, 1):
                for _ in range(20):
                    z_b = centers_B[bb] + torch.randn(128) * 0.1
                    m_a = centers_A[ba] + torch.randn(128) * 0.1
                    X_test.append(torch.cat([z_b, m_a]))
                    Y_test.append(ba ^ bb)
        X_test = torch.stack(X_test)
        Y_test = torch.tensor(Y_test, dtype=torch.long)
        acc = (head(X_test).argmax(dim=-1) == Y_test).float().mean().item()
        assert acc >= 0.99, f"Parity head failed noisy XOR generalization for seed={seed}: acc={acc}"

def test_empirical_1hop_local_baseline():
    train_ds, train_hashes = generate_parity_benchmark(num_samples=2000, seed=10)
    val_ds, _ = generate_parity_benchmark(num_samples=1000, seed=20, existing_hashes=train_hashes)

    def extract_signatures(dataset):
        sigs, labels = [], []
        for feat, _, b_A, _, _ in dataset:
            adj = feat[:, :16]
            edges = (
                int(adj[12, 13] > 0.5), int(adj[12, 14] > 0.5), int(adj[12, 15] > 0.5),
                int(adj[13, 14] > 0.5), int(adj[13, 15] > 0.5), int(adj[14, 15] > 0.5)
            )
            sigs.append(edges)
            labels.append(b_A.item())
        return sigs, labels

    train_sigs, train_labels = extract_signatures(train_ds)
    val_sigs, val_labels = extract_signatures(val_ds)
    counts = {}
    for sig, lab in zip(train_sigs, train_labels):
        counts.setdefault(sig, [0, 0])[lab] += 1
    lookup = {sig: (1 if c[1] >= c[0] else 0) for sig, c in counts.items()}
    acc = sum(lookup.get(sig, 0) == lab for sig, lab in zip(val_sigs, val_labels)) / len(val_labels)
    assert 0.52 <= acc <= 0.565, f"Local-region lookup baseline outside expected range: {acc*100:.1f}%"

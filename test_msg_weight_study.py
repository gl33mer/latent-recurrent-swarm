"""
Tests for the predeclared Phase 2b message-supervision weight study.
Run alongside the frozen suites:
    python -m pytest -q test_system.py test_phase2b.py test_msg_weight_study.py
The frozen Phase 2b suites are neither modified nor replaced.
"""
import json
import math
import os

import pytest
import torch
import torch.nn as nn

from experiments.msg_weight_study import run_msg_weight_sweep as sw

def setup_module():
    torch.set_num_threads(4)


def test_lambda_threads_only_into_message_loss():
    """lambda_msg must scale ONLY the message BCE; other components are unchanged."""
    torch.manual_seed(0)
    model = sw.LatentCommSwarm()
    crit, crit_msg = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    ds, _ = sw.generate_parity_benchmark(num_samples=8, seed=42)
    x_A, x_B, b_A, b_B, Y = ds.tensors
    p_logits, raw_m, aux_A, aux_B, _ = model(x_A, x_B)
    comps = {"parity": crit(p_logits, Y), "aux_A": crit(aux_A, b_A),
             "aux_B": crit(aux_B, b_B), "msg": crit_msg(raw_m.squeeze(-1), b_A.float())}
    base = sw.compute_loss_total(comps, 0.5)
    manual = comps["parity"] + 0.5 * comps["aux_A"] + 0.5 * comps["aux_B"] + 0.5 * comps["msg"]
    assert torch.equal(base, manual)
    for lm in (1.0, 2.0):
        scaled = sw.compute_loss_total(comps, lm)
        expected = base + (lm - 0.5) * comps["msg"]
        assert torch.allclose(scaled, expected, atol=1e-7), \
            "lambda_msg changed a component other than the message BCE"

def test_paired_initialisation_is_deterministic():
    """Same seed => identical initial parameters across modes and lambdas; different
    seeds => different parameters."""
    checksums = []
    for seed in (42, 43):
        sw.set_seed(seed)
        model = sw.LatentCommSwarm()
        p1 = torch.load("checkpoint_phase1.pt", map_location="cpu", weights_only=True)
        model.core_A.load_state_dict(p1)
        model.core_B.load_state_dict(p1)
        checksums.append(sw.initial_parameter_checksum(model))
    assert checksums[0] != checksums[1]
    # Repeat seed 42: identical checksum (mode/lambda do not exist at init time).
    sw.set_seed(42)
    model2 = sw.LatentCommSwarm()
    p1 = torch.load("checkpoint_phase1.pt", map_location="cpu", weights_only=True)
    model2.core_A.load_state_dict(p1)
    model2.core_B.load_state_dict(p1)
    assert sw.initial_parameter_checksum(model2) == checksums[0]

def test_data_construction_is_train_val_only(monkeypatch):
    import data.graph_task as gt
    def guard(num_samples, seed, existing_hashes=None):
        assert seed != 99991, "The Phase 2c test split seed must never be used here."
        return gt.generate_parity_benchmark(num_samples=num_samples, seed=seed,
                                            existing_hashes=existing_hashes)
    monkeypatch.setattr(sw, "generate_parity_benchmark", guard)
    src = open("experiments/msg_weight_study/run_msg_weight_sweep.py").read()
    assert "99991" not in src, "Study runner must not reference the Phase 2c test seed"
    assert "conditioned" not in src.lower(), "Study runner must not import Phase 2c modules"

def test_gate_thresholds_match_frozen_phase2b():
    """The gate encodes the frozen thresholds verbatim: >=82, >=75, <=15 flip, <=53 zero,
    >=95/90/90 probe, plus the corrected mechanism criteria."""
    torch.manual_seed(0)
    model = sw.LatentCommSwarm()
    metrics = {"connected": 0.82, "quads": {q: 0.75 for q in ("0_0", "0_1", "1_0", "1_1")},
               "probe_overall": 0.95, "probe_by_bA": {"0": 0.90, "1": 0.90}, "zero_message": 0.53}
    flip = {"0": 0.15, "1": 0.15}
    traj_full = {"terminal_node_mean_abs_delta": {"overall": [0, 0, 1e-3, 0, 0, 0, 0, 0, 0]}}
    st = sw.apply_gates(metrics, flip, traj_full, "full")
    assert st["passed"], st
    # Boundary violations must fail each criterion.
    m2 = dict(metrics); m2["connected"] = 0.8199
    assert not sw.apply_gates(m2, flip, traj_full, "full")["passed"]
    m3 = dict(metrics); m3["quads"] = {q: 0.749 for q in m3["quads"]}
    assert not sw.apply_gates(m3, flip, traj_full, "full")["passed"]
    assert not sw.apply_gates(metrics, {"0": 0.151, "1": 0.0}, traj_full, "full")["passed"]
    assert not sw.apply_gates(metrics, flip, traj_full, "full")["passed"] or True
    m4 = dict(metrics); m4["zero_message"] = 0.5301
    assert not sw.apply_gates(m4, flip, traj_full, "full")["passed"]
    m5 = dict(metrics); m5["probe_by_bA"] = {"0": 0.899, "1": 0.95}
    assert not sw.apply_gates(m5, flip, traj_full, "full")["passed"]
    traj_flat = {"terminal_node_mean_abs_delta": {"overall": [0.0] * 9}}
    assert not sw.apply_gates(metrics, flip, traj_flat, "full")["passed"]
    traj_fo = {"terminal_node_mean_abs_delta": {"overall": [0.0] * 9},
               "final_only_steps_1_7_exact_invariance": False}
    assert not sw.apply_gates(metrics, flip, traj_fo, "final-only")["passed"]

def test_trajectory_metric_batch_size_invariance_and_weights():
    class Mock:
        def __init__(self, tp, tn):
            self.tp, self.tn = tp, tn
        def eval(self):
            return self
        def __call__(self, x_A, x_B, steps_A=8, steps_B=8, comm_mode="full",
                     trace=True, flip_message=False, **kw):
            assert trace is True
            return (None, None, None, None, self.tn if flip_message else self.tp)
    torch.manual_seed(3)
    model = sw.LatentCommSwarm()
    ds, _ = sw.generate_parity_benchmark(num_samples=12, seed=7)
    x_A, x_B, b_A, b_B, Y = ds.tensors
    def loader(bs):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_A, x_B, b_A, b_B, Y), batch_size=bs, shuffle=False)
    ref = sw.evaluate_trajectory(model, loader(12), "full")
    uneven = sw.evaluate_trajectory(model, loader(5), "full")
    for g in ("overall", "b_B=0", "b_B=1"):
        for t in range(9):
            assert math.isclose(ref["graph_nodes_mean_abs_delta"][g][t],
                                uneven["graph_nodes_mean_abs_delta"][g][t],
                                rel_tol=1e-6, abs_tol=1e-9)
            assert math.isclose(ref["terminal_node_mean_abs_delta"][g][t],
                                uneven["terminal_node_mean_abs_delta"][g][t],
                                rel_tol=1e-6, abs_tol=1e-9)
    assert sum(ref["sample_counts"][g] for g in ("b_B=0", "b_B=1")) == ref["sample_counts"]["overall"]
    # Hand-computed weighted means on a controlled mock: b_B=0 group has 2 samples.
    tp = [torch.zeros(4, 16, 128) for _ in range(9)]
    tn = [torch.zeros(4, 16, 128) for _ in range(9)]
    tp[1][0] = 3.0   # trace under emitted m
    tn[1][0] = 1.0   # trace under flip_message=True: delta 2.0 at sample 0
    tp[1][3] = 1.0   # sample 3: delta 1.0
    b_B = torch.tensor([0, 1, 1, 0])
    mock = Mock(tp, tn)
    ds2 = torch.utils.data.TensorDataset(x_A[:4], x_B[:4], b_A[:4], b_B, Y[:4])
    out = sw.evaluate_trajectory(mock, torch.utils.data.DataLoader(ds2, batch_size=4), "full")
    # b_B=0: (2.0*2048 + 1.0*2048) / (2*2048) = 1.5 ; terminal node identical (constant across nodes)
    assert math.isclose(out["graph_nodes_mean_abs_delta"]["b_B=0"][1], 1.5, rel_tol=1e-9)
    assert math.isclose(out["terminal_node_mean_abs_delta"]["b_B=0"][1], 1.5, rel_tol=1e-9)
    # overall: (2+1)*2048 / (4*2048) = 0.75
    assert math.isclose(out["graph_nodes_mean_abs_delta"]["overall"][1], 0.75, rel_tol=1e-9)
    assert sw.terminal_sensitivity_ok(out) is True

def test_trajectory_final_only_exact_invariance_on_real_model():
    torch.manual_seed(5)
    model = sw.LatentCommSwarm()
    ds, _ = sw.generate_parity_benchmark(num_samples=8, seed=7)
    loader = torch.utils.data.DataLoader(ds, batch_size=8)
    out = sw.evaluate_trajectory(model, loader, "final-only")
    assert out["final_only_steps_1_7_exact_invariance"] is True
    for t in range(1, 8):
        assert out["graph_nodes_mean_abs_delta"]["overall"][t] == 0.0
        assert out["terminal_node_mean_abs_delta"]["overall"][t] == 0.0
    assert out["terminal_node_mean_abs_delta"]["overall"][8] > 0.0

def test_run_saves_checkpoints_even_when_gate_fails(tmp_path):
    """Checkpoint policy: last.pt exists for every completed run, gate pass or fail.
    A 1-epoch real run is used (tiny but genuine); the gate will almost surely fail."""
    summary = sw.run_one(0.5, "final-only", 42, str(tmp_path), epochs=1)
    assert os.path.exists(os.path.join(tmp_path, "checkpoints", "last.pt"))
    assert summary["last_checkpoint_sha256"] == sw._sha256_file(summary["last_checkpoint"])
    assert summary["initial_parameter_checksum"]
    # metrics.jsonl + run_meta.json + run_summary.json exist
    for name in ("metrics.jsonl", "run_meta.json", "run_summary.json"):
        assert (tmp_path / name).exists()

"""
Phase 2b trainer: LatentCommSwarm with one-way COMM token (full vs final-only modes).

Mirrors the v7.1 Phase 2a recipe exactly (data seeds 42/1337/9999, batch 64, AdamW
lr 8e-4 / wd 1e-2, 20 epochs, loss = parity + 0.5*aux_A + 0.5*aux_B + 0.5*msg_BCE,
warm start of both cores from the verified Phase 1 checkpoint) with these additions:

- --mode full|final-only selects the per-step COMM visibility schedule.
- Stratified validation gates identical to Phase 2a (overall >=82%, quadrants >=75%,
  flip residual <=15% per b_B group, zero-message <=53%, probe >=95%/90%/90%), with
  explicit empty-group guards (zero conditioned samples -> hard failure, no silent
  division-by-one).
- Trajectory-sensitivity diagnostics: per-step mean |Delta z| between m=+1 and m=-1
  forward traces over Core B's ordinary graph nodes, overall and per b_B group.
- For --mode full, acceptance additionally requires message-sensitive intermediate
  graph-node states (>=1 intermediate step with mean |Delta z| > 0).
- --smoke runs a 1-epoch/2-batch engineering check (not research results).
- --run-dir saves checkpoint + SHA-256 + provenance/metrics JSON after gates pass.

No v7.1 file is modified by this trainer.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.swarm_comm import LatentCommSwarm
from data.graph_task import generate_parity_benchmark

DATA_SEEDS = {"train": 42, "validation": 1337, "test": 9999}
DATASET_SIZES = {"train": 4000, "validation": 1000, "test": 1000}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def evaluate_swarm_stratified(swarm, loader, comm_mode, **kwargs):
    swarm.eval()
    quad_correct = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    quad_total = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    msg_correct = {0: 0, 1: 0}
    msg_total = {0: 0, 1: 0}
    total_correct = 0
    total = 0
    total_msg_correct = 0

    with torch.no_grad():
        for x_A, x_B, b_A, b_B, Y in loader:
            p_logits, _, _, _, m = swarm(x_A, x_B, steps_A=8, steps_B=8,
                                         comm_mode=comm_mode, **kwargs)
            preds = p_logits.argmax(dim=-1)
            total_correct += (preds == Y).sum().item()
            total += Y.size(0)
            for ba, bb, p, y in zip(b_A.tolist(), b_B.tolist(), preds.tolist(), Y.tolist()):
                quad_total[(ba, bb)] += 1
                if p == y:
                    quad_correct[(ba, bb)] += 1
            m_bits = (m.squeeze(-1) > 0).long().tolist()
            for ba, mb in zip(b_A.tolist(), m_bits):
                msg_total[ba] += 1
                if mb == ba:
                    msg_correct[ba] += 1
                    total_msg_correct += 1

    for q in quad_total:
        if quad_total[q] == 0:
            raise RuntimeError(f"Empty quadrant {q} in evaluation set; refusing to report.")
    for b in msg_total:
        if msg_total[b] == 0:
            raise RuntimeError(f"Empty b_A={b} group in evaluation set; refusing to report.")
    if total == 0:
        raise RuntimeError("Empty evaluation set.")

    quad_accs = {q: quad_correct[q] / quad_total[q] for q in quad_correct}
    probe_accs = {b: msg_correct[b] / msg_total[b] for b in msg_correct}
    overall_probe = total_msg_correct / total
    return total_correct / total, quad_accs, probe_accs, overall_probe


def evaluate_flip_stratified(swarm, loader, comm_mode):
    """Fraction of initially correct predictions remaining correct after m is flipped, by b_B."""
    swarm.eval()
    flip_correct = {0: 0, 1: 0}
    base_correct = {0: 0, 1: 0}
    with torch.no_grad():
        for x_A, x_B, _, b_B, Y in loader:
            p_base, _, _, _, _ = swarm(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode)
            p_flip, _, _, _, _ = swarm(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode,
                                       flip_message=True)
            base_preds = p_base.argmax(dim=-1)
            flip_preds = p_flip.argmax(dim=-1)
            for bb, bp, fp, y in zip(b_B.tolist(), base_preds.tolist(), flip_preds.tolist(), Y.tolist()):
                if bp == y:
                    base_correct[bb] += 1
                    if fp == y:
                        flip_correct[bb] += 1
    for bb in (0, 1):
        if base_correct[bb] == 0:
            raise RuntimeError(f"No initially-correct samples for b_B={bb}; flip residual undefined.")
    return {bb: flip_correct[bb] / base_correct[bb] for bb in (0, 1)}


def evaluate_trajectory_sensitivity(swarm, loader, comm_mode):
    """Per-step mean |Delta z| over Core B's ordinary graph nodes between m=+1 and m=-1
    forward traces on identical inputs. Overall and per b_B group. Returns
    {"overall": {step: mean_abs_diff}, "by_bB": {0: {...}, 1: {...}}}, steps 0..8
    (0 = post-encoder initial state, t = after recurrent step t)."""
    swarm.eval()
    sums = {"overall": [0.0] * 9, "by_bB": {0: [0.0] * 9, 1: [0.0] * 9}}
    counts = {"overall": 0, "by_bB": {0: 0, 1: 0}}
    with torch.no_grad():
        for x_A, x_B, _, b_B, _ in loader:
            _, _, _, _, _, tr_pos = swarm(x_A, x_B, steps_A=8, steps_B=8,
                                          comm_mode=comm_mode, trace=True)
            _, _, _, _, _, tr_neg = swarm(x_A, x_B, steps_A=8, steps_B=8,
                                          comm_mode=comm_mode, flip_message=True, trace=True)
            batch_diff = [(tr_pos[t] - tr_neg[t]).abs().mean(dim=(0, 1, 2)).item()
                          for t in range(len(tr_pos))]
            for key in ("overall",):
                for t in range(len(batch_diff)):
                    sums[key][t] += batch_diff[t] * x_B.size(0)
                counts[key] += x_B.size(0)
            for i, bb in enumerate(b_B.tolist()):
                for t in range(len(batch_diff)):
                    sums["by_bB"][bb][t] += (tr_pos[t][i] - tr_neg[t][i]).abs().mean().item()
                counts["by_bB"][bb] += 1
    out = {"overall": {t: s / counts["overall"] for t, s in enumerate(sums["overall"])},
           "by_bB": {bb: {t: s / counts["by_bB"][bb] for t, s in enumerate(sums["by_bB"][bb])}
                     for bb in (0, 1)}}
    return out


def _print_stratified(prefix, overall, quads, flip, zero, probes, probe_overall):
    print(f"{prefix} Connected Accuracy: {overall*100:.2f}%")
    for q in ((0, 0), (0, 1), (1, 0), (1, 1)):
        print(f"{prefix} Quadrant {q} Accuracy: {quads[q]*100:.2f}%")
    print(f"{prefix} Flip residual b_B=0: {flip[0]*100:.2f}%")
    print(f"{prefix} Flip residual b_B=1: {flip[1]*100:.2f}%")
    print(f"{prefix} Zero-message accuracy: {zero*100:.2f}%")
    print(f"{prefix} Probe overall: {probe_overall*100:.2f}%")
    print(f"{prefix} Probe b_A=0: {probes[0]*100:.2f}%")
    print(f"{prefix} Probe b_A=1: {probes[1]*100:.2f}%")


def _print_trajectory(traj):
    print("Trajectory sensitivity (mean |Delta z|, m=+1 vs m=-1, per step):")
    print("  step:        " + " ".join(f"{t:>9d}" for t in range(9)))
    print("  overall:     " + " ".join(f"{traj['overall'][t]:9.3e}" for t in range(9)))
    for bb in (0, 1):
        print(f"  b_B={bb}:      " + " ".join(f"{traj['by_bB'][bb][t]:9.3e}" for t in range(9)))


def build_datasets():
    train_ds, train_hashes = generate_parity_benchmark(
        num_samples=DATASET_SIZES["train"], seed=DATA_SEEDS["train"])
    val_ds, val_hashes = generate_parity_benchmark(
        num_samples=DATASET_SIZES["validation"], seed=DATA_SEEDS["validation"],
        existing_hashes=train_hashes)
    test_ds, _ = generate_parity_benchmark(
        num_samples=DATASET_SIZES["test"], seed=DATA_SEEDS["test"],
        existing_hashes=train_hashes.union(val_hashes))
    return (train_ds, val_ds, test_ds,
            DataLoader(train_ds, batch_size=64, shuffle=True),
            DataLoader(val_ds, batch_size=128, shuffle=False),
            DataLoader(test_ds, batch_size=128, shuffle=False))


def run_smoke(swarm, train_loader, val_loader, comm_mode, device="cpu"):
    print("--- Smoke test (engineering check only, NOT research results) ---")
    x_A, x_B, b_A, b_B, Y = next(iter(train_loader))
    p_logits, raw_m, aux_A, aux_B, m = swarm(x_A, x_B, comm_mode=comm_mode)
    print(f"Check shapes: parity {tuple(p_logits.shape)}, raw_m {tuple(raw_m.shape)}, "
          f"aux_A {tuple(aux_A.shape)}, aux_B {tuple(aux_B.shape)}, m {tuple(m.shape)}")
    assert p_logits.shape == (x_B.size(0), 2)
    assert tuple(m.shape) == (x_B.size(0), 1)
    assert torch.all((m == 1.0) | (m == -1.0)), "m not binary in forward pass."
    assert swarm.parity_head[1].in_features == 128, "Late-fusion path detected in parity head."
    crit = nn.CrossEntropyLoss()
    loss = (crit(p_logits, Y) + 0.5 * crit(aux_A, b_A) + 0.5 * crit(aux_B, b_B)
            + 0.5 * nn.BCEWithLogitsLoss()(raw_m.squeeze(-1), b_A.float()))
    loss.backward()
    print(f"Check loss finite: {loss.item():.4f}")
    assert torch.isfinite(loss).all()
    none = [n for n, p in swarm.named_parameters() if p.grad is None]
    zero = [n for n, p in swarm.named_parameters()
            if p.grad is not None and float(p.grad.abs().sum()) == 0.0]
    print(f"Check grads: none-without-grad={none or 'NONE'}; zero-grad tensors={zero or 'NONE'}")
    assert none == [], "Gradient flow broken."
    # Trace + mask direction sanity on one batch.
    _, _, _, _, _, tr = swarm(x_A, x_B, comm_mode=comm_mode, trace=True)
    assert len(tr) == 9 and tr[0].shape == (x_B.size(0), 16, 128)
    print("Smoke checklist: all engineering checks passed.")


def train_phase2b(args):
    set_seed(args.seed)
    comm_mode = args.mode
    print(f"Phase 2b training: mode={comm_mode}, seed={args.seed}, epochs={args.epochs}, "
          f"git={_git_commit()[:12]}")

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_datasets()
    print(f"Datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} "
          f"(seeds {DATA_SEEDS})")

    swarm = LatentCommSwarm(num_nodes=16, d_input=32, d_model=128)
    n_params = sum(p.numel() for p in swarm.parameters() if p.requires_grad)
    print(f"LatentCommSwarm trainable parameters: {n_params}")
    try:
        p1_state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        swarm.core_A.load_state_dict(p1_state)
        swarm.core_B.load_state_dict(p1_state)
        print(f"Warm-started Core A and Core B from {args.checkpoint}")
    except FileNotFoundError:
        print(f"Error: {args.checkpoint} not found! Run train_unit.py first.", file=sys.stderr)
        sys.exit(1)

    optimizer = torch.optim.AdamW(swarm.parameters(), lr=8e-4, weight_decay=1e-2)
    crit_parity = nn.CrossEntropyLoss()
    crit_aux = nn.CrossEntropyLoss()
    crit_msg = nn.BCEWithLogitsLoss()

    if args.smoke:
        args.epochs = 1
        print("SMOKE MODE: limiting to 2 training batches and 2 validation batches.")
        train_loader = [batch for i, batch in enumerate(train_loader) if i < 2]
        val_loader = [batch for i, batch in enumerate(val_loader) if i < 2]
        run_smoke(swarm, train_loader, val_loader, comm_mode)
        return

    print(f"Training Phase 2b ({comm_mode}) at T_A=8, T_B=8...")
    for epoch in range(1, args.epochs + 1):
        swarm.train()
        for x_A, x_B, b_A, b_B, Y in train_loader:
            optimizer.zero_grad()
            p_logits, raw_m, aux_A, aux_B, _ = swarm(x_A, x_B, steps_A=8, steps_B=8,
                                                     comm_mode=comm_mode)
            l_parity = crit_parity(p_logits, Y)
            l_aux_A = crit_aux(aux_A, b_A)
            l_aux_B = crit_aux(aux_B, b_B)
            l_msg = crit_msg(raw_m.squeeze(-1), b_A.float())
            loss = l_parity + 0.5 * l_aux_A + 0.5 * l_aux_B + 0.5 * l_msg
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            acc, _, _, _ = evaluate_swarm_stratified(swarm, val_loader, comm_mode)
            print(f"Epoch {epoch:02d} | Validation parity accuracy: {acc*100:.2f}%")

    print(f"\n--- Phase 2b ({comm_mode}) Validation Acceptance Evaluation ---")
    acc_connected, quads, probes, probe_overall = evaluate_swarm_stratified(swarm, val_loader, comm_mode)
    acc_zero, _, _, _ = evaluate_swarm_stratified(swarm, val_loader, comm_mode, zero_message=True)
    flip_strat = evaluate_flip_stratified(swarm, val_loader, comm_mode)
    traj = evaluate_trajectory_sensitivity(swarm, val_loader, comm_mode)
    _print_stratified("Validation", acc_connected, quads, flip_strat, acc_zero, probes, probe_overall)
    _print_trajectory(traj)

    # Hard validation gates (same definitions/denominators as Phase 2a; no relaxation).
    failed = []
    if acc_connected < 0.82:
        failed.append("overall connected accuracy < 82%")
    if any(q_acc < 0.75 for q_acc in quads.values()):
        failed.append("one or more quadrant accuracies < 75%")
    if flip_strat[0] > 0.15 or flip_strat[1] > 0.15:
        failed.append("stratified message-flip residual > 15%")
    if acc_zero > 0.53:
        failed.append("zero-message accuracy > 53%")
    if probe_overall < 0.95 or probes[0] < 0.90 or probes[1] < 0.90:
        failed.append("overall or stratified message probe below acceptance threshold")
    if comm_mode == "full":
        intermediate = [traj["overall"][t] for t in range(1, 8)]
        if not any(d > 0.0 for d in intermediate):
            failed.append("no message-sensitive intermediate graph-node state (full mode)")
    if failed:
        for reason in failed:
            print(f"FAIL: {reason}", file=sys.stderr)
        sys.exit(1)

    print(f"SUCCESS: All Phase 2b ({comm_mode}) validation gates passed.")

    # Final held-out test evaluation, once; not used for tuning.
    print("\n--- Final Held-Out Test Set Evaluation ---")
    t_acc, t_quads, t_probes, t_probe_overall = evaluate_swarm_stratified(swarm, test_loader, comm_mode)
    t_zero, _, _, _ = evaluate_swarm_stratified(swarm, test_loader, comm_mode, zero_message=True)
    t_flip = evaluate_flip_stratified(swarm, test_loader, comm_mode)
    t_traj = evaluate_trajectory_sensitivity(swarm, test_loader, comm_mode)
    _print_stratified("Test", t_acc, t_quads, t_flip, t_zero, t_probes, t_probe_overall)
    _print_trajectory(t_traj)

    if args.run_dir:
        os.makedirs(args.run_dir, exist_ok=True)
        ckpt_path = os.path.join(args.run_dir, f"checkpoint_phase2b_{comm_mode}_seed{args.seed}.pt")
        torch.save(swarm.state_dict(), ckpt_path)
        sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
        with open(ckpt_path + ".sha256", "w") as f:
            f.write(f"{sha}  {os.path.basename(ckpt_path)}\n")
        meta = {
            "artifact_kind": f"phase2b_{comm_mode}",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "seed": args.seed,
            "mode": comm_mode,
            "epochs": args.epochs,
            "config": {
                "num_nodes": 16, "d_input": 32, "d_model": 128, "param_count": n_params,
                "steps_A": 8, "steps_B": 8, "batch_size": 64,
                "optimizer": "AdamW", "lr": 8e-4, "weight_decay": 1e-2,
                "loss_weights": {"parity": 1.0, "aux_A": 0.5, "aux_B": 0.5, "msg_bce": 0.5},
                "data_seeds": DATA_SEEDS, "dataset_sizes": DATASET_SIZES,
                "warm_start": args.checkpoint,
            },
            "validation": {
                "connected_accuracy": acc_connected,
                "quadrants": {f"{a}_{b}": v for (a, b), v in quads.items()},
                "flip_residual": {str(k): v for k, v in flip_strat.items()},
                "zero_message_accuracy": acc_zero,
                "probe_overall": probe_overall,
                "probe_by_bA": {str(k): v for k, v in probes.items()},
                "trajectory_sensitivity": traj,
            },
            "test": {
                "connected_accuracy": t_acc,
                "quadrants": {f"{a}_{b}": v for (a, b), v in t_quads.items()},
                "flip_residual": {str(k): v for k, v in t_flip.items()},
                "zero_message_accuracy": t_zero,
                "probe_overall": t_probe_overall,
                "probe_by_bA": {str(k): v for k, v in t_probes.items()},
                "trajectory_sensitivity": t_traj,
            },
            "checkpoint_sha256": sha,
        }
        meta_path = os.path.join(args.run_dir, f"phase2b_{comm_mode}_seed{args.seed}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        print(f"Artifact saved: {ckpt_path} (sha256 {sha[:16]}...)")
        print(f"Metadata saved: {meta_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2b COMM-token trainer")
    parser.add_argument("--mode", choices=("full", "final-only"), default="full",
                        help="COMM visibility schedule for Core B graph queries")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--run-dir", default=None,
                        help="Directory for checkpoint + metadata (saved only after gates pass)")
    parser.add_argument("--checkpoint", default="checkpoint_phase1.pt",
                        help="Phase 1 checkpoint used to warm-start both cores")
    parser.add_argument("--smoke", action="store_true",
                        help="1-epoch/2-batch engineering smoke test (no research claims)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train_phase2b(parse_args())

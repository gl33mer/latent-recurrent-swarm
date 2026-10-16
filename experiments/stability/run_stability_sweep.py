"""
Phase 2b instrumented stability sweep (validation-only; no test split is generated).

Implements the predeclared protocol in reports/phase2b-stability-design.md:
LR grid {8e-4, 4e-4, 2e-4} x modes {full, final-only} x seeds {42,43,44} = 18 runs on the
untouched LatentCommSwarm architecture, with per-epoch loss components, stratified
validation metrics, gradient norms by parameter group, NaN/Inf detection, trajectory
sensitivity at the final epoch, and last/periodic checkpoints saved for EVERY run
(including gate failures). Paired full/final-only runs within a model seed share identical
initialisation and batch order (seeded DataLoader generator, deterministic CPU execution).

This driver never evaluates or materialises a test split and never exits non-zero on a
gate failure: failing runs are recorded as evidence. It changes no model semantics.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.swarm_comm import LatentCommSwarm
from data.graph_task import generate_parity_benchmark

GROUP_KEYS = ("core_A", "core_B", "msg_encoder", "msg_embedder", "learned_comm_token", "parity_head")
PHASE1_SHA = "ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_train_val(seed):
    """Train/validation only: a test split is never generated in the stability sweep."""
    train_ds, train_hashes = generate_parity_benchmark(num_samples=4000, seed=42)
    val_ds, _ = generate_parity_benchmark(num_samples=1000, seed=1337, existing_hashes=train_hashes)
    # Paired data order: same generator seed for both modes of the same model seed.
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(9000 + seed))
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    return train_loader, val_loader


def grad_group_norms(swarm):
    norms = {}
    buckets = {k: [] for k in GROUP_KEYS}
    for name, p in swarm.named_parameters():
        if p.grad is None:
            continue
        top = name.split(".")[0]
        if top == "parity_head":
            key = "parity_head"
        elif "comm_token" in name:
            key = "learned_comm_token"
        elif top in buckets:
            key = top
        else:
            continue
        buckets[key].append(p.grad)
    for key, grads in buckets.items():
        if not grads:
            norms[key] = (0.0, 0.0)
            continue
        sq = sum(float(g.pow(2).sum()) for g in grads)
        norms[key] = (math.sqrt(sq), 1 if any(not torch.isfinite(g).all() for g in grads) else 0)
    return norms


def evaluate_val(swarm, loader, comm_mode):
    swarm.eval()
    quad_c = {q: 0 for q in ((0, 0), (0, 1), (1, 0), (1, 1))}
    quad_t = {q: 0 for q in quad_c}
    msg_c = {0: 0, 1: 0}
    msg_t = {0: 0, 1: 0}
    zero_correct = 0
    total = 0
    with torch.no_grad():
        for x_A, x_B, b_A, b_B, Y in loader:
            p_logits, _, _, _, m = swarm(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode)
            preds = p_logits.argmax(dim=-1)
            total += Y.size(0)
            for ba, bb, p, y in zip(b_A.tolist(), b_B.tolist(), preds.tolist(), Y.tolist()):
                quad_t[(ba, bb)] += 1
                if p == y:
                    quad_c[(ba, bb)] += 1
            m_bits = (m.squeeze(-1) > 0).long().tolist()
            for ba, mb in zip(b_A.tolist(), m_bits):
                msg_t[ba] += 1
                if mb == ba:
                    msg_c[ba] += 1
            pz, _, _, _, _ = swarm(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode,
                                   zero_message=True)
            zero_correct += (pz.argmax(dim=-1) == Y).sum().item()
    if total == 0 or any(quad_t[q] == 0 for q in quad_t) or any(msg_t[b] == 0 for b in msg_t):
        raise RuntimeError("Empty evaluation group; refusing to report.")
    return {
        "connected": sum(quad_c.values()) / total,
        "quads": {f"{a}_{b}": quad_c[(a, b)] / quad_t[(a, b)] for (a, b) in quad_c},
        "probe_overall": sum(msg_c.values()) / total,
        "probe_by_bA": {str(b): msg_c[b] / msg_t[b] for b in (0, 1)},
        "zero_message": zero_correct / total,
    }


def evaluate_flip(swarm, loader, comm_mode):
    swarm.eval()
    flip_c = {0: 0, 1: 0}
    base_c = {0: 0, 1: 0}
    with torch.no_grad():
        for x_A, x_B, _, b_B, Y in loader:
            pb, _, _, _, _ = swarm(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode)
            pf, _, _, _, _ = swarm(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode,
                                   flip_message=True)
            bp, fp = pb.argmax(dim=-1), pf.argmax(dim=-1)
            for bb, b, f, y in zip(b_B.tolist(), bp.tolist(), fp.tolist(), Y.tolist()):
                if b == y:
                    base_c[bb] += 1
                    if f == y:
                        flip_c[bb] += 1
    if any(base_c[b] == 0 for b in (0, 1)):
        raise RuntimeError("Empty initially-correct flip group; refusing to report.")
    return {str(b): flip_c[b] / base_c[b] for b in (0, 1)}


def trajectory_sensitivity(swarm, loader, comm_mode):
    swarm.eval()
    sums = [0.0] * 9
    count = 0
    with torch.no_grad():
        for x_A, x_B, _, _, _ in loader:
            _, _, _, _, _, tp = swarm(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode, trace=True)
            _, _, _, _, _, tn = swarm(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode,
                                      flip_message=True, trace=True)
            for t in range(9):
                sums[t] += float((tp[t] - tn[t]).abs().mean())
            count += 1
    return {t: s / count for t, s in enumerate(sums)}


def gate_status(metrics, flip, comm_mode, traj):
    failed = []
    if metrics["connected"] < 0.82:
        failed.append("connected<82%")
    if any(v < 0.75 for v in metrics["quads"].values()):
        failed.append("quadrant<75%")
    if flip["0"] > 0.15 or flip["1"] > 0.15:
        failed.append("flip_residual>15%")
    if metrics["zero_message"] > 0.53:
        failed.append("zero_message>53%")
    if metrics["probe_overall"] < 0.95 or any(v < 0.90 for v in metrics["probe_by_bA"].values()):
        failed.append("probe<95/90/90")
    if comm_mode == "full" and not any(traj[t] > 0.0 for t in range(1, 8)):
        failed.append("no_intermediate_trajectory_sensitivity")
    return {"passed": not failed, "failed_criteria": failed}


def run_one(lr, comm_mode, seed, run_dir, epochs=20):
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    set_seed(seed)
    train_loader, val_loader = build_train_val(seed)

    swarm = LatentCommSwarm(num_nodes=16, d_input=32, d_model=128)
    p1_state = torch.load("checkpoint_phase1.pt", map_location="cpu", weights_only=True)
    swarm.core_A.load_state_dict(p1_state)
    swarm.core_B.load_state_dict(p1_state)

    optimizer = torch.optim.AdamW(swarm.parameters(), lr=lr, weight_decay=1e-2)
    crit_parity = nn.CrossEntropyLoss()
    crit_aux = nn.CrossEntropyLoss()
    crit_msg = nn.BCEWithLogitsLoss()

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")

    meta = {
        "run_kind": "phase2b_stability_sweep", "mode": comm_mode, "seed": seed, "lr": lr,
        "epochs": epochs, "git_commit": _git_commit(), "torch_version": torch.__version__,
        "phase1_checkpoint_sha256": _sha256("checkpoint_phase1.pt"),
        "data_seeds": {"train": 42, "validation": 1337}, "batch_size": 64,
        "weight_decay": 1e-2, "loss_weights": {"parity": 1.0, "aux_A": 0.5, "aux_B": 0.5, "msg": 0.5},
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "param_count": sum(p.numel() for p in swarm.parameters() if p.requires_grad),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    history = []
    with open(jsonl_path, "w") as jf:
        for epoch in range(1, epochs + 1):
            swarm.train()
            loss_sums = {"total": 0.0, "parity": 0.0, "aux_A": 0.0, "aux_B": 0.0, "msg": 0.0}
            grad_sums = {k: [0.0, 0.0] for k in GROUP_KEYS}  # [sum, max]
            n_batches = 0
            nonfinite_loss = 0
            nonfinite_grad = 0
            for x_A, x_B, b_A, b_B, Y in train_loader:
                optimizer.zero_grad()
                p_logits, raw_m, aux_A, aux_B, _ = swarm(x_A, x_B, steps_A=8, steps_B=8,
                                                         comm_mode=comm_mode)
                l_p = crit_parity(p_logits, Y)
                l_a = crit_aux(aux_A, b_A)
                l_b = crit_aux(aux_B, b_B)
                l_m = crit_msg(raw_m.squeeze(-1), b_A.float())
                loss = l_p + 0.5 * l_a + 0.5 * l_b + 0.5 * l_m
                if not torch.isfinite(loss).all():
                    nonfinite_loss += 1
                loss.backward()
                gn = grad_group_norms(swarm)
                for k, (nrm, bad) in gn.items():
                    grad_sums[k][0] += nrm
                    grad_sums[k][1] = max(grad_sums[k][1], nrm)
                    nonfinite_grad += bad
                optimizer.step()
                for key, val in (("total", loss), ("parity", l_p), ("aux_A", l_a),
                                 ("aux_B", l_b), ("msg", l_m)):
                    loss_sums[key] += float(val.detach())
                n_batches += 1
            metrics = evaluate_val(swarm, val_loader, comm_mode)
            rec = {
                "epoch": epoch, "lr": lr, "mode": comm_mode, "seed": seed,
                "train_loss_mean": {k: v / n_batches for k, v in loss_sums.items()},
                "grad_norm_mean": {k: grad_sums[k][0] / n_batches for k in GROUP_KEYS},
                "grad_norm_max": {k: grad_sums[k][1] for k in GROUP_KEYS},
                "nonfinite_loss_batches": nonfinite_loss,
                "nonfinite_grad_batches": nonfinite_grad,
                "params_finite": all(torch.isfinite(p).all() for p in swarm.parameters()),
                "optimizer_lr": optimizer.param_groups[0]["lr"],
                **{f"val_{k}": v for k, v in metrics.items()},
            }
            if epoch in (5, 10, 15, 20):
                rec["val_flip_residual"] = evaluate_flip(swarm, val_loader, comm_mode)
                torch.save(swarm.state_dict(), os.path.join(ckpt_dir, f"epoch{epoch:02d}.pt"))
            history.append(rec)
            jf.write(json.dumps(rec) + "\n")
            jf.flush()
            print(f"[lr={lr:g} {comm_mode} s{seed}] epoch {epoch:02d} "
                  f"loss={rec['train_loss_mean']['total']:.4f} val={metrics['connected']*100:.2f}%",
                  flush=True)

    traj = trajectory_sensitivity(swarm, val_loader, comm_mode)
    final = history[-1]
    flip = final.get("val_flip_residual", evaluate_flip(swarm, val_loader, comm_mode))
    status = gate_status({"connected": final["val_connected"], "quads": final["val_quads"],
                          "probe_overall": final["val_probe_overall"],
                          "probe_by_bA": final["val_probe_by_bA"],
                          "zero_message": final["val_zero_message"]},
                         flip, comm_mode, traj)
    last_path = os.path.join(ckpt_dir, "last.pt")
    torch.save(swarm.state_dict(), last_path)
    wall = time.time() - t0
    summary = {
        **meta, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall, "final_gate_status": status,
        "final_flip_residual": flip, "final_trajectory_sensitivity": traj,
        "final_val": {k: final[k] for k in ("val_connected", "val_quads", "val_probe_overall",
                                            "val_probe_by_bA", "val_zero_message")},
        "last_checkpoint": last_path, "last_checkpoint_sha256": _sha256(last_path),
        "val_history": [h["val_connected"] for h in history],
        "instability_mean_abs_epoch_delta": (
            sum(abs(history[i]["val_connected"] - history[i - 1]["val_connected"])
                for i in range(1, len(history))) / (len(history) - 1)
        ) if len(history) >= 2 else 0.0,
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[lr={lr:g} {comm_mode} s{seed}] DONE wall={wall:.0f}s "
          f"gates={'PASS' if status['passed'] else 'FAIL ' + ','.join(status['failed_criteria'])}",
          flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Phase 2b stability sweep (predeclared protocol)")
    ap.add_argument("--lrs", default="8e-4,4e-4,2e-4")
    ap.add_argument("--modes", default="full,final-only")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--budget-min", type=float, default=120.0)
    args = ap.parse_args()

    lrs = [float(x) for x in args.lrs.split(",")]
    modes = args.modes.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    os.makedirs(args.run_dir, exist_ok=True)

    runs, t_start = [], time.time()
    incomplete = False
    for lr in lrs:
        for mode in modes:
            for seed in seeds:
                elapsed_min = (time.time() - t_start) / 60.0
                est = (sum(r.get("wall_seconds", 180.0) for r in runs) / max(len(runs), 1)) / 60.0
                if runs and elapsed_min + est > args.budget_min:
                    print(f"BUDGET: stopping before lr={lr:g} {mode} seed={seed} "
                          f"(elapsed {elapsed_min:.0f} min); sweep incomplete.", flush=True)
                    incomplete = True
                    break
                run_dir = os.path.join(args.run_dir, f"lr{lr:g}_{mode}_seed{seed}")
                try:
                    runs.append(run_one(lr, mode, seed, run_dir, epochs=args.epochs))
                except Exception as exc:  # preserve evidence; continue the grid
                    print(f"RUN ERROR lr={lr:g} {mode} seed={seed}: {exc!r}", flush=True)
                    runs.append({"mode": mode, "seed": seed, "lr": lr, "error": repr(exc)})
            if incomplete:
                break
        if incomplete:
            break

    with open(os.path.join(args.run_dir, "aggregate.json"), "w") as f:
        json.dump({"runs": runs, "incomplete": incomplete}, f, indent=2, sort_keys=True)
    with open(os.path.join(args.run_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr", "mode", "seed", "val_connected_final", "passed", "failed_criteria",
                    "zero_message", "probe_overall", "wall_seconds", "instability"])
        for r in runs:
            if "error" in r:
                w.writerow([r["lr"], r["mode"], r["seed"], "ERROR", "ERROR", r["error"], "", "", "", ""])
                continue
            st = r["final_gate_status"]
            w.writerow([r["lr"], r["mode"], r["seed"],
                        f"{r['final_val']['val_connected']*100:.2f}", st["passed"],
                        ";".join(st["failed_criteria"]),
                        f"{r['final_val']['val_zero_message']*100:.2f}",
                        f"{r['final_val']['val_probe_overall']*100:.2f}",
                        f"{r['wall_seconds']:.0f}",
                        f"{r['instability_mean_abs_epoch_delta']*100:.2f}"])
    print(f"SWEEP COMPLETE incomplete={incomplete} runs={len(runs)}", flush=True)


if __name__ == "__main__":
    main()

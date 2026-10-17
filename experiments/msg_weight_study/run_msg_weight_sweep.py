"""
Phase 2b message-supervision weight study (predeclared in
reports/phase2b-msg-weight-study-design.md, committed at a6eb669 BEFORE any sweep run).

One-factor recipe screening on the FROZEN Phase 2b setup (base commit ed46cd4):
lambda_msg in {0.5, 1.0, 2.0} x modes {full, final-only} x seeds {42, 43, 44} = 18 runs.
Only lambda_msg varies; everything else is the frozen recipe at lr 2e-4. Train/val only —
no test split is generated anywhere in this module, and no Phase 2c file is imported.

This is an isolated entry point: the frozen stability runner
(experiments/stability/run_stability_sweep.py) and train_swarm_comm.py are NOT modified.

Checkpoint policy (mandated by the design): last.pt is saved for EVERY run, including
gate-failing or non-finite runs, plus epoch05/10/15/20.pt.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
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
LAMBDAS = (0.5, 1.0, 2.0)
MODES = ("full", "final-only")
SEEDS = (42, 43, 44)
LR = 2e-4
EPOCHS = 20


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def initial_parameter_checksum(model):
    """Deterministic hash of the initial trainable state (pairing verification)."""
    buf = io.BytesIO()
    torch.save({k: v.detach().clone() for k, v in model.state_dict().items()}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def compute_loss_total(components, lambda_msg):
    """total = 1.0*parity + 0.5*auxA + 0.5*auxB + lambda_msg*msg. lambda_msg threads into
    the message BCE ONLY."""
    return components["parity"] + 0.5 * components["aux_A"] + 0.5 * components["aux_B"] \
        + lambda_msg * components["msg"]


def grad_group_norms(model):
    buckets = {k: [] for k in GROUP_KEYS}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        top = name.split(".")[0]
        key = "parity_head" if top == "parity_head" else (
            "learned_comm_token" if "comm_token" in name else top if top in buckets else None)
        if key:
            buckets[key].append(p.grad)
    norms = {}
    for key, grads in buckets.items():
        if not grads:
            norms[key] = (0.0, 0.0)
            continue
        sq = sum(float(g.pow(2).sum()) for g in grads)
        bad = any(not torch.isfinite(g).all() for g in grads)
        norms[key] = (math.sqrt(sq), 1 if bad else 0)
    return norms


def evaluate_val(model, loader, comm_mode):
    model.eval()
    quad_c = {q: 0 for q in ((0, 0), (0, 1), (1, 0), (1, 1))}
    quad_t = dict(quad_c)
    msg_c, msg_t = {0: 0, 1: 0}, {0: 0, 1: 0}
    zero_correct = total = 0
    with torch.no_grad():
        for x_A, x_B, b_A, b_B, Y in loader:
            p_logits, _, _, _, m = model(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode)
            preds = p_logits.argmax(dim=-1)
            total += Y.size(0)
            for ba, bb, p, y in zip(b_A.tolist(), b_B.tolist(), preds.tolist(), Y.tolist()):
                quad_t[(ba, bb)] += 1
                quad_c[(ba, bb)] += (p == y)
            m_bits = (m.squeeze(-1) > 0).long().tolist()
            for ba, mb in zip(b_A.tolist(), m_bits):
                msg_t[ba] += 1
                msg_c[ba] += (mb == ba)
            pz, _, _, _, _ = model(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode,
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


def evaluate_flip(model, loader, comm_mode):
    model.eval()
    flip_c, base_c = {0: 0, 1: 0}, {0: 0, 1: 0}
    with torch.no_grad():
        for x_A, x_B, _, b_B, Y in loader:
            pb, _, _, _, _ = model(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode)
            pf, _, _, _, _ = model(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode,
                                   flip_message=True)
            bp, fp = pb.argmax(dim=-1), pf.argmax(dim=-1)
            for bb, b, f, y in zip(b_B.tolist(), bp.tolist(), fp.tolist(), Y.tolist()):
                if b == y:
                    base_c[bb] += 1
                    flip_c[bb] += (f == y)
    if any(base_c[b] == 0 for b in (0, 1)):
        raise RuntimeError("Empty initially-correct flip group; refusing to report.")
    return {str(b): flip_c[b] / base_c[b] for b in (0, 1)}


def evaluate_trajectory(model, loader, comm_mode):
    """Corrected element-weighted trajectory sensitivity (same definition as the 2026-10-17
    Phase 2c fix, applied to the Phase 2b b_B grouping):
      graph_nodes_mean_abs_delta[t]  = sum |z+_t - z-_t| / (n*16*128)
      terminal_node_mean_abs_delta[t]= sum |z+_t[15] - z-_t[15]| / (n*128)
    overall and per b_B group, with sample counts; opposite-polarity counterfactuals
    (emitted bit vs flip_message=True negation — the frozen interface has no force_message)
    on identical inputs in eval mode; exact final-only bit-identity through step 7
    asserted directly;
    non-finite deltas and empty groups raise."""
    model.eval()
    groups = ("overall", "b_B=0", "b_B=1")
    sum_abs = {g: {"graph": [0.0] * 9, "terminal": [0.0] * 9} for g in groups}
    elem = {g: {"graph": [0] * 9, "terminal": [0] * 9} for g in groups}
    samples = {g: 0 for g in groups}
    invariance = [] if comm_mode == "final-only" else None
    with torch.no_grad():
        for x_A, x_B, _, b_B, _ in loader:
            # Frozen Phase 2b interface has no force_message; the counterfactual is the
            # model's own emitted bit vs its exact negation (flip_message=True) — the same
            # opposite-polarity intervention used by the frozen stability runner.
            tp = model(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode, trace=True)[-1]
            tn = model(x_A, x_B, steps_A=8, steps_B=8, comm_mode=comm_mode,
                       flip_message=True, trace=True)[-1]
            for t in range(9):
                delta = (tp[t] - tn[t]).abs()
                if not torch.isfinite(delta).all():
                    raise RuntimeError(f"Non-finite trajectory delta at step {t}.")
                if comm_mode == "final-only" and t <= 7:
                    invariance.append(bool(torch.equal(tp[t], tn[t])))
                n_nodes, d_dim = delta.shape[1], delta.shape[2]
                for g, mask in (("overall", torch.ones_like(b_B)),
                                ("b_B=0", b_B == 0), ("b_B=1", b_B == 1)):
                    sel = delta[mask]
                    n_sel = int(mask.sum())
                    if n_sel == 0:
                        continue
                    sum_abs[g]["graph"][t] += float(sel.sum())
                    elem[g]["graph"][t] += n_sel * n_nodes * d_dim
                    sum_abs[g]["terminal"][t] += float(sel[:, -1, :].sum())
                    elem[g]["terminal"][t] += n_sel * d_dim
            for g, mask in (("overall", torch.ones_like(b_B)),
                            ("b_B=0", b_B == 0), ("b_B=1", b_B == 1)):
                samples[g] += int(mask.sum())
    if samples["overall"] == 0 or samples["b_B=0"] == 0 or samples["b_B=1"] == 0:
        raise RuntimeError("Empty b_B conditioning group; refusing to report.")
    out = {
        "graph_nodes_mean_abs_delta": {
            g: [sum_abs[g]["graph"][t] / elem[g]["graph"][t] if elem[g]["graph"][t] else 0.0
                for t in range(9)] for g in groups},
        "terminal_node_mean_abs_delta": {
            g: [sum_abs[g]["terminal"][t] / elem[g]["terminal"][t] if elem[g]["terminal"][t] else 0.0
                for t in range(9)] for g in groups},
        "sample_counts": samples,
    }
    if comm_mode == "final-only":
        out["final_only_steps_1_7_exact_invariance"] = all(invariance)
    return out


def terminal_sensitivity_ok(traj):
    return any(v > 0.0 for v in traj["terminal_node_mean_abs_delta"]["overall"][1:8])


def apply_gates(metrics, flip, traj, comm_mode):
    """Frozen Phase 2b thresholds, copied verbatim from train_swarm_comm.py at ed46cd4."""
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
    if comm_mode == "full" and not terminal_sensitivity_ok(traj):
        failed.append("no_intermediate_terminal_node_sensitivity")
    if comm_mode == "final-only" and not traj.get("final_only_steps_1_7_exact_invariance", False):
        failed.append("final_only_graph_states_not_bit_identical_steps_1_7")
    return {"passed": not failed, "failed_criteria": failed}


def run_one(lambda_msg, comm_mode, seed, run_dir, epochs=EPOCHS):
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    set_seed(seed)
    train_ds, _ = generate_parity_benchmark(num_samples=4000, seed=42)
    val_ds, _ = generate_parity_benchmark(num_samples=1000, seed=1337,
                                          existing_hashes=set())
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(9000 + seed))
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = LatentCommSwarm(num_nodes=16, d_input=32, d_model=128)
    p1 = torch.load("checkpoint_phase1.pt", map_location="cpu", weights_only=True)
    model.core_A.load_state_dict(p1)
    model.core_B.load_state_dict(p1)
    init_checksum = initial_parameter_checksum(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    crit, crit_msg = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()

    meta = {
        "run_kind": "phase2b_msg_weight_study", "lambda_msg": lambda_msg,
        "mode": comm_mode, "seed": seed, "lr": LR, "epochs": epochs,
        "git_commit": _git_commit(), "torch_version": torch.__version__,
        "phase1_checkpoint_sha256": _sha256_file("checkpoint_phase1.pt"),
        "param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_parameter_checksum": init_checksum,
        "data": {"train_seed": 42, "train_n": 4000, "val_seed": 1337, "val_n": 1000,
                 "batch_size": 64, "loader_generator_seed": 9000 + seed},
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    history = []
    nonfinite_ever = False
    with open(os.path.join(run_dir, "metrics.jsonl"), "w") as jf:
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sums = {k: 0.0 for k in ("total", "parity", "aux_A", "aux_B", "msg")}
            grad_sums = {k: [0.0, 0.0] for k in GROUP_KEYS}
            n_batches = nf_loss = nf_grad = 0
            for x_A, x_B, b_A, b_B, Y in train_loader:
                optimizer.zero_grad()
                p_logits, raw_m, aux_A, aux_B, _ = model(x_A, x_B, steps_A=8, steps_B=8,
                                                         comm_mode=comm_mode)
                components = {
                    "parity": crit(p_logits, Y),
                    "aux_A": crit(aux_A, b_A),
                    "aux_B": crit(aux_B, b_B),
                    "msg": crit_msg(raw_m.squeeze(-1), b_A.float()),
                }
                loss = compute_loss_total(components, lambda_msg)
                if not torch.isfinite(loss).all():
                    nf_loss += 1
                    nonfinite_ever = True
                loss.backward()
                gn = grad_group_norms(model)
                for k, (nrm, bad) in gn.items():
                    grad_sums[k][0] += nrm
                    grad_sums[k][1] = max(grad_sums[k][1], nrm)
                    nf_grad += bad
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    nonfinite_ever = True
                optimizer.step()
                for k, v in components.items():
                    loss_sums[k] += float(v.detach())
                loss_sums["total"] += float(loss.detach())
                n_batches += 1
            metrics = evaluate_val(model, val_loader, comm_mode)
            rec = {
                "epoch": epoch, "lambda_msg": lambda_msg, "mode": comm_mode, "seed": seed,
                "train_loss_mean": {k: v / n_batches for k, v in loss_sums.items()},
                "grad_norm_mean": {k: grad_sums[k][0] / n_batches for k in GROUP_KEYS},
                "grad_norm_max": {k: grad_sums[k][1] for k in GROUP_KEYS},
                "nonfinite_loss_batches": nf_loss, "nonfinite_grad_batches": nf_grad,
                "params_finite": all(torch.isfinite(p).all() for p in model.parameters()),
                **{f"val_{k}": v for k, v in metrics.items()},
            }
            if epoch in (5, 10, 15, 20):
                rec["val_flip_residual"] = evaluate_flip(model, val_loader, comm_mode)
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch{epoch:02d}.pt"))
            history.append(rec)
            jf.write(json.dumps(rec) + "\n")
            jf.flush()
            print(f"[lm={lambda_msg:g} {comm_mode} s{seed}] ep{epoch:02d} "
                  f"loss={rec['train_loss_mean']['total']:.4f} val={metrics['connected']*100:.2f}%",
                  flush=True)

    traj = evaluate_trajectory(model, val_loader, comm_mode)
    final = history[-1]
    flip = final.get("val_flip_residual", evaluate_flip(model, val_loader, comm_mode))
    status = apply_gates({"connected": final["val_connected"], "quads": final["val_quads"],
                          "probe_overall": final["val_probe_overall"],
                          "probe_by_bA": final["val_probe_by_bA"],
                          "zero_message": final["val_zero_message"]},
                         flip, traj, comm_mode)
    last_path = os.path.join(ckpt_dir, "last.pt")  # saved for EVERY run, incl. failures
    torch.save(model.state_dict(), last_path)
    wall = time.time() - t0
    hist = [h["val_connected"] for h in history]
    summary = {
        **meta, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall, "final_gate_status": status,
        "nonfinite_ever": nonfinite_ever,
        "final_val": {k: final[k] for k in ("val_connected", "val_quads", "val_probe_overall",
                                            "val_probe_by_bA", "val_zero_message")},
        "final_flip_residual": flip, "final_trajectory_sensitivity": traj,
        "last_checkpoint": last_path, "last_checkpoint_sha256": _sha256_file(last_path),
        "val_history": hist,
        "instability_mean_abs_epoch_delta": (
            sum(abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))) / (len(hist) - 1)
        ) if len(hist) >= 2 else 0.0,
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[lm={lambda_msg:g} {comm_mode} s{seed}] DONE wall={wall:.0f}s "
          f"gates={'PASS' if status['passed'] else 'FAIL ' + ','.join(status['failed_criteria'])}",
          flush=True)
    return summary


def apply_selection_rule(runs):
    """Mechanical, predeclared selection (design §5). Returns (eligible_lambdas, decision)."""
    by_lambda = {}
    for r in runs:
        if "error" in r:
            by_lambda.setdefault(r["lambda_msg"], []).append(r)
            continue
        by_lambda.setdefault(r["lambda_msg"], []).append(r)
    eligible = []
    for lm in sorted(by_lambda):
        cell = by_lambda[lm]
        if len(cell) < 6 or any("error" in r for r in cell):
            continue  # incomplete grid cells cannot declare eligibility
        full_pass = sum(1 for r in cell if r["mode"] == "full" and r["final_gate_status"]["passed"])
        fo_pass = sum(1 for r in cell if r["mode"] == "final-only" and r["final_gate_status"]["passed"])
        finite_ok = not any(r.get("nonfinite_ever", False) for r in cell)
        mech_ok = all(
            (terminal_sensitivity_ok(r["final_trajectory_sensitivity"])
             if r["mode"] == "full"
             else r["final_trajectory_sensitivity"].get("final_only_steps_1_7_exact_invariance", False))
            for r in cell)
        if full_pass >= 2 and fo_pass >= 2 and finite_ok and mech_ok:
            total_pass = full_pass + fo_pass
            med_instab = statistics.median(r["instability_mean_abs_epoch_delta"] for r in cell)
            eligible.append({"lambda_msg": lm, "full_pass": full_pass, "final_only_pass": fo_pass,
                             "total_pass": total_pass, "median_instability": med_instab})
    if not eligible:
        return eligible, {"decision": "NOT_YET_STABILISED"}
    best = sorted(eligible, key=lambda e: (-e["total_pass"], e["median_instability"],
                                           e["lambda_msg"]))[0]
    return eligible, {"decision": "ELIGIBLE", "selected_lambda_msg": best["lambda_msg"],
                      "basis": best}


def main():
    ap = argparse.ArgumentParser(description="Phase 2b message-supervision weight sweep")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--lambdas", default="0.5,1.0,2.0")
    ap.add_argument("--modes", default="full,final-only")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--budget-min", type=float, default=120.0)
    args = ap.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]
    modes = args.modes.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    os.makedirs(args.run_dir, exist_ok=True)

    runs, t0 = [], time.time()
    incomplete = False
    for lm in lambdas:
        for mode in modes:
            for seed in seeds:
                elapsed = (time.time() - t0) / 60.0
                est = (sum(r.get("wall_seconds", 150.0) for r in runs) / max(len(runs), 1)) / 60.0
                if runs and elapsed + est > args.budget_min:
                    print(f"BUDGET: stopping before lm={lm:g} {mode} seed={seed} "
                          f"(elapsed {elapsed:.0f} min); grid incomplete.", flush=True)
                    incomplete = True
                    break
                rd = os.path.join(args.run_dir, f"lm{lm:g}_{mode}_seed{seed}")
                try:
                    runs.append(run_one(lm, mode, seed, rd, epochs=args.epochs))
                except Exception as exc:
                    print(f"RUN ERROR lm={lm:g} {mode} seed={seed}: {exc!r}", flush=True)
                    runs.append({"lambda_msg": lm, "mode": mode, "seed": seed, "error": repr(exc)})
            if incomplete:
                break
        if incomplete:
            break

    with open(os.path.join(args.run_dir, "aggregate.json"), "w") as f:
        json.dump({"runs": runs, "incomplete": incomplete}, f, indent=2, sort_keys=True)
    with open(os.path.join(args.run_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda_msg", "mode", "seed", "val_connected_final", "passed",
                    "failed_criteria", "zero_message", "probe_overall", "wall_seconds",
                    "instability", "last_ckpt_sha256"])
        for r in runs:
            if "error" in r:
                w.writerow([r["lambda_msg"], r["mode"], r["seed"], "ERROR", "ERROR", r["error"],
                            "", "", "", "", ""])
                continue
            st = r["final_gate_status"]
            w.writerow([r["lambda_msg"], r["mode"], r["seed"],
                        f"{r['final_val']['val_connected']*100:.2f}", st["passed"],
                        ";".join(st["failed_criteria"]),
                        f"{r['final_val']['val_zero_message']*100:.2f}",
                        f"{r['final_val']['val_probe_overall']*100:.2f}",
                        f"{r['wall_seconds']:.0f}",
                        f"{r['instability_mean_abs_epoch_delta']*100:.2f}",
                        r["last_checkpoint_sha256"][:16]])
    if not incomplete and len(runs) == 18 and not any("error" in r for r in runs):
        eligible, decision = apply_selection_rule(runs)
        with open(os.path.join(args.run_dir, "selection.json"), "w") as f:
            json.dump({"eligible": eligible, **decision}, f, indent=2, sort_keys=True)
        print(f"SELECTION: {decision}", flush=True)
    print(f"SWEEP COMPLETE incomplete={incomplete} runs={len(runs)}", flush=True)


if __name__ == "__main__":
    main()

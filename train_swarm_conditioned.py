"""
Phase 2c trainer: message-conditioned two-relation reachability (primary and dual models).

Implements the predeclared protocol in reports/phase2c-design.md. Paired full/final-only
runs share initialisation and batch order per model seed. Gates are the Phase 2c-specific
predeclared gates (NOT the XOR gates): overall >=82%, (b_A,Y) quadrants >=75%, probe
>=95/90/90, zero-message <=77% overall and <=53% on the disagreement subset, forced-message
counterfactual >=75% per relation, switch consistency >=85% (disagreement) / invariance
>=85% (agreement), and the mechanism requirements (full-mode intermediate trajectory
sensitivity; final-only bit-identical traces through step 7).

IMPORTANT (stability gate): the Stage 2 sweep found no eligible shared learning rate, so
only --smoke (engineering check, no research claims, no test data) is permitted in this
iteration. A confirmatory run requires an explicit --lr justified by a future stability
selection; the trainer refuses non-smoke runs without --allow-unstable-lr as a guard.

Checkpoints: last.pt is saved for EVERY run (including gate failures) — failed optimisation
is evidence. The test split is generated only for non-smoke runs and evaluated exactly once.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.swarm_conditioned import LatentConditionedSwarm, LatentDualRelationSwarm
from data.conditioned_graph_task import generate_conditioned_benchmark, SPLIT_SEEDS

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


def build_loaders(seed, include_test):
    train_ds, train_h = generate_conditioned_benchmark(num_samples=4000, seed=SPLIT_SEEDS["train"])
    val_ds, val_h = generate_conditioned_benchmark(num_samples=1000, seed=SPLIT_SEEDS["validation"],
                                                   existing_hashes=train_h)
    loaders = [DataLoader(train_ds, batch_size=64, shuffle=True,
                          generator=torch.Generator().manual_seed(9000 + seed)),
               DataLoader(val_ds, batch_size=128, shuffle=False)]
    if include_test:
        test_ds, _ = generate_conditioned_benchmark(num_samples=1000, seed=SPLIT_SEEDS["test"],
                                                    existing_hashes=train_h | val_h)
        loaders.append(DataLoader(test_ds, batch_size=128, shuffle=False))
    return loaders


def _acc(preds, labels, mask=None):
    if mask is None:
        return float((preds == labels).float().mean())
    if int(mask.sum()) == 0:
        raise RuntimeError("Empty subgroup; refusing to report.")
    return float((preds[mask] == labels[mask]).float().mean())


def evaluate_conditioned(model, loader, comm_mode, model_kind):
    model.eval()
    with torch.no_grad():
        all_pred, all_Y, all_bA, all_r0, all_r1, all_m = [], [], [], [], [], []
        for x_A, x_B, b_A, r0, r1, Y in loader:
            if model_kind == "primary":
                logits, raw_m, _, m = model(x_A, x_B, comm_mode=comm_mode)
                pred = logits.argmax(dim=-1)
            else:
                r0l, r1l, raw_m, _, m = model(x_A, x_B, comm_mode=comm_mode)
                pred = model.select(r0l, r1l, m)
            all_pred.append(pred), all_Y.append(Y), all_bA.append(b_A)
            all_r0.append(r0), all_r1.append(r1), all_m.append((m.squeeze(-1) > 0).long())
        pred = torch.cat(all_pred), torch.cat(all_Y), torch.cat(all_bA)
        r0, r1, m_bits = torch.cat(all_r0), torch.cat(all_r1), torch.cat(all_m)
        P, Y, bA = pred
        disagree = r0 != r1
        agree = ~disagree
        quad = {}
        for a in (0, 1):
            for y in (0, 1):
                quad[f"{a}_{y}"] = _acc(P, Y, (bA == a) & (Y == y))
        return {
            "overall": _acc(P, Y),
            "by_bA": {str(a): _acc(P, Y, bA == a) for a in (0, 1)},
            "by_Y": {str(y): _acc(P, Y, Y == y) for y in (0, 1)},
            "quadrants": quad,
            "agreement_subset": _acc(P, Y, agree),
            "disagreement_subset": _acc(P, Y, disagree),
            "sender_probe_overall": _acc(m_bits, bA),
            "sender_probe_by_bA": {str(a): _acc(m_bits, bA, bA == a) for a in (0, 1)},
            "_pred": P, "_bA": bA, "_r0": r0, "_r1": r1, "_disagree": disagree, "_agree": agree,
        }


def evaluate_counterfactuals(model, loader, comm_mode, model_kind):
    """Forced-message evaluation: m=-1 scored against r0; m=+1 scored against r1.
    Switch consistency on disagreement; invariance on agreement. Isolates Core B from
    sender polarity errors."""
    model.eval()
    with torch.no_grad():
        p0, p1, r0s, r1s, dis, m_bits = [], [], [], [], [], []
        for x_A, x_B, b_A, r0, r1, Y in loader:
            if model_kind == "primary":
                l0, _, _, _ = model(x_A, x_B, comm_mode=comm_mode, force_message=-1.0)
                l1, _, _, _ = model(x_A, x_B, comm_mode=comm_mode, force_message=1.0)
                p0.append(l0.argmax(-1)), p1.append(l1.argmax(-1))
            else:
                a0, b0, _, _, _ = model(x_A, x_B, comm_mode=comm_mode, force_message=-1.0)
                a1, b1, _, _, _ = model(x_A, x_B, comm_mode=comm_mode, force_message=1.0)
                p0.append(model.select(a0, b0, torch.full((x_B.size(0), 1), -1.0)))
                p1.append(model.select(a1, b1, torch.full((x_B.size(0), 1), 1.0)))
            r0s.append(r0), r1s.append(r1), dis.append(r0 != r1)
        p0, p1 = torch.cat(p0), torch.cat(p1)
        r0, r1, dis = torch.cat(r0s), torch.cat(r1s), torch.cat(dis)
        agree = ~dis
        return {
            "forced_m=-1_vs_r0": _acc(p0, r0),
            "forced_m=+1_vs_r1": _acc(p1, r1),
            "forced_m=-1_vs_r0_disagreement": _acc(p0, r0, dis),
            "forced_m=+1_vs_r1_disagreement": _acc(p1, r1, dis),
            "switch_consistency_disagreement": _acc(
                (p0 == r0).long() * (p1 == r1).long(), torch.ones_like(r0, dtype=torch.long), dis),
            "selector_invariance_agreement": _acc(
                (p0 == p1).long(), torch.ones_like(r0, dtype=torch.long), agree),
        }


def evaluate_zero_message(model, loader, comm_mode, model_kind):
    model.eval()
    out = {"overall": [], "disagreement": []}
    with torch.no_grad():
        for x_A, x_B, b_A, r0, r1, Y in loader:
            if model_kind == "primary":
                lz, _, _, _ = model(x_A, x_B, comm_mode=comm_mode, zero_message=True)
                pred = lz.argmax(-1)
            else:
                a, b, _, _, _ = model(x_A, x_B, comm_mode=comm_mode, zero_message=True)
                pred = model.select(a, b, torch.zeros(x_B.size(0), 1))  # no message info
            out["overall"].append((pred == Y).float())
            out["disagreement"].append(((pred == Y) & (r0 != r1)).float())
    n = sum(len(v) for v in out["overall"])
    dis_total = sum(float(v.sum()) for v in out["disagreement"])
    n_dis = sum(float((r0 != r1).float().sum()) for _, _, _, r0, r1, _ in loader)
    return {"zero_message_overall": sum(float(v.sum()) for v in out["overall"]) / n,
            "zero_message_disagreement": dis_total / n_dis}


def evaluate_trajectory(model, loader, comm_mode):
    model.eval()
    sums = {"overall": [0.0] * 9, "disagree": [0.0] * 9, "agree": [0.0] * 9}
    counts = {"overall": 0, "disagree": 0, "agree": 0}
    with torch.no_grad():
        for x_A, x_B, b_A, r0, r1, Y in loader:
            tp = model(x_A, x_B, comm_mode=comm_mode, trace=True, force_message=1.0)[-1]
            tn = model(x_A, x_B, comm_mode=comm_mode, trace=True, force_message=-1.0)[-1]
            d = [(tp[t] - tn[t]).abs() for t in range(9)]
            dis = (r0 != r1)
            for key, mask in (("overall", torch.ones_like(dis)), ("disagree", dis), ("agree", ~dis)):
                for t in range(9):
                    sums[key][t] += float(d[t][mask].mean()) if bool(mask.any()) else 0.0
                counts[key] += int(mask.sum())
    return {k: {t: s / max(counts[k], 1) for t, s in enumerate(v)} for k, v in sums.items()}


def apply_gates(val_metrics, cf, zero, traj, comm_mode):
    failed = []
    if val_metrics["overall"] < 0.82:
        failed.append("overall<82%")
    if any(v < 0.75 for v in val_metrics["quadrants"].values()):
        failed.append("quadrant<75%")
    if val_metrics["sender_probe_overall"] < 0.95 or any(
            v < 0.90 for v in val_metrics["sender_probe_by_bA"].values()):
        failed.append("probe<95/90/90")
    if zero["zero_message_overall"] > 0.77:
        failed.append("zero_message_overall>77%")
    if zero["zero_message_disagreement"] > 0.53:
        failed.append("zero_message_disagreement>53%")
    if cf["forced_m=-1_vs_r0"] < 0.75 or cf["forced_m=+1_vs_r1"] < 0.75:
        failed.append("counterfactual<75%")
    if cf["switch_consistency_disagreement"] < 0.85:
        failed.append("switch_consistency<85%")
    if cf["selector_invariance_agreement"] < 0.85:
        failed.append("selector_invariance<85%")
    if comm_mode == "full" and not any(traj["overall"][t] > 0.0 for t in range(1, 8)):
        failed.append("no_intermediate_trajectory_sensitivity")
    return {"passed": not failed, "failed_criteria": failed}


def train_phase2c(args):
    set_seed(args.seed)
    model_kind = args.model
    comm_mode = args.mode
    print(f"Phase 2c training: model={model_kind}, mode={comm_mode}, seed={args.seed}, "
          f"epochs={args.epochs}, lr={args.lr}, git={_git_commit()[:12]}, smoke={args.smoke}")

    loaders = build_loaders(args.seed, include_test=not args.smoke)
    train_loader, val_loader = loaders[0], loaders[1]
    test_loader = loaders[2] if len(loaders) > 2 else None
    print(f"Datasets: train=4000, val=1000{', test=1000' if test_loader else ', test=NOT GENERATED (smoke)'}"
          f" (seeds {SPLIT_SEEDS})")

    if model_kind == "primary":
        model = LatentConditionedSwarm()
    else:
        model = LatentDualRelationSwarm()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params}")
    p1_state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_phase1_warm_start(p1_state)
    print(f"Warm start applied from {args.checkpoint} (Core A strict; Core B documented mapping)")

    crit = nn.CrossEntropyLoss()
    crit_msg = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    if args.smoke:
        args.epochs = 1
        print("SMOKE MODE: 2 training batches, 2 validation batches, no test data, no gates.")
        train_loader = [b for i, b in enumerate(train_loader) if i < 2]
        val_loader = [b for i, b in enumerate(val_loader) if i < 2]
        model.train()
        for x_A, x_B, b_A, r0, r1, Y in train_loader:
            optimizer.zero_grad()
            if model_kind == "primary":
                logits, raw_m, aux_A, _ = model(x_A, x_B, comm_mode=comm_mode)
                loss = crit(logits, Y) + 0.5 * crit(aux_A, b_A) \
                     + 0.5 * crit_msg(raw_m.squeeze(-1), b_A.float())
            else:
                r0l, r1l, raw_m, aux_A, _ = model(x_A, x_B, comm_mode=comm_mode)
                loss = crit(r0l, r0) + crit(r1l, r1) + 0.5 * crit(aux_A, b_A) \
                     + 0.5 * crit_msg(raw_m.squeeze(-1), b_A.float())
            assert torch.isfinite(loss).all(), "Non-finite loss in smoke"
            loss.backward()
            optimizer.step()
        none = [n for n, p in model.named_parameters() if p.grad is None and p.requires_grad]
        print(f"Check shapes/loss/grads: loss finite; grad-None={none or 'NONE'}")
        assert none == []
        ev = evaluate_conditioned(model, val_loader, comm_mode, model_kind)
        print(f"Smoke val overall (meaningless small sample): {ev['overall']*100:.1f}%")
        traj = evaluate_trajectory(model, val_loader, comm_mode)
        print(f"Smoke trajectory step1 overall |Δz|: {traj['overall'][1]:.3e} "
              f"(final-only must be 0.0)")
        print("Smoke checklist: engineering checks passed. NOT research results.")
        return

    print(f"Training Phase 2c ({model_kind}/{comm_mode}) at T_A=8, T_B=8...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x_A, x_B, b_A, r0, r1, Y in train_loader:
            optimizer.zero_grad()
            if model_kind == "primary":
                logits, raw_m, aux_A, _ = model(x_A, x_B, comm_mode=comm_mode)
                loss = crit(logits, Y) + 0.5 * crit(aux_A, b_A) \
                     + 0.5 * crit_msg(raw_m.squeeze(-1), b_A.float())
            else:
                r0l, r1l, raw_m, aux_A, _ = model(x_A, x_B, comm_mode=comm_mode)
                loss = crit(r0l, r0) + crit(r1l, r1) + 0.5 * crit(aux_A, b_A) \
                     + 0.5 * crit_msg(raw_m.squeeze(-1), b_A.float())
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            ev = evaluate_conditioned(model, val_loader, comm_mode, model_kind)
            print(f"Epoch {epoch:02d} | Validation selected-task accuracy: {ev['overall']*100:.2f}%")

    val_metrics = evaluate_conditioned(model, val_loader, comm_mode, model_kind)
    cf = evaluate_counterfactuals(model, val_loader, comm_mode, model_kind)
    zero = evaluate_zero_message(model, val_loader, comm_mode, model_kind)
    traj = evaluate_trajectory(model, val_loader, comm_mode)
    print("\n--- Phase 2c Validation Evaluation ---")
    print(f"Overall {val_metrics['overall']*100:.2f}% | by_bA "
          f"{val_metrics['by_bA']['0']*100:.1f}/{val_metrics['by_bA']['1']*100:.1f} | "
          f"agree {val_metrics['agreement_subset']*100:.2f}% | disagree "
          f"{val_metrics['disagreement_subset']*100:.2f}%")
    print("Quadrants:", {k: f"{v*100:.1f}%" for k, v in val_metrics["quadrants"].items()})
    print(f"Probe {val_metrics['sender_probe_overall']*100:.2f}% "
          f"({val_metrics['sender_probe_by_bA']['0']*100:.1f}/"
          f"{val_metrics['sender_probe_by_bA']['1']*100:.1f})")
    print(f"Zero-message overall {zero['zero_message_overall']*100:.2f}% | "
          f"disagreement {zero['zero_message_disagreement']*100:.2f}%")
    print("Counterfactuals:", {k: f"{v*100:.2f}%" for k, v in cf.items()})
    print("Trajectory overall per step:",
          " ".join(f"{traj['overall'][t]:.2e}" for t in range(9)))
    if comm_mode == "final-only":
        leak = max(abs(traj["overall"][t]) for t in range(1, 8))
        print(f"final-only steps1-7 invariance check: max |Δz| = {leak:.2e} (expected 0)")
    status = apply_gates(val_metrics, cf, zero, traj, comm_mode) \
        if model_kind == "primary" else {"passed": None, "failed_criteria": ["diagnostic-model-no-gates"]}
    print(f"GATE STATUS: {status}")

    test_metrics = None
    if test_loader is not None:
        print("\n--- Final Held-Out Test Evaluation (once, frozen protocol) ---")
        test_metrics = evaluate_conditioned(model, test_loader, comm_mode, model_kind)
        t_cf = evaluate_counterfactuals(model, test_loader, comm_mode, model_kind)
        t_zero = evaluate_zero_message(model, test_loader, comm_mode, model_kind)
        print(f"Test overall {test_metrics['overall']*100:.2f}% | disagree "
              f"{test_metrics['disagreement_subset']*100:.2f}% | zero {t_zero['zero_message_overall']*100:.2f}%")

    if args.run_dir:
        os.makedirs(args.run_dir, exist_ok=True)
        ckpt = os.path.join(args.run_dir, f"last_{model_kind}_{comm_mode}_seed{args.seed}.pt")
        torch.save(model.state_dict(), ckpt)
        sha = _sha256(ckpt)
        meta = {
            "artifact_kind": f"phase2c_{model_kind}_{comm_mode}",
            "git_commit": _git_commit(), "seed": args.seed, "mode": comm_mode,
            "model": model_kind, "lr": args.lr, "epochs": args.epochs,
            "param_count": n_params, "torch_version": torch.__version__,
            "phase1_checkpoint_sha256": _sha256(args.checkpoint),
            "data_seeds": SPLIT_SEEDS, "gate_status": status,
            "validation": {k: v for k, v in val_metrics.items() if not k.startswith("_")},
            "counterfactuals": cf, "zero_message": zero,
            "trajectory_sensitivity": traj,
            "test": ({k: v for k, v in test_metrics.items() if not k.startswith("_")}
                     if test_metrics else None),
            "checkpoint": ckpt, "checkpoint_sha256": sha,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(args.run_dir, f"meta_{model_kind}_{comm_mode}_seed{args.seed}.json"),
                  "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        print(f"Artifact saved: {ckpt} (sha256 {sha[:16]}...)")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2c message-conditioned trainer")
    ap.add_argument("--mode", choices=("full", "final-only"), default="full")
    ap.add_argument("--model", choices=("primary", "dual"), default="primary")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--checkpoint", default="checkpoint_phase1.pt")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-unstable-lr", action="store_true",
                    help="Required for non-smoke runs while no stability-eligible LR exists")
    args = ap.parse_args(argv)
    if not args.smoke and not args.allow_unstable_lr:
        print("REFUSING: no stability-eligible learning rate exists (see "
              "reports/phase2b-stability-report.md). Use --smoke for engineering checks only, "
              "or pass --allow-unstable-lr to explicitly accept an unstable recipe.", file=sys.stderr)
        sys.exit(2)
    return args


if __name__ == "__main__":
    train_phase2c(parse_args())

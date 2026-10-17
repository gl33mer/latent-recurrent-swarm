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


def compute_loss(model_kind, outputs, batch, crit, crit_msg):
    """Predeclared objective wiring (reports/phase2c-design.md §6), factored out for
    testability. Primary: selected-task CE on Y + 0.5·aux_A CE + 0.5·msg BCE.
    Dual diagnostic: r0 head CE against r0, r1 head CE against r1 (relation heads are
    supervised ONLY with their own relation labels — never with Y, so late-selection
    candidates stay relation-pure), + 0.5·aux_A CE + 0.5·msg BCE. The discrete
    selection itself is not part of any loss; the sender's gradient path is the msg BCE."""
    _, _, b_A, r0, r1, Y = batch
    if model_kind == "primary":
        logits, raw_m, aux_A, m = outputs
        return crit(logits, Y) + 0.5 * crit(aux_A, b_A) \
             + 0.5 * crit_msg(raw_m.squeeze(-1), b_A.float())
    r0_logits, r1_logits, raw_m, aux_logits_A, m = outputs
    return crit(r0_logits, r0) + crit(r1_logits, r1) + 0.5 * crit(aux_logits_A, b_A) \
         + 0.5 * crit_msg(raw_m.squeeze(-1), b_A.float())


def build_loaders(seed):
    """Training/validation ONLY. The confirmatory Phase 2c test split (seed 99991) is
    never generated here; held-out evaluation is the exclusive job of the separate,
    explicitly invoked evaluator `evaluate_phase2c_test.py` (Fix 3, 2026-10-17)."""
    train_ds, train_h = generate_conditioned_benchmark(num_samples=4000, seed=SPLIT_SEEDS["train"])
    val_ds, val_h = generate_conditioned_benchmark(num_samples=1000, seed=SPLIT_SEEDS["validation"],
                                                   existing_hashes=train_h)
    return [DataLoader(train_ds, batch_size=64, shuffle=True,
                       generator=torch.Generator().manual_seed(9000 + seed)),
            DataLoader(val_ds, batch_size=128, shuffle=False)]


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
            "_pred": P, "_Y": Y, "_bA": bA, "_r0": r0, "_r1": r1, "_disagree": disagree, "_agree": agree,
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
    """Corrected trajectory-sensitivity metrics (Fix 4, 2026-10-17).

    For identical inputs, model weights and eval mode, run paired counterfactual traces
    with forced m=+1 and m=-1 and report TRUE element-weighted means:
        mean_abs_delta = sum(|delta|) / number_of_elements_included
    per recurrent step (0 = post-encoder, 1..8 = after each step), over
    - all 16 ordinary graph nodes and all 128 latent dims (`graph_nodes_mean_abs_delta`),
    - the terminal node 15 only (`terminal_node_mean_abs_delta`),
    overall and conditioned on relation agreement/disagreement (the Phase 2c analogue of
    the b_B grouping), with sample counts per condition. The previous implementation
    summed per-batch MEANS and divided by sample count, which understates the magnitude
    by a factor of about the batch size; old values are retracted for quantitative use
    (see reports/phase2c-trajectory-metric-erratum.md).

    In final-only mode, ordinary graph-node states must be BIT-IDENTICAL across the two
    messages through step 7; this is asserted directly via exact tensor equality and
    reported as a boolean (step 8 may differ). Empty conditioning groups and non-finite
    deltas raise explicit errors. The COMM register's own message-dependent initial state
    is deliberately excluded: only ordinary graph-node states are measured."""
    model.eval()
    groups = ("overall", "agree", "disagree")
    sum_abs = {g: {"graph": [0.0] * 9, "terminal": [0.0] * 9} for g in groups}
    elem = {g: {"graph": [0] * 9, "terminal": [0] * 9} for g in groups}
    samples = {g: 0 for g in groups}
    invariance_exact = [] if comm_mode == "final-only" else None
    with torch.no_grad():
        for x_A, x_B, b_A, r0, r1, Y in loader:
            tp = model(x_A, x_B, comm_mode=comm_mode, trace=True, force_message=1.0)[-1]
            tn = model(x_A, x_B, comm_mode=comm_mode, trace=True, force_message=-1.0)[-1]
            dis = (r0 != r1)
            group_masks = {"overall": torch.ones_like(dis), "agree": ~dis, "disagree": dis}
            for t in range(9):
                delta = (tp[t] - tn[t]).abs()
                if not torch.isfinite(delta).all():
                    raise RuntimeError(f"Non-finite trajectory delta at step {t}; refusing "
                                       "to report a meaningless metric.")
                if comm_mode == "final-only" and t <= 7:
                    invariance_exact.append(bool(torch.equal(tp[t], tn[t])))
                n_nodes, d_dim = delta.shape[1], delta.shape[2]
                for g, mask in group_masks.items():
                    sel = delta[mask]                      # [n_sel, 16, 128]
                    n_sel = int(mask.sum())
                    if n_sel == 0:
                        continue
                    sum_abs[g]["graph"][t] += float(sel.sum())
                    elem[g]["graph"][t] += n_sel * n_nodes * d_dim
                    sum_abs[g]["terminal"][t] += float(sel[:, -1, :].sum())
                    elem[g]["terminal"][t] += n_sel * d_dim
            for g, mask in group_masks.items():
                samples[g] += int(mask.sum())
    if samples["overall"] == 0:
        raise RuntimeError("Empty evaluation set for trajectory sensitivity.")
    if samples["agree"] == 0 or samples["disagree"] == 0:
        raise RuntimeError("Empty agreement/disagreement conditioning group; refusing to "
                           "report a silently-divided metric.")
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
        out["final_only_steps_1_7_exact_invariance"] = all(invariance_exact)
    return out


def _terminal_sensitivity_ok(traj):
    """Full-mode mechanism criterion: terminal-node (z_B[15]) sensitivity strictly > 0 at
    at least one intermediate step (1..7). No floating tolerance: CPU float32 arithmetic
    here is deterministic, and any positive attention weight on the COMM key makes the
    element mean strictly positive."""
    return any(v > 0.0 for v in traj["terminal_node_mean_abs_delta"]["overall"][1:8])


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
    if comm_mode == "full" and not _terminal_sensitivity_ok(traj):
        failed.append("no_intermediate_terminal_node_sensitivity")
    if comm_mode == "final-only" and not traj.get("final_only_steps_1_7_exact_invariance", False):
        failed.append("final_only_graph_states_not_bit_identical_steps_1_7")
    return {"passed": not failed, "failed_criteria": failed}


def train_phase2c(args):
    set_seed(args.seed)
    model_kind = args.model
    comm_mode = args.mode
    print(f"Phase 2c training: model={model_kind}, mode={comm_mode}, seed={args.seed}, "
          f"epochs={args.epochs}, lr={args.lr}, git={_git_commit()[:12]}, smoke={args.smoke}")

    loaders = build_loaders(args.seed)
    train_loader, val_loader = loaders[0], loaders[1]
    print("Datasets: train=4000, val=1000, test=NOT GENERATED (held-out evaluation is "
          "the separate evaluator's job)" f" (seeds train/val {SPLIT_SEEDS['train']}/{SPLIT_SEEDS['validation']})")

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
        for batch in train_loader:
            optimizer.zero_grad()
            if model_kind == "primary":
                outputs = model(batch[0], batch[1], comm_mode=comm_mode)
            else:
                outputs = model(batch[0], batch[1], comm_mode=comm_mode)
            loss = compute_loss(model_kind, outputs, batch, crit, crit_msg)
            assert torch.isfinite(loss).all(), "Non-finite loss in smoke"
            loss.backward()
            optimizer.step()
        none = [n for n, p in model.named_parameters() if p.grad is None and p.requires_grad]
        print(f"Check shapes/loss/grads: loss finite; grad-None={none or 'NONE'}")
        assert none == []
        ev = evaluate_conditioned(model, val_loader, comm_mode, model_kind)
        print(f"Smoke val overall (meaningless small sample): {ev['overall']*100:.1f}%")
        traj = evaluate_trajectory(model, val_loader, comm_mode)
        print(f"Smoke trajectory step1 graph-nodes mean |Δz|: "
              f"{traj['graph_nodes_mean_abs_delta']['overall'][1]:.3e}; terminal node: "
              f"{traj['terminal_node_mean_abs_delta']['overall'][1]:.3e}"
              + (" (final-only: both must be exactly 0.0)" if comm_mode == "final-only" else ""))
        print("Smoke checklist: engineering checks passed. NOT research results.")
        return

    print(f"Training Phase 2c ({model_kind}/{comm_mode}) at T_A=8, T_B=8...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch[0], batch[1], comm_mode=comm_mode)
            loss = compute_loss(model_kind, outputs, batch, crit, crit_msg)
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
    print("Trajectory graph-nodes mean |Δz| per step (overall):",
          " ".join(f"{v:.2e}" for v in traj["graph_nodes_mean_abs_delta"]["overall"]))
    print("Trajectory terminal-node mean |Δz| per step (overall):",
          " ".join(f"{v:.2e}" for v in traj["terminal_node_mean_abs_delta"]["overall"]))
    if comm_mode == "final-only":
        print(f"final-only steps 1-7 exact bit-identity: "
              f"{traj['final_only_steps_1_7_exact_invariance']} (required: True)")
    status = apply_gates(val_metrics, cf, zero, traj, comm_mode) \
        if model_kind == "primary" else {"passed": None, "failed_criteria": ["diagnostic-model-no-gates"]}
    print(f"GATE STATUS: {status}")
    print("NOTE: no test evaluation occurred and no test split exists; held-out "
          "evaluation must be run via the separate manifest-driven evaluator "
          "evaluate_phase2c_test.py after all configuration is frozen.")

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
            "test": "NOT EVALUATED (separate evaluator required)",
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

"""
Phase 2c Receiver Audit — Stage 1: Matched Single-Relation Mask Ablation.

Directive: phase2c_receiver_audit_and_ablation_directive.md §4.

Question: is the Phase 2c failure caused by the UNION attention mask (R0 ∪ R1) rather
than by the input representation or task difficulty?

Matched design — the ONLY variable is the attention mask:
- Target: reachability in Relation 0 (r0).
- Input features: the full 64-dim Phase 2c vector
  [adj_union(16), adj_R0(16), adj_R1(16), onehot(16)] — identical for both conditions.
- Architecture: LatentReasoner(num_nodes=16, d_input=64, d_model=128); identical
  initialisation for both conditions (torch.manual_seed(42) before each construction;
  initial-state checksums verified equal).
- Training: 20 epochs, AdamW (lr=1e-3, weight_decay=1e-2), batch 64, fixed T=8,
  CrossEntropy on r0.
- Condition A (Native Mask): attention mask constructed strictly from A_R0.
- Condition B (Union Mask): attention mask constructed from A_union = A_R0 OR A_R1
  (the Phase 2c setup).

Data: established Phase 2c splits (train seed 4201 / 4,000; validation seed 13371 /
1,000; hashes chained). The reserved test split (99991) is never generated.

Decision matrix (per directive):
  A >= 90% and B <= 70%  -> union-mask interference confirmed.
  A >= 90% and B >= 90%  -> mask interference rejected (failure lies elsewhere).
  A < 75%  and B < 75%   -> feature-representation failure.

Engineering-only diagnostic; no acceptance thresholds changed; not a research-result
claim about the confirmatory experiment.
"""
import hashlib
import json
import os
import random
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.core import LatentReasoner
from data.conditioned_graph_task import generate_conditioned_benchmark

torch.set_num_threads(4)

EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-2
BATCH = 64
STEPS = 8


def relation_mask(adj_block: torch.Tensor) -> torch.Tensor:
    """Attention mask from a [B, 16, 16] adjacency block, v7.1 orientation:
    query j attends key i iff i == j or adj[i, j] > 0.5. Returns [B, 1, 16, 16] bool."""
    eye = torch.eye(16, dtype=torch.bool, device=adj_block.device).unsqueeze(0)
    return ((adj_block.transpose(-1, -2) > 0.5) | eye).unsqueeze(1)


def evaluate(model, loader, mask_builder):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x_B, r0 in loader:
            mask = mask_builder(x_B)
            logits, _ = model(x_B, steps=STEPS, attn_mask=mask)
            preds = logits.argmax(dim=-1)
            correct += (preds == r0).sum().item()
            total += r0.size(0)
    return correct / total


def run_condition(name, mask_fn, train_loader, val_loader):
    """Train one condition; identical init for both (seed 42 before construction)."""
    torch.manual_seed(42)
    random.seed(42)
    model = LatentReasoner(num_nodes=16, d_input=64, d_model=128)
    init_checksum = hashlib.sha256(
        b"".join(t.detach().numpy().tobytes() for t in model.state_dict().values())
    ).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    curve = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = n = 0
        for x_B, r0 in train_loader:
            mask = mask_fn(x_B)
            optimizer.zero_grad()
            logits, _ = model(x_B, steps=STEPS, attn_mask=mask)
            loss = crit(logits, r0)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            n += 1
        val_acc = evaluate(model, val_loader, mask_fn)
        curve.append({"epoch": epoch, "train_loss": total_loss / n, "val_acc_r0": val_acc})
        print(f"[{name}] epoch {epoch:02d} | train loss {total_loss/n:.4f} | "
              f"val R0-reachability accuracy {val_acc*100:.2f}%", flush=True)
    final_acc = evaluate(model, val_loader, mask_fn)
    print(f"[{name}] FINAL validation R0-reachability accuracy: {final_acc*100:.2f}%")
    return {"name": name, "init_checksum": init_checksum, "curve": curve,
            "final_val_acc": final_acc}


def main(run_dir=None):
    random.seed(42)
    torch.manual_seed(42)
    print("Generating data (train seed 4201 / 4,000; validation seed 13371 / 1,000; "
          "test seed 99991 NOT used)...")
    train_ds, train_h = generate_conditioned_benchmark(num_samples=4000, seed=4201)
    val_ds, _ = generate_conditioned_benchmark(num_samples=1000, seed=13371,
                                               existing_hashes=train_h)
    x_B, r0 = train_ds.tensors[1], train_ds.tensors[3]
    vx_B, vr0 = val_ds.tensors[1], val_ds.tensors[3]
    train_loader = DataLoader(torch.utils.data.TensorDataset(x_B, r0), batch_size=BATCH,
                              shuffle=True,
                              generator=torch.Generator().manual_seed(9000 + 42))
    val_loader = DataLoader(torch.utils.data.TensorDataset(vx_B, vr0), batch_size=256)

    cond_a = run_condition("A native-R0-mask", lambda xb: relation_mask(xb[:, :, 16:32]),
                           train_loader, val_loader)
    cond_b = run_condition("B union-mask", lambda xb: relation_mask(xb[:, :, 0:16]),
                           train_loader, val_loader)

    assert cond_a["init_checksum"] == cond_b["init_checksum"], \
        "Conditions must start from identical weights"
    print(f"\nIdentical initialisation verified for both conditions "
          f"(checksum {cond_a['init_checksum'][:16]}...).")

    a, b = cond_a["final_val_acc"], cond_b["final_val_acc"]
    print("\n--- Stage 1 decision matrix ---")
    print(f"Condition A (native R0 mask):   {a*100:.2f}%")
    print(f"Condition B (union mask):       {b*100:.2f}%")
    if a >= 0.90 and b <= 0.70:
        ruling = "UNION MASK INTERFERENCE CONFIRMED — dot-product attention cannot filter R1 edges from node features alone; close the Phase 2c multi-relation experiment as designed."
    elif a >= 0.90 and b >= 0.90:
        ruling = "MASK INTERFERENCE REJECTED — the receiver can filter relations; the Phase 2c failure lies in COMM-token dynamics or downstream use, not the mask."
    elif a < 0.75 and b < 0.75:
        ruling = "FEATURE REPRESENTATION FAILURE — packing multiple adjacencies into 64-dim node features destabilises the encoder; redesign input encoding."
    else:
        ruling = (f"PARTIAL/AMBIGUOUS — A={a*100:.1f}%, B={b*100:.1f}% falls between the "
                  "directive's matrix cells; report and interpret with caution.")
    print(f"Ruling: {ruling}")

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "stage1_summary.json"), "w") as f:
            json.dump({
                "stage": "phase2c_audit_stage1_mask_ablation",
                "config": {"epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
                           "batch": BATCH, "steps": STEPS, "target": "r0",
                           "data": {"train_seed": 4201, "train_n": 4000,
                                    "val_seed": 13371, "val_n": 1000,
                                    "test_seed_used": None}},
                "condition_A_native_R0_mask": cond_a,
                "condition_B_union_mask": cond_b,
                "final_acc_A": a, "final_acc_B": b,
                "ruling": ruling,
                "git_commit": os.popen("git rev-parse HEAD").read().strip(),
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2, sort_keys=True)
        print(f"Summary saved: {os.path.join(run_dir, 'stage1_summary.json')}")
    return a, b, ruling


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)

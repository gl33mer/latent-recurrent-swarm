"""
Phase 2c Receiver Audit — Stage 0: Frozen Phase 1 Baseline Transfer Check (zero training).

Directive: phase2c_receiver_audit_and_ablation_directive.md §3.

Question: does the existing, known-good Phase 1 checkpoint transfer directly to the
INDIVIDUAL relation graphs produced by the Phase 2c benchmark?

Method:
- Load `checkpoint_phase1.pt` into a standard `LatentReasoner(num_nodes=16, d_input=32,
  d_model=128)` (strict load; no training, no fine-tuning).
- Generate 1,000 validation samples with the Phase 2c conditioned generator using the
  established Phase 2c validation split seed (13371) — NOT the reserved test split
  (99991), which is never touched.
- For each sample, build two single-relation evaluation sets from the stored relation
  adjacency matrices:
    * Eval-R0: node features [A_R0, onehot] (shape [16, 32]); attention mask derived
      strictly from A_R0 (the model's internal mask, whose first 16 input columns are
      exactly A_R0 here); target = r0 (BFS reachability in R0).
    * Eval-R1: node features [A_R1, onehot]; mask strictly from A_R1; target = r1.
- Evaluate at recurrent depth T=8.

Decision logic (per directive):
- accuracy(R0) >= 90% AND accuracy(R1) >= 90%  -> transfer clean; proceed to Stage 1.
- accuracy < 80%                                -> the Phase 2c graph distribution broke
                                                  the base reasoner; stop and report.
- otherwise                                     -> ambiguous; report and continue with caution.

Engineering-only diagnostic; no acceptance thresholds are changed; results are not
research-result claims about Phase 2c training.
"""
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

PHASE1_SHA_EXPECTED = "ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e"


def main(run_dir=None):
    random.seed(42)
    torch.manual_seed(42)

    import hashlib
    sha = hashlib.sha256(open("checkpoint_phase1.pt", "rb").read()).hexdigest()
    assert sha == PHASE1_SHA_EXPECTED, f"Phase 1 checkpoint hash mismatch: {sha}"
    print(f"Phase 1 checkpoint verified: {sha[:16]}...")

    print("Generating 1,000 validation samples (Phase 2c generator, split seed 13371; "
          "test seed 99991 is NOT used)...")
    val_ds, _ = generate_conditioned_benchmark(num_samples=1000, seed=13371)
    x_A, x_B, b_A, r0, r1, Y = val_ds.tensors
    print(f"Samples: {len(Y)}; tensor shapes: x_B {tuple(x_B.shape)}")

    model = LatentReasoner(num_nodes=16, d_input=32, d_model=128)
    state = torch.load("checkpoint_phase1.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # strict
    model.eval()
    print("Phase 1 reasoner loaded (strict; frozen; no training in this stage).")

    eye = torch.eye(16).unsqueeze(0)  # one-hot node id block for 32-dim inputs
    loader = DataLoader(torch.utils.data.TensorDataset(x_B, r0, r1), batch_size=256)

    correct = {"R0": 0, "R1": 0}
    total = 0
    with torch.no_grad():
        for x_Bb, r0b, r1b in loader:
            for key, adj_slice, target in (("R0", slice(16, 32), r0b),
                                           ("R1", slice(32, 48), r1b)):
                adj = x_Bb[:, :, adj_slice]               # [B, 16, 16] relation adjacency
                feat = torch.cat([adj, eye.expand(adj.size(0), -1, -1)], dim=-1)  # [B,16,32]
                logits, _ = model(feat, steps=8)          # internal mask derives from adj
                preds = logits.argmax(dim=-1)
                correct[key] += (preds == target).sum().item()
            total += r0b.size(0)

    acc = {k: correct[k] / total for k in ("R0", "R1")}
    print(f"\n--- Stage 0 results (T=8, frozen Phase 1 reasoner, n={total}) ---")
    print(f"Transfer accuracy on single-relation R0 graphs: {acc['R0']*100:.2f}%")
    print(f"Transfer accuracy on single-relation R1 graphs: {acc['R1']*100:.2f}%")

    if acc["R0"] >= 0.90 and acc["R1"] >= 0.90:
        ruling = "TRANSFER CLEAN — proceed to Stage 1 (mask ablation)."
    elif acc["R0"] < 0.80 or acc["R1"] < 0.80:
        ruling = "DISTRIBUTION DIVERGENCE — the Phase 2c graph distribution broke the base reasoner; stop and report."
    else:
        ruling = "AMBIGUOUS (80-90%) — report and proceed to Stage 1 with caution."
    print(f"Ruling: {ruling}")

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "stage0_summary.json"), "w") as f:
            json.dump({
                "stage": "phase2c_audit_stage0_transfer",
                "data": {"generator": "generate_conditioned_benchmark", "n": 1000,
                         "split_seed": 13371, "test_seed_used": None},
                "phase1_checkpoint_sha256": sha,
                "recurrent_depth": 8,
                "accuracy_R0": acc["R0"], "accuracy_R1": acc["R1"],
                "ruling": ruling,
                "git_commit": os.popen("git rev-parse HEAD").read().strip(),
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2, sort_keys=True)
        print(f"Summary saved: {os.path.join(run_dir, 'stage0_summary.json')}")
    return acc, ruling


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)

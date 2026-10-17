"""
Explicit, manifest-driven held-out test evaluator for Phase 2c (Fix 3, 2026-10-17).

Training and test evaluation are strictly separate entry points:
- `train_swarm_conditioned.py` never generates or evaluates the Phase 2c test split;
- this evaluator is the ONLY component permitted to generate the predeclared test split
  (seed 99991, defined in data.conditioned_graph_task.SPLIT_SEEDS) and it does so only
  when explicitly invoked with a frozen evaluation manifest.

Safety rules enforced here:
- the manifest must be explicitly marked frozen, carry a source commit, the designated
  split seed, and one entry per final model (mode, model kind, paired seed, checkpoint
  path, checkpoint SHA-256, full configuration);
- every checkpoint must exist and match its recorded SHA-256 BEFORE any test data is
  generated; a missing checkpoint or checksum mismatch aborts without generating data;
- ambiguous manifests (duplicate model/mode/seed entries) are rejected;
- each listed model is evaluated exactly once on the same test examples; predictions and
  metrics are written to ignored run storage with provenance;
- this evaluator makes NO training, checkpoint-selection, or acceptance decisions and
  applies NO gates; it reports measurements only.

This module was implemented and unit-tested on 2026-10-17 but deliberately NOT invoked:
no Phase 2c test data has been generated or evaluated as of this commit.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader

from data.conditioned_graph_task import generate_conditioned_benchmark, SPLIT_SEEDS
from model.swarm_conditioned import LatentConditionedSwarm, LatentDualRelationSwarm
# Reuse the trainer's evaluation helpers so metric definitions cannot drift.
from train_swarm_conditioned import (evaluate_conditioned, evaluate_counterfactuals,
                                     evaluate_zero_message, evaluate_trajectory)

MANIFEST_KIND = "phase2c_test_evaluation_manifest"
REQUIRED_MANIFEST_FIELDS = {"kind", "frozen", "created_utc", "source_commit", "split_seed", "entries"}
REQUIRED_ENTRY_FIELDS = {"model", "mode", "seed", "checkpoint", "checkpoint_sha256",
                         "phase1_checkpoint", "phase1_checkpoint_sha256", "config"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    """Load and strictly validate the evaluation manifest. Refuses unfrozen, malformed,
    ambiguous or wrong-seed manifests. Returns the parsed manifest dict."""
    if not os.path.exists(path):
        raise RuntimeError(f"Manifest not found: {path}")
    with open(path) as f:
        manifest = json.load(f)
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise RuntimeError(f"Manifest missing required fields: {sorted(missing)}")
    if manifest["kind"] != MANIFEST_KIND:
        raise RuntimeError(f"Wrong manifest kind: {manifest['kind']!r}")
    if manifest["frozen"] is not True:
        raise RuntimeError("Manifest is not declared frozen; refusing to evaluate. "
                           "Freeze the manifest only after all training/validation "
                           "decisions are complete.")
    if manifest["split_seed"] != SPLIT_SEEDS["test"]:
        raise RuntimeError(f"Manifest split_seed {manifest['split_seed']!r} is not the "
                           f"predeclared Phase 2c test seed {SPLIT_SEEDS['test']}.")
    if not manifest["source_commit"]:
        raise RuntimeError("Manifest must record the source commit of every trained model.")
    entries = manifest["entries"]
    if not entries:
        raise RuntimeError("Manifest has no entries.")
    seen = set()
    for e in entries:
        missing = REQUIRED_ENTRY_FIELDS - set(e)
        if missing:
            raise RuntimeError(f"Entry missing required fields: {sorted(missing)}")
        if e["model"] not in ("primary", "dual"):
            raise RuntimeError(f"Unknown model kind: {e['model']!r}")
        if e["mode"] not in ("full", "final-only"):
            raise RuntimeError(f"Unknown mode: {e['mode']!r}")
        key = (e["model"], e["mode"], e["seed"])
        if key in seen:
            raise RuntimeError(f"Ambiguous manifest: duplicate entry for {key}.")
        seen.add(key)
    return manifest


def verify_artifacts(manifest):
    """Verify every checkpoint (and the Phase 1 warm-start reference) exists and matches
    its recorded SHA-256. Must pass BEFORE any test data is generated."""
    for e in manifest["entries"]:
        if not os.path.exists(e["checkpoint"]):
            raise RuntimeError(f"Checkpoint missing: {e['checkpoint']}")
        actual = _sha256(e["checkpoint"])
        if actual != e["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint checksum mismatch for {e['checkpoint']}: "
                               f"manifest {e['checkpoint_sha256'][:16]}... != actual {actual[:16]}...")
        if not os.path.exists(e["phase1_checkpoint"]):
            raise RuntimeError(f"Phase 1 checkpoint missing: {e['phase1_checkpoint']}")
        p1 = _sha256(e["phase1_checkpoint"])
        if p1 != e["phase1_checkpoint_sha256"]:
            raise RuntimeError(f"Phase 1 checksum mismatch for {e['phase1_checkpoint']}.")
    print(f"Artifacts verified: {len(manifest['entries'])} entr(ies) — checksums OK.")


def build_model(entry):
    model = LatentConditionedSwarm() if entry["model"] == "primary" else LatentDualRelationSwarm()
    phase1 = torch.load(entry["phase1_checkpoint"], map_location="cpu", weights_only=True)
    model.load_phase1_warm_start(phase1)
    state = torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # strict: the checkpoint must be exactly this architecture
    model.eval()
    return model


def run_evaluation(manifest_path, out_dir):
    manifest = load_manifest(manifest_path)      # refuses unfrozen/malformed/ambiguous
    verify_artifacts(manifest)                    # refuses missing/checksum-mismatched
    os.makedirs(out_dir, exist_ok=True)

    # The ONLY legitimate generation point of the predeclared Phase 2c test split.
    test_ds, _ = generate_conditioned_benchmark(num_samples=1000, seed=manifest["split_seed"])
    loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    rows = []
    for e in manifest["entries"]:
        model = build_model(e)
        metrics = evaluate_conditioned(model, loader, e["mode"], e["model"])
        cf = evaluate_counterfactuals(model, loader, e["mode"], e["model"])
        zero = evaluate_zero_message(model, loader, e["mode"], e["model"])
        traj = evaluate_trajectory(model, loader, e["mode"])
        # Persist per-example paired predictions for independent re-analysis (ignored dir).
        preds_path = os.path.join(out_dir, f"predictions_{e['model']}_{e['mode']}_seed{e['seed']}.pt")
        torch.save({"pred": metrics["_pred"], "Y": metrics["_Y"],
                    "b_A": metrics["_bA"], "r0": metrics["_r0"], "r1": metrics["_r1"]},
                   preds_path)
        record = {
            "model": e["model"], "mode": e["mode"], "seed": e["seed"],
            "checkpoint": e["checkpoint"], "checkpoint_sha256": e["checkpoint_sha256"],
            "source_commit": manifest["source_commit"], "split_seed": manifest["split_seed"],
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "overall": metrics["overall"], "quadrants": metrics["quadrants"],
            "disagreement_subset": metrics["disagreement_subset"],
            "agreement_subset": metrics["agreement_subset"],
            "sender_probe_overall": metrics["sender_probe_overall"],
            "counterfactuals": cf, "zero_message": zero,
            "trajectory_sensitivity": traj,
        }
        result_path = os.path.join(out_dir, f"test_metrics_{e['model']}_{e['mode']}_seed{e['seed']}.json")
        with open(result_path, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True)
        rows.append([e["model"], e["mode"], e["seed"], f"{metrics['overall']*100:.2f}",
                     f"{metrics['disagreement_subset']*100:.2f}",
                     f"{metrics['sender_probe_overall']*100:.2f}"])
        print(f"[{e['model']}/{e['mode']} seed {e['seed']}] test overall "
              f"{metrics['overall']*100:.2f}% (reported only; no decisions made)")

    with open(os.path.join(out_dir, "test_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mode", "seed", "overall_acc", "disagreement_acc", "probe_overall"])
        w.writerows(rows)
    with open(os.path.join(out_dir, "evaluation_manifest_used.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print("Evaluation complete. No acceptance decisions were applied to test results.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2c held-out test evaluator (explicit invocation only)")
    ap.add_argument("--manifest", required=True, help="Frozen evaluation manifest JSON")
    ap.add_argument("--out-dir", required=True, help="Ignored output directory for metrics/predictions")
    args = ap.parse_args(argv)
    run_evaluation(args.manifest, args.out_dir)


if __name__ == "__main__":
    main()

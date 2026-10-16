# Phase 2b Analysis Addendum (dated 2026-10-16, post-run correction)

**Status:** addendum to `reports/phase2b-run-report.md` (commit `d82f555`). The original
report, its tables, and all six run outcomes are preserved unchanged; this addendum
corrects one causal interpretation, adds transparent paired arithmetic, and records
artifact provenance. Nothing here re-opens or re-runs the original runs.

## 1. Correction to the causal interpretation of optimisation instability

The original report stated (§3, preserved verbatim below for the record):

> "Every Phase 2b run shows strong non-monotonicity (collapses of 20–50 points followed by
> partial or no recovery), whereas Phase 2a at the identical recipe was deterministic
> (97.50% reproduced exactly). Routing the message through the recurrent trajectory
> demonstrably destabilises optimisation at this learning rate/epoch budget."

The final clause is **stronger than the experiment identifies**. Both `full` and
`final-only` Phase 2b modes exhibited instability, and those two modes share most of the
architectural differences from Phase 2a (17-token Core B, COMM register and its embedding
path, terminal-node-only readout); they differ only in the per-step COMM visibility
schedule. The corrected wording for all future summaries is:

> Phase 2b showed substantial seed- and epoch-varying validation performance under the
> frozen recipe. The existing experiment does not isolate whether recurrent message
> integration itself, other Phase 2b architectural differences, or their interaction with
> optimisation caused the instability.

Resolving this is the purpose of the follow-up instrumented stability study (per-step loss
components and gradient norms under a predeclared learning-rate grid); its design is
committed separately before any sweep run.

## 2. Transparent paired validation arithmetic (exploratory, descriptive only)

Using the six recorded validation connected accuracies, the paired full-minus-final-only
differences are:

| Seed | full | final-only | Δ (full − final-only) |
|---|---:|---:|---:|
| 42 | 91.7% | 99.7% | −8.0 pp |
| 43 | 95.0% | 67.6% | +27.4 pp |
| 44 | 82.3% | 74.8% | +7.5 pp |
| **mean** | | | **+8.97 pp** |
| **median** | | | **+7.5 pp** |

These are **exploratory descriptive comparisons only**: only three paired seeds exist, the
spread is very large (−8.0 to +27.4 pp), no paired seed had both modes pass every
acceptance gate, and with n=3 no meaningful inferential statistic is available. The two
passing models' held-out test numbers (full/seed-43: 94.80%; final-only/seed-42: 100.00%)
are **different seeds** and must not be treated as a paired test comparison.

## 3. Artifact availability and checksums (verified 2026-10-16)

| Artifact | SHA-256 (first 16) | Location | Status |
|---|---|---|---|
| Phase 1 checkpoint | `ef4514342b65e16f` | `checkpoint_phase1.pt`, `runs/20261016T211522Z/` | present, matches recorded hash exactly |
| Phase 2a rerun checkpoint | `05a163e23e240173` | `runs/20261016T215726Z/` | present, exact-metric reproduction |
| Phase 2b full/seed43 | `38c505a5f9ae938f` | `runs/20261016T220921Z_comm_full_seed43/` | present, matches stored `.sha256` and metadata JSON |
| Phase 2b final-only/seed42 | `79aadaf7880c369c` | `runs/20261016T221820Z_comm_final-only_seed42/` | present, matches stored `.sha256` and metadata JSON |
| Failed-run logs (4 runs) | — | `runs/20261016T220650Z/_221141Z_/_221356Z_/_221607Z_comm_*` | logs + env snapshots present; no checkpoints (artifact save is post-gate only) |
| Smoke logs (both modes) | — | `runs/20261016T220607Z_smoke/` | present |

A dated checksummed archive of all Phase 2b run directories, the smoke logs, the Phase 2a
reproduction run, and `checkpoint_phase1.pt` was copied to
`/workspace/latent-recurrent-swarm-archive/phase2b-artifacts_20261016T225525Z.tar.gz`
(SHA-256 `2e9c8d96afe0aedc…`, 4.1 MB). **Storage caveat:** this VM exposes a single 20 GB
volume; `/workspace` is the only persistent location available and no external/durable
mount exists, so the archive resides outside the source tree but on the same volume and
should be treated as potentially ephemeral on VM destruction. No model binaries are or
will be uploaded to the public repository.

## 4. Metadata consistency

- Repository visibility: **public** (owner-requested change, recorded in `CHANGELOG.md`);
  verified via API on 2026-10-16. `reports/first-run-report.md` already carries the
  corrected visibility statement with provenance (housekeeping commit `d450847` on
  `main`, included in this branch's history), so no further edit is needed here.
- Baseline regression on this branch before any Phase 2c/stability work:
  `pytest -q test_system.py test_phase2b.py` → **18 passed** (log: run directory
  `runs/20261016T225501Z_stage0_audit/regression.log`).

## 5. No test reuse

The Phase 2b seed-9999 test evaluations (two passing checkpoints) remain historical
results. They will not be reused as evidence for the stability study or Phase 2c; new
task-specific splits with new seeds are predeclared in the Phase 2c design.

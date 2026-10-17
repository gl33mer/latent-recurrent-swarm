# Phase 2c Receiver-Capability Audit & Matched Mask Ablation — Report (2026-10-17)

**Branch:** `experiment/phase2c-receiver-audit` (from `experiment/phase2c-lambda2-paired-confirmatory` @ `bc75501`)
**Run ID:** `runs/20261017T151354Z_phase2c_audit/` (stage0.log, stage1.log, stage0_summary.json, stage1_summary.json, environment.freeze.txt)
**Environment:** Debian 12, Python 3.11.2, torch 2.14.1+cpu (CUDA off), 4 intra-op threads, float32, seed 42 for model initialisation.
**Total diagnostic wall time:** ≈ 2 m 30 s (Stage 0 ≈ 8 s; Stage 1 ≈ 2 m 07 s) — within the directive's <5-minute budget.
**Test-split sanctity:** the reserved test split (seed 99991) and the Phase 2b seed 9999 were **never generated or evaluated**. Data: established Phase 2c validation split (seed 13371) for Stage 0 and Stage 1 validation; established train split (seed 4201) for Stage 1 training.

## Motivation

The 15-run confirmatory study ended `PARTIAL/UNSTABLE` (overall ~60%, disagreement subset at chance, dual diagnostic also at chance). Before further sweeps or redesigns, this audit tests two hypotheses:

1. **Task/Distribution Mismatch:** the Phase 1 reasoner cannot solve the new graph distribution even with a single relation given directly.
2. **Union-Mask Interference:** the reasoner solves single-relation reachability easily, but masking on R0 ∪ R1 while encoding relations only in 64-dim node features destroys the topological inductive bias.

## Stage 0 — Frozen Phase 1 Baseline Transfer Check (zero training)

Frozen `checkpoint_phase1.pt` (SHA-256 verified `ef4514…49e`), strict load into `LatentReasoner(16, 32, 128)`; 1,000 Phase 2c validation samples; per-sample single-relation evaluation sets ([A_R0, onehot] with mask strictly from A_R0 → target r0; same for R1 → r1); T=8; no training.

| Evaluation set | Accuracy |
|---|---|
| R0 graphs | **99.80%** |
| R1 graphs | **99.80%** |

**Ruling: TRANSFER CLEAN** (≥90% on both). Hypothesis 1 is **rejected**: the Phase 2c graph distribution itself is entirely solvable by the frozen Phase 1 reasoner; the confirmatory failure is not a distribution/representation mismatch at the single-relation level.

## Stage 1 — Matched Single-Relation Mask Ablation

Identical everything (64-dim inputs `[union, R0, R1, onehot]`, target r0, 20 epochs, AdamW lr 1e-3 / wd 1e-2, batch 64, T=8, **identical initial weights** — checksums verified equal, `af5b47a2c084fc5f…`) except the attention mask:

| | Condition A (native R0 mask) | Condition B (union mask, Phase 2c setup) |
|---|---|---|
| epoch 1 | 59.7% | 54.4% |
| epoch 2 | 66.6% | 56.9% |
| epoch 3 | 74.8% | 63.6% |
| epoch 5 | 90.8% | 68.6% |
| epoch 10 | 96.5% | 78.8% |
| epoch 15 | **99.2%** | 83.5% |
| epoch 20 (final) | **89.8%** | **80.4%** |

Observations:
- The native-R0 mask learns the 64-dim representation easily (90.8% by epoch 5; 99.2% at epoch 15) — the input encoding is **not** fundamentally destabilised by packing three adjacency blocks + one-hot into 64 dims.
- The union mask retards learning severely: at epoch 10 it is still below condition A's epoch-3 level, and at the directive's fixed 20-epoch budget it reaches 80.4% (its best being 83.5% at epoch 15). The gap at every checkpoint after epoch 2 is ≥11 pp and grows to ≈16–19 pp mid-training.
- Condition A shows a late wobble (99.2% → 89.8% over epochs 15–20 with train-loss spikes ~0.27–0.35), i.e. near-threshold behaviour at exactly 20 epochs; condition B's curve had not clearly converged either. The fixed 20-epoch budget leaves both curves in a noisy regime.

## Ruling against the directive's decision matrix

| Cell | Requirement | Outcome |
|---|---|---|
| Union interference confirmed | A ≥ 90% **and** B ≤ 70% | **Not met** — A = 89.80% (0.2 pp under), B = 80.40% |
| Mask interference rejected | A ≥ 90% **and** B ≥ 90% | **Not met** |
| Feature representation failure | A < 75% and B < 75% | **Not met** |
| **Result** | | **PARTIAL/AMBIGUOUS** (as pre-specified by the script for between-cell outcomes) |

**Honest interpretation:**

1. Hypothesis 1 (distribution mismatch) is definitively **rejected** (Stage 0: 99.8%/99.8%).
2. Hypothesis 2 (union-mask interference) is **substantially supported but not confirmed under the strict matrix**: the union topology demonstrably retards and degrades single-relation reachability learning on identical inputs (≈19 pp peak-accuracy gap; condition B never reaches condition A's epoch-5 level within the budget), yet at 20 epochs condition B exceeds the 70% "confirmed" ceiling and condition A finishes just 0.2 pp under the 90% floor. The 20-epoch budget places both curves in a noisy near-threshold regime, so the strict matrix cannot be satisfied cleanly in either direction.
3. What the audit **does** establish: the Phase 2c receiver must learn per-relation path computation *through a topology that mixes both relations*; that mixed topology is a major learning-speed and accuracy handicap even when the relation identity is explicitly available in the node features and the target is a single fixed relation. This is consistent with the confirmatory result (disagreement subset at chance for every model including the dual diagnostic): the union-masked receiver could not reliably compute the relation-specific candidates the selector must choose between.

## Strategic recommendation

Per the directive's horizon, the evidence leans toward **not investing in union-mask relational edge-attention engineering** for Phase 2c-as-designed: the audit shows the union topology is a real handicap (though the strict "confirmed" cell was not met at 20 epochs), and the confirmatory study shows the full stack is far from the gates. Both diagnostics point away from "more COMM training on the union topology" and toward either (a) a topology/mask redesign if Phase 2c is continued (e.g., relation-separated attention — a new predeclared experiment), or (b) the **Phase 3 transition (Asymmetric Cooperative Rendezvous)**, which returns to two cores with asymmetric information on a single shared relation and does not require multi-relation filtering. The choice between (a) and (b) is the owner's; the audit supplies the evidence that the current Phase 2c representation/mask combination is the binding constraint, not the data distribution and not the communication channel.

## Artefacts & provenance

- `audit_stage0_transfer.py`, `audit_stage1_ablation.py` (committed); `model/core.py` gained an optional, backward-compatible `attn_mask` override on `LatentReasoner.forward` (directive-authorised; verified exactly equivalent to the internal mask when not overridden; full suite **53 passed** after the change).
- Run directory `runs/20261017T151354Z_phase2c_audit/` with both logs, both summary JSONs (incl. full 20-epoch curves and initial-state checksums), and `environment.freeze.txt`.
- Housekeeping: historical note prepended to `reports/phase2c-run-report.md`; trainer docstring corrected (held-out evaluation is exclusively the manifest-driven evaluator's job); `CHANGELOG.md` updated.

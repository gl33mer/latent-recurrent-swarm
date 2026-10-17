# Phase 2b Message-Supervision Weight Study — Report (18 instrumented runs, validation-only)

**Branch:** `experiment/phase2b-msg-supervision-weight-study` · **Base commit:** `ed46cd4`
(the verified Phase 2b stability state; frozen `LatentCommSwarm` untouched — 284,107
params re-verified) · **Design (predeclared, committed before any run):** `a6eb669` ·
**Runner commits:** `6d12304`, `5a28a2a` (overall-group masking fix caught by tests before
any sweep run) · **Sweep directory:** `runs/20261017T101804Z_msg_weight_stability/`

**Final status: `CANDIDATE RECIPE SCREENED IN` — λ_msg = 2.0 is eligible under the
predeclared rule (6/6 all-gate passes). Confirmatory Phase 2c training was NOT run and
remains blocked pending owner review; the Phase 2c test split was not generated and the
Phase 2c evaluator was not invoked.**

## 1. Environment and execution

Debian 12, Python 3.11.2, torch 2.14.1+cpu (CUDA unavailable), numpy 2.4.6, 4 vCPUs,
4 intra-op threads, ~8 GiB RAM. 18/18 planned runs completed in **44.6 minutes**
(budget 120 min); grid not truncated; no errors; no test data generated (train/val only:
seeds 42/4,000 and 1337/1,000).

## 2. All 18 runs (final-epoch validation; frozen Phase 2b gates)

| λ_msg | Mode | Seed | Conn% | Pass | Failed criteria | Zero% | Probe% | Flip 0/1% | Instab% |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | full | 42 | 54.90 | ✗ | connected, quadrant, flip, probe | 49.7 | 62.2 | 18.18/69.05 | 4.75 |
| 0.5 | full | 43 | 96.30 | ✓ | — | 49.8 | 96.3 | 0.00/0.21 | 2.30 |
| 0.5 | full | 44 | 94.30 | ✗ | probe | 49.2 | 94.3 | 0.00/0.00 | 6.56 |
| 0.5 | final-only | 42 | 94.60 | ✗ | probe | 50.4 | 94.6 | 0.00/0.21 | 6.50 |
| 0.5 | final-only | 43 | 99.50 | ✓ | — | 49.6 | 99.5 | 0.00/0.20 | 4.62 |
| 0.5 | final-only | 44 | 99.40 | ✓ | — | 49.9 | 99.5 | 0.00/0.20 | 3.21 |
| 1.0 | full | 42 | 87.80 | ✗ | quadrant, probe | 49.1 | 88.1 | 0.67/0.23 | 4.83 |
| 1.0 | full | 43 | 96.30 | ✓ | — | 49.5 | 96.5 | 0.00/0.00 | 2.34 |
| 1.0 | full | 44 | 95.80 | ✓ | — | 49.8 | 96.8 | 0.21/1.89 | 4.66 |
| 1.0 | final-only | 42 | 98.20 | ✓ | — | 50.4 | 98.2 | 0.00/0.20 | 6.64 |
| 1.0 | final-only | 43 | 84.90 | ✗ | probe | 49.5 | 85.2 | 0.00/3.76 | 7.09 |
| 1.0 | final-only | 44 | 98.50 | ✓ | — | 47.4 | 98.5 | 0.00/0.00 | 6.40 |
| **2.0** | **full** | **42** | **98.50** | **✓** | — | 49.8 | 98.5 | 0.00/0.00 | 3.04 |
| **2.0** | **full** | **43** | **99.50** | **✓** | — | 50.3 | 99.6 | 0.00/0.00 | 2.13 |
| **2.0** | **full** | **44** | **99.30** | **✓** | — | 49.9 | 99.5 | 0.00/0.00 | 2.54 |
| **2.0** | **final-only** | **42** | **99.90** | **✓** | — | 49.0 | 99.9 | 0.00/0.00 | 2.41 |
| **2.0** | **final-only** | **43** | **99.80** | **✓** | — | 50.1 | 99.8 | 0.00/0.00 | 2.45 |
| **2.0** | **final-only** | **44** | **99.70** | **✓** | — | 50.0 | 99.7 | 0.00/0.00 | 2.54 |

Zero-message accuracy sits at chance (47.4–50.4%) in every run — the bit is always
necessary; flip residuals collapse to ≈0 at λ=2.0.

## 3. Selection rule applied mechanically (as predeclared in the design, §5)

Per-lambda gate-pass counts (full, final-only), non-finite events, mechanism checks,
median instability (corrected definition: per-run mean absolute epoch-to-epoch change of
validation connected accuracy; median across the six runs):

| λ_msg | full | final-only | nonfinite | mechanism | median instab | Eligible? |
|---|---|---|---|---|---|---|
| 0.5 | 1/3 | 2/3 | none | ok | 4.68% | **No** (full < 2/3) |
| 1.0 | 2/3 | 2/3 | none | ok | 5.62% | **Yes** |
| 2.0 | 3/3 | 3/3 | none | ok | 2.47% | **Yes** |

Among eligible: highest total pass count → λ=2.0 (6) vs λ=1.0 (4). **Selected:
λ_msg = 2.0** (also the lower median instability — no tie-break needed). The
`selection.json` in the sweep directory records the machine-computed decision.

**Criterion 4 (no pathology) evidence for λ=2.0:** zero >15-point validation drops in all
six runs; zero non-finite loss/gradient batches; parameters finite throughout; median
instability 2.47% (lowest of the grid); msg_encoder gradient norms are higher (maxima
16.7–37.9, expected with a 4× message weight) but produce no collapses — in contrast to
the λ=0.5/1.0 cells, where probe failures co-occurred with message-BCE spikes and
validation collapses (see §5).

## 4. Pairing verification

For each model seed, all six cells (3 λ × 2 modes) started from **one identical initial
parameter checksum** (SHA-256 over the serialised initial state dict, recorded in each
`run_meta.json`): seed 42 → 1 distinct checksum/6 cells; same for 43 and 44. Paired
full/final-only runs shared batch order via `torch.Generator().manual_seed(9000+seed)`.
No optimizer state was shared across runs.

## 5. Diagnosis: why λ matters (consistent with the Stage C audit)

Stage C3 established from the first sweep's raw logs that sender-probe failures are
late-training collapses — probe accuracy often starts at 94–99% and degrades in lockstep
with parity accuracy while the message BCE spikes (e.g. 0.058 → 1.916 with probe
97.1% → 10.0%), accompanied by msg_encoder/core_A gradient spikes; it is not a
sign/metric bug (probe is high early; the metric is trivially correct when training is
stable).

This study's dose-response supports the interference hypothesis: raising the weight on
the *existing* BCE sender supervision from 0.5 → 2.0 monotonically (i) eliminates probe
failures (failing probe checks per lambda: 4/6 → 2/6 → 0/6), (ii) reduces validation
instability (median 4.68% → 5.62% → 2.47%), and (iii) turns 2/6 → 4/6 → 6/6 all-gate
passes at lr 2e-4. The message channel is anchored against the joint-objective
interference rather than being destabilised by the stronger weight. Residual
uncertainties: 3 seeds per cell, one LR, observational collapse evidence from per-epoch
means (not per-batch); λ=1.0's two probe failures show the transition is not fully
smooth. This is a recipe-screening finding, not a scientific claim about Phase 2c.

## 6. Checkpoints and logs (all under ignored `runs/`; never committed)

`last.pt` was saved for **every** run including gate failures (mandatory policy,
correcting the earlier Phase 2b post-gate-only deviation), plus `epoch05/10/15/20.pt`.
Per-cell directories `runs/20261017T101804Z_msg_weight_stability/lm<λ>_<mode>_seed<seed>/`
contain `last.pt` (+ SHA-256 in `run_summary.json`), metrics.jsonl, run_meta.json,
run_summary.json. Sweep-level: `aggregate.json` (SHA-256 `d40ca0341f86b651…`),
`summary.csv`, `selection.json`, `sweep.log`, `git-commit.txt`, `environment.freeze.txt`,
`phase1.sha256`. `last.pt` SHA-256s (first 16 hex):

| λ | mode | seed | last.pt SHA-256 |
|---|---|---|---|
| 0.5 | full | 42/43/44 | `ea54cc3ffc3e80bf…` / `f60ccf4f468ef4c2…` / `edd40247a55bf2a8…` |
| 0.5 | final-only | 42/43/44 | `59f564418d346947…` / `0d856f207f8b8658…` / `2a10cee8c92e04ab…` |
| 1.0 | full | 42/43/44 | `962653818a3f5f8d…` / `64e92438cdd4a0ee…` / `20806574e2035c46…` |
| 1.0 | final-only | 42/43/44 | `f26c8075730cba0f…` / `3ae19d00bc12673f…` / `1879ea8220d64a0f…` |
| 2.0 | full | 42/43/44 | `488e452742a16e13…` / `5355bcc03e641725…` / `e87d92a8ff920609…` |
| 2.0 | final-only | 42/43/44 | `b8a81b62632bd5c4…` / `8645a3e71afbe21c…` / `06423b0bb8c4840f…` |

## 7. Corrected instability summaries (also fixing the earlier misreport)

Corrected medians of per-run mean |Δval| across all six runs per LR (first sweep):
**12.61% (8e-4), 11.99% (4e-4), 4.68% (2e-4)** — the previously reported "medians"
(9.5/7.9/3.2) were single per-run values; see the dated addendum to
`reports/phase2b-stability-report.md`. In this study the same corrected definition gives
medians 4.68% (λ=0.5), 5.62% (λ=1.0), 2.47% (λ=2.0).

## 8. Limitations and scope

- Recipe screening only: 3 seeds/cell, one LR (2e-4), one architectural family, XOR task;
  no claim of seed-robustness beyond this grid, no Phase 2c relevance claim tested.
- λ=1.0 also satisfies the ≥2/3 rule; λ=2.0 dominates it on every predeclared criterion.
- The Phase 2b stop-gate is hereby **superseded only for the specific recipe
  (lr 2e-4, λ_msg 2.0)** pending owner review; no Phase 2c file, datum, or evaluation was
  touched: the seed-99991 split was never generated, `evaluate_phase2c_test.py` was never
  invoked, and no Phase 2c training occurred.

---

**Dated erratum (2026-10-17):** two statements in §5 were corrected by
`reports/phase2b-msg-weight-study-erratum-2026-10-17.md` after independent recomputation
from the raw run data: (E1) probe-gate failures at λ=0.5 are **3/6**, not 4/6; (E2) the
instability medians (4.68% → 5.62% → 2.47%) are **non-monotonic** in λ_msg — only the
total all-gate pass count (3/6 → 4/6 → 6/6) and the probe pass rate rise monotonically.
The §2 run table, §3 selection arithmetic, and the selected candidate (λ_msg = 2.0) are
unchanged.

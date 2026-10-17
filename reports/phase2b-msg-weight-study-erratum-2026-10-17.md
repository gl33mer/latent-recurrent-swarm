# Erratum — Phase 2b Message-Supervision Weight Study Report (2026-10-17)

**Applies to:** `reports/phase2b-msg-weight-study-report.md` (branch
`experiment/phase2b-msg-supervision-weight-study`). The original report, its tables and the
run artifacts are preserved unchanged; this dated erratum corrects two statements in its
§5 narrative. All corrections were **independently recomputed from the raw run data**
(`runs/20261017T101804Z_msg_weight_stability/`: `aggregate.json`, `selection.json`,
`summary.csv`, per-cell `run_summary.json` and `metrics.jsonl`), not transcribed.

## E1. Sender-probe gate-failure counts (§5, item (i))

Original §5 stated "failing probe checks per lambda: 4/6 → 2/6 → 0/6".
**Corrected: 3/6 → 2/6 → 0/6.** At λ_msg=0.5 exactly three runs failed the probe gate
(full/seed-42 at 62.2%, full/seed-44 at 94.3%, final-only/seed-42 at 94.6%); full/seed-43
(96.3%) passed. The qualitative conclusion is unchanged — probe failures shrink
monotonically as λ_msg rises and vanish at λ_msg=2.0 — but the λ=0.5 count was overstated
by one.

## E2. Instability trend is non-monotonic (§5, item (ii))

Original §5 described "reduces validation instability (median 4.68% → 5.62% → 2.47%)"
under the heading "monotonically". **Correction: the median instability is
non-monotonic** — it *increases* from λ=0.5 (4.68%) to λ=1.0 (5.62%), then decreases at
λ=2.0 (2.47%). What does increase monotonically is the total all-gate pass count
(3/6 → 4/6 → 6/6) and, per E1, the probe-gate pass rate. The λ=2.0 cell is simultaneously
the best on all three summary measures, but the path between λ=0.5 and λ=2.0 is not a
simple monotone improvement, and the λ=1.0 cell's two probe failures show the transition
is not smooth.

## E3. Numbers verified against raw data (no discrepancies found)

| λ_msg | all-gate full | all-gate final-only | total | probe-gate failures | median instab |
|---|---|---|---|---|---|
| 0.5 | 1/3 | 2/3 | 3/6 | 3/6 | 4.68% |
| 1.0 | 2/3 | 2/3 | 4/6 | 2/6 | 5.62% |
| 2.0 | 3/3 | 3/3 | 6/6 | 0/6 | 2.47% |

These match the report's §2 run table and §3 selection table (which were correct); only
the §5 narrative items E1/E2 were wrong. The predeclared selection rule is unaffected:
λ_msg=2.0 remains the selected candidate (highest total all-gate pass count, lowest
median instability among eligible lambdas; `selection.json` records the machine-computed
decision).

## E4. Limitations (unchanged)

Three seeds per cell, one learning rate (2e-4), one task/model family (Phase 2b XOR,
`LatentCommSwarm`); a screened candidate recipe, not an established optimum and not
evidence about Phase 2c.

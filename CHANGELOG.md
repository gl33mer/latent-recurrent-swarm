# Changelog

## 2026-10-16 — Phase 2b stage B: baseline housekeeping & artifact provenance

Branch `chore/preserve-v7.1-artifacts`; baseline source frozen at tag `v7.1-baseline`
(commit `9f94151`); immutable tag `v7.1-pre-run` untouched.

1. **B1 — report metadata fix:** `reports/first-run-report.md` said the repository was
   *private*; it has been public since the owner-requested visibility change recorded in
   the 2026-10-16 housekeeping entry above. Corrected the visibility statement and added
   the provenance note. No historical metric was altered.
2. **B2 — Phase 1 checkpoint verified:** root `checkpoint_phase1.pt` and
   `runs/20261016T211522Z/checkpoint_phase1.pt` both hash to
   `ef4514342b65e16f901774134d03a0c4db8f69fd38dccd56eda1dcf29e50049e`, matching the
   recorded first-run checksum exactly. No reproduction needed for Phase 1.
3. **B3 — Phase 2a artifact capture:** the original v7.1 `train_swarm.py` never saved a
   trained swarm checkpoint, so none exists. Added an optional `--artifact-dir` argument:
   when provided, it saves `checkpoint_phase2a.pt` plus a provenance/metrics JSON **only
   after all validation gates pass** and the test evaluation completes. The change adds no
   model computation, no RNG consumption, and no gate change (verified: `compileall` clean,
   `test_system.py` 7/7 pass, behaviour identical without the flag).
4. A clearly-labelled **Phase 2a reproducibility rerun** (frozen source at
   `v7.1-baseline` + the artifact-capture option, same seeds 42/1337/9999, seed 42) was
   executed under a new run ID; outcome recorded in `WORKLOG.md` and the run directory.
   It is a reproduction artifact, not the original trained state.

## 2026-10-16 — v7.1 baseline bootstrap (test-only/dependency hardening)

1. **Dependency addition (no architecture change):** the canonical code calls
   `Tensor.numpy()` (`data/graph_task.py::_hash_graph`, `test_system.py::test_conditional_parity_balancing`),
   but the official CPU-only PyTorch wheel does not bundle NumPy. Added `numpy>=1.26`
   to `requirements-cpu.txt` and installed it in `.venv`. No model, loss, threshold,
   or training change.
2. **Scope clarification (per v7.1 spec):** Phase 2a here is *decoupled* recurrent
   reasoning with receiver-side one-bit decision fusion (Core B never ingests Core A's
   message into its recurrence; the message enters only the final parity readout).
   This is **not** Phase 2b: no COMM token, no global workspace, no recurrent message
   integration is implemented or claimed.
3. **Housekeeping:** added `*.swp` to `.gitignore` for a pre-existing editor swap file
   in the working directory. No project code affected.
4. **Repository visibility (2026-10-16, post-run, owner request):** repository switched
   from private to **public** for external review at the owner's explicit instruction,
   after the completed baseline run and a secret scan of all tracked content. The v7.1
   spec's private-by-default rule governed the initial setup only; provenance of this
   change is recorded here and in `WORKLOG.md`.
4. **Test hardening already present in the canonical suite** (recorded, not modified):
   `test_binary_channel_contract` requires the seeded batch to exercise both message
   polarities plus exact flip inversion; `test_parity_head_generalization_on_noisy_inputs`
   runs five fixed seeds with a separate noisy test set at a 99% threshold.

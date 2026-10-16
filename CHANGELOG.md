# Changelog

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

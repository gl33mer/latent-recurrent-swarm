# Latent Recurrent Swarm — v7.1 Baseline

A small CPU-only research experiment on graph-masked latent recurrence and one-bit distributed parity.

## Scope

- **Phase 1:** one parameter-tied latent reasoner predicts directed reachability on 16-node DAGs.
- **Phase 2a:** two independently recurrent reasoners solve `Y = b_A XOR b_B` via a single binary message from Core A and a nonlinear receiver-side fusion head.
- **Phase 2b is not implemented here:** recurrent integration of a received message through a COMM token / global workspace is a separate future experiment.

## Runtime

Target: CPU-only VM, 4 vCPUs, about 8 GiB RAM, PyTorch float32. Use `.venv`; do not install project packages into system Python.

Example setup on Linux (install a CPU-only PyTorch wheel first):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
```

## Run

```bash
python -m pytest -q test_system.py
python -u train_unit.py
python -u train_swarm.py
```

Run Phase 2 only after Phase 1 passes and creates `checkpoint_phase1.pt`.

## Reproducibility and outputs

Training scripts seed Python and PyTorch and limit PyTorch intra-op threading to four threads. The data generator uses independent explicit seeds for train, validation and held-out test sets and enforces graph-hash separation. Run logs, checkpoints, environment snapshots and checksums should be stored under `runs/`, which is ignored by Git. Commit source and a concise, sanitised report; do not commit the virtual environment or checkpoints.

## Acceptance criteria

See `PROJECT_SPEC.md`. A system-test pass does not imply a training gate has passed. Do not tune on the held-out test set.

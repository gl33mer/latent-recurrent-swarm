# Latent Swarm Project: Phase 1 & 2 Specification (v7.1)

## Research scope

Phase 1 tests graph-masked latent recurrence for directed graph reachability. Phase 2a tests two decoupled recurrent reasoners solving distributed parity through a strict 1-bit channel and receiver-side decision fusion. Phase 2a is a baseline; recurrent integration of the incoming message via a COMM token is reserved for Phase 2b.

## Target hardware and runtime
- Platform: CPU-only Smol Machine VM, nominally 4 vCPUs and 8 GiB RAM.
- Precision: `torch.float32`.
- Threads: `torch.set_num_threads(4)` at script entry.
- Python and PyTorch versions must be recorded per run.
- Seeds: Python `random.seed(42)` and `torch.manual_seed(42)` for training scripts; dataset generation uses explicit seeds.

## Architecture
- Number of node tokens: 16.
- Node input dimension: 32 (16 adjacency-row features + 16 one-hot node-ID features).
- Latent dimension: 128.
- Attention heads: 4 (head dimension 32).
- Feed-forward dimension: 256.
- Single `LatentReasoner` parameter count: 137,476.
- `LatentSwarm` parameter count: 292,427.
- Parity head: `LayerNorm(256) -> Linear(256, 64) -> GELU -> Linear(64, 2)`.
- Recurrent depth: 8 steps in Phase 1 training and both Phase 2 cores.
- Communication: exactly one forward-pass binary scalar `m ∈ {-1.0, +1.0}` through a straight-through estimator.

## Graph task
- Directed acyclic graphs have 16 nodes and only forward edges `i -> j` where `j - i ∈ {1, 2, 3}`.
- `b_A = Reach(Graph A, 0 -> 15)`.
- `b_B = Reach(Graph B, 0 -> 15)`.
- `Y = b_A XOR b_B`.
- All graphs have at least one outgoing edge from Node 0 and at least one incoming edge to Node 15.
- Reachable graphs are accepted only when shortest path length `d` satisfies `5 <= d <= 8`.
- Unreachable graphs have shortest path length `-1`.
- Samples are exactly balanced across the four `(b_A, b_B)` quadrants.
- SHA-256 graph hashes prevent any graph from being reused across generated splits.

## Phase 1 acceptance gates (`train_unit.py`)
- `pytest -q test_system.py` passes before training.
- T=1 validation accuracy <= 56.5% (fixed empirical local-baseline ceiling).
- T=8 validation accuracy >= 90.0%.
- Report the untouched test-set T=8 result separately.
- Save `checkpoint_phase1.pt` only after both validation gates pass.

## Phase 2a acceptance gates (`train_swarm.py`)
- Overall connected validation parity accuracy >= 82.0%.
- Each of the four `(b_A, b_B)` quadrants has accuracy >= 75.0%.
- Message-flip residual among previously correct samples <= 15.0% separately for `b_B=0` and `b_B=1`.
- Zero-vector message ablation validation accuracy <= 53.0%.
- Message polarity probe accuracy >= 95.0% overall and >= 90.0% separately for `b_A=0` and `b_A=1`.
- After validation gates are assessed, report the same stratified metrics and interventions on an untouched held-out test split. Do not use test metrics to tune the run.

## Interpretation limits
- A passing system test is not proof that training gates pass.
- An empirical local signature lookup is an approximate baseline, not a proof of the exact information available to the T=1 reasoner. It uses the induced subgraph on nodes 12–15 and should be described precisely.
- Phase 2a is decoupled recurrent reasoning with 1-bit late decision fusion. It does not prove recurrent message integration or a fully collaborative latent swarm.
- Phase 2b may separately test a global workspace / COMM token while preserving graph-local attention among graph nodes.

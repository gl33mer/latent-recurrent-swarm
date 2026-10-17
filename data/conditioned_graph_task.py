"""
Phase 2c: message-conditioned two-relation reachability benchmark.

Task: Core A reasons over its single-relation Graph A and sends one binary selector
m = +1 iff b_A = Reach_A(0 -> 15), else -1. Core B receives Graph B containing two
independent directed edge relations R0 and R1 over the same 16 nodes and must report

    Y = r0 if b_A == 0 else r1,

where r0 = Reach_R0(0 -> 15) and r1 = Reach_R1(0 -> 15). The bit selects which
relation's reachability predicate is relevant; it is not an XOR combination.

Generation contract (predeclared in reports/phase2c-design.md):
- exactly balanced across all eight (b_A, r0, r1) combinations; num_samples % 8 == 0;
- every relation matrix is sampled by the v7.1 matched-DAG sampler: endpoint degree
  constraints (Node 0 out-degree >= 1, Node 15 in-degree >= 1), reachable graphs have
  shortest path length in [5, 8], unreachable graphs have no 0 -> 15 path;
- Graph A uses the same sampler with label b_A;
- **raw structural graph uniqueness** (amended by the 2026-10-17 fix-only pass, see
  reports/phase2c-fix-only-report.md): every individual graph (A, R0, R1) is hashed by
  its canonical raw SHA-256 with NO role prefix; a raw hash is rejected if it has already
  appeared in the current dataset under any role or in any role of a supplied
  `existing_hashes` set from a prior split. The returned set contains exactly
  `3 * num_samples` raw hashes for a valid split. Role lineage (A/R0/R1) is tracking
  metadata only and is never part of the uniqueness key;
- x_B layout (64 features per node): [adj_union(16), adj_R0(16), adj_R1(16),
  one_hot_node_id(16)], where adj_union is the logical OR of the two relation matrices;
  x_A remains the v7.1 layout [adj_A(16), one_hot(16)].
"""
import hashlib
import random
from typing import Optional, Set, Tuple

import torch
from torch.utils.data import TensorDataset

from data.graph_task import _sample_matched_dag

SPLIT_SEEDS = {"train": 4201, "validation": 13371, "test": 99991}

MAX_ATTEMPTS_PER_GRAPH = 200_000  # bounded failure if the sampler cannot satisfy uniqueness


def canonical_raw_graph_hash(adj: torch.Tensor) -> str:
    """Canonical raw structural SHA-256 of an adjacency matrix (no role prefix).

    Canonicalisation: values are cast to float32, moved to CPU and made contiguous before
    hashing, so equivalent binary adjacency matrices cannot produce different hashes
    because of dtype or storage layout. Entries are 0.0/1.0 as produced by the sampler."""
    canon = adj.detach().to(dtype=torch.float32).contiguous().cpu()
    return hashlib.sha256(canon.numpy().tobytes()).hexdigest()


def generate_conditioned_benchmark(
    num_samples: int = 4000,
    num_nodes: int = 16,
    seed: int = SPLIT_SEEDS["train"],
    existing_hashes: Optional[Set[str]] = None,
) -> Tuple[TensorDataset, Set[str]]:
    assert num_nodes == 16, "Current benchmark supports exactly 16 nodes."
    assert num_samples % 8 == 0, "num_samples must be divisible by 8 for exact (b_A, r0, r1) balancing."
    rng = random.Random(seed)
    seen = set() if existing_hashes is None else set(existing_hashes)
    generated: Set[str] = set()
    eye = torch.eye(num_nodes)

    combos = [(bA, r0, r1)
              for bA in (0, 1) for r0 in (0, 1) for r1 in (0, 1)] * (num_samples // 8)

    samples_A, samples_B, bits = [], [], []
    for b_A, r0, r1 in combos:
        graphs = {}
        for slot, reach in (("A", b_A), ("R0", r0), ("R1", r1)):
            # Raw structural uniqueness: the role is NOT part of the key. A graph that
            # already appeared (under any role, in this split or a prior one) is rejected.
            for _attempt in range(MAX_ATTEMPTS_PER_GRAPH):
                adj, _ = _sample_matched_dag(num_nodes, bool(reach), rng)
                raw_h = canonical_raw_graph_hash(adj)
                if raw_h not in seen and raw_h not in generated:
                    generated.add(raw_h)
                    graphs[slot] = adj
                    break
            else:
                raise RuntimeError(
                    f"Could not sample a structurally unique graph for role {slot} after "
                    f"{MAX_ATTEMPTS_PER_GRAPH} attempts; the requested task is too large "
                    f"for the 16-node forward-DAG sampler.")

        adj_A, adj_R0, adj_R1 = graphs["A"], graphs["R0"], graphs["R1"]
        adj_union = ((adj_R0 > 0.5) | (adj_R1 > 0.5)).float()
        feat_A = torch.cat([adj_A, eye], dim=-1)                                  # [16, 32]
        feat_B = torch.cat([adj_union, adj_R0, adj_R1, eye], dim=-1)              # [16, 64]

        samples_A.append(feat_A)
        samples_B.append(feat_B)
        bits.append((b_A, r0, r1, r0 if b_A == 0 else r1))

    b_A_t = torch.tensor([b[0] for b in bits], dtype=torch.long)
    r0_t = torch.tensor([b[1] for b in bits], dtype=torch.long)
    r1_t = torch.tensor([b[2] for b in bits], dtype=torch.long)
    Y_t = torch.tensor([b[3] for b in bits], dtype=torch.long)

    dataset = TensorDataset(torch.stack(samples_A), torch.stack(samples_B),
                            b_A_t, r0_t, r1_t, Y_t)
    return dataset, generated


def benchmark_statistics(dataset: TensorDataset) -> dict:
    """Data-only baselines and balance checks (no model involved)."""
    _, _, b_A, r0, r1, Y = dataset.tensors
    b_A, r0, r1, Y = b_A.numpy(), r0.numpy(), r1.numpy(), Y.numpy()
    disagree = r0 != r1
    stats = {
        "n": int(len(Y)),
        "combo_counts": {f"{a}{i}{j}": int(((b_A == a) & (r0 == i) & (r1 == j)).sum())
                         for a in (0, 1) for i in (0, 1) for j in (0, 1)},
        "disagreement_fraction": float(disagree.mean()),
        "P(Y=1)": float(Y.mean()),
        "P(Y=1|b_A=0)": float(Y[b_A == 0].mean()),
        "P(Y=1|b_A=1)": float(Y[b_A == 1].mean()),
        # No-selector optimum: answer r0 (=r1) on agreement examples, guess on disagreement.
        "no_selector_optimum": float(1.0 * (1 - disagree.mean()) + 0.5 * disagree.mean()),
        "no_selector_disagreement": 0.5,
    }
    return stats

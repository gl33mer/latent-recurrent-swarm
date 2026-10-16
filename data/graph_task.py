"""
Balanced 16-Node Parity Reachability Task.
Enforces endpoint degree constraints, bounded reachable path lengths, and hash separation.
"""
import torch
from torch.utils.data import TensorDataset
import random
import hashlib
from typing import Optional, Set, Tuple

def _hash_graph(adj: torch.Tensor) -> str:
    return hashlib.sha256(adj.numpy().tobytes()).hexdigest()

def _shortest_path_length(adj: torch.Tensor, start: int, target: int) -> int:
    visited = {start: 0}
    queue = [start]
    while queue:
        curr = queue.pop(0)
        dist = visited[curr]
        if curr == target:
            return dist
        neighbors = torch.where(adj[curr] == 1.0)[0].tolist()
        for n in neighbors:
            if n not in visited:
                visited[n] = dist + 1
                queue.append(n)
    return -1

def _sample_matched_dag(num_nodes: int, force_reachable: bool, rng: random.Random) -> Tuple[torch.Tensor, str]:
    assert num_nodes == 16, "Current benchmark implementation strictly parameterized for 16 nodes."
    while True:
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

        out_0 = rng.choice([1, 2, 3])
        adj[0, out_0] = 1.0

        in_15 = rng.choice([12, 13, 14])
        adj[in_15, 15] = 1.0

        for i in range(num_nodes):
            for j in range(i + 1, min(i + 4, num_nodes)):
                if (i == 0 and j == out_0) or (i == in_15 and j == 15):
                    continue
                if rng.random() < 0.38:
                    adj[i, j] = 1.0

        d = _shortest_path_length(adj, 0, num_nodes - 1)
        reachable = (d != -1)

        if force_reachable:
            if reachable and (5 <= d <= 8):
                return adj, _hash_graph(adj)
        else:
            if not reachable:
                return adj, _hash_graph(adj)

def generate_parity_benchmark(
    num_samples: int = 6000,
    num_nodes: int = 16,
    seed: int = 42,
    existing_hashes: Optional[Set[str]] = None
) -> Tuple[TensorDataset, Set[str]]:
    assert num_nodes == 16, "Current benchmark supports exactly 16 nodes."
    assert num_samples % 4 == 0, "num_samples must be divisible by 4 for exact quadrant balancing."
    rng = random.Random(seed)
    seen_hashes = set() if existing_hashes is None else set(existing_hashes)
    generated_hashes = set()
    eye = torch.eye(num_nodes)

    samples_A = []
    samples_B = []
    bits_A = []
    bits_B = []
    targets = []

    combos = [(False, False), (False, True), (True, False), (True, True)] * (num_samples // 4)

    for reach_A, reach_B in combos:
        while True:
            adj_A, h_A = _sample_matched_dag(num_nodes, reach_A, rng)
            if h_A not in seen_hashes and h_A not in generated_hashes:
                generated_hashes.add(h_A)
                break

        while True:
            adj_B, h_B = _sample_matched_dag(num_nodes, reach_B, rng)
            if h_B not in seen_hashes and h_B not in generated_hashes:
                generated_hashes.add(h_B)
                break

        feat_A = torch.cat([adj_A, eye], dim=-1)
        feat_B = torch.cat([adj_B, eye], dim=-1)
        bit_A_val = 1 if reach_A else 0
        bit_B_val = 1 if reach_B else 0

        samples_A.append(feat_A)
        samples_B.append(feat_B)
        bits_A.append(bit_A_val)
        bits_B.append(bit_B_val)
        targets.append(bit_A_val ^ bit_B_val)

    dataset = TensorDataset(
        torch.stack(samples_A),
        torch.stack(samples_B),
        torch.tensor(bits_A, dtype=torch.long),
        torch.tensor(bits_B, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long)
    )
    return dataset, generated_hashes

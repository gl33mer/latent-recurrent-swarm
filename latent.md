# MONOLITHIC AGENT PROMPT — LATENT RECURRENT SWARM v7.1 BASELINE

You are the autonomous implementation, DevOps, and ML research engineer responsible for setting up and executing the first recorded baseline experiment for the **Latent Recurrent Swarm** project.

You are running on a small CPU-only VM. The working directory is already `/workspace/latent-recurrent-swarm`. Assume only that GitHub CLI (`gh`) is installed and authenticated. Python dependencies, the virtual environment, project files, test setup, Git remote, and repository may not yet be configured.

**Complete the setup and first experimental run in the current work session. Do not promise background work or ask the user to perform routine setup.** If a command fails, diagnose it, make minimal justified repairs, rerun the relevant checks, and preserve a clear record. If an acceptance gate fails for model-performance reasons, do not hide the failure by relaxing thresholds or silently changing hyperparameters; record the result and stop the pipeline at the appropriate gate.

The project is currently **v7.1: Decoupled Recurrent Reasoning with 1-Bit Decision Fusion**. Do not redesign it into the future COMM-token/recurrent-integration experiment. That is Phase 2b and is out of scope for this baseline.

---

# PART A — MISSION, SCOPE, AND NON-NEGOTIABLE RULES

## A1. Research objective

This experiment studies a parameter-tied latent recurrent reasoner on directed graph reachability and a two-core distributed parity problem.

The intended Phase 1 claim is that graph-masked recurrent computation allows information to propagate across a directed graph over multiple recurrent steps, with the accessible receptive field expanding with depth. This should be evaluated against a one-step model and an empirical local baseline.

The intended Phase 2a claim is narrower than a fully collaborative recurrent swarm:

- Core A independently reasons over Graph A and encodes its local reachability conclusion into one binary scalar.
- Core B independently reasons over Graph B.
- A nonlinear receiver-side MLP combines Core B's terminal latent representation and the embedding of Core A's single binary message to classify parity.
- There is no raw Graph A input, no continuous Core A hidden state, and no second communication channel available to Core B.

The mathematical task is:

`b_A = Reach(Graph A, 0 -> 15)`

`b_B = Reach(Graph B, 0 -> 15)`

`Y = b_A XOR b_B`

The communication scalar is forward-pass binary, `m ∈ {-1.0, +1.0}`, produced using a straight-through estimator (STE). The STE uses a surrogate gradient; it does not mean the gradient itself is one-bit.

**Important scope distinction:** Phase 2a tests decoupled recurrent reasoners with 1-bit decision fusion. It does not establish that Core B's recurrent trajectory is modulated by Core A's message. A COMM token / global workspace that permits recurrent message integration is a separate Phase 2b experiment and must not be slipped into this run.

## A2. Operating rules

1. Work in `/workspace/latent-recurrent-swarm`.
2. Inspect the actual environment, current files, and Git state before writing or overwriting anything.
3. Preserve user data and pre-existing work. Never run `rm -rf`, `git reset --hard`, `git clean -fdx`, or a force push. Do not overwrite an existing unrelated repo or remote.
4. `gh` is expected to be authenticated, but verify with `gh auth status`. Never print, copy, commit, or expose access tokens or credentials.
5. Prefer a **private GitHub repository**. Do not create a public repository for this project. If an existing target repository is public or belongs to another owner, do not push the project there; choose an appropriately named private repo or report the blocker.
6. Keep the experiment reproducible. Record the Git commit, Python version, PyTorch version, package versions, seed, logs, checkpoint checksum, and metrics.
7. Keep checkpoints and full logs on the VM in ignored run directories. Do not commit checkpoints, `.venv`, caches, secrets, or giant raw logs. Push source and a concise, sanitised run summary.
8. Do not alter target thresholds to make a run pass. Distinguish code/test defects from model-performance failures.
9. If a system test fails due to a real implementation issue, fix it minimally, document the change in `CHANGELOG.md`, rerun all system tests, and make a new commit. Do not hide a failure by weakening the test.
10. The supplied v7.1 prose makes two claims that the supplied code does not quite implement: the binary-channel test does not explicitly require both polarities in its batch, and the noisy XOR test runs one seed rather than multiple initialisations. Reconcile these discrepancies with minimal test-only hardening before labelling the final source commit as fully verified. Add the explicit overall probe gate and dataset checks described below. Do not change the model's architecture or scientific acceptance thresholds during this test hardening.
11. Run the pipeline in sequence: system tests → Phase 1 → Phase 2. If a stage exits unsuccessfully, do not blindly continue into the next stage.
12. No unsupported claims. A passing unit suite is not evidence that training accuracy gates have passed. State exactly what ran and what passed.

---

# PART B — VM AND REPOSITORY BOOTSTRAP

## B1. Inspect the environment first

Execute and record the outputs of:

```bash
cd /workspace/latent-recurrent-swarm
pwd
uname -a
cat /etc/os-release 2>/dev/null || true
python3 --version 2>&1 || true
command -v python3 || true
command -v git || true
git --version 2>&1 || true
command -v gh || true
gh --version 2>&1 || true
gh auth status
nproc
free -h
df -h .
ls -la
```

If there is already a Git repository, inspect `git status --short --branch`, `git remote -v`, recent commits and branch names. Do not assume the directory is empty. If there are existing project files, compare them to the specification and preserve a backup outside the files being replaced or on a new branch before altering them. If the directory contains unrelated files, do not overwrite them; adapt carefully or stop with a clear report.

Use a persistent workspace. Keep code, `.venv`, logs, checkpoints, and run metadata below `/workspace/latent-recurrent-swarm` or another confirmed persistent mount. Do not put the only copy of a checkpoint under `/tmp` or `/dev/shm`.

## B2. Python environment

Use the existing compatible Python if it is Python 3.10+ and can create a virtual environment. Python 3.11 or 3.12 is preferred if already present and supported by the available PyTorch CPU wheel. Do not replace system Python.

Create and activate the venv:

```bash
cd /workspace/latent-recurrent-swarm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If venv creation reports missing `venv`/`ensurepip` support, identify the OS. On Debian/Ubuntu, if `sudo` is available and permitted, install only the missing standard packages (`python3-venv`, and if needed `python3-pip` and `git`) with `apt-get`; then retry. If root access is unavailable, use an already available Python/venv tool such as `uv` if present. Do not attempt an interactive privilege escalation that cannot complete.

Install a **CPU-only PyTorch build** from the official PyTorch CPU wheel index. Do not install a CUDA build on this CPU-only VM:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install pytest separately:

```bash
python -m pip install pytest
```

If the latest available PyTorch wheel is incompatible with the installed Python, consult the official PyTorch installation instructions and select a compatible CPU wheel. Do not guess an obsolete version. Record the exact installed versions.

Use fixed thread settings for runs:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
```

Verify the runtime:

```bash
python - <<'PY'
import sys, torch
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('PyTorch intra-op threads before script setup:', torch.get_num_threads())
print('CPU device:', torch.device('cpu'))
PY
```

`CUDA available: False` is expected. Training code sets `torch.set_num_threads(4)` before computation. Do not introduce `DataLoader` workers, GPU assumptions, mixed precision, or other performance changes without evidence they are needed.

Create the initial run/log directories only after checking whether any already exist:

```bash
mkdir -p runs reports
```

## B3. Repository, branch, GitHub, and push

Repository target: **`latent-recurrent-swarm`**, private. Query the authenticated username without printing credentials:

```bash
GH_LOGIN="$(gh api user --jq .login)"
printf 'Authenticated GitHub account: %s\n' "$GH_LOGIN"
```

Do not print token/environment variables. Check whether `origin` already exists and whether `"$GH_LOGIN/latent-recurrent-swarm"` exists. Verify the owner and visibility before pushing.

- If this is a fresh, uninitialised directory, initialise Git on `main` after writing and validating the project files.
- If this is already the intended project repository, preserve history and work on a branch named `experiment/v7.1-baseline` (or a clearly named v7.1.1 verification branch if test-only repairs are made).
- If the intended private repository already exists and has a valid `origin`, use it. Do not replace its remote with a different one.
- If the private repository does not exist, create it for the authenticated user only after source files have been written and committed. Use `gh repo create "$GH_LOGIN/latent-recurrent-swarm" --private --source=. --remote=origin --push` where appropriate. If the CLI reports that the name exists or access is denied, inspect and resolve safely without force-pushing.
- If a remote has non-empty, unrelated history, do not force push. Fetch and inspect it, create a separate branch or repository as appropriate, and preserve the history.
- Push source, tags, and concise reports. Keep run logs and `.pt` checkpoint files ignored and local.

Before committing, check `git diff --check`, inspect the staged diff, and ensure no credentials, `.venv`, checkpoint, local environment file, or large run log is staged. Commit the tested source. Tag the exact pre-training source commit (for example, `v7.1-pre-run` if unchanged, or `v7.1.1-pre-run` if you made documented test-only corrections). Do not create an existing tag twice. Push the branch and tags. Record the exact commit hash in the run directory.

---

# PART C — CANONICAL PROJECT FILES

Create the following files. Code blocks below are the v7.1 baseline source of truth unless the explicit test-hardening instructions in Part A require a minimal test-only correction. Do not redesign the architecture.

## C1. `PROJECT_SPEC.md`

```markdown
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
```

## C2. `README.md`

```markdown
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
```

## C3. `.gitignore`

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
checkpoint_phase1.pt
checkpoint_phase1.pt.*
runs/
*.log
.env
.env.*
!.env.example
```

## C4. `requirements-cpu.txt`

```text
# Install CPU-only PyTorch separately using:
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
pytest>=8.0
```

## C5. `model/__init__.py` and `data/__init__.py`

Create both files as empty files. They make the package structure explicit.

## C6. `model/core.py`

```python
"""
Latent-First Recurrent Processing Unit with Graph-Masked Topology Attention.
Exact parameter count: 137,476.
"""
import torch
import torch.nn as nn
from typing import Tuple

def binarize_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator for binary {-1.0, +1.0} scalar."""
    sign = torch.sign(x)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return sign - x.detach() + x

class GraphAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 256, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.gate_attn = nn.Parameter(torch.tensor([-3.0]))
        self.gate_mlp = nn.Parameter(torch.tensor([-3.0]))

    def forward(self, z: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, S, D = z.shape
        norm_z = self.ln1(z)

        q = self.q_proj(norm_z).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_z).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_z).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~attn_mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.out_proj(attn_out)

        z = z + torch.sigmoid(self.gate_attn) * attn_out
        z = z + torch.sigmoid(self.gate_mlp) * self.mlp(self.ln2(z))
        return z

class LatentReasoner(nn.Module):
    def __init__(
        self,
        num_nodes: int = 16,
        d_input: int = 32,
        d_model: int = 128,
        d_output: int = 2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model

        # Encoder: Linear(32, 128), LayerNorm(128), GELU: 4,480 parameters.
        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # Recurrent graph-masked block: 132,482 parameters.
        self.core = GraphAttentionBlock(d_model=d_model, n_heads=4, d_ff=256)

        # Terminal task decoder: LayerNorm(128), Linear(128, 2): 514 parameters.
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_output)
        )

    def build_attention_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Query node j may attend to key node i iff i -> j or i == j."""
        adj = x[:, :, :self.num_nodes]
        eye = torch.eye(self.num_nodes, dtype=torch.bool, device=x.device).unsqueeze(0)
        mask = (adj.transpose(-1, -2) > 0.5) | eye
        return mask.unsqueeze(1)

    def step(self, z: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self.core(z, attn_mask)

    def forward(self, x: torch.Tensor, steps: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_mask = self.build_attention_mask(x)
        z = self.encoder(x)
        for _ in range(steps):
            z = self.step(z, attn_mask)
        logits = self.decoder(z[:, -1, :])
        return logits, z
```

## C7. `model/swarm.py`

```python
"""
Dual-Core Swarm with Nonlinear MLP Parity Head.
Exact parameter count: 292,427.
"""
import torch
import torch.nn as nn
from typing import Tuple
from model.core import LatentReasoner, binarize_ste

class LatentSwarm(nn.Module):
    def __init__(self, num_nodes: int = 16, d_input: int = 32, d_model: int = 128):
        super().__init__()
        # Core A: Transmitter (137,476 parameters).
        self.core_A = LatentReasoner(num_nodes=num_nodes, d_input=d_input, d_model=d_model, d_output=2)
        # Core B: Receiver reasoner (137,476 parameters).
        self.core_B = LatentReasoner(num_nodes=num_nodes, d_input=d_input, d_model=d_model, d_output=2)

        # Single-scalar binary channel interfaces.
        self.msg_encoder = nn.Linear(d_model, 1)    # 129 parameters.
        self.msg_embedder = nn.Linear(1, d_model)   # 256 parameters.

        # Nonlinear receiver-side parity head (17,090 parameters):
        # LayerNorm(256) 512 + Linear(256, 64) 16,448 + Linear(64, 2) 130.
        self.parity_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(
        self,
        x_A: torch.Tensor,
        x_B: torch.Tensor,
        steps_A: int = 8,
        steps_B: int = 8,
        flip_message: bool = False,
        zero_message: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_A = self.core_A.build_attention_mask(x_A)
        mask_B = self.core_B.build_attention_mask(x_B)

        # Core A independently reasons over Graph A.
        zA = self.core_A.encoder(x_A)
        for _ in range(steps_A):
            zA = self.core_A.step(zA, mask_A)

        aux_logits_A = self.core_A.decoder(zA[:, -1, :])
        raw_m = self.msg_encoder(zA[:, -1, :])
        m = binarize_ste(raw_m)

        if flip_message:
            m = -m

        # Core B independently reasons over Graph B. It does not ingest m recurrently.
        zB = self.core_B.encoder(x_B)
        for _ in range(steps_B):
            zB = self.core_B.step(zB, mask_B)

        aux_logits_B = self.core_B.decoder(zB[:, -1, :])

        # Receiver-side message ablation bypasses the embedder and its bias.
        if zero_message:
            msg_repr = torch.zeros(
                x_B.size(0), self.core_B.d_model,
                dtype=zB.dtype, device=x_B.device
            )
        else:
            msg_repr = self.msg_embedder(m)

        joint_readout = torch.cat([zB[:, -1, :], msg_repr], dim=-1)
        parity_logits = self.parity_head(joint_readout)
        return parity_logits, raw_m, aux_logits_A, aux_logits_B, m
```

## C8. `data/graph_task.py`

```python
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
```

## C9. `test_system.py`

Use the following as the baseline, then make the minimal test-hardening corrections specified immediately after the code. Do not change the model architecture or acceptance thresholds to force a pass.

```python
"""
System tests for architecture, receptive-field horizon, data integrity,
parity head expressivity, binary-channel behaviour and a 1-hop local baseline.
Run: python -m pytest -q test_system.py
"""
import pytest
import torch
import torch.nn as nn
from model.core import LatentReasoner
from model.swarm import LatentSwarm
from data.graph_task import generate_parity_benchmark, _shortest_path_length

def setup_module():
    torch.set_num_threads(4)

def test_exact_parameter_counts():
    reasoner = LatentReasoner(num_nodes=16, d_input=32, d_model=128, d_output=2)
    assert sum(p.numel() for p in reasoner.parameters() if p.requires_grad) == 137_476

    swarm = LatentSwarm(num_nodes=16, d_input=32, d_model=128)
    assert sum(p.numel() for p in swarm.parameters() if p.requires_grad) == 292_427

def test_receptive_field_horizon_theorem():
    torch.manual_seed(123)
    model = LatentReasoner(num_nodes=16, d_input=32, d_model=128)
    adj = torch.zeros((1, 16, 16), dtype=torch.float32)
    for i in range(15):
        adj[0, i, i + 1] = 1.0

    eye = torch.eye(16).unsqueeze(0)
    feat = torch.cat([adj, eye], dim=-1)
    feat.requires_grad_(True)

    mask = model.build_attention_mask(feat)
    z = model.encoder(feat)
    for _ in range(14):
        z = model.step(z, mask)

    loss_14 = z[0, 15, :].sum()
    loss_14.backward(retain_graph=True)
    assert torch.all(feat.grad[0, 0, :] == 0.0), "Information leaked before T=15!"

    feat.grad.zero_()
    z = model.step(z, mask)
    loss_15 = z[0, 15, :].sum()
    loss_15.backward()
    assert torch.any(feat.grad[0, 0, :] != 0.0), "Signal failed to reach Node 15 at T=15!"

def test_binary_channel_contract():
    torch.manual_seed(321)
    swarm = LatentSwarm()
    dummy_A = torch.randn(64, 16, 32)
    dummy_B = torch.randn(64, 16, 32)

    _, _, _, _, m = swarm(dummy_A, dummy_B)
    assert torch.all((m == 1.0) | (m == -1.0)), "Message is not strictly binary {-1, 1}!"
    # The seeded, sufficiently large random batch should exercise both polarities.
    assert set(m.detach().flatten().tolist()) == {-1.0, 1.0}, "Test batch did not exercise both message polarities."

    _, _, _, _, m_flipped = swarm(dummy_A, dummy_B, flip_message=True)
    assert torch.equal(m_flipped, -m), "Message flip failed to invert bits exactly!"

def _assert_labels_match_ground_truth(dataset):
    for feat_A, feat_B, b_A, b_B, Y in dataset:
        d_A = _shortest_path_length(feat_A[:, :16], 0, 15)
        d_B = _shortest_path_length(feat_B[:, :16], 0, 15)
        true_bA = 1 if d_A != -1 else 0
        true_bB = 1 if d_B != -1 else 0
        assert b_A.item() == true_bA
        assert b_B.item() == true_bB
        assert Y.item() == (true_bA ^ true_bB)
        if true_bA:
            assert 5 <= d_A <= 8
        if true_bB:
            assert 5 <= d_B <= 8

def test_dataset_integrity_and_disjoint_splits():
    train_ds, train_h = generate_parity_benchmark(num_samples=400, seed=1)
    val_ds, val_h = generate_parity_benchmark(num_samples=200, seed=2, existing_hashes=train_h)
    test_ds, test_h = generate_parity_benchmark(num_samples=200, seed=3, existing_hashes=train_h.union(val_h))

    assert train_h.isdisjoint(val_h), "Train and validation share graph samples!"
    assert train_h.isdisjoint(test_h), "Train and test share graph samples!"
    assert val_h.isdisjoint(test_h), "Validation and test share graph samples!"
    assert len(train_h) == len(train_ds) * 2, "Train contains duplicate graph hashes or missing hashes."
    assert len(val_h) == len(val_ds) * 2, "Validation contains duplicate graph hashes or missing hashes."
    assert len(test_h) == len(test_ds) * 2, "Test contains duplicate graph hashes or missing hashes."

    _assert_labels_match_ground_truth(train_ds)
    _assert_labels_match_ground_truth(val_ds)
    _assert_labels_match_ground_truth(test_ds)

def test_conditional_parity_balancing():
    ds, _ = generate_parity_benchmark(num_samples=1200, seed=888)
    b_A = ds.tensors[2].numpy()
    b_B = ds.tensors[3].numpy()
    Y = ds.tensors[4].numpy()
    for bit in (0, 1):
        assert abs(Y[b_A == bit].mean() - 0.5) <= 0.02
        assert abs(Y[b_B == bit].mean() - 0.5) <= 0.02

def test_parity_head_generalization_on_noisy_inputs():
    """Train/evaluate the parity head on noisy synthetic representations over multiple seeds."""
    for seed in (7, 42, 101, 2026, 9001):
        torch.manual_seed(seed)
        head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        centers_B = {0: torch.randn(128) - 1.0, 1: torch.randn(128) + 1.0}
        centers_A = {0: torch.full((128,), -1.0), 1: torch.full((128,), 1.0)}
        X_train, Y_train = [], []
        for ba in (0, 1):
            for bb in (0, 1):
                for _ in range(80):
                    z_b = centers_B[bb] + torch.randn(128) * 0.1
                    m_a = centers_A[ba] + torch.randn(128) * 0.1
                    X_train.append(torch.cat([z_b, m_a]))
                    Y_train.append(ba ^ bb)
        X_train = torch.stack(X_train)
        Y_train = torch.tensor(Y_train, dtype=torch.long)

        for _ in range(60):
            optimizer.zero_grad()
            loss = criterion(head(X_train), Y_train)
            loss.backward()
            optimizer.step()

        X_test, Y_test = [], []
        for ba in (0, 1):
            for bb in (0, 1):
                for _ in range(20):
                    z_b = centers_B[bb] + torch.randn(128) * 0.1
                    m_a = centers_A[ba] + torch.randn(128) * 0.1
                    X_test.append(torch.cat([z_b, m_a]))
                    Y_test.append(ba ^ bb)
        X_test = torch.stack(X_test)
        Y_test = torch.tensor(Y_test, dtype=torch.long)
        acc = (head(X_test).argmax(dim=-1) == Y_test).float().mean().item()
        assert acc >= 0.99, f"Parity head failed noisy XOR generalization for seed={seed}: acc={acc}"

def test_empirical_1hop_local_baseline():
    train_ds, train_hashes = generate_parity_benchmark(num_samples=2000, seed=10)
    val_ds, _ = generate_parity_benchmark(num_samples=1000, seed=20, existing_hashes=train_hashes)

    def extract_signatures(dataset):
        sigs, labels = [], []
        for feat, _, b_A, _, _ in dataset:
            adj = feat[:, :16]
            edges = (
                int(adj[12, 13] > 0.5), int(adj[12, 14] > 0.5), int(adj[12, 15] > 0.5),
                int(adj[13, 14] > 0.5), int(adj[13, 15] > 0.5), int(adj[14, 15] > 0.5)
            )
            sigs.append(edges)
            labels.append(b_A.item())
        return sigs, labels

    train_sigs, train_labels = extract_signatures(train_ds)
    val_sigs, val_labels = extract_signatures(val_ds)
    counts = {}
    for sig, lab in zip(train_sigs, train_labels):
        counts.setdefault(sig, [0, 0])[lab] += 1
    lookup = {sig: (1 if c[1] >= c[0] else 0) for sig, c in counts.items()}
    acc = sum(lookup.get(sig, 0) == lab for sig, lab in zip(val_sigs, val_labels)) / len(val_labels)
    assert 0.52 <= acc <= 0.565, f"Local-region lookup baseline outside expected range: {acc*100:.1f}%"
```

### Required test-hardening notes

- Run the suite and make sure `test_binary_channel_contract` is stable for the fixed seed. If it fails because a random untrained model happened to emit one polarity only, redesign this specific test deterministically; do not simply remove the polarity requirement. The fundamental contract test should test binary values and exact flipping, while the trained probe gate tests that the trained model learns both label polarities.
- The local lookup baseline is deliberately described as a **local-region lookup baseline**, not a mathematical proof of the precise T=1 receptive field. It encodes the six edges among Nodes 12–15; it is an empirical diagnostic.
- The noisy parity-head capacity test uses synthetic, clustered representations. Passing it shows the chosen head can learn this synthetic XOR setup; it does not prove that the learned Core B latents are naturally separable in the same way.

## C10. `train_unit.py`

```python
"""
Phase 1: Unit Cell Training with Hard Programmatic Acceptance Gates.
Trains at fixed T=8, checks T=1 against the empirical local ceiling, and evaluates a separate test split.
"""
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.core import LatentReasoner
from data.graph_task import generate_parity_benchmark

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)

def evaluate(model, loader, steps):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_A, _, b_A, _, _ in loader:
            logits, _ = model(x_A, steps=steps)
            preds = logits.argmax(dim=-1)
            correct += (preds == b_A).sum().item()
            total += b_A.size(0)
    return correct / total

def train_phase1():
    set_seed(42)
    print("Generating datasets...")
    train_ds, train_hashes = generate_parity_benchmark(num_samples=4000, seed=42)
    val_ds, val_hashes = generate_parity_benchmark(num_samples=1000, seed=1337, existing_hashes=train_hashes)
    test_ds, _ = generate_parity_benchmark(num_samples=1000, seed=9999, existing_hashes=train_hashes.union(val_hashes))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = LatentReasoner(num_nodes=16, d_input=32, d_model=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    print("Training LatentReasoner at fixed recurrent depth T=8...")
    for epoch in range(1, 26):
        model.train()
        for x_A, _, b_A, _, _ in train_loader:
            optimizer.zero_grad()
            logits, _ = model(x_A, steps=8)
            loss = criterion(logits, b_A)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            acc_8 = evaluate(model, val_loader, steps=8)
            print(f"Epoch {epoch:02d} | Val Acc (T=8): {acc_8*100:.1f}%")

    print("\n--- Phase 1 Acceptance Gate Evaluation ---")
    acc_1 = evaluate(model, val_loader, steps=1)
    acc_8 = evaluate(model, val_loader, steps=8)
    print(f"Gate 1: T=1 Accuracy = {acc_1*100:.1f}% (Required: <= 56.5%)")
    print(f"Gate 2: T=8 Accuracy = {acc_8*100:.1f}% (Required: >= 90.0%)")

    if acc_1 > 0.565:
        print(f"FAIL: Gate 1 missed ({acc_1*100:.1f}% > 56.5%)!", file=sys.stderr)
        sys.exit(1)
    if acc_8 < 0.90:
        print(f"FAIL: Gate 2 missed ({acc_8*100:.1f}% < 90.0%)!", file=sys.stderr)
        sys.exit(1)

    acc_test_8 = evaluate(model, test_loader, steps=8)
    print(f"SUCCESS: Phase 1 validation gates passed. Held-out Test Acc (T=8): {acc_test_8*100:.1f}%")
    torch.save(model.state_dict(), "checkpoint_phase1.pt")

if __name__ == "__main__":
    train_phase1()
```

## C11. `train_swarm.py`

```python
"""
Phase 2a: 1-Bit Latent Swarm with Stratified Acceptance Gates.
Evaluates all four (b_A, b_B) quadrants, stratified message-flip and probe recovery.
Reports validation acceptance and held-out test results.
"""
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.swarm import LatentSwarm
from data.graph_task import generate_parity_benchmark

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)

def evaluate_swarm_stratified(swarm, loader, **kwargs):
    swarm.eval()
    quad_correct = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    quad_total = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    msg_correct = {0: 0, 1: 0}
    msg_total = {0: 0, 1: 0}
    total_correct = 0
    total = 0
    total_msg_correct = 0

    with torch.no_grad():
        for x_A, x_B, b_A, b_B, Y in loader:
            p_logits, _, _, _, m = swarm(x_A, x_B, steps_A=8, steps_B=8, **kwargs)
            preds = p_logits.argmax(dim=-1)
            total_correct += (preds == Y).sum().item()
            total += Y.size(0)

            for ba, bb, p, y in zip(b_A.tolist(), b_B.tolist(), preds.tolist(), Y.tolist()):
                quad_total[(ba, bb)] += 1
                if p == y:
                    quad_correct[(ba, bb)] += 1

            m_bits = (m.squeeze(-1) > 0).long().tolist()
            for ba, mb in zip(b_A.tolist(), m_bits):
                msg_total[ba] += 1
                if mb == ba:
                    msg_correct[ba] += 1
                    total_msg_correct += 1

    quad_accs = {q: quad_correct[q] / max(quad_total[q], 1) for q in quad_correct}
    probe_accs = {b: msg_correct[b] / max(msg_total[b], 1) for b in msg_correct}
    overall_probe = total_msg_correct / max(total, 1)
    return total_correct / max(total, 1), quad_accs, probe_accs, overall_probe

def evaluate_flip_stratified(swarm, loader):
    """Fraction of initially correct predictions remaining correct after m is flipped, by b_B."""
    swarm.eval()
    flip_correct = {0: 0, 1: 0}
    base_correct = {0: 0, 1: 0}
    with torch.no_grad():
        for x_A, x_B, _, b_B, Y in loader:
            p_base, _, _, _, _ = swarm(x_A, x_B, steps_A=8, steps_B=8)
            p_flip, _, _, _, _ = swarm(x_A, x_B, steps_A=8, steps_B=8, flip_message=True)
            base_preds = p_base.argmax(dim=-1)
            flip_preds = p_flip.argmax(dim=-1)
            for bb, bp, fp, y in zip(b_B.tolist(), base_preds.tolist(), flip_preds.tolist(), Y.tolist()):
                if bp == y:
                    base_correct[bb] += 1
                    if fp == y:
                        flip_correct[bb] += 1
    return {bb: flip_correct[bb] / max(base_correct[bb], 1) for bb in (0, 1)}

def _print_stratified(prefix, overall, quads, flip, zero, probes, probe_overall):
    print(f"{prefix} Connected Accuracy: {overall*100:.2f}%")
    for q in ((0, 0), (0, 1), (1, 0), (1, 1)):
        print(f"{prefix} Quadrant {q} Accuracy: {quads[q]*100:.2f}%")
    print(f"{prefix} Flip residual b_B=0: {flip[0]*100:.2f}%")
    print(f"{prefix} Flip residual b_B=1: {flip[1]*100:.2f}%")
    print(f"{prefix} Zero-message accuracy: {zero*100:.2f}%")
    print(f"{prefix} Probe overall: {probe_overall*100:.2f}%")
    print(f"{prefix} Probe b_A=0: {probes[0]*100:.2f}%")
    print(f"{prefix} Probe b_A=1: {probes[1]*100:.2f}%")

def train_phase2():
    set_seed(42)
    train_ds, train_hashes = generate_parity_benchmark(num_samples=4000, seed=42)
    val_ds, val_hashes = generate_parity_benchmark(num_samples=1000, seed=1337, existing_hashes=train_hashes)
    test_ds, _ = generate_parity_benchmark(num_samples=1000, seed=9999, existing_hashes=train_hashes.union(val_hashes))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    swarm = LatentSwarm(num_nodes=16, d_input=32, d_model=128)
    try:
        # State dict contains tensors only; all computation is CPU-only.
        p1_state = torch.load("checkpoint_phase1.pt", map_location="cpu", weights_only=True)
        swarm.core_A.load_state_dict(p1_state)
        swarm.core_B.load_state_dict(p1_state)
        print("Warm-started Core A and Core B from checkpoint_phase1.pt")
    except FileNotFoundError:
        print("Error: checkpoint_phase1.pt not found! Run train_unit.py first.", file=sys.stderr)
        sys.exit(1)

    optimizer = torch.optim.AdamW(swarm.parameters(), lr=8e-4, weight_decay=1e-2)
    crit_parity = nn.CrossEntropyLoss()
    crit_aux = nn.CrossEntropyLoss()
    crit_msg = nn.BCEWithLogitsLoss()

    print("Training Phase 2a at T_A=8, T_B=8...")
    for epoch in range(1, 21):
        swarm.train()
        for x_A, x_B, b_A, b_B, Y in train_loader:
            optimizer.zero_grad()
            p_logits, raw_m, aux_A, aux_B, _ = swarm(x_A, x_B, steps_A=8, steps_B=8)
            l_parity = crit_parity(p_logits, Y)
            l_aux_A = crit_aux(aux_A, b_A)
            l_aux_B = crit_aux(aux_B, b_B)
            l_msg = crit_msg(raw_m.squeeze(-1), b_A.float())
            loss = l_parity + 0.5 * l_aux_A + 0.5 * l_aux_B + 0.5 * l_msg
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            acc, _, _, _ = evaluate_swarm_stratified(swarm, val_loader)
            print(f"Epoch {epoch:02d} | Validation parity accuracy: {acc*100:.2f}%")

    print("\n--- Phase 2a Validation Acceptance Evaluation ---")
    acc_connected, quads, probes, probe_overall = evaluate_swarm_stratified(swarm, val_loader)
    acc_zero, _, _, _ = evaluate_swarm_stratified(swarm, val_loader, zero_message=True)
    flip_strat = evaluate_flip_stratified(swarm, val_loader)
    _print_stratified("Validation", acc_connected, quads, flip_strat, acc_zero, probes, probe_overall)

    # Hard validation gates; do not relax these to force a pass.
    failed = []
    if acc_connected < 0.82:
        failed.append("overall connected accuracy < 82%")
    if any(q_acc < 0.75 for q_acc in quads.values()):
        failed.append("one or more quadrant accuracies < 75%")
    if flip_strat[0] > 0.15 or flip_strat[1] > 0.15:
        failed.append("stratified message-flip residual > 15%")
    if acc_zero > 0.53:
        failed.append("zero-message accuracy > 53%")
    if probe_overall < 0.95 or probes[0] < 0.90 or probes[1] < 0.90:
        failed.append("overall or stratified message probe below acceptance threshold")
    if failed:
        for reason in failed:
            print(f"FAIL: {reason}", file=sys.stderr)
        sys.exit(1)

    print("SUCCESS: All Phase 2a validation gates passed.")

    # Final held-out test evaluation; report all metrics but do not tune on test data.
    print("\n--- Final Held-Out Test Set Evaluation ---")
    t_acc, t_quads, t_probes, t_probe_overall = evaluate_swarm_stratified(swarm, test_loader)
    t_zero, _, _, _ = evaluate_swarm_stratified(swarm, test_loader, zero_message=True)
    t_flip = evaluate_flip_stratified(swarm, test_loader)
    _print_stratified("Test", t_acc, t_quads, t_flip, t_zero, t_probes, t_probe_overall)

if __name__ == "__main__":
    train_phase2()
```

---

# PART D — REQUIRED MINIMAL CONSISTENCY CHECKS BEFORE THE RUN

The source above is the intended v7.1 baseline. Before training, perform these small, documented hardening checks; do not change the research architecture or thresholds:

1. **Binary channel test:** ensure its chosen seeded test batch exercises both message polarities, while also checking `m` contains only `-1.0` or `+1.0`, and `flip_message=True` produces exactly `-m`. If a model-output batch test is flaky, test the mathematical binarizer directly for both polarities and keep a separate integration assertion for the model output's binary values. Do not pretend a random initial model must always emit both polarities for arbitrary seeds.
2. **XOR head test:** use several fixed seeds and a separate noisy test set, as the supplied hardened test does. Ensure the test is stable and completes within a reasonable time on 4 vCPUs. A 99% threshold is acceptable for this synthetic diagnostic; never weaken a failing test without identifying why.
3. **Dataset integrity:** check train, validation and test graph-hash sets are pairwise disjoint; there are exactly two unique graph hashes per parity sample within each split; and labels match reachability and XOR across all three splits. Check all four quadrants are exactly balanced.
4. **Conditional parity balancing:** verify `P(Y=1 | b_A)` and `P(Y=1 | b_B)` are approximately 0.5 for both binary values.
5. **Probe gate:** Phase 2 validation must enforce all three requirements: overall message probe ≥95%, probe for `b_A=0` ≥90%, and probe for `b_A=1` ≥90%.
6. **Empirical local baseline:** leave it as an explicitly empirical induced-subgraph baseline. The fixed Phase 1 T=1 gate remains ≤56.5% for this experiment. Print and preserve the measured baseline value.
7. **Generator guardrails:** ensure `num_nodes == 16` and `num_samples % 4 == 0` are asserted, as included above.
8. **Parameter counts:** verify exactly 137,476 for `LatentReasoner` and 292,427 for `LatentSwarm`.
9. **Code syntax:** run `python -m compileall -q model data test_system.py train_unit.py train_swarm.py`.
10. **Gradient flow smoke test:** on one small training batch, run a backward pass and report any parameter with `grad is None`. This is diagnostic only; do not require every parameter tensor to have a nonzero gradient on every possible batch without checking whether that is mathematically appropriate.

Document any deviations or code fixes from the supplied canonical version in `CHANGELOG.md`. Keep them minimal. Do not alter model architecture, optimisers, epochs, batch sizes, loss weights, or scientific thresholds simply to get a pass.

---

# PART E — GIT COMMIT AND PUSH PROCEDURE

After source creation and minimal test consistency repairs:

1. Run `git diff --check`.
2. Inspect `git status --short` and `git diff --stat`.
3. Verify `.gitignore` excludes `.venv/`, `runs/`, checkpoints, caches and logs.
4. Stage only intended source files and project documentation.
5. Commit with a message such as `Add v7.1 latent recurrent swarm baseline` or, if test-only hardening has been made, `Harden v7.1 baseline verification tests`.
6. Add a changelog entry explaining that Phase 2a is decoupled reasoning with receiver-side one-bit decision fusion and is not Phase 2b.
7. Create a unique pre-run tag (`v7.1-pre-run` if unchanged, or `v7.1.1-pre-run` after documented test-only repairs).
8. Push the branch and tag to the correct private GitHub repository. Never use `--force`.
9. Verify the pushed commit and tag with Git commands / `gh repo view`.

If GitHub repo creation or push is blocked by authentication, repository-name conflicts, visibility, or permission, preserve the complete local commit and give a precise diagnostic. Do not make the repository public or push to an unrelated remote as a workaround.

---

# PART F — RECORDED FIRST RUN

## F1. Run metadata

Create a UTC-stamped run directory. Do not reuse an existing run directory. For example:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/git-commit.txt"
git status --short --branch > "$RUN_DIR/git-status.txt"
python --version > "$RUN_DIR/python-version.txt"
python -m pip freeze > "$RUN_DIR/environment.freeze.txt"
python - <<'PY' > "$RUN_DIR/runtime.txt"
import sys, torch
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Threads before training-script setup:', torch.get_num_threads())
PY
free -h > "$RUN_DIR/memory-before.txt"
```

Set thread environment variables in the same shell that runs training:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
set -Eeuo pipefail
```

## F2. Stage 1 — test suite

Run:

```bash
python -m pytest -q test_system.py 2>&1 | tee "$RUN_DIR/pytest.log"
```

If any system test fails:

- Do not start training.
- Determine whether it is a genuine implementation defect, a test flake, an environment incompatibility, or an unsupported expectation.
- Fix only the underlying issue; do not lower acceptance thresholds to make the suite pass.
- Record the issue and fix in `CHANGELOG.md`, make a new commit/tag if applicable, push it, and rerun the entire system suite.
- If the fix cannot be made safely, stop and report the exact failure and evidence.

## F3. Stage 2 — Phase 1

Before training, remove only a stale checkpoint at the exact project path if one exists and has been confirmed to be generated output from this project. Preserve or back it up into the active run directory first. This prevents Phase 2 from accidentally loading an old checkpoint if Phase 1 fails.

Run:

```bash
python -u train_unit.py 2>&1 | tee "$RUN_DIR/train_unit.log"
```

The Phase 1 script must complete successfully and create a non-empty fresh `checkpoint_phase1.pt`. If it fails either acceptance gate, stop. Preserve the log, report T=1 and T=8 validation accuracy, and do not run Phase 2 on the failed checkpoint.

If Phase 1 succeeds:

```bash
test -s checkpoint_phase1.pt
cp checkpoint_phase1.pt "$RUN_DIR/checkpoint_phase1.pt"
sha256sum checkpoint_phase1.pt > "$RUN_DIR/checkpoint_phase1.sha256"
```

Do not commit the checkpoint to Git. Preserve it on the VM and preserve its checksum in the report.

## F4. Stage 3 — Phase 2a

Run only after Phase 1 gates pass and the new checkpoint has been verified:

```bash
python -u train_swarm.py 2>&1 | tee "$RUN_DIR/train_swarm.log"
```

The script must enforce all validation gates:

- overall connected validation accuracy ≥82%;
- each of the four `(b_A, b_B)` quadrants ≥75%;
- message-flip residual ≤15% separately for `b_B=0` and `b_B=1` among previously correct samples;
- zero-message validation accuracy ≤53%;
- message probe ≥95% overall and ≥90% separately for `b_A=0` and `b_A=1`.

When validation passes, the script reports the corresponding held-out test metrics and interventions. Do not tune on the test split.

If a Phase 2 validation gate fails, stop this baseline run and preserve every metric. Do not alter a threshold, silently add epochs, change the optimiser, or retry under the same run ID. A diagnostic follow-up may be proposed separately, but it must be clearly identified as a new experiment.

## F5. Logs and interruption safety

Keep standard output and standard error together in the run log using `tee` and `pipefail`. If the command interface has a hard timeout, check whether `tmux` or a similar persistent terminal is available and use it transparently; do not start an unmonitored background process and claim it completed. Check VM memory and disk before starting each training stage. Do not change batch size, depth, or precision pre-emptively; the specified model is small, and changes require evidence.

---

# PART G — FINAL REPORT AND GIT PUSH

Create `reports/first-run-report.md` after the run. Include:

1. Date/time in UTC and run ID.
2. Repository URL, branch, Git commit, and source tag.
3. Python/PyTorch versions, CPU count, memory, thread settings, OS, and dependency snapshot reference.
4. Exact system-test summary, run time, and whether any tests required a documented correction.
5. Phase 1 epoch logs summary, T=1 and T=8 validation accuracy, held-out T=8 test accuracy, pass/fail status, and checkpoint SHA-256.
6. Measured local-region lookup baseline and its limitations.
7. Phase 2a epoch logs summary, overall connected validation accuracy, all four validation quadrant accuracies, stratified flip residuals, zero-message validation accuracy, overall/subgroup probe metrics, and held-out test counterparts.
8. Explicit pass/fail for each acceptance gate.
9. Any error, interruption, modification, or deviation from the plan.
10. Interpretation limits: this is the decoupled-reasoner/receiver-fusion baseline, not evidence for recurrent message integration; one run does not establish seed robustness.

Do not invent results for stages that did not run. Mark them `NOT RUN` or `FAILED`, as appropriate.

After generating the report, inspect it for secrets, inaccurate claims and unreported failures. Commit only the concise report and relevant non-sensitive source/documentation changes; leave `runs/`, full logs, and checkpoints ignored. Push the report commit to the same private repo and branch. Never force-push. Ensure the final `git status` is clean except for intentionally ignored run artefacts.

---

# PART H — FINAL RESPONSE TO THE USER

Return a direct operational report, not a promise to do work later. State:

- the repository URL and privacy status;
- the branch, latest commit hash, and tag;
- VM/Python/PyTorch setup and CPU-only confirmation;
- system test results;
- whether Phase 1 and Phase 2a actually ran;
- every validation gate result and held-out test metric that was measured;
- paths to the run directory, logs, checkpoint checksum, and report;
- any deviations, failed gates, or blockers;
- whether the next correct step is Phase 2b planning or baseline debugging.

Be exact about what was and was not executed. A passed unit suite is not a passed training run. A trained validation pass is not by itself proof of generalisation or of the original recurrent-collaboration hypothesis.

**Begin now:** inspect the VM and repository, then carry out the setup and first run according to this prompt.

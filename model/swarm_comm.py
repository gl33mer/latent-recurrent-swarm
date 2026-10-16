"""
Phase 2b: Dual-Core Swarm with a one-way COMM-token workspace.

Core A runs the unchanged v7.1 recurrent reasoner on Graph A and sends exactly one
forward-pass binary scalar m in {-1.0, +1.0} (straight-through estimator).

Core B runs as a 17-token sequence: 16 graph-node tokens plus one COMM token (index 16).
The COMM register is a one-way message workspace:

- Graph query node j may attend graph key node i iff i == j or edge i -> j exists
  (v7.1 orientation: adj[b, i, j] == 1 means edge i -> j), and additionally may attend
  the COMM key while COMM visibility is enabled for the current recurrent step.
- The COMM query may attend ONLY to itself (index 16). It can never read graph nodes,
  so Graph B information cannot bypass graph-local receptive-field paths through the
  workspace.
- No unmasked all-to-all attention fallback is ever used; every query keeps at least
  one allowed key (self / COMM self-loop).

Step masks (comm_mode):
- "full":       graph queries may read the COMM key on every recurrent step.
- "final-only": graph queries may read the COMM key on the final recurrent step only;
  on earlier steps the graph-node computation is bit-identical for m = +1 and m = -1.

The terminal parity head consumes ONLY the terminal graph-node state z_B[15]
(shape [batch, d_model]). It receives no message tensor, no message embedding and no
COMM state, so the incoming bit can influence the output only by modulating Core B's
recurrent graph-node states.

This module does not modify model/core.py or model/swarm.py; the v7.1 baseline stays
available and checkpoint-compatible (Phase 1 state dicts load into both cores).
"""
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from model.core import LatentReasoner, binarize_ste

COMM_MODES = ("full", "final-only")


def comm_step_mask(
    comm_mode: str,
    step: int,
    steps_B: int,
    mask_off: torch.Tensor,
    mask_on: torch.Tensor,
) -> torch.Tensor:
    """Select the attention mask for recurrent step `step` (0-indexed) of `steps_B`.

    "full": COMM visible on every step. "final-only": COMM visible on the final step
    only (graph queries cannot read COMM on steps 0..steps_B-2).
    """
    if comm_mode == "full":
        return mask_on
    if comm_mode == "final-only":
        return mask_on if step == steps_B - 1 else mask_off
    raise ValueError(f"Unknown comm_mode: {comm_mode!r}; expected one of {COMM_MODES}")


class LatentCommSwarm(nn.Module):
    """Phase 2b swarm: one-bit message integrated into Core B's recurrence via a
    read-only COMM token. Parameter count (num_nodes=16, d_input=32, d_model=128):
    284,107 (documented in reports/phase2b-design.md; asserted in test_phase2b.py)."""

    def __init__(self, num_nodes: int = 16, d_input: int = 32, d_model: int = 128):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model

        # Core A: transmitter (v7.1 architecture, 137,476 parameters).
        self.core_A = LatentReasoner(num_nodes=num_nodes, d_input=d_input, d_model=d_model, d_output=2)
        # Core B: receiver reasoner (v7.1 architecture, 137,476 parameters); here driven
        # over a 17-token sequence with externally supplied per-step masks.
        self.core_B = LatentReasoner(num_nodes=num_nodes, d_input=d_input, d_model=d_model, d_output=2)

        # Single-scalar binary channel interfaces (as v7.1).
        self.msg_encoder = nn.Linear(d_model, 1)    # 129 parameters.
        self.msg_embedder = nn.Linear(1, d_model)   # 256 parameters.

        # Learned COMM token/register: shape [1, 1, d_model], small random init.
        self.learned_comm_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)  # 128 parameters.

        # Terminal parity head: consumes ONLY z_B[15] (8,642 parameters):
        # LayerNorm(128) 256 + Linear(128, 64) 8,256 + Linear(64, 2) 130.
        self.parity_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    # ------------------------------------------------------------------ masks
    def build_comm_masks(self, x_B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the two 17-token attention masks for Core B.

        Returns (mask_off, mask_on), each [batch, 1, 17, 17] boolean:
        - mask_off: graph->graph self+predecessors; COMM query -> COMM only;
          graph queries canNOT read the COMM key.
        - mask_on:  identical, plus graph queries may read the COMM key.
        """
        adj = x_B[:, :, :self.num_nodes]
        B = x_B.size(0)
        S = self.num_nodes + 1
        eye = torch.eye(self.num_nodes, dtype=torch.bool, device=x_B.device).unsqueeze(0)
        graph_graph = (adj.transpose(-1, -2) > 0.5) | eye  # [B, 16, 16]

        mask_off = torch.zeros(B, S, S, dtype=torch.bool, device=x_B.device)
        mask_off[:, :self.num_nodes, :self.num_nodes] = graph_graph
        mask_off[:, self.num_nodes, self.num_nodes] = True  # COMM self-loop only.

        mask_on = mask_off.clone()
        mask_on[:, :self.num_nodes, self.num_nodes] = True  # graph queries may read COMM.
        return mask_off.unsqueeze(1), mask_on.unsqueeze(1)

    # ----------------------------------------------------------- message path
    def build_comm_state(
        self,
        m: torch.Tensor,
        batch_size: int,
        zero_message: bool = False,
    ) -> torch.Tensor:
        """COMM register initial state: learned token + embedded message [batch, 1, d_model].

        With zero_message=True the information-bearing message contribution is replaced
        by an exact zero tensor before the addition (the msg_embedder — including its
        bias — is bypassed, so no bias term can encode the bit). The constant learned
        COMM token remains: it is not sample-specific information from Core A.
        """
        if zero_message:
            msg_repr = torch.zeros(
                batch_size, self.d_model, dtype=m.dtype, device=m.device
            )
        else:
            msg_repr = self.msg_embedder(m)
        return self.learned_comm_token.expand(batch_size, 1, self.d_model) + msg_repr.unsqueeze(1)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x_A: torch.Tensor,
        x_B: torch.Tensor,
        steps_A: int = 8,
        steps_B: int = 8,
        flip_message: bool = False,
        zero_message: bool = False,
        comm_mode: str = "full",
        trace: bool = False,
    ):
        """Full forward pass.

        Returns (parity_logits, raw_m, aux_logits_A, aux_logits_B, m), and additionally
        a list `traces` of graph-node state tensors [batch, 16, d_model] when trace=True:
        traces[0] is the post-encoder initial state, traces[t] (t = 1..steps_B) is the
        state after recurrent step t. Trace tensors are detached; tracing changes no
        computed value, only what is retained.
        """
        if comm_mode not in COMM_MODES:
            raise ValueError(f"Unknown comm_mode: {comm_mode!r}; expected one of {COMM_MODES}")

        # ---- Core A: independent recurrent reasoning over Graph A (identical to v7.1).
        mask_A = self.core_A.build_attention_mask(x_A)
        zA = self.core_A.encoder(x_A)
        for _ in range(steps_A):
            zA = self.core_A.step(zA, mask_A)

        aux_logits_A = self.core_A.decoder(zA[:, -1, :])
        raw_m = self.msg_encoder(zA[:, -1, :])
        m = binarize_ste(raw_m)

        if flip_message:
            m = -m

        # ---- Core B: 17-token sequence with one-way COMM workspace.
        mask_off, mask_on = self.build_comm_masks(x_B)
        comm_state = self.build_comm_state(m, x_B.size(0), zero_message=zero_message)
        zB = torch.cat([self.core_B.encoder(x_B), comm_state], dim=1)  # [B, 17, d_model]

        traces: Optional[List[torch.Tensor]] = [] if trace else None
        if trace:
            traces.append(zB[:, :self.num_nodes, :].detach().clone())

        for step in range(steps_B):
            step_mask = comm_step_mask(comm_mode, step, steps_B, mask_off, mask_on)
            zB = self.core_B.step(zB, step_mask)
            if trace:
                traces.append(zB[:, :self.num_nodes, :].detach().clone())

        # Terminal graph-node readout: explicit index 15 (token 16 is the COMM register,
        # so v7.1's z[:, -1, :] indexing must NOT be reused here).
        terminal_state = zB[:, self.num_nodes - 1, :]
        aux_logits_B = self.core_B.decoder(terminal_state)
        parity_logits = self.parity_head(terminal_state)

        if trace:
            return parity_logits, raw_m, aux_logits_A, aux_logits_B, m, traces
        return parity_logits, raw_m, aux_logits_A, aux_logits_B, m

"""
Phase 2c: message-conditioned reachability models.

`LatentConditionedSwarm` (primary): Core A is the unchanged v7.1 reasoner (32-dim input)
sending one binary selector m in {-1, +1} (STE). Core B is a v7.1 reasoner whose encoder
accepts 64-dim inputs: [adj_union(16), adj_R0(16), adj_R1(16), one_hot(16)]. Core B runs
over a 17-token sequence (16 graph nodes + one read-only COMM register at index 16) with
the same one-way mask rules as Phase 2b (`model/swarm_comm.py`): graph queries may read
the COMM key per the visibility schedule; the COMM query attends only to itself; the
graph-to-graph mask uses the union adjacency and is independent of the message. The task
readout is Core B's existing terminal decoder applied to z_B[15] ONLY — no message
tensor, message embedding, or COMM state reaches the head.

Warm-start mapping (documented transfer, NOT an exact checkpoint load):
Core A loads the Phase 1 state dict strictly. Core B transfers every shape-compatible
Phase 1 weight (encoder LayerNorm, recurrent block, decoder); for the first encoder
linear layer, Phase 1 adjacency-input columns (0:16) map to the new union columns (0:16),
the new relation channels (16:48) start at exactly zero, and Phase 1 node-ID columns
(16:32) map to new columns (48:64); the bias is copied. With relation columns zero, the
new encoder reproduces the Phase 1 encoder output on equivalent union/ID inputs
(unit-tested).

`LatentDualRelationSwarm` (explicit diagnostic, isolated from the primary comparison):
same Core B backbone and input representation (including the COMM register), but two
terminal binary heads supervised separately for r0 and r1; at evaluation the one-bit
selector chooses between the two candidate predictions only AFTER both are produced. The
unused Phase 1-style decoder is replaced by nn.Identity() so no trainable parameter is
dead. It never uses training labels at inference.
"""
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from model.core import LatentReasoner, binarize_ste
from model.swarm_comm import COMM_MODES, comm_step_mask


class _CommSwarmBase(nn.Module):
    """Shared Core A / channel / COMM-mask machinery for Phase 2c models."""

    def __init__(self, d_input_B: int, d_model: int = 128):
        super().__init__()
        self.num_nodes = 16
        self.d_model = d_model
        self.core_A = LatentReasoner(num_nodes=16, d_input=32, d_model=d_model, d_output=2)
        self.core_B = LatentReasoner(num_nodes=16, d_input=d_input_B, d_model=d_model, d_output=2)
        self.msg_encoder = nn.Linear(d_model, 1)
        self.msg_embedder = nn.Linear(1, d_model)
        self.learned_comm_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    # Same one-way 17-token masks as Phase 2b; adj_union is input columns 0:16.
    def build_comm_masks(self, x_B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        adj = x_B[:, :, :self.num_nodes]
        B = x_B.size(0)
        S = self.num_nodes + 1
        eye = torch.eye(self.num_nodes, dtype=torch.bool, device=x_B.device).unsqueeze(0)
        graph_graph = (adj.transpose(-1, -2) > 0.5) | eye
        mask_off = torch.zeros(B, S, S, dtype=torch.bool, device=x_B.device)
        mask_off[:, :self.num_nodes, :self.num_nodes] = graph_graph
        mask_off[:, self.num_nodes, self.num_nodes] = True
        mask_on = mask_off.clone()
        mask_on[:, :self.num_nodes, self.num_nodes] = True
        return mask_off.unsqueeze(1), mask_on.unsqueeze(1)

    def build_comm_state(self, m: torch.Tensor, batch_size: int,
                         zero_message: bool = False) -> torch.Tensor:
        if zero_message:
            msg_repr = torch.zeros(batch_size, self.d_model, dtype=m.dtype, device=m.device)
        else:
            msg_repr = self.msg_embedder(m)
        return self.learned_comm_token.expand(batch_size, 1, self.d_model) + msg_repr.unsqueeze(1)

    def _run_core_A(self, x_A, steps_A):
        mask_A = self.core_A.build_attention_mask(x_A)
        zA = self.core_A.encoder(x_A)
        for _ in range(steps_A):
            zA = self.core_A.step(zA, mask_A)
        aux_logits_A = self.core_A.decoder(zA[:, -1, :])
        raw_m = self.msg_encoder(zA[:, -1, :])
        return aux_logits_A, raw_m

    def _message(self, raw_m, flip_message, force_message):
        m = binarize_ste(raw_m)
        if flip_message:
            m = -m
        if force_message is not None:
            if force_message not in (-1.0, 1.0):
                raise ValueError("force_message must be None, -1.0 or +1.0")
            m = torch.full_like(m, force_message)
        return m

    def _run_core_B(self, x_B, m, steps_B, comm_mode, zero_message, trace):
        mask_off, mask_on = self.build_comm_masks(x_B)
        comm_state = self.build_comm_state(m, x_B.size(0), zero_message=zero_message)
        zB = torch.cat([self.core_B.encoder(x_B), comm_state], dim=1)
        traces: Optional[List[torch.Tensor]] = [] if trace else None
        if trace:
            traces.append(zB[:, :self.num_nodes, :].detach().clone())
        for step in range(steps_B):
            step_mask = comm_step_mask(comm_mode, step, steps_B, mask_off, mask_on)
            zB = self.core_B.step(zB, step_mask)
            if trace:
                traces.append(zB[:, :self.num_nodes, :].detach().clone())
        terminal = zB[:, self.num_nodes - 1, :]
        return terminal, traces

    def load_phase1_warm_start(self, phase1_state: dict) -> None:
        """Documented transfer. Core A: strict load. Core B: shape-compatible transfer
        with the documented first-layer column mapping (see module docstring).
        If a subclass replaced core_B.decoder (e.g. the dual-relation diagnostic), the
        decoder keys are intentionally dropped from the transfer: that model's readout
        heads are freshly initialised by design."""
        self.core_A.load_state_dict(phase1_state)  # strict
        decoder_replaced = isinstance(self.core_B.decoder, nn.Identity)
        mapped = {}
        for key, value in phase1_state.items():
            if decoder_replaced and key.startswith("decoder."):
                continue
            if key == "encoder.0.weight":  # [128, 32] -> [128, 64]
                w = torch.zeros(value.size(0), self.core_B.encoder[0].in_features,
                                dtype=value.dtype)
                w[:, :16] = value[:, :16]        # adjacency columns -> union columns
                w[:, 16:48] = 0.0                # relation channels start at zero
                w[:, 48:64] = value[:, 16:32]    # node-ID columns
                mapped[key] = w
            else:
                mapped[key] = value
        self.core_B.load_state_dict(mapped)      # strict: all keys, shapes verified


class LatentConditionedSwarm(_CommSwarmBase):
    """Primary Phase 2c model. Task head = Core B's terminal decoder on z_B[15].
    Parameters (d_input_B=64): 137,476 + 141,572 + 129 + 256 + 128 = 279,561."""

    def __init__(self, num_nodes: int = 16, d_input_A: int = 32, d_input_B: int = 64,
                 d_model: int = 128):
        super().__init__(d_input_B=d_input_B, d_model=d_model)

    def forward(self, x_A, x_B, steps_A: int = 8, steps_B: int = 8,
                flip_message: bool = False, zero_message: bool = False,
                comm_mode: str = "full", trace: bool = False,
                force_message: Optional[float] = None):
        if comm_mode not in COMM_MODES:
            raise ValueError(f"Unknown comm_mode: {comm_mode!r}; expected one of {COMM_MODES}")
        aux_logits_A, raw_m = self._run_core_A(x_A, steps_A)
        m = self._message(raw_m, flip_message, force_message)
        terminal, traces = self._run_core_B(x_B, m, steps_B, comm_mode, zero_message, trace)
        task_logits = self.core_B.decoder(terminal)  # [batch, 2] from z_B[15] only.
        if trace:
            return task_logits, raw_m, aux_logits_A, m, traces
        return task_logits, raw_m, aux_logits_A, m


class LatentDualRelationSwarm(_CommSwarmBase):
    """Compute-both-then-select diagnostic. Two terminal heads supervised separately for
    r0 and r1; the selector is applied only after both predictions exist. The unused
    Phase 1-style decoder is replaced by nn.Identity() (no dead parameters).
    Parameters (d_input_B=64): 137,476 + 141,058 + 129 + 256 + 128 + 2*514 = 280,075."""

    def __init__(self, num_nodes: int = 16, d_input_A: int = 32, d_input_B: int = 64,
                 d_model: int = 128):
        super().__init__(d_input_B=d_input_B, d_model=d_model)
        self.core_B.decoder = nn.Identity()  # replaced by the two relation heads below
        self.r0_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))
        self.r1_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def forward(self, x_A, x_B, steps_A: int = 8, steps_B: int = 8,
                flip_message: bool = False, zero_message: bool = False,
                comm_mode: str = "full", trace: bool = False,
                force_message: Optional[float] = None):
        if comm_mode not in COMM_MODES:
            raise ValueError(f"Unknown comm_mode: {comm_mode!r}; expected one of {COMM_MODES}")
        aux_logits_A, raw_m = self._run_core_A(x_A, steps_A)
        m = self._message(raw_m, flip_message, force_message)
        terminal, traces = self._run_core_B(x_B, m, steps_B, comm_mode, zero_message, trace)
        r0_logits = self.r0_head(terminal)  # per-relation heads: each sees z_B[15] only.
        r1_logits = self.r1_head(terminal)
        if trace:
            return r0_logits, r1_logits, raw_m, aux_logits_A, m, traces
        return r0_logits, r1_logits, raw_m, aux_logits_A, m

    @staticmethod
    def select(r0_logits, r1_logits, m):
        """Late selection: the one-bit selector chooses the candidate AFTER both heads
        have produced their predictions (never uses labels)."""
        bits = (m.squeeze(-1) > 0).long()
        chosen = torch.where(bits.bool().unsqueeze(-1), r1_logits, r0_logits)
        return chosen.argmax(dim=-1)

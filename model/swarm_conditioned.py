"""
Phase 2c: message-conditioned reachability models.

`LatentConditionedSwarm` (primary): Core A is the unchanged v7.1 reasoner (32-dim input)
sending one binary selector m in {-1, +1} (STE). Core B is a v7.1 reasoner whose encoder
accepts 64-dim inputs: [adj_union(16), adj_R0(16), adj_R1(16), one_hot(16)]. Core B runs
over a 17-token sequence (16 graph nodes + one read-only COMM register at index 16) with
the same one-way mask rules as Phase 2b (`model/swarm_comm.py`): graph queries may read
the COMM key per the visibility schedule; the COMM query attends only to itself;
the graph-to-graph mask uses the union adjacency and is independent of the message. The
task readout is Core B's existing terminal decoder applied to z_B[15] ONLY — no message
tensor, message embedding, or COMM state reaches the head.

Warm-start mapping (documented transfer, NOT an exact checkpoint load):
Core A loads the Phase 1 state dict strictly. Core B transfers every shape-compatible
Phase 1 weight (encoder LayerNorm, recurrent block, decoder); for the first encoder
linear layer, Phase 1 adjacency-input columns (0:16) map to the new union columns (0:16),
the new relation channels (16:48) start at exactly zero, and Phase 1 node-ID columns
(16:32) map to new columns (48:64); the bias is copied. With relation columns zero, the
new encoder reproduces the Phase 1 encoder output on equivalent union/ID inputs
(unit-tested).

`LatentDualRelationSwarm` (explicit diagnostic; corrected by the 2026-10-17 fix-only
pass — see reports/phase2c-fix-only-report.md): a genuine compute-both-then-select
late-selection control. Core B receives ONLY x_B and the fixed union-graph attention
mask: its entire recurrent trajectory is independent of m, force_message, flip_message
and the COMM visibility schedule, because this diagnostic has NO COMM register and NO
message embedding at all. It produces two candidate outputs r0_logits and r1_logits from
that selector-independent representation, and only afterwards uses the transmitted bit to
select between them (never using labels at inference). The Phase 1-style decoder is
replaced by nn.Identity() (no dead parameters), and the unused msg_embedder and learned
COMM token are NOT instantiated here (no unused trainable parameters); msg_encoder is
retained because the learned selector bit and its BCE supervision require it. Shared
low-level helpers (union-mask construction, Core A runner, message binariser, warm-start
mapping) are used by both models; the selector independence of the dual path is explicit
and unit-tested.

Parameter counts (d_input_B=64, verified programmatically and unit-asserted):
- LatentConditionedSwarm: 137,476 + 141,572 + 129 + 256 + 128 = 279,561.
- LatentDualRelationSwarm: 137,476 + 141,058 (Core B without decoder) + 129
  + 2*514 (r0/r1 heads) = 279,691 (no msg_embedder, no COMM token).
"""
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from model.core import LatentReasoner, binarize_ste
from model.swarm_comm import COMM_MODES, comm_step_mask


class _ConditionedBase(nn.Module):
    """Shared, selector-independent machinery: cores, message channel interface,
    union-graph mask construction, Core A runner, STE message binariser and the
    documented Phase 1 warm-start mapping. COMM-specific state lives only in the
    primary model."""

    def __init__(self, d_input_B: int, d_model: int = 128):
        super().__init__()
        self.num_nodes = 16
        self.d_model = d_model
        self.core_A = LatentReasoner(num_nodes=16, d_input=32, d_model=d_model, d_output=2)
        self.core_B = LatentReasoner(num_nodes=16, d_input=d_input_B, d_model=d_model, d_output=2)
        self.msg_encoder = nn.Linear(d_model, 1)

    def build_union_mask(self, x_B: torch.Tensor) -> torch.Tensor:
        """Graph-only attention mask over the 16 node tokens: query j may attend key i
        iff i == j or edge i -> j exists in the UNION adjacency (input columns 0:16).
        Independent of the message by construction (it never sees one)."""
        adj = x_B[:, :, :self.num_nodes]
        eye = torch.eye(self.num_nodes, dtype=torch.bool, device=x_B.device).unsqueeze(0)
        graph_graph = (adj.transpose(-1, -2) > 0.5) | eye
        return graph_graph.unsqueeze(1)  # [B, 1, 16, 16]

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


class LatentConditionedSwarm(_ConditionedBase):
    """Primary Phase 2c model. Task head = Core B's terminal decoder on z_B[15].
    Parameters (d_input_B=64): 137,476 + 141,572 + 129 + 256 + 128 = 279,561."""

    def __init__(self, num_nodes: int = 16, d_input_A: int = 32, d_input_B: int = 64,
                 d_model: int = 128):
        super().__init__(d_input_B=d_input_B, d_model=d_model)
        self.msg_embedder = nn.Linear(1, d_model)   # 256 parameters.
        self.learned_comm_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)  # 128.

    # One-way 17-token COMM masks (Phase 2b semantics) built on the shared union mask.
    def build_comm_masks(self, x_B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x_B.size(0)
        S = self.num_nodes + 1
        union = self.build_union_mask(x_B)  # [B, 1, 16, 16]
        mask_off = torch.zeros(B, 1, S, S, dtype=torch.bool, device=x_B.device)
        mask_off[:, :, :self.num_nodes, :self.num_nodes] = union
        mask_off[:, :, self.num_nodes, self.num_nodes] = True  # COMM self-loop only.
        mask_on = mask_off.clone()
        mask_on[:, :, :self.num_nodes, self.num_nodes] = True  # graph queries may read COMM.
        return mask_off, mask_on

    def build_comm_state(self, m: torch.Tensor, batch_size: int,
                         zero_message: bool = False) -> torch.Tensor:
        """COMM register initial state: learned token + embedded message. With
        zero_message=True the information-bearing message contribution is replaced by an
        exact zero before the addition (the msg_embedder — including its bias — is
        bypassed); the constant learned COMM token remains."""
        if zero_message:
            msg_repr = torch.zeros(batch_size, self.d_model, dtype=m.dtype, device=m.device)
        else:
            msg_repr = self.msg_embedder(m)
        return self.learned_comm_token.expand(batch_size, 1, self.d_model) + msg_repr.unsqueeze(1)

    def forward(self, x_A, x_B, steps_A: int = 8, steps_B: int = 8,
                flip_message: bool = False, zero_message: bool = False,
                comm_mode: str = "full", trace: bool = False,
                force_message: Optional[float] = None):
        if comm_mode not in COMM_MODES:
            raise ValueError(f"Unknown comm_mode: {comm_mode!r}; expected one of {COMM_MODES}")
        aux_logits_A, raw_m = self._run_core_A(x_A, steps_A)
        m = self._message(raw_m, flip_message, force_message)

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

        # Terminal graph-node readout: explicit index 15 (token 16 is the COMM register).
        terminal = zB[:, self.num_nodes - 1, :]
        task_logits = self.core_B.decoder(terminal)  # [batch, 2] from z_B[15] only.
        if trace:
            return task_logits, raw_m, aux_logits_A, m, traces
        return task_logits, raw_m, aux_logits_A, m


class LatentDualRelationSwarm(_ConditionedBase):
    """Compute-both-then-select diagnostic (selector-independent Core B).

    Core B's recurrent trajectory depends ONLY on x_B and the fixed union-graph mask:
    there is no COMM register and no message embedding in this model, so r0_logits and
    r1_logits are bit-identical under any message intervention. The transmitted bit is
    used exclusively by `select` AFTER both candidate outputs exist (never using labels).
    comm_mode and zero_message are accepted for interface compatibility and documented
    as no-ops here (there is no COMM schedule and no message contribution to ablate).
    Parameters (d_input_B=64): 137,476 + 141,058 + 129 + 2*514 = 279,691."""

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
        aux_logits_A, raw_m = self._run_core_A(x_A, steps_A)
        m = self._message(raw_m, flip_message, force_message)  # selection input only.

        # Selector-independent Core B: 16 graph tokens, fixed union mask, no COMM.
        union_mask = self.build_union_mask(x_B)
        zB = self.core_B.encoder(x_B)
        traces: Optional[List[torch.Tensor]] = [] if trace else None
        if trace:
            traces.append(zB.detach().clone())
        for _ in range(steps_B):
            zB = self.core_B.step(zB, union_mask)
            if trace:
                traces.append(zB.detach().clone())

        terminal = zB[:, self.num_nodes - 1, :]
        r0_logits = self.r0_head(terminal)  # each head sees z_B[15] only.
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

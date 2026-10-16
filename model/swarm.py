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

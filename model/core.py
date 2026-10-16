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

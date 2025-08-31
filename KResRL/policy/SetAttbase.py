
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, embed_dim: int, d_ff: int, num_experts: int, k: int=1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        # Experts = small FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, embed_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating network (decides which experts to use per token)
        self.gate = nn.Linear(embed_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        B, L, D = x.shape
        
        # Compute gate logits
        gate_logits = self.gate(x)  # (B, L, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Pick top-k experts
        topk_vals, topk_idx = torch.topk(gate_probs, self.k, dim=-1)  # (B, L, k)
        
        # Compute expert outputs
        out = torch.zeros_like(x)
        for i in range(self.k):
            expert_idx = topk_idx[..., i]  # (B, L)
            mask = F.one_hot(expert_idx, num_classes=self.num_experts).float()  # (B, L, num_experts)
            mask = mask.unsqueeze(-2)  # (B, L, 1, num_experts)
            
            # Run all experts in parallel
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # (B, L, D, num_experts)
            
            # Weighted sum
            out += (expert_outputs * mask).sum(dim=-1) * topk_vals[..., i].unsqueeze(-1)
        
        return out

class MAB(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True,
        used_moe: bool = False,
        num_experts: int = 4,
        moe_top_k: int = 2
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        if used_moe:
            self.ff = MoE(embed_dim, d_ff, num_experts=num_experts, k=moe_top_k)
        else:
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, d_ff),
                nn.ReLU(),   # or GELU
                nn.Linear(d_ff, embed_dim)
            )
        self.ff_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)

        # ---- Multi-Head Attention ----
        attn_out, _ = self.attn.forward(x, y, y, attn_mask=mask)  # (B, L, d_model)
        x = x + self.dropout(attn_out)   # Residual
        x = self.attn_norm(x)            # LayerNorm

        # ---- Feed Forward ----
        ff_out = self.ff(x)              # (B, L, d_model)
        x = x + self.dropout(ff_out)     # Residual
        x = self.ff_norm(x)              # LayerNorm

        return x  # (B, L, d_model)

class SAB(MAB):
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return super().forward(x, x, mask)

class ISAB(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_inducing: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True,
        used_moe: bool = False,
        num_experts: int = 4,
        moe_top_k: int = 2
    ):
        super().__init__()
        
        # Learnable inducing points (m, d)
        self.inducing_points = nn.Parameter(torch.randn(num_inducing, embed_dim))

        self.mab1 = MAB(
            embed_dim=embed_dim,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
            used_moe=used_moe,
            num_experts=num_experts,
            moe_top_k=moe_top_k
        )

        self.mab2 = MAB(
            embed_dim=embed_dim,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
            used_moe=used_moe,
            num_experts=num_experts,
            moe_top_k=moe_top_k
        )


    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, N, d)  input set
        mask: optional attention mask
        """

        B = x.shape[0]

        H = self.mab1(self.inducing_points.unsqueeze(0).expand(B, -1, -1), x, mask)  # (B, m, d)

        X = self.mab2(x, H)  # (B, N, d)

        return X


def main():
    from torchinfo import summary

    model = SAB(
        embed_dim=128,
        num_heads=8,
        d_ff=512,
        used_moe=False,
        num_experts=4,
        moe_top_k=2
    )

    summary(model, input_size=(1, 10, 128), col_names=["input_size", "output_size", "num_params", "trainable"], depth=5)


    model = ISAB(
        embed_dim=128,
        num_heads=8,
        num_inducing=32,
        d_ff=512,
        used_moe=False,
        num_experts=4,
        moe_top_k=2
    )
    summary(model, input_size=(1, 10, 128), col_names=["input_size", "output_size", "num_params", "trainable"], depth=5)

if __name__ == "__main__":
    main()

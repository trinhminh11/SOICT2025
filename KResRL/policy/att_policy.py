from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from gymnasium import spaces

from .ac_policy import NodeLevelFeatureExtractor, PolicyOptions
from .att_base import ISAB, SAB, AttBlock


class AttFeaturesExtractor(NodeLevelFeatureExtractor):
    """
    GCN-based feature extractor for Stable Baselines3

    This extracts features from graph-structured observations where:
    - obs[:, :, :n_drones] is the adjacency matrix
    - obs[:, :, n_drones:] contains node features
    """

    supported: list[AttBlock] = [SAB, ISAB]

    def __init__(
        self,
        observation_space: spaces.Box,
        encoder_hidden_dim: int = 64,
        NNBlock: AttBlock = SAB,
        n_layers: int = 2,
        embed_dim=128,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        bias: bool = True,
        used_moe: bool = False,
        num_experts: int = 4,
        moe_top_k: int = 2,
        block_kwargs: dict[str, Any] = None,
    ):
        # The features_dim is the output dimension of this feature extractor
        super().__init__(observation_space, embed_dim)

        self.n_drones = observation_space.shape[0]
        self.n_node_features = observation_space.shape[1] - self.n_drones

        # encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.n_node_features, encoder_hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(encoder_hidden_dim),  # helps stabilize before attention
            nn.Dropout(dropout, inplace=True),
            nn.Linear(encoder_hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dim),
        )

        if block_kwargs is None:
            block_kwargs = {}

        self.att_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.att_layers.append(
                NNBlock(
                    embed_dim,
                    num_heads,
                    d_ff,
                    dropout,
                    bias,
                    used_moe,
                    num_experts,
                    moe_top_k,
                    **block_kwargs,
                )
            )

        self.dropout = nn.Dropout(p=dropout) if 0 <= dropout < 1 else nn.Identity()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            observations: [batch_size, n_drones, n_drones + n_node_features]

        Returns:
            features: [batch_size, features_dim]
        """

        # Extract adjacency matrix and node features

        # don't need adj maxtrix
        X = observations[:, :, self.n_drones :]  # batch_size, n_drones, n_node_features

        X = self.encoder(X)  # batch_size, n_drones, embed_dim

        for layer in self.att_layers:
            X = layer(X)

        X = self.dropout(X)

        return X  # batch_size, n_drones, embed_dim


@dataclass(frozen=True)
class AttFeatureOptions:
    encoder_hidden_dim: int = 64
    NNBlock: AttBlock = SAB
    n_layers: int = 2
    embed_dim: int = 128
    num_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    bias: bool = True
    used_moe: bool = False
    num_experts: int = 4
    moe_top_k: int = 2
    block_kwargs: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class AttPolicyOptions(PolicyOptions):
    features_extractor_class: type[AttFeaturesExtractor] = field(
        default=AttFeaturesExtractor, init=False
    )
    features_extractor_kwargs: AttFeatureOptions = AttFeatureOptions()

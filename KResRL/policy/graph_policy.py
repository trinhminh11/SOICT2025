from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from gymnasium import spaces

from .ac_policy import NodeLevelFeatureExtractor, PolicyOptions
from .graph_base import GATBlock, GCNAdjBlock, GCNBlock, GraphBase, ResBn


class GraphFeaturesExtractor(NodeLevelFeatureExtractor):
    """
    GCN-based feature extractor for Stable Baselines3

    This extracts features from graph-structured observations where:
    - obs[:, :, :n_drones] is the adjacency matrix
    - obs[:, :, n_drones:] contains node features
    """

    supported: list[GraphBase] = [GCNBlock, GCNAdjBlock, GATBlock]

    def __init__(
        self,
        observation_space: spaces.Box,
        NNBlock: GraphBase = GCNBlock,
        hidden_dims: list[int] = [64, 64],
        used_res: list[bool] | bool = [True, False],
        norm: Optional[str | Callable] = "layer",
        norm_kwargs: Optional[dict[str, Any]] = None,
        act: Optional[str | Callable] = "relu",
        act_kwargs: Optional[dict[str, Any]] = None,
        gnn_layer_kwargs: dict[str, Any] = None,
        dropout: float = 0.1,
    ):
        # The features_dim is the output dimension of this feature extractor
        super().__init__(observation_space, hidden_dims[-1])

        self.n_drones = observation_space.shape[0]
        self.n_node_features = observation_space.shape[1] - self.n_drones
        self.hidden_dims = hidden_dims

        if isinstance(used_res, bool):
            used_res = [used_res] * len(hidden_dims)

        if len(used_res) < len(hidden_dims):
            used_res.extend([False] * (len(hidden_dims) - len(used_res)))

        # GCN layers
        self.gcn_layers = nn.ModuleList()

        if gnn_layer_kwargs is None:
            gnn_layer_kwargs = {}

        # First layer: node_features -> hidden_dim
        block = NNBlock(
            self.n_node_features,
            self.hidden_dims[0],
            norm=norm,
            norm_kwargs=norm_kwargs,
            act=act,
            act_kwargs=act_kwargs,
            **gnn_layer_kwargs,
        )
        if used_res[0]:
            block = ResBn(block)

        self.gcn_layers.append(block)

        for i in range(1, len(hidden_dims)):
            block = NNBlock(
                self.hidden_dims[i - 1],
                self.hidden_dims[i],
                norm=norm,
                norm_kwargs=norm_kwargs,
                act=act,
                act_kwargs=act_kwargs,
                **gnn_layer_kwargs,
            )
            if used_res[i]:
                block = ResBn(block)
            self.gcn_layers.append(block)

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

        X = observations[:, :, self.n_drones :]
        adj = observations[:, :, : self.n_drones]

        for i, gcn_layer in enumerate(self.gcn_layers):
            X = gcn_layer(X, adj)  # [batch_size, n_drones, hidden_dim]

        X = self.dropout(X)

        return X  # batch_size, n_drones, hidden_dims[-1]


@dataclass(frozen=True)
class GraphFeatureOptions:
    NNBlock: type[GraphBase] = GCNBlock
    hidden_dims: list[int] = field(default_factory=lambda: [64, 64])
    used_res: bool | list[bool] = True
    norm: str = "layer"
    norm_kwargs: Optional[dict[str, Any]] = None
    act: str = "relu"
    act_kwargs: Optional[dict[str, Any]] = None
    gnn_layer_kwargs: Optional[dict] = None
    dropout: float = 0.5


@dataclass(frozen=True)
class GraphPolicyOptions(PolicyOptions):
    features_extractor_class: type[GraphFeaturesExtractor] = field(
        default=GraphFeaturesExtractor, init=False
    )
    features_extractor_kwargs: GraphFeatureOptions = GraphFeatureOptions()

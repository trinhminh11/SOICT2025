"""Graph Neural Network base classes and building blocks.

This module provides a flexible framework for building Graph Neural Networks (GNNs)
with support for both dense adjacency matrices and sparse edge indices. It includes:

1. **Core Layer**: GCNConvAdj - A GCN layer that works with dense adjacency matrices
2. **Base Classes**: GraphBase and EdgeIndexBlock - Abstract bases for GNN blocks
3. **Concrete Blocks**: GCNAdjBlock, GCNBlock, GATBlock - Ready-to-use GNN layers
4. **Utilities**: Residual connections, normalization, activation handling
5. **Composition**: ResBn wrapper for adding residual connections

The design allows for easy composition of different GNN architectures while handling
the complexities of batching, normalization, and format conversion between dense
adjacency matrices and sparse edge representations.

Typical usage:
    >>> # Create a GCN block with residual connections
    >>> gcn_block = GCNBlock(input_dim, hidden_dim, norm="layer", act="relu")
    >>> gcn_with_residual = ResBn(gcn_block)
    >>>
    >>> # Use with batched input
    >>> output = gcn_with_residual(node_features, adjacency_matrix)

    >>> # Create a GAT block with multiple heads
    >>> gat_block = GATBlock(hidden_dim, output_dim, heads=8, dropout=0.1)
    >>> output = gat_block(node_features, adjacency_matrix)

For reinforcement learning applications, these blocks can be used as policy network
components when dealing with graph-structured state representations.
"""

from typing import Any, Callable, Optional

import torch
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv
from torch_geometric.nn.dense.linear import Linear as GraphLinear
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)

from KResRL.utils import to_batch


class GCNConvAdj(torch.nn.Module):
    """Graph Convolutional Network layer that operates on dense adjacency matrices.

    This implementation provides a GCN layer that takes adjacency matrices as input
    instead of edge indices, making it suitable for dense graph representations.
    It implements the standard GCN message passing with symmetric normalization.

    The layer computes: H' = D^(-1/2) * (A+I) * D^(-1/2) * H * W + b
    where:
    - A is the adjacency matrix with added self-loops
    - D is the degree matrix
    - H is the input node features
    - W is the learnable weight matrix
    - b is the optional bias term

    Args:
        in_channels (int): Number of input node features
        out_channels (int): Number of output node features
        improved (bool, optional): If True, uses improved GCN with 2*I instead of I
            for self-loops. Defaults to False.
        cached (bool, optional): If True, caches the normalized adjacency matrix.
            Currently not implemented. Defaults to False.
        bias (bool, optional): If True, adds a learnable bias. Defaults to True.

    Shape:
        - Input: (B, N, F_in) where B is batch size, N is number of nodes,
                 F_in is number of input features
        - Adjacency: (B, N, N) dense adjacency matrices
        - Output: (B, N, F_out) where F_out is number of output features
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        # Linear layer for feature transformation
        self.lin = GraphLinear(
            in_channels, out_channels, bias=False, weight_initializer="glorot"
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Cache for normalized adjacency
        self._cached_adj = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_adj = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Batched GCN forward for dense adjacency.

        Args:
            x (Tensor): Node features [B, N, F_in]
            adj (Tensor): Adjacency matrices [B, N, N]
        Returns:
            Tensor: Updated node features [B, N, F_out]
        """
        B, N, _ = x.shape
        fill_value = 2.0 if self.improved else 1.0

        # Add self-loops
        adj = adj + fill_value * torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(
            0
        )

        # Compute degree
        deg = adj.sum(dim=-1)  # [B, N]
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        # Normalize adjacency: D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)  # [B, N, N]
        adj_norm = torch.bmm(torch.bmm(D_inv_sqrt, adj), D_inv_sqrt)  # [B, N, N]

        # Apply linear transformation
        x = self.lin(x)  # [B, N, F_out]

        # Message passing
        out = torch.bmm(adj_norm, x)  # [B, N, F_out]

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias  # broadcast over batch and nodes

        return out


class EdgeInfo(torch.nn.Identity):
    """Identity layer used as a placeholder for edge information debugging.

    This class extends torch.nn.Identity and serves as a debugging utility
    for tracking edge information flow through the network. It performs
    no transformation on its input.

    This is typically used within GraphBase for development and debugging
    purposes to monitor edge-related data without affecting computation.
    """


class Residual(torch.nn.Module):
    """Residual connection module that handles dimension matching.

    This module creates a residual connection that can adapt to different
    input and output channel dimensions. If the dimensions match, it uses
    an identity mapping. If they differ, it applies a linear transformation
    to match the dimensions.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels

    Shape:
        - Input: (*, in_channels) where * means any number of dimensions
        - Output: (*, out_channels)

    Examples:
        >>> residual = Residual(64, 64)  # Identity mapping
        >>> residual = Residual(32, 64)  # Linear projection
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = (
            torch.nn.Identity()
            if in_channels == out_channels
            else GraphLinear(in_channels, out_channels)
        )

    def forward(self, x):
        """Apply residual transformation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Transformed tensor with appropriate dimensions
        """
        return self.residual(x)


class GraphBase(torch.nn.Module):
    """Abstract base class for GNN building blocks.

    This class provides a common interface for different types of GNN layers,
    handling normalization, activation, and batching logic. Concrete subclasses
    must implement the `init_conv` method to specify the actual convolution layer.

    The forward pass follows the pattern:
    1. Ensure inputs are batched (add batch dimension if needed)
    2. Transform inputs (convert adjacency to edge indices if needed)
    3. Apply convolution
    4. Apply normalization
    5. Apply activation
    6. Reshape output to match expected batch format

    Args:
        in_channels (int): Number of input node features
        out_channels (int): Number of output node features
        norm (str | Callable, optional): Normalization type or callable.
            Defaults to "layer".
        norm_kwargs (dict, optional): Additional arguments for normalization.
        act (str | Callable, optional): Activation function type or callable.
            Defaults to "relu".
        act_kwargs (dict, optional): Additional arguments for activation.
        **kwargs: Additional arguments passed to the convolution layer.

    Shape:
        - Input: (B, N, F_in) or (N, F_in) where B is batch size, N is number of nodes
        - Adjacency: (B, N, N) or (N, N) dense adjacency matrices
        - Output: (B, N, F_out)

    Note:
        Subclasses must implement `init_conv` to return the appropriate convolution layer.
        The `transform` method can be overridden to preprocess inputs (e.g., convert
        adjacency matrices to edge indices).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: Optional[str | Callable] = "layer",
        norm_kwargs: Optional[dict[str, Any]] = None,
        act: Optional[str | Callable] = "relu",
        act_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        self.edge_debuger = EdgeInfo()

        kwargs = kwargs | {"bias": norm is None}

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm = normalization_resolver(
            norm,
            out_channels,
            **(norm_kwargs or {}),
        )

        if self.norm is None:
            self.norm = torch.nn.Identity()

        self.act = activation_resolver(act, **(act_kwargs or {}))
        if self.act is None:
            self.act = torch.nn.Identity()

        self.conv = self.init_conv(in_channels, out_channels, **kwargs)

    def __one_to_batch(self, x, adjacency_matrix):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if adjacency_matrix.dim() == 2:
            adjacency_matrix = adjacency_matrix.unsqueeze(0)
        return x, adjacency_matrix

    def transform(self, x: torch.Tensor, adjacency_matrix: torch.Tensor) -> Any:
        return x, adjacency_matrix

    def init_conv(
        self, in_channels: int | tuple[int, int], out_channels: int, **kwargs
    ) -> torch.nn.Module:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor):
        """Forward pass for the block.

        Args:
            x (torch.Tensor): Node features.
            adjacency_matrix (torch.Tensor): adjacency matrix.

        Returns:
            _type_: _description_
        """

        x, adjacency_matrix = self.__one_to_batch(x, adjacency_matrix)

        B, N, _ = x.shape

        x, edge_info = self.transform(x, adjacency_matrix)

        X: torch.Tensor = self.conv(x, edge_info)  # [B*N, out_channels]
        X = self.norm(X)
        X = self.act(X)

        if X.dim() == 2:
            X = X.view(B, N, self.out_channels)

        return X


class GCNAdjBlock(GraphBase):
    """GCN block that operates directly on dense adjacency matrices.

    This block uses the custom GCNConvAdj layer to perform graph convolution
    on dense adjacency matrices without converting to edge indices. This is
    more efficient for dense graphs or when adjacency matrices are already
    available in dense format.

    Inherits all parameters from GraphBase. Additional kwargs are passed
    to GCNConvAdj (e.g., improved, cached, bias).

    Examples:
        >>> block = GCNAdjBlock(64, 128, improved=True)
        >>> output = block(node_features, adjacency_matrix)
    """

    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return GCNConvAdj(in_channels, out_channels, **kwargs)


class EdgeIndexBlock(GraphBase):
    """Base block for PyTorch Geometric layers that use edge indices.

    This class handles the conversion from dense adjacency matrices to
    edge indices, which are required by most PyTorch Geometric layers.
    It uses the `to_batch` utility function to create a PyG batch object.

    The transform method converts the dense adjacency representation to
    sparse edge indices format, allowing the use of standard PyG layers.

    Note:
        This is an intermediate base class. Use concrete implementations
        like GCNBlock or GATBlock for actual usage.
    """

    def transform(self, x: torch.Tensor, adjacency_matrix: torch.Tensor) -> Any:
        """Convert adjacency matrix to edge indices.

        Args:
            x (torch.Tensor): Node features
            adjacency_matrix (torch.Tensor): Dense adjacency matrix

        Returns:
            tuple: (node_features, edge_index) suitable for PyG layers
        """
        batch = to_batch(x, adjacency_matrix)
        return batch.x, batch.edge_index


class GCNBlock(EdgeIndexBlock):
    """GCN block using PyTorch Geometric's GCNConv layer.

    This block converts dense adjacency matrices to edge indices and then
    applies the standard PyTorch Geometric GCNConv layer. It's suitable
    when you want to use the optimized sparse implementations from PyG.

    Inherits all parameters from GraphBase. Additional kwargs are passed
    to GCNConv (e.g., improved, cached, bias, add_self_loops).

    Examples:
        >>> block = GCNBlock(64, 128, improved=True, cached=True)
        >>> output = block(node_features, adjacency_matrix)
    """

    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return GCNConv(in_channels, out_channels, **kwargs)


class GATBlock(EdgeIndexBlock):
    """Graph Attention Network block using PyTorch Geometric's GAT layers.

    This block converts dense adjacency matrices to edge indices and applies
    Graph Attention Network convolution. It supports both GATConv and GATv2Conv
    with configurable attention heads, concatenation, and dropout.

    Inherits all parameters from GraphBase. Additional GAT-specific parameters:

    Args:
        v2 (bool, optional): If True, uses GATv2Conv instead of GATConv.
            Defaults to False.
        heads (int, optional): Number of attention heads. Defaults to 1.
        concat (bool, optional): If True, concatenate multi-head outputs.
            If False, average them. Defaults to True.
        dropout (float, optional): Dropout rate for attention coefficients.
            Defaults to 0.0.

    Note:
        When concat=True, the output channels must be divisible by the number
        of heads. The layer will automatically adjust the per-head output size.

    Examples:
        >>> block = GATBlock(64, 128, heads=4, dropout=0.1)
        >>> block = GATBlock(64, 128, v2=True, heads=8, concat=False)
        >>> output = block(node_features, adjacency_matrix)
    """

    def init_conv(
        self, in_channels: int | tuple[int, int], out_channels: int, **kwargs
    ) -> torch.nn.Module:
        """Initialize the GAT convolution layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            **kwargs: Additional arguments including GAT-specific parameters

        Returns:
            torch.nn.Module: Configured GAT layer (GATConv or GATv2Conv)

        Raises:
            ValueError: If output channels not divisible by number of heads
                when concatenation is enabled
        """
        v2 = kwargs.pop("v2", False)
        heads = kwargs.pop("heads", 1)
        concat = kwargs.pop("concat", True)
        dropout = kwargs.pop("dropout", 0.0)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):

        if concat and out_channels % heads != 0:
            raise ValueError(
                f"Ensure that the number of output channels of "
                f"'GATConv' (got '{out_channels}') is divisible "
                f"by the number of heads (got '{heads}')"
            )

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(
            in_channels,
            out_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            **kwargs,
        )


class ResBn(torch.nn.Module):
    """Residual wrapper that adds skip connections to any GNN block.

    This class wraps any GNN module and adds residual connections, implementing
    the ResNet-style skip connection pattern: output = module(x) + residual(x).
    The residual connection is adaptively handled based on input/output dimensions.

    Args:
        module (torch.nn.Module): The GNN module to wrap. Must have `in_channels`
            and `out_channels` attributes.
        skip_if_different (bool, optional): If True and input/output dimensions
            differ, the residual connection is disabled (set to 0). If False,
            a linear projection is used to match dimensions. Defaults to False.

    Shape:
        - Input: Same as the wrapped module
        - Output: Same as the wrapped module

    Examples:
        >>> gcn_block = GCNBlock(64, 64)
        >>> res_gcn = ResBn(gcn_block)  # Identity residual
        >>>
        >>> gcn_block = GCNBlock(32, 64)
        >>> res_gcn = ResBn(gcn_block)  # Linear projection residual
        >>>
        >>> res_gcn = ResBn(gcn_block, skip_if_different=True)  # No residual
    """

    def __init__(self, module: torch.nn.Module, skip_if_different=False):
        super().__init__()
        self.module = module

        if skip_if_different and module.in_channels != module.out_channels:
            self.residual = lambda x: 0
        else:
            self.residual = Residual(module.in_channels, module.out_channels)

    def forward(self, x, adjacency_matrix):
        """Forward pass with residual connection.

        Args:
            x (torch.Tensor): Input node features
            adjacency_matrix (torch.Tensor): Adjacency matrix

        Returns:
            torch.Tensor: Output with residual connection applied
        """
        res = self.residual(x)
        X = self.module(x, adjacency_matrix)
        X = X + res
        return X


def main():
    from torchinfo import summary

    batch_size = 3
    n_node = 3
    node_features = 10
    X = torch.randn((batch_size, n_node, node_features))

    adj = torch.randn((batch_size, n_node, n_node))
    adj = torch.zeros((batch_size, n_node, n_node))
    adj[0][0][0] = 1
    adj[2][0][1] = 1

    model = ResBn(GCNBlock(node_features, 64), skip_if_different=False)

    summary(
        model,
        input_data=(X, adj),
        col_names=[
            "input_size",
            "output_size",
            "params_percent",
            "trainable",
        ],
    )

    print(model(X, adj).shape)


if __name__ == "__main__":
    main()

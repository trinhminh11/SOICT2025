from typing import Callable, Optional, Any
import torch
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv


from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)

from utils import to_batch

from torch_geometric.nn.inits import zeros
from torch_geometric.nn.dense.linear import Linear as GraphLinear

class GCNConvAdj(torch.nn.Module):
    """GCN layer taking adjacency matrix instead of edge_index."""

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        # Linear layer for feature transformation
        self.lin = GraphLinear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

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
        adj = adj + fill_value * torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0)

        # Compute degree
        deg = adj.sum(dim=-1)  # [B, N]
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

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
    pass

class Residual(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = torch.nn.Identity() if in_channels == out_channels else GraphLinear(in_channels, out_channels)
    def forward(self, x):
        return self.residual(x)

class BlockBase(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: Optional[str | Callable] = "layer",
        norm_kwargs: Optional[dict[str, Any]] = None,
        act: Optional[str | Callable] = "relu",
        act_kwargs: Optional[dict[str, Any]] = None,
        **kwargs
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

    def init_conv(self, in_channels: int | tuple[int, int],
                out_channels: int, **kwargs) -> torch.nn.Module:
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

        X: torch.Tensor = self.conv(x, edge_info)    # [B*N, out_channels]
        X = self.norm(X)
        X = self.act(X)

        if X.dim() == 2:
            X = X.view(B, N, self.out_channels)

        return X
    
class GCNAdjBlock(BlockBase):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return GCNConvAdj(in_channels, out_channels, **kwargs)

class EdgeIndexBlock(BlockBase):
    def transform(self, x: torch.Tensor, adjacency_matrix: torch.Tensor) -> Any:
        batch = to_batch(x, adjacency_matrix)
        return batch.x, batch.edge_index

class GCNBlock(EdgeIndexBlock):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return GCNConv(in_channels, out_channels, **kwargs)

class GATBlock(EdgeIndexBlock):
    def init_conv(self, in_channels: int | tuple[int, int],
                  out_channels: int, **kwargs) -> torch.nn.Module:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)
        dropout = kwargs.pop('dropout', 0.0)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=dropout, **kwargs)

class ResBn(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, skip_if_different=False):
        super().__init__()
        self.module = module

        if skip_if_different and module.in_channels != module.out_channels:
            self.residual = lambda x: 0
        else:
            self.residual = Residual(module.in_channels, module.out_channels)

    def forward(self, x, adjacency_matrix):
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

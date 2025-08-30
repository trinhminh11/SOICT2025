import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_edge_index

def to_batch(x: torch.Tensor, adj: torch.Tensor) -> Batch:
    B = x.shape[0]
    assert adj.dim() == 3
    assert adj.shape[0] == B, f"Expected {B} batches, got {adj.shape[0]}"

    edge_indices, edge_attrs = to_edge_index(adj.to_sparse())

    data_list: list[Data] = []

    for i in range(B):
        mask = (edge_indices[0] == i)

        data_list.append(Data(
            x = x[i],
            edge_index = edge_indices[1:, mask],
            edge_attr = edge_attrs[mask]
        ))

    batch = Batch.from_data_list(data_list)

    return batch



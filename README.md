# DropPath
Implementation of DropPath: A Structured Dropout for \\Graph Neural Networks in PyTorch.

# Usage
Currently we have made a simple implementation of DropPath based on PyTorch Geometric.
```python
import torch
import torch_cluster

def drop_path(edge_index, r: float = 0.5,
              walks_per_node: int = 2,
              walk_length: int = 4,
              p: float = 1, q: float = 1, 
              num_nodes=None):

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    row, col = edge_index
    deg = torch.zeros(num_nodes, device=row.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=row.device))
    num_starts = int(r * num_nodes)
    start = torch.randperm(num_nodes, device=edge_index.device)[:num_starts]

    if walks_per_node:
        start = start.repeat(walks_per_node)

    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    n_id, e_id = torch.ops.torch_cluster.random_walk(rowptr, col, start, walk_length, p, q)
    mask = row.new_ones(row.size(0), dtype=torch.bool)
    mask[e_id.view(-1)] = False
    return edge_index[:, mask]
```

DropPath can be seamlessly integrated into PyG based codes.
```python
# We support PyG like input
edge_index = drop_edge(edge_index)
```

# Cite


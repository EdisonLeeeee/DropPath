import torch
import torch_cluster

def drop_edge(edge_index, p: float = 0.7):
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask]

def drop_path(edge_index, r: float = 0.5,
              walks_per_node: int = 2,
              walk_length: int = 4,
              p: float = 1, q: float = 1, 
              num_nodes=None,
              by='degree'):

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    assert by in {'degree', 'uniform'}

    row, col = edge_index
    deg = torch.zeros(num_nodes, device=row.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=row.device))

    if isinstance(r, float):
        assert 0 < r <= 1
        num_starts = int(r * num_nodes)
        if by == 'degree':
            prob = deg.float() / deg.sum()
            start = prob.multinomial(num_samples=num_starts, replacement=True)
        else:
            start = torch.randperm(num_nodes, device=edge_index.device)[:num_starts]
    elif torch.is_tensor(r):
        start = r.to(edge_index)
    else:
        raise ValueError(r)

    if walks_per_node:
        start = start.repeat(walks_per_node)

    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    n_id, e_id = torch.ops.torch_cluster.random_walk(rowptr, col, start, walk_length, p, q)
    mask = row.new_ones(row.size(0), dtype=torch.bool)
    mask[e_id.view(-1)] = False
    return edge_index[:, mask]

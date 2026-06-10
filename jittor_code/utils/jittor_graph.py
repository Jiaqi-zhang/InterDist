"""Small graph layers used to replace torch_geometric pieces."""

from .jittor_compat import jt, nn


class SAGEConv(nn.Module):
    """Mean-aggregation GraphSAGE convolution for fixed edge_index graphs."""

    def __init__(self, in_channels, out_channels, project=True, bias=True):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)

    def execute(self, x, edge_index):
        # x: (num_nodes, channels), edge_index: (2, num_edges) with source -> target.
        src, dst = edge_index[0].int32(), edge_index[1].int32()
        neigh = jt.zeros_like(x)
        deg = jt.zeros((x.shape[0], 1), dtype=x.dtype)
        neigh = neigh.scatter(0, dst.unsqueeze(-1).expand(-1, x.shape[1]), x[src], reduce="add")
        deg = deg.scatter(0, dst.unsqueeze(-1), jt.ones((dst.shape[0], 1), dtype=x.dtype), reduce="add")
        neigh = neigh / deg.maximum(1.0)
        return self.lin_self(x) + self.lin_neigh(neigh)

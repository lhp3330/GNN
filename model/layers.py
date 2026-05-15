from typing import Callable

import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops


class MolConv(MessagePassing):
    def __init__(self, aggr='add'):
        super().__init__(aggr=aggr)  # 'add', 'mean' or 'max'

    def message(self, x_j: Tensor, edge_attr: OptTensor, edge_weight: OptTensor = None) -> Tensor:
        if edge_attr is None:
            if edge_weight is None:
                return x_j
            else:
                return edge_weight.view(-1, 1) * x_j
        else:
            if edge_weight is None:
                return x_j + edge_attr
            else:
                return edge_weight.view(-1, 1) * (x_j + edge_attr)

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out


class WeightedGCNConv(MolConv):
    def __init__(self, in_channels: int, out_channels: int, bias: bool, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = True
        self.improved = False

        self.lin = Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        edge_index = remove_self_loops(edge_index=edge_index)[0]
        _, edge_attr = add_remaining_self_loops(edge_index, edge_attr, fill_value=1, num_nodes=x.shape[0])

        edge_index, edge_weight = gcn_norm(
            edge_index, edge_weight, x.size(self.node_dim),
            self.improved, self.add_self_loops, self.flow, x.dtype)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None, edge_attr=edge_attr)
        out = self.lin(out)
        return out


class WeightedGINConv(MolConv):
    def __init__(self, in_channels: int, out_channels: int, bias: bool, mlp_func: Callable):
        super(WeightedGINConv, self).__init__(aggr="add")

        self.mlp = mlp_func(in_channels=in_channels, out_channels=out_channels, bias=bias)
        self.eps = torch.Tensor([0])
        self.eps = torch.nn.Parameter(self.eps)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None, edge_attr=edge_attr)

        return self.mlp((1 + self.eps.to(x.device)) * x + out)


class WeightedGNNConv(MolConv):
    def __init__(self, in_channels: int, out_channels: int, aggr='add', bias=True):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(2 * in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight)
        out = self.lin(torch.cat((x, out), dim=-1))
        return out


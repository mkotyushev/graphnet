from typing import Callable, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Sigmoid

from torch_geometric.nn.conv import ResGatedGraphConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, PairTensor


class ResGatedGraphConvEdge(ResGatedGraphConv):
    r"""Modified version of the ResGatedGraphConv layer from PyG that allows for
    edge features to be used in the message passing step.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        act: Optional[Callable] = Sigmoid(),
        root_weight: bool = True,
        bias: bool = True,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, act, root_weight, bias, **kwargs)

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, out_channels)
        else:
            self.lin_edge = None
        
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # hasattr to avoid AttributeError due to 
        # reset_parameters called in super().__init__
        if hasattr(self, 'lin_edge') and self.lin_edge is not None:
            self.lin_edge.reset_parameters()

    def forward(
        self, 
        x: Union[Tensor, PairTensor], 
        edge_index: Adj, 
        edge_attr: Optional[Tensor] = None
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        k = self.lin_key(x[1])
        q = self.lin_query(x[0])
        v = self.lin_value(x[0])
        if self.lin_edge is not None:
            e = self.lin_edge(edge_attr)

        # propagate_type: (k: Tensor, q: Tensor, v: Tensor)
        out = self.propagate(edge_index, k=k, q=q, v=v, size=None, e=e)

        if self.root_weight:
            out = out + self.lin_skip(x[1])

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, k_i: Tensor, q_j: Tensor, v_j: Tensor, e_ij: Optional[Tensor] = None) -> Tensor:
        if e_ij is not None:
            return self.act(k_i + q_j + e_ij) * v_j
        else:
            return self.act(k_i + q_j) * v_j

"""Class(es) implementing layers to be used in `graphnet` models."""

import torch

from typing import Any, Callable, Optional, Sequence, Union
from torch.functional import Tensor
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import knn_graph
from torch_geometric.typing import Adj, PairTensor
from pytorch_lightning import LightningModule



class EdgeConvEdge(EdgeConv):
    r"""EdgeConv with edge attributes.
    """
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Optional[Tensor] = None) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None, edge_attr=edge_attr)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr=None) -> Tensor:
        if edge_attr is not None:
            return self.nn(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))
        else:
            return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))


class DynEdgeConv(EdgeConvEdge, LightningModule):
    """Dynamical edge convolution layer."""

    def __init__(
        self,
        nn: Callable,
        aggr: str = "max",
        nb_neighbors: int = 8,
        features_subset: Optional[Union[Sequence[int], slice]] = None,
        **kwargs: Any,
    ):
        """Construct `DynEdgeConv`.

        Args:
            nn: The MLP/torch.Module to be used within the `EdgeConv`.
            aggr: Aggregation method to be used with `EdgeConv`.
            nb_neighbors: Number of neighbours to be clustered after the
                `EdgeConv` operation.
            features_subset: Subset of features in `Data.x` that should be used
                when dynamically performing the new graph clustering after the
                `EdgeConv` operation. Defaults to all features.
            **kwargs: Additional features to be passed to `EdgeConv`.
        """
        # Check(s)
        if features_subset is None:
            features_subset = slice(None)  # Use all features
        assert isinstance(features_subset, (list, slice))

        # Base class constructor
        super().__init__(nn=nn, aggr=aggr, **kwargs)

        # Additional member variables
        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset

    def forward(
        self, 
        x: Tensor, 
        edge_index: Adj, 
        batch: Optional[Tensor] = None, 
        edge_attr: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass."""
        # Standard EdgeConv forward pass
        x = super().forward(x, edge_index, edge_attr=edge_attr)

        # Recompute adjacency
        edge_index = knn_graph(
            x=x[:, self.features_subset],
            k=self.nb_neighbors,
            batch=batch,
        ).to(self.device)

        return x, edge_index

from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.conv import GPSConv
from torch_geometric.nn.pool import knn_graph
from torch_geometric.typing import Adj

from graphnet.models.detector.detector import Detector


class DynGPSConv(GPSConv):
    r"""GPS with dynamic graph rebuilding.
    """
    def __init__(
        self,
        nb_neighbors: int,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nb_neighbors = nb_neighbors

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        
        # Run the conv forward pass
        x = super().forward(x, edge_index, batch, **kwargs)
        
        # Recompute adjacency
        # Note: if edge_attr is used, it need to be
        # recomputed based on first conv input
        # because edge_index is updated
        edge_index = knn_graph(
            x=x,
            k=self.nb_neighbors,
            batch=batch,
        ).to(x.device)

        return x, edge_index

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')

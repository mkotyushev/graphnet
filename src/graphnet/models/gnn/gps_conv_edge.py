from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.conv import GPSConv


class GPSConvEdge(GPSConv):
    r"""GPSConv with explicit edge_attr handling.
    """
    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h, edge_attr = self.conv(x, edge_index, edge_attr=edge_attr, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Global attention transformer-style model.
        h, mask = to_dense_batch(x, batch)
        h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                h = self.norm3(h, batch=batch)
            else:
                h = self.norm3(h)

        return out, edge_attr

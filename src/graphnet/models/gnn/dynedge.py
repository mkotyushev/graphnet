"""Implementation of the DynEdge GNN model architecture."""
from collections import OrderedDict
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum
from torch_geometric.nn.conv import GPSConv, GINEConv

from graphnet.models.components.layers import DynEdgeConv
from graphnet.utilities.config import save_model_config
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily
from simplex.models.simplex_models import Linear as SimplexLinear, BatchNorm1d as SimplexBatchNorm1d


GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}


def remap_values(remapping: LongTensor, x: LongTensor) -> LongTensor:
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)


class Float32BatchNorm1d(torch.nn.BatchNorm1d):
    def forward(self, input):
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.float32):
            return super().forward(input)


class LinearBlock(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        linear_builder: Optional[Union[Type, Callable]] = None,
        activation_builder: Optional[Union[Type, Callable]] = None,
        dropout: Optional[float] = None,
        bn_builder: Optional[Union[Type, Callable]] = None,
    ):
        """Construct `LinearBlock`."""
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if linear_builder is None:
            linear_builder = torch.nn.Linear

        self.linear = linear_builder(in_features, out_features, bias=bias)
        
        self.activation = None
        if activation_builder is not None:
            self.activation = activation_builder()

        self.dropout = None
        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)

        self.bn = None
        if bn_builder is not None:
            self.bn = bn_builder(out_features)
        

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class DynEdge(GNN):
    """DynEdge (dynamical edge convolutional) model."""

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_edge_attrs: int = 0,
        nb_neighbours: int = 8,
        features_subset: Optional[Union[List[int], slice]] = None,
        dynedge_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        post_processing_layer_sizes: Optional[List[int]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        global_pooling_schemes: Optional[Union[str, List[str]]] = None,
        add_global_variables_after_pooling: bool = False,
        fix_points: Optional[List[bool]] = None,
        bias: bool = True,
        bn: bool = False,
        dropout: Optional[float] = None,
        repeat_input: int = 1,
        gps: bool = False,
        gps_heads: int = 1,
    ):
        """Construct `DynEdge`.

        Args:
            nb_inputs: Number of input features on each node.
            nb_edge_attrs: Number of edge attributes
            nb_neighbours: Number of neighbours to used in the k-nearest
                neighbour clustering which is performed after each (dynamical)
                edge convolution.
            features_subset: The subset of latent features on each node that
                are used as metric dimensions when performing the k-nearest
                neighbours clustering. Defaults to [0,1,2].
            dynedge_layer_sizes: The layer sizes, or latent feature dimenions,
                used in the `DynEdgeConv` layer. Each entry in
                `dynedge_layer_sizes` corresponds to a single `DynEdgeConv`
                layer; the integers in the corresponding tuple corresponds to
                the layer sizes in the multi-layer perceptron (MLP) that is
                applied within each `DynEdgeConv` layer. That is, a list of
                size-two tuples means that all `DynEdgeConv` layers contain a
                two-layer MLP.
                Defaults to [(128, 256), (336, 256), (336, 256), (336, 256)].
            post_processing_layer_sizes: Hidden layer sizes in the MLP
                following the skip-concatenation of the outputs of each
                `DynEdgeConv` layer. Defaults to [336, 256].
            readout_layer_sizes: Hidden layer sizes in the MLP following the
                post-processing _and_ optional global pooling. As this is the
                last layer(s) in the model, the last layer in the read-out
                yields the output of the `DynEdge` model. Defaults to [128,].
            global_pooling_schemes: The list global pooling schemes to use.
                Options are: "min", "max", "mean", and "sum".
            add_global_variables_after_pooling: Whether to add global variables
                after global pooling. The alternative is to  added (distribute)
                them to the individual nodes before any convolutional
                operations.
            bias: if `True`, then the layers use bias weights else they don't.
            bn: if `True`, then the layers use batch normalization else they
                don't.
            dropout: if not `None`, then the layers use dropout with the
                specified probability.
            repeat_input: Number of times to repeat the input features in 
                preprocess layer input.
            gps: if `True`, then the model uses the GPS conv
                with DyneEdgeConv as local model, otherwise it uses the
                DyneEdgeConv conv.
            gps_heads: number of heads in the GPS conv.
        """
        self.dropout_builder, self.dropout, self.bn_builder = None, None, None
        if fix_points is not None:
            if bn:
                # TODO: replace with to autocast version
                self.bn_builder = lambda *args, **kwargs: SimplexBatchNorm1d(
                    *args, **kwargs, fix_points=fix_points
                )
            self.linear_builder = lambda *args, **kwargs: SimplexLinear(
                *args, **kwargs, fix_points=fix_points
            )
        else:
            if bn:
                self.bn_builder = Float32BatchNorm1d
            self.linear_builder = torch.nn.Linear
        if dropout is not None:
            self.dropout_builder = torch.nn.Dropout
            self.dropout = dropout
        self.activation_builder = torch.nn.LeakyReLU
        # Latent feature subset for computing nearest neighbours in DynEdge.
        if features_subset is None:
            features_subset = slice(0, 3)

        # DynEdge layer sizes
        if dynedge_layer_sizes is None:
            dynedge_layer_sizes = [
                (
                    128,
                    256,
                ),
                (
                    336,
                    256,
                ),
                (
                    336,
                    256,
                ),
                (
                    336,
                    256,
                ),
            ]

        assert isinstance(dynedge_layer_sizes, list)
        assert len(dynedge_layer_sizes)
        assert all(isinstance(sizes, tuple) for sizes in dynedge_layer_sizes)
        assert all(len(sizes) > 0 for sizes in dynedge_layer_sizes)
        assert all(
            all(size > 0 for size in sizes) for sizes in dynedge_layer_sizes
        )

        self._dynedge_layer_sizes = dynedge_layer_sizes

        # Post-processing layer sizes
        if post_processing_layer_sizes is None:
            post_processing_layer_sizes = [
                336,
                256,
            ]

        assert isinstance(post_processing_layer_sizes, list)
        assert len(post_processing_layer_sizes)
        assert all(size > 0 for size in post_processing_layer_sizes)

        self._post_processing_layer_sizes = post_processing_layer_sizes

        # Read-out layer sizes
        if readout_layer_sizes is None:
            readout_layer_sizes = [
                128,
            ]

        assert isinstance(readout_layer_sizes, list)
        assert len(readout_layer_sizes)
        assert all(size > 0 for size in readout_layer_sizes)

        self._readout_layer_sizes = readout_layer_sizes

        self._gps = gps
        self._gps_heads = gps_heads
        # Check that all the hidden dimentions are the same
        if gps:
            self._gps_hidden_size = dynedge_layer_sizes[0][0]
            assert all(
                sizes[0] == self._gps_hidden_size and sizes[1] == self._gps_hidden_size
                for sizes in dynedge_layer_sizes
            ), \
                f'If using GPS, all conv hidden dimensions must be the same. ' \
                f'Got {dynedge_layer_sizes}'
            assert all(
                size == self._gps_hidden_size for size in post_processing_layer_sizes
            ), \
                f'If using GPS, all post-processing hidden dimensions must be the same. ' \
                f'Got {post_processing_layer_sizes}'
            assert all(
                size == self._gps_hidden_size for size in readout_layer_sizes
            ), \
                f'If using GPS, all readout hidden dimensions must be the same. ' \
                f'Got {readout_layer_sizes}'

        # Global pooling scheme(s)
        if isinstance(global_pooling_schemes, str):
            global_pooling_schemes = [global_pooling_schemes]

        if isinstance(global_pooling_schemes, list):
            for pooling_scheme in global_pooling_schemes:
                assert (
                    pooling_scheme in GLOBAL_POOLINGS
                ), f"Global pooling scheme {pooling_scheme} not supported."
        else:
            assert global_pooling_schemes is None

        self._global_pooling_schemes = global_pooling_schemes

        if add_global_variables_after_pooling:
            assert self._global_pooling_schemes, (
                "No global pooling schemes were request, so cannot add global"
                " variables after pooling."
            )
        self._add_global_variables_after_pooling = (
            add_global_variables_after_pooling
        )

        self.repeat_input = repeat_input

        # Base class constructor
        super().__init__(nb_inputs, self._readout_layer_sizes[-1])

        # Remaining member variables()
        self._nb_inputs = nb_inputs
        self._nb_edge_attrs = nb_edge_attrs
        self._nb_global_variables = 5 + nb_inputs
        self._nb_neighbours = nb_neighbours
        self._features_subset = features_subset
        self._bias = bias

        self._construct_layers()

    def _construct_layers(self) -> None:
        """Construct layers (torch.nn.Modules)."""
        nb_input_features = self._nb_inputs
        if not self._add_global_variables_after_pooling:
            nb_input_features += self._nb_global_variables

        # Input processing for GPS
        self.node_emb, self.edge_emb, self.pe_lin = None, None, None
        if self._gps:
            self.node_emb = torch.nn.Linear(nb_input_features, self._gps_hidden_size)
            self.pe_lin = torch.nn.Linear(20, self._gps_hidden_size)
            self.edge_emb = torch.nn.Linear(self._nb_edge_attrs, self._gps_hidden_size)

        # Convolutional operations
        self._conv_layers = torch.nn.ModuleList()
        nb_latent_features = nb_input_features
        nb_edge_attrs = self._nb_edge_attrs
        for conv_ix, sizes in enumerate(self._dynedge_layer_sizes):
            layers = []
            layer_sizes = []
            if self._gps:
                layer_sizes.append(self._gps_hidden_size)
            else:
                layer_sizes.append(nb_latent_features)
            layer_sizes = layer_sizes + list(sizes)
            for ix, (nb_in, nb_out) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
            ):
                if ix == 0 and not self._gps:
                    nb_in *= 2
                    if conv_ix == 0:
                        nb_in += nb_edge_attrs
                linear_block = LinearBlock(
                    nb_in,
                    nb_out,
                    bias=self._bias,
                    linear_builder=self.linear_builder,
                    bn_builder=None,  # no batch norm for local conv
                    activation_builder=self.activation_builder,
                    dropout=self.dropout,
                )
                layers.append(linear_block)
            nn = torch.nn.Sequential(*layers)
            if not self._gps:
                conv_layer = DynEdgeConv(
                    nn,
                    aggr="add",
                    nb_neighbors=self._nb_neighbours,
                    features_subset=self._features_subset,
                )
            else:
                conv_layer = GPSConv(
                    self._gps_hidden_size,
                    GINEConv(nn, edge_dim=self._gps_hidden_size),
                    heads=self._gps_heads,
                )
            self._conv_layers.append(conv_layer)

            nb_latent_features = nb_out

        # Post-processing operations
        if self._gps:
            post_processing_input_cat_size = self._gps_hidden_size
        else:
            post_processing_input_cat_size = nb_input_features
        nb_latent_features = (
            sum(sizes[-1] for sizes in self._dynedge_layer_sizes)
            + post_processing_input_cat_size * self.repeat_input
        )

        post_processing_layers = []
        layer_sizes = [nb_latent_features] + list(
            self._post_processing_layer_sizes
        )
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            linear_block = LinearBlock(
                nb_in,
                nb_out,
                bias=self._bias,
                linear_builder=self.linear_builder,
                bn_builder=self.bn_builder,
                activation_builder=self.activation_builder,
                dropout=self.dropout,
            )
            post_processing_layers.append(linear_block)

        self._post_processing = torch.nn.Sequential(*post_processing_layers)

        # Read-out operations
        nb_poolings = (
            len(self._global_pooling_schemes)
            if self._global_pooling_schemes
            else 1
        )
        nb_latent_features = nb_out * nb_poolings
        if self._add_global_variables_after_pooling:
            nb_latent_features += self._nb_global_variables

        readout_layers = []
        layer_sizes = [nb_latent_features] + list(self._readout_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            linear_block = LinearBlock(
                nb_in,
                nb_out,
                bias=self._bias,
                linear_builder=self.linear_builder,
                bn_builder=self.bn_builder,
                activation_builder=self.activation_builder,
                dropout=self.dropout,
            )
            readout_layers.append(linear_block)

        self._readout = torch.nn.Sequential(*readout_layers)

    def _global_pooling(self, x: Tensor, batch: LongTensor) -> Tensor:
        """Perform global pooling."""
        assert self._global_pooling_schemes
        pooled = []
        for pooling_scheme in self._global_pooling_schemes:
            pooling_fn = GLOBAL_POOLINGS[pooling_scheme]
            pooled_x = pooling_fn(x, index=batch, dim=0)
            if isinstance(pooled_x, tuple) and len(pooled_x) == 2:
                # `scatter_{min,max}`, which return also an argument, vs.
                # `scatter_{mean,sum}`
                pooled_x, _ = pooled_x
            pooled.append(pooled_x)

        return torch.cat(pooled, dim=1)

    def _calculate_global_variables(
        self,
        x: Tensor,
        edge_index: LongTensor,
        batch: LongTensor,
        *additional_attributes: Tensor,
    ) -> Tensor:
        """Calculate global variables."""
        # Calculate homophily (scalar variables)
        h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        # Calculate mean features
        global_means = scatter_mean(x, batch, dim=0)

        # Add global variables
        global_variables = torch.cat(
            [
                global_means,
                h_x,
                h_y,
                h_z,
                h_t,
            ]
            + [attr.unsqueeze(dim=1) for attr in additional_attributes],
            dim=1,
        )

        return global_variables

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        # Convenience variables
        x, edge_index, batch, n_pulses, edge_attr = \
            data.x, data.edge_index, data.batch, data.n_pulses, data.edge_attr

        global_variables = self._calculate_global_variables(
            x,
            edge_index,
            batch,
            torch.log10(n_pulses),
        )

        # Distribute global variables out to each node
        if not self._add_global_variables_after_pooling:
            distribute = (
                batch.unsqueeze(dim=1) == torch.unique(batch).unsqueeze(dim=0)
            ).type(torch.float)

            global_variables_distributed = torch.sum(
                distribute.unsqueeze(dim=2)
                * global_variables.unsqueeze(dim=0),
                dim=1,
            )

            x = torch.cat((x, global_variables_distributed), dim=1)
        
        if self._gps:
            pe = data.pe
            x = self.node_emb(x.squeeze(-1)) + self.pe_lin(pe)
            edge_attr = self.edge_emb(edge_attr)

        # DynEdge-convolutions
        skip_connections = [x] * self.repeat_input
        for i, conv_layer in enumerate(self._conv_layers):
            if not self._gps:
                edge_attr_to_pass = None
                if i == 0:
                    edge_attr_to_pass = edge_attr
                x, edge_index = conv_layer(x, edge_index, batch, edge_attr=edge_attr_to_pass)
            else:
                x = conv_layer(x, edge_index, batch, edge_attr=edge_attr)
            skip_connections.append(x)

        # Skip-cat
        x = torch.cat(skip_connections, dim=1)

        # Post-processing
        x = self._post_processing(x)

        # (Optional) Global pooling
        if self._global_pooling_schemes:
            x = self._global_pooling(x, batch=batch)
            if self._add_global_variables_after_pooling:
                x = torch.cat(
                    [
                        x,
                        global_variables,
                    ],
                    dim=1,
                )

        # Read-out
        x = self._readout(x)

        return x

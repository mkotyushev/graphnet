import numpy as np
import torch
import polars as pl
import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union
from graphnet.data.dataset import ColumnMissingException, Dataset
from torch_geometric.data import Data


def build_geometry_table(geometry_path):
    geometry = pl.read_csv(geometry_path)

    geometry = geometry.with_columns([
        (pl.col('x') / 500).alias('x'),
        (pl.col('y') / 500).alias('y'),
        (pl.col('z') / 500).alias('z'),
        pl.col('sensor_id').cast(pl.Int16).alias('sensor_id'), 
    ])
        
    return geometry


# https://www.kaggle.com/code/iafoss/chunk-based-data-loading-with-caching
class SequentialParquetDataset(Dataset):
    def __init__(
        self, 
        path,
        pulsemaps,
        filepathes, 
        geometry_path,
        features,
        truth,
        *,
        shuffle_outer: bool = False,
        shuffle_inner: bool = False,
        meta_path: Optional[str] = None,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[Union[str, List[int], List[List[int]]]] = None,
        dtype: torch.dtype = torch.float32,
        loss_weight_table: Optional[str] = None,
        loss_weight_columns: Optional[Union[List[str], str]] = None,
        loss_weight_default_value: Optional[float] = None,
        loss_weight_transform: Optional[Callable[[float], float]] = None,
        seed: Optional[int] = None,
        max_n_pulses: Optional[int] = None,
        max_n_pulses_strategy: Optional[str] = None,
        transforms: Optional[List[Callable]] = None,
    ):
        self.filepathes = list(filepathes)
        self.meta_path = meta_path
        self.shuffle_outer = shuffle_outer
        self.shuffle_inner = shuffle_inner
        self.label_fns = dict()

        # Internal sequence state
        self.current_outer_index = 0
        self.current_inner_index = 0
        self.current_inner_index_permutation = None
        self.filepath_to_len = {
            filepath: 
            len(pl.read_parquet(self._filepath_to_meta_filepath(filepath))) 
            for filepath in self.filepathes
        }

        if self.shuffle_outer:
            np.random.shuffle(self.filepathes)
        
        filepath = self.filepathes[self.current_outer_index]      
        meta_filepath = self._filepath_to_meta_filepath(filepath)          
        
        self.current_inner_index_permutation = list(range(self.filepath_to_len[filepath]))
        if self.shuffle_inner:
            np.random.shuffle(self.current_inner_index_permutation)

        geometry = build_geometry_table(geometry_path)
        self.current_tables = {
            'data': self._load_data(filepath, geometry),
            'meta': self._load_meta(meta_filepath) if meta_filepath is not None else None,
            'geometry': geometry,
        }

        self.length = sum(self.filepath_to_len.values())
        self.lock = torch.multiprocessing.Lock()

        super().__init__(
            path=path, 
            pulsemaps='data', 
            features=features, 
            truth=truth, 
            node_truth=node_truth,
            index_column=index_column,
            truth_table='meta',
            node_truth_table='meta',
            string_selection=string_selection,
            selection=selection,
            dtype=torch.float32,
            loss_weight_table='meta' if loss_weight_columns is not None else None,
            loss_weight_columns=loss_weight_columns,
            loss_weight_default_value=loss_weight_default_value,
            loss_weight_transform=loss_weight_transform,
            seed=seed,
            max_n_pulses=max_n_pulses,
            max_n_pulses_strategy=max_n_pulses_strategy,
            transforms=transforms,
        )

    def _filepath_to_meta_filepath(self, filepath):
        if self.meta_path is None:
            return None
        batch_id = filepath.stem.split('_')[1]
        return self.meta_path / f'train_meta_{batch_id}.parquet'
        
    # def _load_data(self, filepath):
    #     df = pl.read_parquet(filepath).sort('event_id')
    #     return df

    def _load_data(self, filepath, geometry):
        df = pl.read_parquet(filepath)
        df = df.join(geometry, on='sensor_id', how="inner")
        df = df.groupby("event_id").agg([
            pl.count(),
            pl.col("sensor_id").list(),
            pl.col("x").list(),
            pl.col("y").list(),
            pl.col("z").list(),
            pl.col("time").list(),
            pl.col("charge").list(),
            pl.col("auxiliary").list(),]).sort('event_id')
        return df
                
    def _load_meta(self, meta_filepath):
        df = pl.read_parquet(meta_filepath).sort('event_id')
        return df

    def _advance(self, idx):
        with self.lock:
            if self.current_inner_index < len(self.current_tables['data']):
                self.current_inner_index += 1
            else:
                self.current_inner_index = 0
                self.current_outer_index += 1

            if self.current_outer_index >= len(self.filepathes):
                self.current_outer_index = 0

                # Shuffle outer
                if self.shuffle_outer:
                    np.random.shuffle(self.filepathes)
            
            # Shuffle inner
            filepath = self.filepathes[self.current_outer_index]      
            meta_filepath = self._filepath_to_meta_filepath(filepath)          
            
            self.current_inner_index_permutation = list(range(self.filepath_to_len[filepath]))
            if self.current_inner_index == 0 and self.shuffle_inner:
                np.random.shuffle(self.current_inner_index_permutation)

            # Load data if needed
            if self.current_inner_index == 0:
                self.current_tables['data'] = self._load_data(filepath, self.current_tables['geometry'])
                if meta_filepath is not None:
                    self.current_tables['meta'] = self._load_meta(meta_filepath)

    # Override abstract method(s)
    def _init(self) -> None:
        """Set internal representation needed to read data from input file."""
        pass

    def _post_init(self) -> None:
        """Implemenation-specific code to be run after the main constructor."""
        pass

    def _get_all_indices(self) -> List[int]:
        indices = []
        for filepath in self.filepathes:
            meta = pl.read_parquet(self._filepath_to_meta_filepath(filepath))
            indices.extend(meta['event_id'].to_list())
        return indices

    def _get_event_index(
        self, sequential_index: Optional[int]
    ) -> Optional[int]:
        """Return a the event index corresponding to a `sequential_index`."""
        with self.lock:
            return self.current_inner_index_permutation[self.current_inner_index]

    def query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> List[Tuple[Any, ...]]:
        """Query a table at a specific index, optionally with some selection.

        Args:
            table: Table to be queried.
            columns: Columns to read out.
            sequential_index: Sequentially numbered index
                (i.e. in [0,len(self))) of the event to query. This _may_
                differ from the indexation used in `self._indices`. If no value
                is provided, the entire column is returned.
            selection: Selection to be imposed before reading out data.
                Defaults to None.

        Returns:
            List of tuples containing the values in `columns`. If the `table`
                contains only scalar data for `columns`, a list of length 1 is
                returned

        Raises:
            ColumnMissingException: If one or more element in `columns` is not
                present in `table`.
        """
        with self.lock:
            t = self.current_tables[table]
            index = self.current_inner_index_permutation[self.current_inner_index]
        
        if len(set(t.columns).intersection(columns)) != len(columns):
            raise ColumnMissingException

        return t[index][columns].rows()

    def _create_graph(
        self,
        features: List[Tuple[float, ...]],
        truth: Tuple[Any, ...],
        node_truth: Optional[List[Tuple[Any, ...]]] = None,
        loss_weight: Optional[float] = None,
    ) -> Data:
        """Create Pytorch Data (i.e. graph) object.

        No preprocessing is performed at this stage, just as no node adjancency
        is imposed. This means that the `edge_attr` and `edge_weight`
        attributes are not set.

        Args:
            features: List of tuples, containing event features.
            truth: List of tuples, containing truth information.
            node_truth: List of tuples, containing node-level truth.
            loss_weight: A weight associated with the event for weighing the
                loss.

        Returns:
            Graph object.
        """
        # Convert nested list to simple dict
        truth_dict = {
            key: truth[index] for index, key in enumerate(self._truth)
        }

        # Define custom labels
        labels_dict = self._get_labels(truth_dict)

        # Convert nested list to simple dict
        if node_truth is not None:
            node_truth_array = np.asarray(node_truth)
            assert self._node_truth is not None
            node_truth_dict = {
                key: node_truth_array[:, index]
                for index, key in enumerate(self._node_truth)
            }

        # Catch cases with no reconstructed pulses
        features = [feature[0] for feature in features[0][1:]]
        if len(features):
            data = np.asarray(features).T
        else:
            data = np.array([]).reshape((0, len(self._features) - 1))

        # Apply max_n_pulses strategy
        if self._max_n_pulses is not None and len(data) > self._max_n_pulses:
            if self._max_n_pulses_strategy == 'clamp':
                data = data[:self._max_n_pulses]
            elif self._max_n_pulses_strategy == 'random':
                indices, _ = torch.sort(torch.randperm(len(data))[:self._max_n_pulses])
                data = data[indices]
            elif self._max_n_pulses_strategy == 'each_nth':
                data = data[::len(data) // self._max_n_pulses]
                data = data[:self._max_n_pulses]
        
        # Apply transforms
        if self._transforms is not None:
            for transform in self._transforms:
                if truth_dict is not None:
                    data, truth_dict = transform(data, truth_dict)
                else:
                    data = transform(data)
                
        # Construct graph data object
        x = torch.tensor(data, dtype=self._dtype)  # pylint: disable=C0103
        n_pulses = torch.tensor(len(x), dtype=torch.int32)

        graph = Data(x=x, edge_index=None)
        graph.n_pulses = n_pulses
        graph.features = self._features[1:]

        # Add loss weight to graph.
        if loss_weight is not None and self._loss_weight_columns is not None:
            # No loss weight was retrieved, i.e., it is missing for the current
            # event.
            if loss_weight < 0:
                if self._loss_weight_default_value is None:
                    raise ValueError(
                        "At least one event is missing an entry in "
                        f"{self._loss_weight_columns} "
                        "but loss_weight_default_value is None."
                    )
                graph['loss_weight'] = torch.tensor(
                    self._loss_weight_default_value, dtype=self._dtype
                ).reshape(-1, 1)
            else:
                graph['loss_weight'] = torch.tensor(
                    loss_weight, dtype=self._dtype
                ).reshape(-1, 1)

        # Write attributes, either target labels, truth info or original
        # features.
        add_these_to_graph = [labels_dict, truth_dict]
        if node_truth is not None:
            add_these_to_graph.append(node_truth_dict)
        for write_dict in add_these_to_graph:
            for key, value in write_dict.items():
                try:
                    graph[key] = torch.tensor(value)
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type,
                    # e.g. `str`.
                    self.debug(
                        (
                            f"Could not assign `{key}` with type "
                            f"'{type(value).__name__}' as attribute to graph."
                        )
                    )

        # Additionally add original features as (static) attributes
        for index, feature in enumerate(graph.features):
            if feature not in ["x"]:
                graph[feature] = graph.x[:, index].detach()

        # Add custom labels to the graph
        for key, fn in self._label_fns.items():
            graph[key] = fn(graph)

        # Add Dataset Path. Useful if multiple datasets are concatenated.
        graph["dataset_path"] = self._path

        return graph

    # def _get_current_index(self) -> int:
    #     """Return the current index."""
    #     with self.lock:
    #         return self.current_inner_index_permutation[self.current_inner_index]

    def __getitem__(self, sequential_index: int) -> Data:
        result = super().__getitem__(sequential_index)
        
        # Advance the sequence state and load new chunk 
        # if current chunk is over or not initialized
        self._advance(sequential_index)
        
        return result

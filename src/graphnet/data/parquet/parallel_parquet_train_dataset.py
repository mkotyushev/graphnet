import gc
import os
import random
import time
import numpy as np
import torch
import numpy as np
import pandas as pd
import polars as pl
from typing import Any, Callable, List, Optional, Tuple, Union
from graphnet.data.dataset import ColumnMissingException, Dataset
from torch_geometric.data import Data
from .icecube_kaggle_lengths import parquet_file_stem_to_length


class ListMockWithLen:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


# desync_timeout_s should be larger than single file reading time
desync_timeout_s = 20
def parallel_parquet_worker_init_fn(worker_id):
    # Ensure reproducibility for all the libraries
    seed = worker_id
    random.seed(seed)
    # AFAIK env vars are not inherited by subprocesses 
    # if spawn context is used
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Sleep to avoid synchronous reading from disk
    time.sleep(desync_timeout_s * worker_id)


def build_geometry_table(geometry_path):
    geometry = pl.read_csv(geometry_path)

    geometry = geometry.with_columns([
        pl.col('sensor_id').cast(pl.Int16).alias('sensor_id'), 
    ])
        
    return geometry


# https://www.kaggle.com/code/iafoss/chunk-based-data-loading-with-caching
class ParallelParquetTrainDataset(Dataset):
    def __init__(
        self, 
        path,
        pulsemaps,
        filepathes, 
        geometry_path,
        features,
        truth,
        *,
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
        self.geometry_path = geometry_path   
        self.meta_path = meta_path   

        # Individual state: for internal checks of Dataset
        # it needs to be set to tables with same structure
        # as in real chunks
        self.tables = {
            'data': None,
            'meta': None,
        }
        self.index = None
        self.order = None
        self.indexes = None
        self.read_files_count = 0
        
        self.geometry_table = None
        self.is_initialized = False

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
            dtype=dtype,
            loss_weight_table='meta' if loss_weight_columns is not None else None,
            loss_weight_columns=loss_weight_columns,
            loss_weight_default_value=loss_weight_default_value,
            loss_weight_transform=loss_weight_transform,
            seed=seed,
            max_n_pulses=max_n_pulses,
            max_n_pulses_strategy=max_n_pulses_strategy,
            transforms=transforms,
        )

    def _is_init(self, table: str) -> bool:
        assert table in self.tables, f'Unknown table: {table}'
        # If initialized, all the tables should be loaded
        # TODO: add checks for all the internal state
        if self.is_initialized:
            assert all(t is not None for t in self.tables.values())
        return self.is_initialized

    def _get_inner_index(self):
        return int(self.order[self.index])

    def _filepath_to_meta_filepath(self, filepath):
        if self.meta_path is None:
            return None
        batch_id = filepath.stem.split('_')[1]
        return self.meta_path / f'train_meta_{batch_id}.parquet'

    def _load_data(self, filepath):
        df = pl.read_parquet(filepath)
        df = df.join(self.geometry_table, on='sensor_id', how="inner")
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

    def _load_next(self):
        # Reading from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py
        # worker_info.id is in [0, num_workers - 1] sequential.
        # So we can use it to select file for current worker.
        # 
        # And persistent_workers is somehow not used ???
        # Use persistent_workers = True just in case

        # Select file for current worker
        worker_info = torch.utils.data.get_worker_info()
        worker_id, num_workers = 0, 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            assert len(self.filepathes) > worker_info.num_workers, \
                f'Use less workers than number of files: {len(self.filepathes)} <= {worker_id}'
        
        # TODO: now in case when len(self.filepathes) % num_workers != 0 
        # some batches are read more frequently: not really
        # affecting the result for len(self.filepathes) >> num_workers, 
        # but it would be nice to have it fixed
        worker_filepathes = self.filepathes[
            worker_id::
            num_workers
        ]
        filepath = worker_filepathes[self.read_files_count]

        # Update internal state
        del self.tables['data'], self.tables['meta']
        gc.collect()
        self.tables = {
            'data': self._load_data(filepath),
            'meta': self._load_meta(self._filepath_to_meta_filepath(filepath)),
        }
        self.index = 0
        self.order = np.random.permutation(len(self.tables['meta']))
        self.indexes = self.tables['meta'][self._index_column]

        # Advance read_files_count
        self.read_files_count = (self.read_files_count + 1) % len(worker_filepathes)

    def _advance(self):
        assert self.is_initialized, 'Dataset is not initialized'
        
        self.index += 1

        if self.index >= len(self.tables['data']):
            self._load_next()

        assert self.index < len(self.tables['data'])

    # Override abstract method(s)
    def _init(self) -> None:
        """Set internal representation needed to read data from input file."""
        pass

    def _post_init(self) -> None:
        """Implemenation-specific code to be run after the main constructor."""
        pass

    def _get_all_indices(self) -> List[int]:
        # To aviod storing all the indices 
        # (not really needed here or in Dataset other than checking length)
        # use ListMockWithLen
        length = 0
        for filepath in self.filepathes:
            length += parquet_file_stem_to_length[filepath.stem]
        return ListMockWithLen(length)

    def _get_event_index(
        self, sequential_index: Optional[int]
    ) -> Optional[int]:
        """Return a the event index corresponding to a `sequential_index`."""
        result = self.indexes[self._get_inner_index()]
        if not isinstance(result, np.int64):
            result = result.compute()
        return result

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
        t = self.tables[table]
        
        if len(set(t.columns).intersection(columns)) != len(columns):
            raise ColumnMissingException

        return t[self._get_inner_index()][columns].rows()

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

    def __getitem__(self, sequential_index: int) -> Data:
        # Load first chunk inside worker
        if not self.is_initialized:
            self.geometry_table = build_geometry_table(self.geometry_path)
            self._load_next()
            self.is_initialized = True
        
        # Here sequential_index is actually not relevant
        # because we are using the inner index
        # later in query_table
        result = super().__getitem__(sequential_index)
        
        # Advance the sequence state and load new chunk 
        # if current chunk is over or not initialized
        self._advance()
        
        return result

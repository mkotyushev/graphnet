"""Parquet-specific implementation of data classes."""

from graphnet.utilities.imports import has_torch_package

from .parquet_dataconverter import ParquetDataConverter

if has_torch_package():
    from .parquet_dataset import ParquetDataset
    from .parallel_parquet_train_dataset import (
        ParallelParquetTrainDataset, 
        parallel_parquet_worker_init_fn
    )

del has_torch_package

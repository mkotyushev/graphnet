"""Parquet-specific implementation of data classes."""

from graphnet.utilities.imports import has_torch_package

from .parquet_dataconverter import ParquetDataConverter

if has_torch_package():
    from .parquet_dataset import ParquetDataset
    from .sequential_parquet_dataset import SequentialParquetDataset

del has_torch_package

"""
Data Ingestion Module
Handles loading of diverse file formats (CSV, TSV, Excel), schema inspection,
memory profiling, and robust ingestion validation while preserving raw data integrity.
"""

import os
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Production-ready data ingestion engine supporting CSV and Excel files.
    Ensures raw immutability, encoding resilience, and upfront schema inspection.
    """

    SUPPORTED_EXTENSIONS = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".parquet": "parquet"
    }

    ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]

    def __init__(self, raw_storage_dir: Optional[str] = "data/raw"):
        self.raw_storage_dir = raw_storage_dir
        if raw_storage_dir:
            os.makedirs(raw_storage_dir, exist_ok=True)

    def detect_format(self, file_path: str) -> str:
        """Determines the file format based on extension."""
        _, ext = os.path.splitext(file_path.lower())
        if ext not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(self.SUPPORTED_EXTENSIONS.keys())
            raise ValueError(
                f"Unsupported file format '{ext}' for file '{file_path}'. "
                f"Supported extensions: {supported}"
            )
        return self.SUPPORTED_EXTENSIONS[ext]

    def load_dataset(self, file_path: str, sheet_name: Any = 0, **kwargs) -> pd.DataFrame:
        """
        Loads dataset with encoding fallback and error handling.
        Preserves raw input immutability by returning an independent DataFrame copy.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found at path: {file_path}")

        file_format = self.detect_format(file_path)
        logger.info(f"Ingesting file: {file_path} (Detected format: {file_format})")

        df: Optional[pd.DataFrame] = None
        load_error: Optional[Exception] = None

        if file_format in ("csv", "tsv"):
            sep = "\t" if file_format == "tsv" else ","
            for encoding in self.ENCODINGS_TO_TRY:
                try:
                    df = pd.read_csv(file_path, sep=sep, encoding=encoding, low_memory=False, **kwargs)
                    logger.info(f"Successfully loaded CSV with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    load_error = e
                    break
            if df is None:
                raise ValueError(
                    f"Failed to decode CSV file {file_path} using tried encodings: {self.ENCODINGS_TO_TRY}. "
                    f"Original error: {load_error}"
                )

        elif file_format == "excel":
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl", **kwargs)
                logger.info(f"Successfully loaded Excel workbook sheet: {sheet_name}")
            except Exception as e:
                raise IOError(f"Failed to read Excel file '{file_path}': {str(e)}")

        elif file_format == "parquet":
            try:
                df = pd.read_parquet(file_path, **kwargs)
                logger.info(f"Successfully loaded Parquet dataset: {file_path}")
            except Exception as e:
                raise IOError(f"Failed to read Parquet file '{file_path}': {str(e)}")

        if df is None or df.empty:
            logger.warning(f"File {file_path} was loaded but contains no records.")
            return pd.DataFrame()

        # Always return deep copy to protect raw source data
        return df.copy(deep=True)

    @staticmethod
    def profile_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs comprehensive schema and memory profiling on raw ingested data.
        """
        if df.empty:
            return {
                "total_rows": 0,
                "total_columns": 0,
                "memory_usage_mb": 0.0,
                "empty_rows_count": 0,
                "empty_columns_count": 0,
                "columns": [],
                "dtypes": {},
                "missing_counts": {},
                "missing_percentages": {}
            }

        total_rows, total_cols = df.shape
        memory_usage_bytes = df.memory_usage(deep=True).sum()
        memory_usage_mb = round(memory_usage_bytes / (1024 * 1024), 2)

        # Identify completely empty rows (all values NaN or blank strings)
        is_empty_cell = df.isna() | df.map(lambda x: str(x).strip() in ("", "nan", "None") if pd.notna(x) else True)
        empty_rows_count = int(is_empty_cell.all(axis=1).sum())
        empty_cols_count = int(is_empty_cell.all(axis=0).sum())

        missing_counts = df.isna().sum().to_dict()
        missing_percentages = {col: round((count / total_rows) * 100, 2) for col, count in missing_counts.items()}
        dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

        profile = {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "memory_usage_mb": memory_usage_mb,
            "empty_rows_count": empty_rows_count,
            "empty_columns_count": empty_cols_count,
            "columns": list(df.columns),
            "dtypes": dtypes_dict,
            "missing_counts": missing_counts,
            "missing_percentages": missing_percentages
        }

        logger.info(
            f"Dataset Ingested: {total_rows:,} rows, {total_cols} columns, "
            f"{memory_usage_mb} MB memory, {empty_rows_count} empty rows detected."
        )
        return profile

    @staticmethod
    def get_sample_records(df: pd.DataFrame, n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns the first n and last n records as separate views."""
        return df.head(n).copy(), df.tail(n).copy()

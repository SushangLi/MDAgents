"""
Data analysis utilities using pandas and numpy.
"""

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


# Global dataframe storage
_dataframes: Dict[str, pd.DataFrame] = {}


def load_csv(file_path: str) -> Dict[str, Any]:
    """
    Load CSV file into pandas DataFrame.

    Args:
        file_path: Path to CSV file

    Returns:
        Dict with df_id and summary information

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file can't be parsed
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    try:
        df = pd.read_csv(path)
        df_id = f"df_{uuid.uuid4().hex[:8]}"
        _dataframes[df_id] = df

        logger.info(f"Loaded CSV: {file_path} -> {df_id} ({len(df)} rows, {len(df.columns)} columns)")

        return {
            "df_id": df_id,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head().to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}")
        raise ValueError(f"Failed to load CSV: {e}")


def load_excel(file_path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load Excel file into pandas DataFrame.

    Args:
        file_path: Path to Excel file
        sheet_name: Optional sheet name (defaults to first sheet)

    Returns:
        Dict with df_id and summary information

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file can't be parsed
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    try:
        df = pd.read_excel(path, sheet_name=sheet_name or 0)
        df_id = f"df_{uuid.uuid4().hex[:8]}"
        _dataframes[df_id] = df

        logger.info(f"Loaded Excel: {file_path} -> {df_id} ({len(df)} rows, {len(df.columns)} columns)")

        return {
            "df_id": df_id,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head().to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"Error loading Excel {file_path}: {e}")
        raise ValueError(f"Failed to load Excel: {e}")


def analyze_dataframe(df_id: str) -> Dict[str, Any]:
    """
    Analyze DataFrame and return statistics.

    Args:
        df_id: DataFrame identifier

    Returns:
        Dict with analysis results

    Raises:
        KeyError: If df_id doesn't exist
    """
    if df_id not in _dataframes:
        raise KeyError(f"DataFrame not found: {df_id}")

    df = _dataframes[df_id]

    # Get basic info
    info = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
    }

    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info["numeric_stats"] = df[numeric_cols].describe().to_dict()

    # Categorical columns info
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        info["categorical_info"] = {
            col: {
                "unique_values": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict(),
            }
            for col in categorical_cols
        }

    logger.debug(f"Analyzed DataFrame {df_id}")
    return info


def get_dataframe_head(df_id: str, n: int = 10) -> Dict[str, Any]:
    """
    Get first n rows of DataFrame.

    Args:
        df_id: DataFrame identifier
        n: Number of rows to return

    Returns:
        Dict with head data

    Raises:
        KeyError: If df_id doesn't exist
    """
    if df_id not in _dataframes:
        raise KeyError(f"DataFrame not found: {df_id}")

    df = _dataframes[df_id]
    head_df = df.head(n)

    return {
        "df_id": df_id,
        "rows_shown": len(head_df),
        "total_rows": len(df),
        "data": head_df.to_dict(orient="records"),
        "columns": head_df.columns.tolist(),
    }


def query_dataframe(df_id: str, query: str) -> str:
    """
    Execute pandas query on DataFrame.

    Args:
        df_id: DataFrame identifier
        query: Pandas query string

    Returns:
        Result as string

    Raises:
        KeyError: If df_id doesn't exist
        ValueError: If query is invalid
    """
    if df_id not in _dataframes:
        raise KeyError(f"DataFrame not found: {df_id}")

    df = _dataframes[df_id]

    try:
        # Try query method first
        result = df.query(query)
        logger.debug(f"Query executed on {df_id}: {query}")
        return result.to_string()
    except Exception as e:
        logger.error(f"Query failed on {df_id}: {e}")
        raise ValueError(f"Invalid query: {e}")


def compute_statistics(df_id: str, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute statistics for specified columns.

    Args:
        df_id: DataFrame identifier
        columns: List of column names (None for all numeric columns)

    Returns:
        Dict with statistics

    Raises:
        KeyError: If df_id or columns don't exist
    """
    if df_id not in _dataframes:
        raise KeyError(f"DataFrame not found: {df_id}")

    df = _dataframes[df_id]

    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        return {"message": "No numeric columns found"}

    try:
        stats = {}
        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column not found: {col}")

            col_data = df[col]

            if pd.api.types.is_numeric_dtype(col_data):
                stats[col] = {
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "count": int(col_data.count()),
                    "missing": int(col_data.isnull().sum()),
                }
            else:
                stats[col] = {
                    "unique": int(col_data.nunique()),
                    "top": str(col_data.mode().iloc[0]) if len(col_data.mode()) > 0 else None,
                    "count": int(col_data.count()),
                    "missing": int(col_data.isnull().sum()),
                }

        logger.debug(f"Computed statistics for {df_id}")
        return stats
    except Exception as e:
        logger.error(f"Error computing statistics: {e}")
        raise


def get_dataframe(df_id: str) -> pd.DataFrame:
    """
    Get DataFrame by ID (for internal use).

    Args:
        df_id: DataFrame identifier

    Returns:
        pandas DataFrame

    Raises:
        KeyError: If df_id doesn't exist
    """
    if df_id not in _dataframes:
        raise KeyError(f"DataFrame not found: {df_id}")
    return _dataframes[df_id]


def list_dataframes() -> List[Dict[str, Any]]:
    """
    List all loaded DataFrames.

    Returns:
        List of DataFrame info
    """
    return [
        {
            "df_id": df_id,
            "shape": df.shape,
            "columns": df.columns.tolist(),
        }
        for df_id, df in _dataframes.items()
    ]


def delete_dataframe(df_id: str) -> bool:
    """
    Delete DataFrame from memory.

    Args:
        df_id: DataFrame identifier

    Returns:
        True if successful

    Raises:
        KeyError: If df_id doesn't exist
    """
    if df_id not in _dataframes:
        raise KeyError(f"DataFrame not found: {df_id}")

    del _dataframes[df_id]
    logger.debug(f"Deleted DataFrame {df_id}")
    return True


def clear_all_dataframes() -> None:
    """Clear all DataFrames from memory."""
    _dataframes.clear()
    logger.info("Cleared all DataFrames")

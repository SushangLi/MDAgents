"""
Base Preprocessor for Omics Data.

This module defines the abstract base class for omics data preprocessing.
All omics-specific preprocessors should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class PreprocessingResult:
    """Result of preprocessing with metadata."""

    data: pd.DataFrame
    feature_names: List[str]
    n_samples: int
    n_features: int
    missing_values_filled: int
    outliers_handled: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data": self.data.to_dict(),
            "feature_names": self.feature_names,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "missing_values_filled": self.missing_values_filled,
            "outliers_handled": self.outliers_handled,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        result_dict = self.to_dict()
        # Convert DataFrame to JSON-serializable format
        result_dict["data"] = self.data.to_json()
        return json.dumps(result_dict, indent=indent)


class BasePreprocessor(ABC):
    """
    Abstract base class for omics data preprocessing.

    All preprocessors should implement:
    - fit(): Learn preprocessing parameters from training data
    - transform(): Apply preprocessing to data
    - fit_transform(): Fit and transform in one step
    """

    def __init__(
        self,
        omics_type: str,
        missing_value_strategy: str = "median",
        outlier_threshold: float = 3.0,
        normalization_method: str = "zscore"
    ):
        """
        Initialize base preprocessor.

        Args:
            omics_type: Type of omics data ("microbiome", "metabolome", "proteome")
            missing_value_strategy: How to handle missing values ("median", "mean", "zero", "drop")
            outlier_threshold: Z-score threshold for outlier detection
            normalization_method: Normalization method ("zscore", "minmax", "robust")
        """
        self.omics_type = omics_type
        self.missing_value_strategy = missing_value_strategy
        self.outlier_threshold = outlier_threshold
        self.normalization_method = normalization_method

        # Will be set during fit()
        self.feature_names_: Optional[List[str]] = None
        self.fill_values_: Optional[Dict[str, float]] = None
        self.normalization_params_: Optional[Dict[str, Any]] = None
        self.is_fitted_ = False

    def fit(self, data: pd.DataFrame) -> "BasePreprocessor":
        """
        Fit the preprocessor on training data.

        Args:
            data: Training data

        Returns:
            self
        """
        self.feature_names_ = list(data.columns)

        # Calculate fill values for missing data
        self.fill_values_ = self._calculate_fill_values(data)

        # Calculate normalization parameters
        self.normalization_params_ = self._calculate_normalization_params(data)

        self.is_fitted_ = True
        return self

    def transform(self, data: pd.DataFrame) -> PreprocessingResult:
        """
        Transform data using fitted parameters.

        Args:
            data: Data to transform

        Returns:
            PreprocessingResult with processed data and metadata
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform()")

        # Track preprocessing statistics
        missing_count = 0
        outlier_count = 0

        # 1. Handle missing values
        data_clean, missing_count = self._handle_missing_values(data)

        # 2. Handle outliers
        data_clean, outlier_count = self._handle_outliers(data_clean)

        # 3. Normalize
        data_normalized = self._normalize(data_clean)

        # 4. Omics-specific processing
        data_final = self._omics_specific_processing(data_normalized)

        # Create result
        result = PreprocessingResult(
            data=data_final,
            feature_names=list(data_final.columns),
            n_samples=len(data_final),
            n_features=len(data_final.columns),
            missing_values_filled=missing_count,
            outliers_handled=outlier_count,
            metadata={
                "omics_type": self.omics_type,
                "missing_value_strategy": self.missing_value_strategy,
                "outlier_threshold": self.outlier_threshold,
                "normalization_method": self.normalization_method
            }
        )

        return result

    def fit_transform(self, data: pd.DataFrame) -> PreprocessingResult:
        """
        Fit and transform data in one step.

        Args:
            data: Data to fit and transform

        Returns:
            PreprocessingResult with processed data and metadata
        """
        self.fit(data)
        return self.transform(data)

    def _calculate_fill_values(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate values to fill missing data."""
        fill_values = {}

        for col in data.columns:
            if self.missing_value_strategy == "median":
                fill_values[col] = data[col].median()
            elif self.missing_value_strategy == "mean":
                fill_values[col] = data[col].mean()
            elif self.missing_value_strategy == "zero":
                fill_values[col] = 0.0
            else:
                fill_values[col] = data[col].median()  # Default to median

        return fill_values

    def _calculate_normalization_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate normalization parameters."""
        params = {}

        if self.normalization_method == "zscore":
            params["mean"] = data.mean()
            params["std"] = data.std()
        elif self.normalization_method == "minmax":
            params["min"] = data.min()
            params["max"] = data.max()
        elif self.normalization_method == "robust":
            params["median"] = data.median()
            params["iqr"] = data.quantile(0.75) - data.quantile(0.25)

        return params

    def _handle_missing_values(self, data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Handle missing values in data."""
        data_clean = data.copy()
        missing_count = data_clean.isna().sum().sum()

        if self.missing_value_strategy == "drop":
            data_clean = data_clean.dropna()
        else:
            for col in data_clean.columns:
                fill_value = self.fill_values_.get(col, 0.0)
                data_clean[col] = data_clean[col].fillna(fill_value)

        return data_clean, missing_count

    def _handle_outliers(self, data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Handle outliers using z-score method."""
        data_clean = data.copy()
        outlier_count = 0

        for col in data_clean.columns:
            # Calculate z-scores
            mean = data_clean[col].mean()
            std = data_clean[col].std()

            if std > 0:
                z_scores = np.abs((data_clean[col] - mean) / std)
                outliers = z_scores > self.outlier_threshold

                # Cap outliers at threshold
                if outliers.any():
                    outlier_count += outliers.sum()
                    upper_bound = mean + self.outlier_threshold * std
                    lower_bound = mean - self.outlier_threshold * std

                    data_clean.loc[data_clean[col] > upper_bound, col] = upper_bound
                    data_clean.loc[data_clean[col] < lower_bound, col] = lower_bound

        return data_clean, outlier_count

    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data using fitted parameters."""
        data_normalized = data.copy()

        if self.normalization_method == "zscore":
            mean = self.normalization_params_["mean"]
            std = self.normalization_params_["std"]
            data_normalized = (data_normalized - mean) / (std + 1e-8)

        elif self.normalization_method == "minmax":
            min_val = self.normalization_params_["min"]
            max_val = self.normalization_params_["max"]
            data_normalized = (data_normalized - min_val) / (max_val - min_val + 1e-8)

        elif self.normalization_method == "robust":
            median = self.normalization_params_["median"]
            iqr = self.normalization_params_["iqr"]
            data_normalized = (data_normalized - median) / (iqr + 1e-8)

        return data_normalized

    @abstractmethod
    def _omics_specific_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply omics-specific processing.

        This method should be implemented by subclasses to apply
        domain-specific transformations (e.g., CLR for microbiome,
        log transform for metabolome).

        Args:
            data: Normalized data

        Returns:
            Processed data
        """
        pass

    def save_params(self, filepath: str):
        """Save fitted parameters to file."""
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before saving")

        params = {
            "omics_type": self.omics_type,
            "missing_value_strategy": self.missing_value_strategy,
            "outlier_threshold": self.outlier_threshold,
            "normalization_method": self.normalization_method,
            "feature_names": self.feature_names_,
            "fill_values": {k: float(v) for k, v in self.fill_values_.items()},
            "normalization_params": {
                k: v.to_dict() if hasattr(v, "to_dict") else v.tolist() if hasattr(v, "tolist") else v
                for k, v in self.normalization_params_.items()
            }
        }

        with open(filepath, "w") as f:
            json.dump(params, f, indent=2)

    def load_params(self, filepath: str):
        """Load fitted parameters from file."""
        with open(filepath, "r") as f:
            params = json.load(f)

        self.omics_type = params["omics_type"]
        self.missing_value_strategy = params["missing_value_strategy"]
        self.outlier_threshold = params["outlier_threshold"]
        self.normalization_method = params["normalization_method"]
        self.feature_names_ = params["feature_names"]
        self.fill_values_ = params["fill_values"]
        self.normalization_params_ = params["normalization_params"]
        self.is_fitted_ = True

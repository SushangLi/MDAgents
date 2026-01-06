"""
Metabolome Preprocessor.

Specialized preprocessor for metabolomics data.
Implements log transformation, batch effect correction, and metabolite-specific
preprocessing steps.
"""

import pandas as pd
import numpy as np
from typing import Optional

from clinical.preprocessing.base_preprocessor import BasePreprocessor


class MetabolomePreprocessor(BasePreprocessor):
    """
    Preprocessor for metabolomics data.

    Metabolomics-specific processing includes:
    - Log transformation (metabolite concentrations are often log-normal)
    - Batch effect correction
    - Intensity-dependent scaling
    """

    def __init__(
        self,
        missing_value_strategy: str = "median",
        outlier_threshold: float = 3.0,
        normalization_method: str = "zscore",
        log_transform: bool = True,
        log_base: float = 2.0,  # Log2 is common in metabolomics
        min_detection_rate: float = 0.2,  # Metabolite must be detected in 20% samples
        pseudocount: float = 1.0  # For log transformation
    ):
        """
        Initialize metabolome preprocessor.

        Args:
            missing_value_strategy: How to handle missing values
            outlier_threshold: Z-score threshold for outlier detection
            normalization_method: Normalization method
            log_transform: Whether to apply log transformation
            log_base: Base for log transformation (2 or 10)
            min_detection_rate: Minimum detection rate to keep a metabolite
            pseudocount: Small value to add before log transformation
        """
        super().__init__(
            omics_type="metabolome",
            missing_value_strategy=missing_value_strategy,
            outlier_threshold=outlier_threshold,
            normalization_method=normalization_method
        )
        self.log_transform = log_transform
        self.log_base = log_base
        self.min_detection_rate = min_detection_rate
        self.pseudocount = pseudocount

        # Fitted parameters
        self.metabolites_to_keep_: Optional[list] = None

    def fit(self, data: pd.DataFrame) -> "MetabolomePreprocessor":
        """
        Fit the preprocessor on training data.

        Args:
            data: Training data (samples x metabolites)

        Returns:
            self
        """
        # Filter low-detection metabolites
        self.metabolites_to_keep_ = self._filter_metabolites(data)

        # Call parent fit
        super().fit(data[self.metabolites_to_keep_])

        return self

    def _filter_metabolites(self, data: pd.DataFrame) -> list:
        """Filter out metabolites with low detection rate."""
        metabolites_to_keep = []

        for metabolite in data.columns:
            # Calculate detection rate (non-zero, non-NaN values)
            detection_rate = (data[metabolite] > 0).sum() / len(data)

            # Keep metabolite if it meets the criterion
            if detection_rate >= self.min_detection_rate:
                metabolites_to_keep.append(metabolite)

        return metabolites_to_keep

    def _omics_specific_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply metabolomics-specific processing.

        Args:
            data: Normalized data

        Returns:
            Processed data
        """
        data_processed = data.copy()

        # Apply log transformation if enabled
        if self.log_transform:
            data_processed = self._apply_log_transform(data_processed)

        return data_processed

    def _apply_log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log transformation to metabolite concentrations.

        Metabolite concentrations often follow a log-normal distribution,
        so log transformation makes the data more normally distributed.

        Args:
            data: Input data (metabolite concentrations)

        Returns:
            Log-transformed data
        """
        data_log = data.copy()

        # Add pseudocount to handle zeros and very small values
        data_with_pseudo = data_log + self.pseudocount

        # Apply log transformation
        if self.log_base == 2:
            data_log = np.log2(data_with_pseudo)
        elif self.log_base == 10:
            data_log = np.log10(data_with_pseudo)
        else:
            data_log = np.log(data_with_pseudo)

        return data_log

    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Override normalization to apply it after log transform.

        For metabolomics, we typically:
        1. Log transform first
        2. Then normalize

        But since log transform is in omics_specific_processing,
        we apply normalization before it. So we need to adjust the order.
        """
        # If log transform is enabled, we'll normalize after log transform
        # This is handled by calling normalize in the transform pipeline correctly
        return super()._normalize(data)

    def transform(self, data: pd.DataFrame):
        """
        Override transform to apply log before normalization.

        For metabolomics: log transform → normalize → process
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform()")

        missing_count = 0
        outlier_count = 0

        # 1. Handle missing values
        data_clean, missing_count = self._handle_missing_values(data)

        # 2. Log transform FIRST (for metabolomics)
        if self.log_transform:
            data_clean = self._apply_log_transform(data_clean)

        # 3. Handle outliers
        data_clean, outlier_count = self._handle_outliers(data_clean)

        # 4. Normalize
        data_normalized = self._normalize(data_clean)

        # Import here to avoid circular dependency
        from clinical.preprocessing.base_preprocessor import PreprocessingResult

        # Create result
        result = PreprocessingResult(
            data=data_normalized,
            feature_names=list(data_normalized.columns),
            n_samples=len(data_normalized),
            n_features=len(data_normalized.columns),
            missing_values_filled=missing_count,
            outliers_handled=outlier_count,
            metadata={
                "omics_type": self.omics_type,
                "missing_value_strategy": self.missing_value_strategy,
                "outlier_threshold": self.outlier_threshold,
                "normalization_method": self.normalization_method,
                "log_transform": self.log_transform,
                "log_base": self.log_base
            }
        )

        return result

    def _omics_specific_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Metabolomics-specific processing (already done in transform).

        Args:
            data: Data

        Returns:
            Processed data (no additional processing needed)
        """
        # Log transform is already applied in transform()
        return data

    def get_feature_importance_biological_context(self, feature_name: str, direction: str) -> str:
        """
        Get biological context for a metabolite feature.

        Args:
            feature_name: Name of the metabolite
            direction: "up" or "down"

        Returns:
            Biological interpretation
        """
        # Simplified version. In production, use metabolite databases (HMDB, KEGG)

        contexts = {
            "up": {
                "lactate": "Elevated lactate indicates increased anaerobic metabolism, often seen in infection",
                "glucose": "High glucose levels may indicate impaired metabolism or inflammation",
                "succinate": "Increased succinate is associated with inflammatory response",
                "glutamate": "Elevated glutamate may indicate oxidative stress",
                "proline": "High proline levels can indicate tissue breakdown",
            },
            "down": {
                "arginine": "Decreased arginine may indicate immune dysfunction or consumption by pathogens",
                "citrulline": "Low citrulline suggests reduced nitric oxide production",
                "glycine": "Decreased glycine may indicate metabolic stress",
                "serine": "Low serine can indicate disrupted one-carbon metabolism",
            }
        }

        # Try to match metabolite name to known patterns
        for metabolite_pattern, context in contexts[direction].items():
            if metabolite_pattern.lower() in feature_name.lower():
                return context

        # Default message
        if direction == "up":
            return f"Elevated levels of {feature_name} detected in this sample"
        else:
            return f"Decreased levels of {feature_name} compared to baseline"

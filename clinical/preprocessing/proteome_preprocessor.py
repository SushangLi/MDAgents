"""
Proteome Preprocessor.

Specialized preprocessor for proteomics data.
Implements protein-specific normalization, missing value imputation,
and intensity-based filtering.
"""

import pandas as pd
import numpy as np
from typing import Optional

from clinical.preprocessing.base_preprocessor import BasePreprocessor


class ProteomePreprocessor(BasePreprocessor):
    """
    Preprocessor for proteomics data.

    Proteomics-specific processing includes:
    - Protein intensity normalization
    - Missing value imputation (proteins not detected)
    - Quantile normalization across samples
    """

    def __init__(
        self,
        missing_value_strategy: str = "median",
        outlier_threshold: float = 3.0,
        normalization_method: str = "zscore",
        log_transform: bool = True,
        log_base: float = 2.0,
        min_detection_rate: float = 0.3,  # Protein must be detected in 30% samples
        use_quantile_normalization: bool = False,
        pseudocount: float = 1.0
    ):
        """
        Initialize proteome preprocessor.

        Args:
            missing_value_strategy: How to handle missing values
            outlier_threshold: Z-score threshold for outlier detection
            normalization_method: Normalization method
            log_transform: Whether to apply log transformation
            log_base: Base for log transformation
            min_detection_rate: Minimum detection rate to keep a protein
            use_quantile_normalization: Whether to use quantile normalization
            pseudocount: Small value to add before log transformation
        """
        super().__init__(
            omics_type="proteome",
            missing_value_strategy=missing_value_strategy,
            outlier_threshold=outlier_threshold,
            normalization_method=normalization_method
        )
        self.log_transform = log_transform
        self.log_base = log_base
        self.min_detection_rate = min_detection_rate
        self.use_quantile_normalization = use_quantile_normalization
        self.pseudocount = pseudocount

        # Fitted parameters
        self.proteins_to_keep_: Optional[list] = None
        self.quantile_reference_: Optional[np.ndarray] = None

    def fit(self, data: pd.DataFrame) -> "ProteomePreprocessor":
        """
        Fit the preprocessor on training data.

        Args:
            data: Training data (samples x proteins)

        Returns:
            self
        """
        # Filter low-detection proteins
        self.proteins_to_keep_ = self._filter_proteins(data)

        # If using quantile normalization, calculate reference distribution
        if self.use_quantile_normalization:
            self.quantile_reference_ = self._calculate_quantile_reference(
                data[self.proteins_to_keep_]
            )

        # Call parent fit
        super().fit(data[self.proteins_to_keep_])

        return self

    def _filter_proteins(self, data: pd.DataFrame) -> list:
        """Filter out proteins with low detection rate."""
        proteins_to_keep = []

        for protein in data.columns:
            # Calculate detection rate (non-zero, non-NaN values)
            detection_rate = (data[protein] > 0).sum() / len(data)

            # Keep protein if it meets the criterion
            if detection_rate >= self.min_detection_rate:
                proteins_to_keep.append(protein)

        return proteins_to_keep

    def _calculate_quantile_reference(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate reference distribution for quantile normalization.

        Args:
            data: Training data

        Returns:
            Reference quantile distribution
        """
        # Calculate mean quantiles across all samples
        quantiles = []
        for col in data.columns:
            sorted_values = np.sort(data[col].values)
            quantiles.append(sorted_values)

        # Average quantiles across all proteins
        reference = np.mean(quantiles, axis=0)
        return reference

    def _apply_quantile_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply quantile normalization to make distributions identical across samples.

        Args:
            data: Input data

        Returns:
            Quantile-normalized data
        """
        data_qnorm = data.copy()

        for col in data_qnorm.columns:
            # Get ranks of original values
            ranks = data_qnorm[col].rank(method='average')

            # Map ranks to reference quantiles
            # Convert ranks to indices (0-based)
            indices = (ranks - 1).astype(int)
            indices = np.clip(indices, 0, len(self.quantile_reference_) - 1)

            # Replace with reference values
            data_qnorm[col] = self.quantile_reference_[indices]

        return data_qnorm

    def _apply_log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log transformation to protein intensities.

        Args:
            data: Input data (protein intensities)

        Returns:
            Log-transformed data
        """
        data_log = data.copy()

        # Add pseudocount to handle zeros
        data_with_pseudo = data_log + self.pseudocount

        # Apply log transformation
        if self.log_base == 2:
            data_log = np.log2(data_with_pseudo)
        elif self.log_base == 10:
            data_log = np.log10(data_with_pseudo)
        else:
            data_log = np.log(data_with_pseudo)

        return data_log

    def transform(self, data: pd.DataFrame):
        """
        Override transform to apply log and quantile normalization.

        For proteomics: log transform → quantile norm → normalize → process
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform()")

        missing_count = 0
        outlier_count = 0

        # 1. Handle missing values
        data_clean, missing_count = self._handle_missing_values(data)

        # 2. Log transform FIRST
        if self.log_transform:
            data_clean = self._apply_log_transform(data_clean)

        # 3. Quantile normalization (if enabled)
        if self.use_quantile_normalization and self.quantile_reference_ is not None:
            data_clean = self._apply_quantile_normalization(data_clean)

        # 4. Handle outliers
        data_clean, outlier_count = self._handle_outliers(data_clean)

        # 5. Normalize
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
                "log_base": self.log_base,
                "use_quantile_normalization": self.use_quantile_normalization
            }
        )

        return result

    def _omics_specific_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Proteomics-specific processing (already done in transform).

        Args:
            data: Data

        Returns:
            Processed data (no additional processing needed)
        """
        # Log transform and quantile normalization already applied in transform()
        return data

    def get_feature_importance_biological_context(self, feature_name: str, direction: str) -> str:
        """
        Get biological context for a protein feature.

        Args:
            feature_name: Name of the protein
            direction: "up" or "down"

        Returns:
            Biological interpretation
        """
        # Simplified version. In production, use protein databases (UniProt, etc.)

        contexts = {
            "up": {
                "MMP": "Elevated matrix metalloproteinase indicates tissue remodeling and destruction",
                "IL-": "Increased interleukin suggests active inflammatory response",
                "TNF": "High tumor necrosis factor indicates systemic inflammation",
                "CRP": "Elevated C-reactive protein is a marker of acute inflammation",
                "albumin": "High albumin may indicate dehydration or other metabolic changes",
                "fibrinogen": "Elevated fibrinogen suggests coagulation activation",
            },
            "down": {
                "albumin": "Decreased albumin may indicate malnutrition or chronic inflammation",
                "transferrin": "Low transferrin can indicate iron deficiency or inflammation",
                "IgG": "Decreased immunoglobulin G may indicate immune suppression",
                "complement": "Low complement levels suggest immune consumption",
            }
        }

        # Try to match protein name to known patterns
        for protein_pattern, context in contexts[direction].items():
            if protein_pattern.lower() in feature_name.lower():
                return context

        # Default message
        if direction == "up":
            return f"Increased expression of {feature_name} detected in this sample"
        else:
            return f"Decreased expression of {feature_name} compared to baseline"

"""
Microbiome Preprocessor.

Specialized preprocessor for microbiome (16S rRNA, metagenomics) data.
Implements CLR (Centered Log-Ratio) transformation and other microbiome-specific
preprocessing steps.
"""

import pandas as pd
import numpy as np
from typing import Optional

from clinical.preprocessing.base_preprocessor import BasePreprocessor


class MicrobiomePreprocessor(BasePreprocessor):
    """
    Preprocessor for microbiome data.

    Microbiome-specific processing includes:
    - Abundance filtering (remove low-abundance taxa)
    - CLR (Centered Log-Ratio) transformation
    - Compositional data handling
    """

    def __init__(
        self,
        missing_value_strategy: str = "zero",  # Microbiome: missing = absent
        outlier_threshold: float = 3.0,
        normalization_method: str = "clr",  # CLR is standard for microbiome
        min_abundance: float = 0.001,  # Filter taxa below 0.1%
        min_prevalence: float = 0.1,  # Present in at least 10% of samples
        pseudocount: float = 1e-6  # For log transformation
    ):
        """
        Initialize microbiome preprocessor.

        Args:
            missing_value_strategy: How to handle missing values
            outlier_threshold: Z-score threshold for outlier detection
            normalization_method: Normalization method ("clr", "zscore")
            min_abundance: Minimum relative abundance to keep a taxon
            min_prevalence: Minimum prevalence (fraction of samples) to keep a taxon
            pseudocount: Small value to add before log transformation
        """
        super().__init__(
            omics_type="microbiome",
            missing_value_strategy=missing_value_strategy,
            outlier_threshold=outlier_threshold,
            normalization_method=normalization_method
        )
        self.min_abundance = min_abundance
        self.min_prevalence = min_prevalence
        self.pseudocount = pseudocount

        # Fitted parameters
        self.taxa_to_keep_: Optional[list] = None

    def fit(self, data: pd.DataFrame) -> "MicrobiomePreprocessor":
        """
        Fit the preprocessor on training data.

        Args:
            data: Training data (samples x taxa)

        Returns:
            self
        """
        # Filter low-abundance and low-prevalence taxa
        self.taxa_to_keep_ = self._filter_taxa(data)

        # Call parent fit
        super().fit(data[self.taxa_to_keep_])

        return self

    def _filter_taxa(self, data: pd.DataFrame) -> list:
        """Filter out low-abundance and low-prevalence taxa."""
        taxa_to_keep = []

        for taxon in data.columns:
            # Calculate mean relative abundance
            mean_abundance = data[taxon].mean()

            # Calculate prevalence (fraction of non-zero samples)
            prevalence = (data[taxon] > 0).sum() / len(data)

            # Keep taxon if it meets both criteria
            if mean_abundance >= self.min_abundance and prevalence >= self.min_prevalence:
                taxa_to_keep.append(taxon)

        return taxa_to_keep

    def _omics_specific_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply microbiome-specific processing (CLR transformation).

        CLR (Centered Log-Ratio) transformation is the standard for compositional
        microbiome data. It handles the compositional nature of relative abundances.

        Args:
            data: Normalized data

        Returns:
            CLR-transformed data
        """
        if self.normalization_method == "clr":
            return self._clr_transform(data)
        else:
            # If not using CLR, just return the data
            return data

    def _clr_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Centered Log-Ratio (CLR) transformation.

        CLR(x) = log(x / geometric_mean(x))

        Args:
            data: Input data (relative abundances)

        Returns:
            CLR-transformed data
        """
        data_clr = data.copy()

        # Add pseudocount to avoid log(0)
        data_with_pseudo = data_clr + self.pseudocount

        # Calculate geometric mean for each sample (row-wise)
        geometric_means = data_with_pseudo.apply(
            lambda row: np.exp(np.log(row).mean()),
            axis=1
        )

        # Apply CLR transformation
        for col in data_clr.columns:
            data_clr[col] = np.log(data_with_pseudo[col] / geometric_means)

        return data_clr

    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Override normalization for microbiome data.

        If using CLR, skip standard normalization (it's done in omics-specific processing).
        Otherwise, use parent's normalization.
        """
        if self.normalization_method == "clr":
            # Don't apply standard normalization, CLR will handle it
            return data
        else:
            return super()._normalize(data)

    def get_feature_importance_biological_context(self, feature_name: str, direction: str) -> str:
        """
        Get biological context for a microbiome feature.

        Args:
            feature_name: Name of the taxon
            direction: "up" or "down"

        Returns:
            Biological interpretation
        """
        # This is a simplified version. In production, you'd have a database
        # of taxa and their associations with diseases.

        contexts = {
            "up": {
                "Porphyromonas": "Increased Porphyromonas is associated with periodontal disease and tissue destruction",
                "Treponema": "Elevated Treponema levels indicate active periodontal infection",
                "Tannerella": "High Tannerella forsythia is a marker of severe periodontitis",
                "Prevotella": "Elevated Prevotella may indicate dysbiosis in oral microbiome",
            },
            "down": {
                "Streptococcus": "Decreased Streptococcus may indicate loss of beneficial oral flora",
                "Actinomyces": "Reduced Actinomyces suggests disruption of healthy biofilm",
                "Veillonella": "Lower Veillonella may indicate shift away from health-associated microbiome",
            }
        }

        # Try to match taxon name to known patterns
        for taxon_pattern, context in contexts[direction].items():
            if taxon_pattern.lower() in feature_name.lower():
                return context

        # Default message
        if direction == "up":
            return f"Increased abundance of {feature_name} detected in this sample"
        else:
            return f"Decreased abundance of {feature_name} compared to baseline"

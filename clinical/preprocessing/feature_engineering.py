"""
Feature Engineering Module.

This module provides functions for extracting and selecting the most
informative features from preprocessed omics data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


@dataclass
class FeatureRanking:
    """Feature ranking with statistical information."""

    feature_name: str
    score: float
    p_value: Optional[float]
    fold_change: Optional[float]
    rank: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "score": self.score,
            "p_value": self.p_value,
            "fold_change": self.fold_change,
            "rank": self.rank
        }


class FeatureEngineer:
    """
    Feature engineering for omics data.

    Provides methods for:
    - Differential feature selection (t-test, ANOVA)
    - Feature ranking by importance
    - Dimensionality reduction
    """

    def __init__(
        self,
        method: str = "ttest",  # "ttest", "anova", "mutual_info"
        p_value_threshold: float = 0.05,
        fold_change_threshold: float = 1.5,
        max_features: Optional[int] = None
    ):
        """
        Initialize feature engineer.

        Args:
            method: Feature selection method
            p_value_threshold: P-value threshold for significance
            fold_change_threshold: Minimum fold change to consider
            max_features: Maximum number of features to select (None = no limit)
        """
        self.method = method
        self.p_value_threshold = p_value_threshold
        self.fold_change_threshold = fold_change_threshold
        self.max_features = max_features

        # Fitted parameters
        self.selected_features_: Optional[List[str]] = None
        self.feature_rankings_: Optional[List[FeatureRanking]] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> "FeatureEngineer":
        """
        Fit feature engineer on training data.

        Args:
            X: Feature matrix (samples x features)
            y: Target labels

        Returns:
            self
        """
        # Perform feature selection based on method
        if self.method == "ttest":
            self.feature_rankings_ = self._ttest_selection(X, y)
        elif self.method == "anova":
            self.feature_rankings_ = self._anova_selection(X, y)
        elif self.method == "mutual_info":
            self.feature_rankings_ = self._mutual_info_selection(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Filter by p-value and fold change
        self.selected_features_ = self._filter_features(self.feature_rankings_)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting important features.

        Args:
            X: Feature matrix

        Returns:
            Transformed data with selected features
        """
        if self.selected_features_ is None:
            raise ValueError("FeatureEngineer must be fitted before transform")

        return X[self.selected_features_]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Transformed data
        """
        self.fit(X, y)
        return self.transform(X)

    def _ttest_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[FeatureRanking]:
        """
        Select features using t-test (for binary classification).

        Args:
            X: Feature matrix
            y: Target labels (binary)

        Returns:
            List of feature rankings
        """
        rankings = []
        classes = y.unique()

        if len(classes) != 2:
            raise ValueError("T-test requires binary labels")

        class_0, class_1 = classes

        for feature in X.columns:
            # Split data by class
            group_0 = X.loc[y == class_0, feature]
            group_1 = X.loc[y == class_1, feature]

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group_0, group_1, equal_var=False)

            # Calculate fold change
            mean_0 = group_0.mean()
            mean_1 = group_1.mean()
            fold_change = mean_1 / (mean_0 + 1e-8)

            rankings.append(
                FeatureRanking(
                    feature_name=feature,
                    score=abs(t_stat),
                    p_value=p_value,
                    fold_change=fold_change,
                    rank=0  # Will be set later
                )
            )

        # Sort by score (descending)
        rankings.sort(key=lambda x: x.score, reverse=True)

        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1

        return rankings

    def _anova_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[FeatureRanking]:
        """
        Select features using ANOVA (for multi-class classification).

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            List of feature rankings
        """
        rankings = []
        classes = y.unique()

        for feature in X.columns:
            # Split data by class
            groups = [X.loc[y == c, feature].values for c in classes]

            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups)

            rankings.append(
                FeatureRanking(
                    feature_name=feature,
                    score=f_stat,
                    p_value=p_value,
                    fold_change=None,  # Not applicable for multi-class
                    rank=0
                )
            )

        # Sort by score (descending)
        rankings.sort(key=lambda x: x.score, reverse=True)

        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1

        return rankings

    def _mutual_info_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[FeatureRanking]:
        """
        Select features using mutual information.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            List of feature rankings
        """
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)

        rankings = []
        for feature, score in zip(X.columns, mi_scores):
            rankings.append(
                FeatureRanking(
                    feature_name=feature,
                    score=score,
                    p_value=None,  # MI doesn't provide p-values
                    fold_change=None,
                    rank=0
                )
            )

        # Sort by score (descending)
        rankings.sort(key=lambda x: x.score, reverse=True)

        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1

        return rankings

    def _filter_features(
        self,
        rankings: List[FeatureRanking]
    ) -> List[str]:
        """
        Filter features by p-value and fold change thresholds.

        Args:
            rankings: Feature rankings

        Returns:
            List of selected feature names
        """
        selected = []

        for ranking in rankings:
            # Check p-value (if available)
            if ranking.p_value is not None:
                if ranking.p_value > self.p_value_threshold:
                    continue

            # Check fold change (if available and applicable)
            if ranking.fold_change is not None:
                if abs(ranking.fold_change) < self.fold_change_threshold and \
                   abs(1 / ranking.fold_change) < self.fold_change_threshold:
                    continue

            selected.append(ranking.feature_name)

            # Check max features limit
            if self.max_features is not None and len(selected) >= self.max_features:
                break

        return selected

    def get_top_features(self, n: int = 10) -> List[FeatureRanking]:
        """
        Get top N features by ranking.

        Args:
            n: Number of top features to return

        Returns:
            List of top feature rankings
        """
        if self.feature_rankings_ is None:
            raise ValueError("FeatureEngineer must be fitted first")

        return self.feature_rankings_[:n]

    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get summary of all features as DataFrame.

        Returns:
            DataFrame with feature rankings and statistics
        """
        if self.feature_rankings_ is None:
            raise ValueError("FeatureEngineer must be fitted first")

        summary = pd.DataFrame([r.to_dict() for r in self.feature_rankings_])
        summary["selected"] = summary["feature_name"].isin(self.selected_features_)

        return summary


def calculate_differential_expression(
    X: pd.DataFrame,
    y: pd.Series,
    class_0_label: str,
    class_1_label: str
) -> pd.DataFrame:
    """
    Calculate differential expression between two classes.

    Args:
        X: Feature matrix
        y: Target labels
        class_0_label: Label for class 0 (e.g., "healthy")
        class_1_label: Label for class 1 (e.g., "disease")

    Returns:
        DataFrame with differential expression statistics
    """
    results = []

    for feature in X.columns:
        # Split by class
        group_0 = X.loc[y == class_0_label, feature]
        group_1 = X.loc[y == class_1_label, feature]

        # Calculate statistics
        mean_0 = group_0.mean()
        mean_1 = group_1.mean()
        std_0 = group_0.std()
        std_1 = group_1.std()

        # T-test
        t_stat, p_value = stats.ttest_ind(group_0, group_1, equal_var=False)

        # Fold change
        fold_change = mean_1 / (mean_0 + 1e-8)
        log2_fold_change = np.log2(fold_change + 1e-8)

        results.append({
            "feature": feature,
            f"{class_0_label}_mean": mean_0,
            f"{class_1_label}_mean": mean_1,
            f"{class_0_label}_std": std_0,
            f"{class_1_label}_std": std_1,
            "fold_change": fold_change,
            "log2_fold_change": log2_fold_change,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05 and abs(log2_fold_change) > 0.58  # log2(1.5)
        })

    return pd.DataFrame(results).sort_values("p_value")

"""
Base Expert Agent.

This module defines the abstract base class for expert agents.
Each expert specializes in one type of omics data and provides
structured opinions about patient diagnosis.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from clinical.models.expert_opinion import ExpertOpinion, FeatureImportance


class BaseExpert(ABC):
    """
    Abstract base class for expert agents.

    All expert agents should implement:
    - fit(): Train the expert model on labeled data
    - predict(): Make predictions and generate expert opinions
    - explain(): Provide detailed explanations for predictions
    """

    def __init__(
        self,
        expert_name: str,
        omics_type: str,
        model_version: str = "1.0.0"
    ):
        """
        Initialize base expert.

        Args:
            expert_name: Name of the expert (e.g., "microbiome_expert")
            omics_type: Type of omics data ("microbiome", "metabolome", "proteome")
            model_version: Version of the model
        """
        self.expert_name = expert_name
        self.omics_type = omics_type
        self.model_version = model_version

        # Will be set during fit()
        self.model_ = None
        self.feature_names_ = None
        self.classes_ = None
        self.is_fitted_ = False

        # Threshold for decision-making (used in debate)
        self.decision_threshold_ = 0.5  # Default threshold

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> "BaseExpert":
        """
        Train the expert model on labeled data.

        Args:
            X: Feature matrix (samples x features)
            y: Target labels

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame
    ) -> List[ExpertOpinion]:
        """
        Make predictions and generate expert opinions.

        Args:
            X: Feature matrix (samples x features)

        Returns:
            List of expert opinions (one per sample)
        """
        pass

    @abstractmethod
    def _calculate_confidence(
        self,
        probabilities: np.ndarray,
        feature_importances: np.ndarray
    ) -> float:
        """
        Calculate confidence score for prediction.

        Args:
            probabilities: Class probabilities
            feature_importances: Feature importance values

        Returns:
            Confidence score (0-1)
        """
        pass

    @abstractmethod
    def _get_top_features(
        self,
        sample_data: pd.Series,
        feature_importances: np.ndarray,
        n_features: int = 10
    ) -> List[FeatureImportance]:
        """
        Get top important features for a sample.

        Args:
            sample_data: Data for one sample
            feature_importances: Feature importance values from model
            n_features: Number of top features to return

        Returns:
            List of FeatureImportance objects
        """
        pass

    @abstractmethod
    def _generate_biological_explanation(
        self,
        diagnosis: str,
        top_features: List[FeatureImportance],
        probability: float
    ) -> str:
        """
        Generate natural language biological explanation.

        Args:
            diagnosis: Predicted diagnosis
            top_features: Top important features
            probability: Prediction probability

        Returns:
            Biological explanation text
        """
        pass

    def _generate_evidence_chain(
        self,
        diagnosis: str,
        top_features: List[FeatureImportance],
        probability: float,
        confidence: float
    ) -> List[str]:
        """
        Generate evidence chain for the decision.

        Args:
            diagnosis: Predicted diagnosis
            top_features: Top important features
            probability: Prediction probability
            confidence: Confidence score

        Returns:
            List of evidence statements
        """
        evidence = []

        # Step 1: Model prediction
        evidence.append(
            f"Model predicted '{diagnosis}' with {probability:.1%} probability"
        )

        # Step 2: Feature evidence
        if len(top_features) > 0:
            feature_summary = ", ".join([
                f"{f.feature_name} ({f.direction})"
                for f in top_features[:3]
            ])
            evidence.append(
                f"Key biomarkers identified: {feature_summary}"
            )

        # Step 3: Confidence assessment
        if confidence >= 0.8:
            evidence.append(f"High confidence ({confidence:.1%}) in this prediction")
        elif confidence >= 0.6:
            evidence.append(f"Moderate confidence ({confidence:.1%}) in this prediction")
        else:
            evidence.append(f"Low confidence ({confidence:.1%}) - recommend additional validation")

        # Step 4: Sample quality
        evidence.append(
            f"Based on {len(top_features)} informative features from {self.omics_type} analysis"
        )

        return evidence

    def save_model(self, filepath: str):
        """
        Save trained model to file.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "expert_name": self.expert_name,
            "omics_type": self.omics_type,
            "model_version": self.model_version,
            "model": self.model_,
            "feature_names": self.feature_names_,
            "classes": self.classes_
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """
        Load trained model from file.

        Args:
            filepath: Path to the saved model
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.expert_name = model_data["expert_name"]
        self.omics_type = model_data["omics_type"]
        self.model_version = model_data["model_version"]
        self.model_ = model_data["model"]
        self.feature_names_ = model_data["feature_names"]
        self.classes_ = model_data["classes"]
        self.is_fitted_ = True

    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dictionary with model information
        """
        if not self.is_fitted_:
            return {
                "expert_name": self.expert_name,
                "omics_type": self.omics_type,
                "model_version": self.model_version,
                "is_fitted": False
            }

        return {
            "expert_name": self.expert_name,
            "omics_type": self.omics_type,
            "model_version": self.model_version,
            "is_fitted": True,
            "n_features": len(self.feature_names_) if self.feature_names_ else 0,
            "classes": list(self.classes_) if self.classes_ is not None else [],
            "model_type": type(self.model_).__name__ if self.model_ else None
        }

    def validate_input(self, X: pd.DataFrame):
        """
        Validate input data.

        Args:
            X: Input feature matrix

        Raises:
            ValueError: If input is invalid
        """
        if not self.is_fitted_:
            raise ValueError(f"{self.expert_name} must be fitted before prediction")

        if X.empty:
            raise ValueError("Input data is empty")

        # Check for required features
        missing_features = set(self.feature_names_) - set(X.columns)
        if missing_features:
            raise ValueError(
                f"Input data missing required features: {missing_features}"
            )

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"name='{self.expert_name}', "
            f"omics='{self.omics_type}', "
            f"version='{self.model_version}', "
            f"status='{status}')"
        )

    def adjust_threshold(self, new_threshold: float) -> "BaseExpert":
        """
        Adjust decision threshold for classification.

        This is used in the debate process to check if the expert's opinion
        is near the decision boundary (borderline case).

        Args:
            new_threshold: New threshold for decision-making (0-1)

        Returns:
            self
        """
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        self.decision_threshold_ = new_threshold
        return self

    def predict_with_threshold(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> List[ExpertOpinion]:
        """
        Make predictions using a specific threshold.

        This method re-evaluates the diagnosis based on a different threshold,
        which is useful for debate scenarios where we want to check if the
        expert is on the decision boundary.

        Args:
            X: Feature matrix
            threshold: Classification threshold (default: use self.decision_threshold_)

        Returns:
            List of expert opinions with adjusted decisions
        """
        if threshold is None:
            threshold = self.decision_threshold_

        # Get original predictions
        opinions = self.predict(X)

        # Re-evaluate with new threshold
        adjusted_opinions = []
        for opinion in opinions:
            # If probability is close to threshold, it might be a borderline case
            is_borderline = abs(opinion.probability - threshold) < 0.1

            # Adjust diagnosis if needed
            adjusted_diagnosis = opinion.diagnosis
            if opinion.probability < threshold:
                # Probability below threshold, might need to change diagnosis
                # Find the class with highest probability
                pass  # Keep original for now, subclasses will implement detailed logic

            # Add metadata about threshold adjustment
            adjusted_metadata = opinion.model_metadata.copy()
            adjusted_metadata["decision_threshold"] = threshold
            adjusted_metadata["is_borderline"] = is_borderline

            # Create adjusted opinion
            adjusted_opinion = ExpertOpinion(
                expert_name=opinion.expert_name,
                omics_type=opinion.omics_type,
                diagnosis=adjusted_diagnosis,
                probability=opinion.probability,
                confidence=opinion.confidence,
                top_features=opinion.top_features,
                biological_explanation=opinion.biological_explanation,
                evidence_chain=opinion.evidence_chain,
                model_metadata=adjusted_metadata,
                timestamp=opinion.timestamp
            )

            adjusted_opinions.append(adjusted_opinion)

        return adjusted_opinions


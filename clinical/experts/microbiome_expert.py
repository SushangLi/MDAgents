"""
Microbiome Expert Agent.

Specialized expert for microbiome data analysis using RandomForest classifier
with SHAP for interpretability.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import shap

from clinical.experts.base_expert import BaseExpert
from clinical.models.expert_opinion import ExpertOpinion, FeatureImportance
from clinical.preprocessing.microbiome_preprocessor import MicrobiomePreprocessor


class MicrobiomeExpert(BaseExpert):
    """
    Expert agent for microbiome data.

    Uses RandomForest classifier optimized for high-dimensional sparse features
    typical of microbiome data.
    """

    def __init__(
        self,
        expert_name: str = "microbiome_expert",
        model_version: str = "1.0.0",
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        random_state: int = 42
    ):
        """
        Initialize microbiome expert.

        Args:
            expert_name: Name of the expert
            model_version: Version of the model
            n_estimators: Number of trees in RandomForest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            random_state: Random seed for reproducibility
        """
        super().__init__(
            expert_name=expert_name,
            omics_type="microbiome",
            model_version=model_version
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        # SHAP explainer (will be set after training)
        self.explainer_ = None
        self.feature_importances_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_grid_search: bool = False
    ) -> "MicrobiomeExpert":
        """
        Train the microbiome expert model.

        Args:
            X: Feature matrix (samples x taxa)
            y: Target labels
            use_grid_search: Whether to use grid search for hyperparameter tuning

        Returns:
            self
        """
        self.feature_names_ = list(X.columns)
        self.classes_ = y.unique()

        if use_grid_search:
            # Grid search for hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }

            base_model = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid_search.fit(X, y)

            self.model_ = grid_search.best_estimator_
            # Update hyperparameters
            self.n_estimators = grid_search.best_params_['n_estimators']
            self.max_depth = grid_search.best_params_['max_depth']
            self.min_samples_split = grid_search.best_params_['min_samples_split']
        else:
            # Train with specified hyperparameters
            self.model_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                class_weight='balanced'  # Handle class imbalance
            )
            self.model_.fit(X, y)

        # Store feature importances
        self.feature_importances_ = self.model_.feature_importances_

        # Initialize SHAP explainer
        self.explainer_ = shap.TreeExplainer(self.model_)

        self.is_fitted_ = True
        return self

    def predict(
        self,
        X: pd.DataFrame
    ) -> List[ExpertOpinion]:
        """
        Make predictions and generate expert opinions.

        Args:
            X: Feature matrix (samples x taxa)

        Returns:
            List of expert opinions (one per sample)
        """
        self.validate_input(X)

        # Ensure features are in correct order
        X_aligned = X[self.feature_names_]

        # Get predictions and probabilities
        predictions = self.model_.predict(X_aligned)
        probabilities = self.model_.predict_proba(X_aligned)

        # Calculate SHAP values for explanations
        shap_values = self.explainer_.shap_values(X_aligned)

        # Generate opinion for each sample
        opinions = []
        for i, sample_id in enumerate(X.index):
            diagnosis = predictions[i]
            probability = probabilities[i, list(self.classes_).index(diagnosis)]

            # Get SHAP values for this sample
            if isinstance(shap_values, list):
                # Multi-class: get SHAP for predicted class
                class_idx = list(self.classes_).index(diagnosis)
                sample_shap = shap_values[class_idx][i]
            else:
                # Binary: use SHAP values directly
                sample_shap = shap_values[i]

            # Calculate confidence
            confidence = self._calculate_confidence(probabilities[i], sample_shap)

            # Get top features
            top_features = self._get_top_features(
                X_aligned.iloc[i],
                sample_shap,
                n_features=10
            )

            # Generate explanation
            explanation = self._generate_biological_explanation(
                diagnosis,
                top_features,
                probability
            )

            # Generate evidence chain
            evidence_chain = self._generate_evidence_chain(
                diagnosis,
                top_features,
                probability,
                confidence
            )

            # Create expert opinion
            opinion = ExpertOpinion(
                expert_name=self.expert_name,
                omics_type=self.omics_type,
                diagnosis=diagnosis,
                probability=float(probability),
                confidence=float(confidence),
                top_features=top_features,
                biological_explanation=explanation,
                evidence_chain=evidence_chain,
                model_metadata={
                    "model_type": "RandomForest",
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "model_version": self.model_version
                }
            )

            opinions.append(opinion)

        return opinions

    def _calculate_confidence(
        self,
        probabilities: np.ndarray,
        shap_values: np.ndarray
    ) -> float:
        """
        Calculate confidence score based on prediction probability and SHAP values.

        Args:
            probabilities: Class probabilities
            shap_values: SHAP values for the sample

        Returns:
            Confidence score (0-1)
        """
        # Max probability (certainty of prediction)
        max_prob = probabilities.max()

        # Probability margin (difference between top 2 classes)
        sorted_probs = np.sort(probabilities)
        prob_margin = sorted_probs[-1] - sorted_probs[-2] if len(sorted_probs) > 1 else sorted_probs[-1]

        # SHAP value concentration (how concentrated are important features)
        abs_shap = np.abs(shap_values)
        top_10_shap_sum = np.sum(np.sort(abs_shap)[-10:])
        total_shap_sum = np.sum(abs_shap)
        shap_concentration = top_10_shap_sum / (total_shap_sum + 1e-8)

        # Combined confidence score
        confidence = (
            0.5 * max_prob +
            0.3 * prob_margin +
            0.2 * shap_concentration
        )

        return np.clip(confidence, 0.0, 1.0)

    def _get_top_features(
        self,
        sample_data: pd.Series,
        shap_values: np.ndarray,
        n_features: int = 10
    ) -> List[FeatureImportance]:
        """
        Get top important features using SHAP values.

        Args:
            sample_data: Data for one sample
            shap_values: SHAP values for the sample
            n_features: Number of top features to return

        Returns:
            List of FeatureImportance objects
        """
        # Get indices of top features by absolute SHAP value
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-n_features:][::-1]

        top_features = []
        for idx in top_indices:
            feature_name = self.feature_names_[idx]
            shap_value = shap_values[idx]
            feature_value = sample_data.iloc[idx]

            # Determine direction (up or down regulated)
            direction = "up" if shap_value > 0 else "down"

            # Get biological context from preprocessor
            preprocessor = MicrobiomePreprocessor()
            biological_meaning = preprocessor.get_feature_importance_biological_context(
                feature_name,
                direction
            )

            top_features.append(
                FeatureImportance(
                    feature_name=feature_name,
                    importance_score=float(abs_shap[idx]),
                    direction=direction,
                    biological_meaning=biological_meaning,
                    p_value=None  # SHAP doesn't provide p-values
                )
            )

        return top_features

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
        # Start with diagnosis statement
        explanation = f"Based on microbiome analysis, this sample shows characteristics consistent with {diagnosis} "
        explanation += f"(confidence: {probability:.1%}). "

        # Add feature-based reasoning
        if len(top_features) > 0:
            explanation += "Key microbial signatures include: "

            up_regulated = [f for f in top_features[:3] if f.direction == "up"]
            down_regulated = [f for f in top_features[:3] if f.direction == "down"]

            if up_regulated:
                up_taxa = ", ".join([f.feature_name for f in up_regulated])
                explanation += f"increased abundance of {up_taxa}; "

            if down_regulated:
                down_taxa = ", ".join([f.feature_name for f in down_regulated])
                explanation += f"decreased abundance of {down_taxa}. "

        # Add clinical context based on diagnosis
        if diagnosis.lower() in ["periodontitis", "periodontal disease"]:
            explanation += (
                "The observed microbial dysbiosis is characteristic of periodontal disease, "
                "with an increase in pathogenic bacteria and decrease in health-associated flora."
            )
        elif diagnosis.lower() == "healthy":
            explanation += (
                "The microbiome composition is within normal ranges, "
                "with balanced abundance of commensal and beneficial bacteria."
            )

        return explanation

    def get_model_performance(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.

        Args:
            X_test: Test feature matrix
            y_test: Test labels

        Returns:
            Dictionary with performance metrics
        """
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        predictions = self.model_.predict(X_test[self.feature_names_])

        return {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions, average='weighted'),
            "classification_report": classification_report(y_test, predictions, output_dict=True)
        }

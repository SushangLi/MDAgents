"""
Metabolome Expert Agent.

Specialized expert for metabolomics data analysis using XGBoost classifier
with SHAP for interpretability.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import shap

from clinical.experts.base_expert import BaseExpert
from clinical.models.expert_opinion import ExpertOpinion, FeatureImportance
from clinical.preprocessing.metabolome_preprocessor import MetabolomePreprocessor


class MetabolomeExpert(BaseExpert):
    """
    Expert agent for metabolomics data.

    Uses XGBoost classifier optimized for continuous metabolite concentration features.
    """

    def __init__(
        self,
        expert_name: str = "metabolome_expert",
        model_version: str = "1.0.0",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize metabolome expert.

        Args:
            expert_name: Name of the expert
            model_version: Version of the model
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            random_state: Random seed for reproducibility
        """
        super().__init__(
            expert_name=expert_name,
            omics_type="metabolome",
            model_version=model_version
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        # SHAP explainer (will be set after training)
        self.explainer_ = None
        self.feature_importances_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_grid_search: bool = False
    ) -> "MetabolomeExpert":
        """
        Train the metabolome expert model.

        Args:
            X: Feature matrix (samples x metabolites)
            y: Target labels
            use_grid_search: Whether to use grid search for hyperparameter tuning

        Returns:
            self
        """
        self.feature_names_ = list(X.columns)
        self.classes_ = y.unique()

        # Encode labels for XGBoost
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        self.label_encoder_ = label_encoder

        if use_grid_search:
            # Grid search for hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3]
            }

            base_model = XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid_search.fit(X, y_encoded)

            self.model_ = grid_search.best_estimator_
            # Update hyperparameters
            self.n_estimators = grid_search.best_params_['n_estimators']
            self.max_depth = grid_search.best_params_['max_depth']
            self.learning_rate = grid_search.best_params_['learning_rate']
        else:
            # Train with specified hyperparameters
            self.model_ = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=self._calculate_scale_pos_weight(y_encoded)
            )
            self.model_.fit(X, y_encoded)

        # Store feature importances
        self.feature_importances_ = self.model_.feature_importances_

        # Initialize SHAP explainer
        self.explainer_ = shap.TreeExplainer(self.model_)

        self.is_fitted_ = True
        return self

    def _calculate_scale_pos_weight(self, y: np.ndarray) -> float:
        """Calculate scale_pos_weight for handling class imbalance."""
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) == 2:
            # Binary classification
            return counts[0] / counts[1]
        else:
            # Multi-class: return 1.0 (no scaling)
            return 1.0

    def predict(
        self,
        X: pd.DataFrame
    ) -> List[ExpertOpinion]:
        """
        Make predictions and generate expert opinions.

        Args:
            X: Feature matrix (samples x metabolites)

        Returns:
            List of expert opinions (one per sample)
        """
        self.validate_input(X)

        # Ensure features are in correct order
        X_aligned = X[self.feature_names_]

        # Get predictions and probabilities
        predictions_encoded = self.model_.predict(X_aligned)
        predictions = self.label_encoder_.inverse_transform(predictions_encoded)
        probabilities = self.model_.predict_proba(X_aligned)

        # Calculate SHAP values for explanations
        shap_values = self.explainer_.shap_values(X_aligned)

        # Generate opinion for each sample
        opinions = []
        for i, sample_id in enumerate(X.index):
            diagnosis = predictions[i]
            class_idx = list(self.label_encoder_.classes_).index(diagnosis)
            probability = probabilities[i, class_idx]

            # Get SHAP values for this sample
            if isinstance(shap_values, list):
                # Multi-class: get SHAP for predicted class
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
                    "model_type": "XGBoost",
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
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
        # Max probability
        max_prob = probabilities.max()

        # Probability margin
        sorted_probs = np.sort(probabilities)
        prob_margin = sorted_probs[-1] - sorted_probs[-2] if len(sorted_probs) > 1 else sorted_probs[-1]

        # SHAP value consistency (how consistent are SHAP values)
        abs_shap = np.abs(shap_values)
        top_20_percent = int(len(abs_shap) * 0.2)
        top_shap_sum = np.sum(np.sort(abs_shap)[-top_20_percent:])
        total_shap_sum = np.sum(abs_shap)
        shap_consistency = top_shap_sum / (total_shap_sum + 1e-8)

        # Combined confidence score
        confidence = (
            0.5 * max_prob +
            0.3 * prob_margin +
            0.2 * shap_consistency
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

            # Determine direction
            direction = "up" if shap_value > 0 else "down"

            # Get biological context from preprocessor
            preprocessor = MetabolomePreprocessor()
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
                    p_value=None
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
        explanation = f"Metabolomics analysis indicates {diagnosis} "
        explanation += f"with {probability:.1%} probability. "

        # Add feature-based reasoning
        if len(top_features) > 0:
            explanation += "Key metabolic alterations include: "

            elevated = [f for f in top_features[:3] if f.direction == "up"]
            reduced = [f for f in top_features[:3] if f.direction == "down"]

            if elevated:
                metabolites = ", ".join([f.feature_name for f in elevated])
                explanation += f"elevated levels of {metabolites}; "

            if reduced:
                metabolites = ", ".join([f.feature_name for f in reduced])
                explanation += f"reduced levels of {metabolites}. "

        # Add clinical context
        if diagnosis.lower() in ["periodontitis", "periodontal disease"]:
            explanation += (
                "These metabolic changes suggest active inflammation and tissue breakdown, "
                "consistent with periodontal disease pathology."
            )
        elif diagnosis.lower() == "healthy":
            explanation += (
                "Metabolite profiles are within normal physiological ranges, "
                "indicating balanced metabolic homeostasis."
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

        y_test_encoded = self.label_encoder_.transform(y_test)
        predictions_encoded = self.model_.predict(X_test[self.feature_names_])
        predictions = self.label_encoder_.inverse_transform(predictions_encoded)

        return {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions, average='weighted'),
            "classification_report": classification_report(y_test, predictions, output_dict=True)
        }

"""
Expert Opinion Data Model.

This module defines the data structure for expert agent opinions,
including diagnosis predictions, confidence scores, feature importance,
and biological explanations.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import json


@dataclass
class FeatureImportance:
    """Feature importance with biological context."""

    feature_name: str
    importance_score: float
    direction: str  # "up" or "down"
    biological_meaning: str
    p_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "importance_score": self.importance_score,
            "direction": self.direction,
            "biological_meaning": self.biological_meaning,
            "p_value": self.p_value
        }


@dataclass
class ExpertOpinion:
    """
    Expert opinion from a specialized omics agent.

    Attributes:
        expert_name: Name of the expert (e.g., "microbiome_expert")
        omics_type: Type of omics ("microbiome", "metabolome", "proteome")
        diagnosis: Predicted diagnosis category
        probability: Prediction probability (0-1)
        confidence: Expert confidence score (0-1)
        top_features: Most important features for this prediction
        biological_explanation: Natural language explanation
        evidence_chain: List of reasoning steps
        model_metadata: Model information (version, parameters, etc.)
        timestamp: When the prediction was made
    """

    expert_name: str
    omics_type: str
    diagnosis: str
    probability: float
    confidence: float
    top_features: List[FeatureImportance]
    biological_explanation: str
    evidence_chain: List[str] = field(default_factory=list)
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Validate field values."""
        assert 0 <= self.probability <= 1, "Probability must be between 0 and 1"
        assert 0 <= self.confidence <= 1, "Confidence must be between 0 and 1"
        assert self.omics_type in ["microbiome", "metabolome", "proteome"], \
            f"Invalid omics_type: {self.omics_type}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "expert_name": self.expert_name,
            "omics_type": self.omics_type,
            "diagnosis": self.diagnosis,
            "probability": self.probability,
            "confidence": self.confidence,
            "top_features": [f.to_dict() for f in self.top_features],
            "biological_explanation": self.biological_explanation,
            "evidence_chain": self.evidence_chain,
            "model_metadata": self.model_metadata,
            "timestamp": self.timestamp
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpertOpinion":
        """Create ExpertOpinion from dictionary."""
        # Convert feature importance dicts back to objects
        top_features = [
            FeatureImportance(**f) if isinstance(f, dict) else f
            for f in data.get("top_features", [])
        ]

        return cls(
            expert_name=data["expert_name"],
            omics_type=data["omics_type"],
            diagnosis=data["diagnosis"],
            probability=data["probability"],
            confidence=data["confidence"],
            top_features=top_features,
            biological_explanation=data["biological_explanation"],
            evidence_chain=data.get("evidence_chain", []),
            model_metadata=data.get("model_metadata", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ExpertOpinion":
        """Create ExpertOpinion from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        top_features_str = ", ".join([
            f"{f.feature_name} ({f.direction})"
            for f in self.top_features[:3]
        ])

        return (
            f"{self.expert_name} ({self.omics_type}):\n"
            f"  Diagnosis: {self.diagnosis}\n"
            f"  Probability: {self.probability:.2%}\n"
            f"  Confidence: {self.confidence:.2%}\n"
            f"  Key Features: {top_features_str}\n"
            f"  Explanation: {self.biological_explanation[:100]}..."
        )

    def agrees_with(self, other: "ExpertOpinion", threshold: float = 0.1) -> bool:
        """
        Check if this opinion agrees with another expert opinion.

        Args:
            other: Another expert opinion
            threshold: Probability difference threshold for agreement

        Returns:
            True if diagnoses match and probabilities are within threshold
        """
        return (
            self.diagnosis == other.diagnosis and
            abs(self.probability - other.probability) <= threshold
        )

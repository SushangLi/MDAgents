"""
Diagnosis Result Data Model.

This module defines the data structure for the final diagnosis result,
including expert opinions, conflict resolution details, and final decision.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

from clinical.models.expert_opinion import ExpertOpinion


@dataclass
class ConflictResolution:
    """Details about how expert opinion conflicts were resolved."""

    conflicts_detected: List[str]
    resolution_method: str  # "voting", "rag", "cag", "cmo_reasoning"
    rag_evidence: List[Dict[str, Any]] = field(default_factory=list)
    cag_cases: List[Dict[str, Any]] = field(default_factory=list)
    cmo_reasoning: str = ""
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conflicts_detected": self.conflicts_detected,
            "resolution_method": self.resolution_method,
            "rag_evidence": self.rag_evidence,
            "cag_cases": self.cag_cases,
            "cmo_reasoning": self.cmo_reasoning,
            "confidence_score": self.confidence_score
        }


@dataclass
class DiagnosisResult:
    """
    Final diagnosis result from the clinical diagnosis system.

    Attributes:
        patient_id: Patient identifier
        diagnosis: Final diagnosis category
        confidence: Overall confidence score (0-1)
        expert_opinions: List of expert opinions from all agents
        conflict_resolution: Details about conflict resolution (if any)
        key_biomarkers: Most important biomarkers across all omics
        clinical_recommendations: Recommended next steps
        explanation: Natural language explanation of the decision
        references: Supporting literature references
        metadata: Additional metadata (model versions, timing, etc.)
        timestamp: When the diagnosis was made
    """

    patient_id: str
    diagnosis: str
    confidence: float
    expert_opinions: List[ExpertOpinion]
    key_biomarkers: List[Dict[str, Any]]
    clinical_recommendations: List[str]
    explanation: str
    conflict_resolution: Optional[ConflictResolution] = None
    references: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Validate field values."""
        assert 0 <= self.confidence <= 1, "Confidence must be between 0 and 1"
        assert len(self.expert_opinions) > 0, "At least one expert opinion required"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "patient_id": self.patient_id,
            "diagnosis": self.diagnosis,
            "confidence": self.confidence,
            "expert_opinions": [op.to_dict() for op in self.expert_opinions],
            "conflict_resolution": self.conflict_resolution.to_dict() if self.conflict_resolution else None,
            "key_biomarkers": self.key_biomarkers,
            "clinical_recommendations": self.clinical_recommendations,
            "explanation": self.explanation,
            "references": self.references,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosisResult":
        """Create DiagnosisResult from dictionary."""
        expert_opinions = [
            ExpertOpinion.from_dict(op) if isinstance(op, dict) else op
            for op in data["expert_opinions"]
        ]

        conflict_resolution = None
        if data.get("conflict_resolution"):
            conflict_resolution = ConflictResolution(**data["conflict_resolution"])

        return cls(
            patient_id=data["patient_id"],
            diagnosis=data["diagnosis"],
            confidence=data["confidence"],
            expert_opinions=expert_opinions,
            conflict_resolution=conflict_resolution,
            key_biomarkers=data["key_biomarkers"],
            clinical_recommendations=data["clinical_recommendations"],
            explanation=data["explanation"],
            references=data.get("references", []),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DiagnosisResult":
        """Create DiagnosisResult from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        expert_summary = "\n".join([
            f"  - {op.expert_name}: {op.diagnosis} ({op.probability:.2%})"
            for op in self.expert_opinions
        ])

        conflict_info = ""
        if self.conflict_resolution:
            conflict_info = f"\n  Conflicts Resolved: {', '.join(self.conflict_resolution.conflicts_detected)}"

        return (
            f"=== Diagnosis Report for Patient {self.patient_id} ===\n"
            f"Final Diagnosis: {self.diagnosis}\n"
            f"Confidence: {self.confidence:.2%}\n"
            f"\nExpert Opinions:\n{expert_summary}"
            f"{conflict_info}\n"
            f"\nKey Biomarkers: {len(self.key_biomarkers)}\n"
            f"Recommendations: {len(self.clinical_recommendations)}\n"
            f"References: {len(self.references)}\n"
            f"Timestamp: {self.timestamp}\n"
            f"{'=' * 50}"
        )

    def has_conflict(self) -> bool:
        """Check if there were conflicts among expert opinions."""
        if len(self.expert_opinions) < 2:
            return False

        # Check if all experts agree on diagnosis
        diagnoses = {op.diagnosis for op in self.expert_opinions}
        return len(diagnoses) > 1

    def get_expert_consensus_rate(self) -> float:
        """Get the rate of expert consensus."""
        if not self.expert_opinions:
            return 0.0

        diagnoses = [op.diagnosis for op in self.expert_opinions]
        most_common_diagnosis = max(set(diagnoses), key=diagnoses.count)
        consensus_count = diagnoses.count(most_common_diagnosis)

        return consensus_count / len(diagnoses)

    def get_average_expert_confidence(self) -> float:
        """Get the average confidence across all experts."""
        if not self.expert_opinions:
            return 0.0

        return sum(op.confidence for op in self.expert_opinions) / len(self.expert_opinions)

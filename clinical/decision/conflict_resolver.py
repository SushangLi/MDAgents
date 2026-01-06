"""
Conflict Resolver Module.

Detects and classifies conflicts between expert opinions to determine
if debate mechanism should be triggered.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from clinical.models.expert_opinion import ExpertOpinion


class ConflictType(Enum):
    """Types of conflicts between expert opinions."""

    NO_CONFLICT = "no_conflict"
    DIAGNOSIS_DISAGREEMENT = "diagnosis_disagreement"
    LOW_CONFIDENCE = "low_confidence"
    BORDERLINE_CASE = "borderline_case"
    HIGH_UNCERTAINTY = "high_uncertainty"


@dataclass
class ConflictAnalysis:
    """Analysis of expert opinion conflicts."""

    has_conflict: bool
    conflict_types: List[ConflictType]
    diagnosis_distribution: Dict[str, int]
    confidence_scores: List[float]
    avg_confidence: float
    requires_debate: bool
    requires_rag: bool
    requires_cag: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_conflict": self.has_conflict,
            "conflict_types": [ct.value for ct in self.conflict_types],
            "diagnosis_distribution": self.diagnosis_distribution,
            "confidence_scores": self.confidence_scores,
            "avg_confidence": self.avg_confidence,
            "requires_debate": self.requires_debate,
            "requires_rag": self.requires_rag,
            "requires_cag": self.requires_cag,
            "metadata": self.metadata
        }


class ConflictResolver:
    """
    Detects and resolves conflicts between expert opinions.

    Implements conflict detection logic to determine when debate
    mechanism should be triggered.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        borderline_threshold: float = 0.1,
        unanimity_required: bool = False
    ):
        """
        Initialize conflict resolver.

        Args:
            confidence_threshold: Minimum confidence for non-conflict
            borderline_threshold: Threshold for borderline probability
            unanimity_required: If True, require all experts to agree
        """
        self.confidence_threshold = confidence_threshold
        self.borderline_threshold = borderline_threshold
        self.unanimity_required = unanimity_required

    def detect_conflict(
        self,
        expert_opinions: List[ExpertOpinion]
    ) -> ConflictAnalysis:
        """
        Detect conflicts in expert opinions.

        Args:
            expert_opinions: List of expert opinions

        Returns:
            ConflictAnalysis with conflict details
        """
        if not expert_opinions:
            raise ValueError("No expert opinions provided")

        # Collect diagnoses and confidences
        diagnoses = [op.diagnosis for op in expert_opinions]
        confidences = [op.confidence for op in expert_opinions]

        # Calculate diagnosis distribution
        diagnosis_dist = {}
        for diagnosis in diagnoses:
            diagnosis_dist[diagnosis] = diagnosis_dist.get(diagnosis, 0) + 1

        # Calculate average confidence
        avg_confidence = np.mean(confidences)

        # Detect conflict types
        conflict_types = []

        # 1. Diagnosis disagreement
        if len(diagnosis_dist) > 1:
            # Multiple different diagnoses
            if self.unanimity_required:
                conflict_types.append(ConflictType.DIAGNOSIS_DISAGREEMENT)
            else:
                # Check if there's a clear majority
                max_count = max(diagnosis_dist.values())
                total_count = len(expert_opinions)

                # No clear majority (< 2/3 agreement)
                if max_count / total_count < 0.67:
                    conflict_types.append(ConflictType.DIAGNOSIS_DISAGREEMENT)

        # 2. Low confidence
        if avg_confidence < self.confidence_threshold:
            conflict_types.append(ConflictType.LOW_CONFIDENCE)

        # 3. Borderline case (probability near decision threshold)
        borderline_count = 0
        for opinion in expert_opinions:
            # Check if probability is near 0.5 (decision boundary)
            if abs(opinion.probability - 0.5) < self.borderline_threshold:
                borderline_count += 1

        if borderline_count >= 2:  # At least 2 experts near boundary
            conflict_types.append(ConflictType.BORDERLINE_CASE)

        # 4. High uncertainty (large variance in confidence)
        confidence_variance = np.var(confidences)
        if confidence_variance > 0.05:  # High variance
            conflict_types.append(ConflictType.HIGH_UNCERTAINTY)

        # Determine if conflict exists
        has_conflict = len(conflict_types) > 0

        # Determine if debate is required
        requires_debate = has_conflict and any([
            ConflictType.DIAGNOSIS_DISAGREEMENT in conflict_types,
            ConflictType.BORDERLINE_CASE in conflict_types
        ])

        # Determine if RAG is needed (for literature evidence)
        requires_rag = ConflictType.DIAGNOSIS_DISAGREEMENT in conflict_types

        # Determine if CAG is needed (for similar cases)
        requires_cag = any([
            ConflictType.LOW_CONFIDENCE in conflict_types,
            ConflictType.BORDERLINE_CASE in conflict_types
        ])

        # Add no conflict type if no conflicts detected
        if not has_conflict:
            conflict_types.append(ConflictType.NO_CONFLICT)

        # Build metadata
        metadata = {
            "n_experts": len(expert_opinions),
            "n_unique_diagnoses": len(diagnosis_dist),
            "confidence_variance": float(confidence_variance),
            "borderline_count": borderline_count,
            "expert_names": [op.expert_name for op in expert_opinions]
        }

        return ConflictAnalysis(
            has_conflict=has_conflict,
            conflict_types=conflict_types,
            diagnosis_distribution=diagnosis_dist,
            confidence_scores=confidences,
            avg_confidence=avg_confidence,
            requires_debate=requires_debate,
            requires_rag=requires_rag,
            requires_cag=requires_cag,
            metadata=metadata
        )

    def get_majority_diagnosis(
        self,
        expert_opinions: List[ExpertOpinion]
    ) -> Optional[str]:
        """
        Get majority diagnosis from expert opinions.

        Args:
            expert_opinions: List of expert opinions

        Returns:
            Majority diagnosis or None if no majority
        """
        if not expert_opinions:
            return None

        # Count diagnoses
        diagnosis_counts = {}
        for opinion in expert_opinions:
            diagnosis_counts[opinion.diagnosis] = \
                diagnosis_counts.get(opinion.diagnosis, 0) + 1

        # Get diagnosis with max count
        max_count = max(diagnosis_counts.values())
        majority_diagnoses = [
            d for d, count in diagnosis_counts.items()
            if count == max_count
        ]

        # Return if unique majority
        if len(majority_diagnoses) == 1:
            return majority_diagnoses[0]

        return None

    def get_weighted_diagnosis(
        self,
        expert_opinions: List[ExpertOpinion]
    ) -> Optional[str]:
        """
        Get diagnosis weighted by confidence scores.

        Args:
            expert_opinions: List of expert opinions

        Returns:
            Weighted diagnosis
        """
        if not expert_opinions:
            return None

        # Calculate weighted votes
        diagnosis_weights = {}
        for opinion in expert_opinions:
            if opinion.diagnosis not in diagnosis_weights:
                diagnosis_weights[opinion.diagnosis] = 0.0

            # Weight by confidence
            diagnosis_weights[opinion.diagnosis] += opinion.confidence

        # Get diagnosis with max weight
        max_weight = max(diagnosis_weights.values())
        weighted_diagnoses = [
            d for d, weight in diagnosis_weights.items()
            if weight == max_weight
        ]

        # Return if unique
        if len(weighted_diagnoses) == 1:
            return weighted_diagnoses[0]

        return None

    def format_conflict_summary(
        self,
        conflict_analysis: ConflictAnalysis
    ) -> str:
        """
        Format conflict analysis as human-readable summary.

        Args:
            conflict_analysis: Conflict analysis result

        Returns:
            Formatted summary string
        """
        lines = []

        lines.append("## Conflict Analysis Summary\n")

        # Conflict status
        if conflict_analysis.has_conflict:
            lines.append("**Status**: ⚠ Conflict Detected\n")
        else:
            lines.append("**Status**: ✓ No Conflict\n")

        # Conflict types
        if conflict_analysis.conflict_types:
            lines.append("\n**Conflict Types**:")
            for ct in conflict_analysis.conflict_types:
                if ct != ConflictType.NO_CONFLICT:
                    lines.append(f"- {ct.value.replace('_', ' ').title()}")

        # Diagnosis distribution
        lines.append("\n**Diagnosis Distribution**:")
        for diagnosis, count in sorted(
            conflict_analysis.diagnosis_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percentage = (count / sum(conflict_analysis.diagnosis_distribution.values())) * 100
            lines.append(f"- {diagnosis}: {count} experts ({percentage:.0f}%)")

        # Confidence scores
        lines.append(f"\n**Average Confidence**: {conflict_analysis.avg_confidence:.1%}")
        lines.append(f"**Confidence Range**: {min(conflict_analysis.confidence_scores):.1%} - {max(conflict_analysis.confidence_scores):.1%}")

        # Actions required
        lines.append("\n**Actions Required**:")
        if conflict_analysis.requires_debate:
            lines.append("- ✓ Debate mechanism")
        if conflict_analysis.requires_rag:
            lines.append("- ✓ RAG literature search")
        if conflict_analysis.requires_cag:
            lines.append("- ✓ CAG case retrieval")
        if not any([conflict_analysis.requires_debate,
                   conflict_analysis.requires_rag,
                   conflict_analysis.requires_cag]):
            lines.append("- None (proceed with majority vote)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConflictResolver("
            f"confidence_threshold={self.confidence_threshold}, "
            f"borderline_threshold={self.borderline_threshold})"
        )

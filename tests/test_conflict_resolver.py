"""
Test Conflict Resolution and Debate System.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clinical.decision.conflict_resolver import ConflictResolver, ConflictType
from clinical.decision.debate_system import DebateSystem, DebateConfig
from clinical.models.expert_opinion import ExpertOpinion, FeatureImportance


def create_test_opinions(scenario="consensus"):
    """Create test expert opinions for different scenarios."""
    if scenario == "consensus":
        # All experts agree
        opinions = [
            ExpertOpinion(
                expert_name="microbiome_expert",
                omics_type="microbiome",
                diagnosis="Periodontitis",
                probability=0.85,
                confidence=0.82,
                top_features=[
                    FeatureImportance("Porphyromonas_gingivalis", 0.25, "up"),
                    FeatureImportance("Treponema_denticola", 0.18, "up")
                ],
                biological_explanation="Elevated periodontal pathogens",
                evidence_chain=["High pathogen abundance", "Low beneficial bacteria"],
                model_metadata={},
                timestamp=""
            ),
            ExpertOpinion(
                expert_name="metabolome_expert",
                omics_type="metabolome",
                diagnosis="Periodontitis",
                probability=0.78,
                confidence=0.75,
                top_features=[
                    FeatureImportance("Butyrate", 0.22, "up"),
                    FeatureImportance("Propionate", 0.15, "up")
                ],
                biological_explanation="Elevated inflammatory metabolites",
                evidence_chain=["High SCFA levels", "Metabolic dysbiosis"],
                model_metadata={},
                timestamp=""
            ),
            ExpertOpinion(
                expert_name="proteome_expert",
                omics_type="proteome",
                diagnosis="Periodontitis",
                probability=0.82,
                confidence=0.80,
                top_features=[
                    FeatureImportance("MMP9", 0.28, "up"),
                    FeatureImportance("IL6", 0.20, "up")
                ],
                biological_explanation="Elevated inflammatory markers",
                evidence_chain=["High MMP9", "Elevated cytokines"],
                model_metadata={},
                timestamp=""
            )
        ]

    elif scenario == "conflict":
        # Experts disagree
        opinions = [
            ExpertOpinion(
                expert_name="microbiome_expert",
                omics_type="microbiome",
                diagnosis="Periodontitis",
                probability=0.75,
                confidence=0.70,
                top_features=[FeatureImportance("Porphyromonas_gingivalis", 0.25, "up")],
                biological_explanation="Moderate periodontal pathogens",
                evidence_chain=["Pathogen detected"],
                model_metadata={},
                timestamp=""
            ),
            ExpertOpinion(
                expert_name="metabolome_expert",
                omics_type="metabolome",
                diagnosis="Healthy",
                probability=0.65,
                confidence=0.60,
                top_features=[FeatureImportance("Acetate", 0.18, "down")],
                biological_explanation="Near-normal metabolite profile",
                evidence_chain=["Low metabolic dysbiosis"],
                model_metadata={},
                timestamp=""
            ),
            ExpertOpinion(
                expert_name="proteome_expert",
                omics_type="proteome",
                diagnosis="Periodontitis",
                probability=0.70,
                confidence=0.68,
                top_features=[FeatureImportance("MMP9", 0.22, "up")],
                biological_explanation="Mild inflammation",
                evidence_chain=["Moderate MMP9 elevation"],
                model_metadata={},
                timestamp=""
            )
        ]

    elif scenario == "low_confidence":
        # Low confidence across experts
        opinions = [
            ExpertOpinion(
                expert_name="microbiome_expert",
                omics_type="microbiome",
                diagnosis="Periodontitis",
                probability=0.55,
                confidence=0.50,
                top_features=[],
                biological_explanation="Unclear pattern",
                evidence_chain=[],
                model_metadata={},
                timestamp=""
            ),
            ExpertOpinion(
                expert_name="metabolome_expert",
                omics_type="metabolome",
                diagnosis="Periodontitis",
                probability=0.58,
                confidence=0.52,
                top_features=[],
                biological_explanation="Weak signal",
                evidence_chain=[],
                model_metadata={},
                timestamp=""
            ),
            ExpertOpinion(
                expert_name="proteome_expert",
                omics_type="proteome",
                diagnosis="Periodontitis",
                probability=0.60,
                confidence=0.55,
                top_features=[],
                biological_explanation="Low certainty",
                evidence_chain=[],
                model_metadata={},
                timestamp=""
            )
        ]

    return opinions


def test_conflict_detection_consensus():
    """Test conflict detection with consensus."""
    resolver = ConflictResolver()
    opinions = create_test_opinions("consensus")

    conflict = resolver.detect_conflict(opinions)

    assert conflict is not None
    assert not conflict.has_conflict or ConflictType.NO_CONFLICT in conflict.conflict_types
    assert not conflict.requires_debate

    print(f"\nConsensus Scenario:")
    print(f"  Conflict detected: {conflict.has_conflict}")
    print(f"  Requires debate: {conflict.requires_debate}")
    print(f"  Average confidence: {conflict.avg_confidence:.2%}")


def test_conflict_detection_disagreement():
    """Test conflict detection with disagreement."""
    resolver = ConflictResolver()
    opinions = create_test_opinions("conflict")

    conflict = resolver.detect_conflict(opinions)

    assert conflict is not None
    assert conflict.has_conflict
    assert ConflictType.DIAGNOSIS_DISAGREEMENT in conflict.conflict_types
    assert conflict.requires_debate

    print(f"\nConflict Scenario:")
    print(f"  Conflict detected: {conflict.has_conflict}")
    print(f"  Conflict types: {[ct.value for ct in conflict.conflict_types]}")
    print(f"  Requires debate: {conflict.requires_debate}")
    print(f"  Diagnosis distribution: {conflict.diagnosis_distribution}")


def test_conflict_detection_low_confidence():
    """Test conflict detection with low confidence."""
    resolver = ConflictResolver(confidence_threshold=0.7)
    opinions = create_test_opinions("low_confidence")

    conflict = resolver.detect_conflict(opinions)

    assert conflict is not None
    assert conflict.has_conflict
    assert ConflictType.LOW_CONFIDENCE in conflict.conflict_types

    print(f"\nLow Confidence Scenario:")
    print(f"  Conflict types: {[ct.value for ct in conflict.conflict_types]}")
    print(f"  Average confidence: {conflict.avg_confidence:.2%}")
    print(f"  Requires CAG: {conflict.requires_cag}")


def test_majority_diagnosis():
    """Test majority diagnosis extraction."""
    resolver = ConflictResolver()
    opinions = create_test_opinions("conflict")

    majority = resolver.get_majority_diagnosis(opinions)

    assert majority is not None
    assert isinstance(majority, str)

    print(f"\nMajority Diagnosis: {majority}")


def test_weighted_diagnosis():
    """Test weighted diagnosis calculation."""
    resolver = ConflictResolver()
    opinions = create_test_opinions("conflict")

    weighted = resolver.get_weighted_diagnosis(opinions)

    assert weighted is not None
    assert isinstance(weighted, str)

    print(f"\nWeighted Diagnosis: {weighted}")


def test_conflict_summary():
    """Test conflict summary formatting."""
    resolver = ConflictResolver()
    opinions = create_test_opinions("conflict")

    conflict = resolver.detect_conflict(opinions)
    summary = resolver.format_conflict_summary(conflict)

    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "Conflict Analysis Summary" in summary

    print(f"\n{summary}")


def test_debate_system_initialization():
    """Test debate system initialization."""
    debate_system = DebateSystem(
        config=DebateConfig(
            max_rounds=3,
            threshold_adjustment=0.1
        )
    )

    assert debate_system is not None
    assert debate_system.config.max_rounds == 3
    assert debate_system.config.threshold_adjustment == 0.1


@pytest.mark.asyncio
async def test_debate_workflow_consensus():
    """Test debate workflow with consensus (should skip debate)."""
    debate_system = DebateSystem()
    opinions = create_test_opinions("consensus")

    result = debate_system.run_debate(opinions)

    assert result is not None
    assert "final_diagnosis" in result
    assert result["final_diagnosis"] is not None
    assert not result.get("requires_debate", True)

    print(f"\nDebate Result (Consensus):")
    print(f"  Final diagnosis: {result['final_diagnosis']}")
    print(f"  Debate required: {result.get('requires_debate', False)}")


@pytest.mark.asyncio
async def test_debate_workflow_conflict():
    """Test debate workflow with conflict."""
    debate_system = DebateSystem(
        config=DebateConfig(max_rounds=2)
    )
    opinions = create_test_opinions("conflict")

    result = debate_system.run_debate(opinions)

    assert result is not None
    assert "final_diagnosis" in result
    assert "current_round" in result

    print(f"\nDebate Result (Conflict):")
    print(f"  Final diagnosis: {result['final_diagnosis']}")
    print(f"  Rounds completed: {result['current_round']}")
    print(f"  Debate resolved: {result.get('debate_resolved', False)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

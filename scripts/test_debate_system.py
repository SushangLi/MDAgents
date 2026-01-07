"""
Test Debate System with Deliberately Conflicting Expert Opinions.

This script demonstrates the LangGraph debate mechanism by creating
conflicting expert opinions and showing how the system resolves them
through threshold adjustment and RAG/CAG queries.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clinical.models.expert_opinion import ExpertOpinion, FeatureImportance
from clinical.decision.conflict_resolver import ConflictResolver
import asyncio


def create_feature_importance(name: str, score: float, direction: str, meaning: str) -> FeatureImportance:
    """Helper to create FeatureImportance objects."""
    return FeatureImportance(
        feature_name=name,
        importance_score=score,
        direction=direction,
        biological_meaning=meaning
    )


def create_conflicting_scenario_1():
    """
    Scenario 1: Strong conflict - all experts disagree
    Expected: Conflict detected
    """
    print("\n" + "="*70)
    print("SCENARIO 1: Strong Conflict - All Experts Disagree")
    print("="*70)

    opinions = [
        ExpertOpinion(
            expert_name="microbiome_expert",
            omics_type="microbiome",
            diagnosis="Periodontitis",
            probability=0.85,
            confidence=0.90,
            biological_explanation="HIGH levels of P.gingivalis (20x) and T.denticola (18x) detected",
            top_features=[
                create_feature_importance("Porphyromonas_gingivalis", 0.35, "up", "Pathogenic bacteria"),
                create_feature_importance("Treponema_denticola", 0.28, "up", "Periodontal pathogen"),
                create_feature_importance("Fusobacterium_nucleatum", 0.15, "up", "Inflammatory bacteria")
            ],
            evidence_chain=["High pathogenic bacteria detected", "Consistent with periodontal disease"],
            model_metadata={"version": "v1.0.0", "threshold": 0.5}
        ),
        ExpertOpinion(
            expert_name="metabolome_expert",
            omics_type="metabolome",
            diagnosis="Diabetes",
            probability=0.82,
            confidence=0.88,
            biological_explanation="HIGH glucose (25x) and lactate (22x) levels",
            top_features=[
                create_feature_importance("Glucose", 0.40, "up", "Elevated blood sugar"),
                create_feature_importance("Lactate", 0.32, "up", "Metabolic dysfunction"),
                create_feature_importance("Pyruvate", 0.12, "up", "Energy metabolism issue")
            ],
            evidence_chain=["Elevated glucose detected", "Metabolic profile suggests diabetes"],
            model_metadata={"version": "v1.0.0", "threshold": 0.5}
        ),
        ExpertOpinion(
            expert_name="proteome_expert",
            omics_type="proteome",
            diagnosis="Healthy",
            probability=0.78,
            confidence=0.85,
            biological_explanation="HIGH protective IgA (22x) and Lactoferrin (20x)",
            top_features=[
                create_feature_importance("IgA", 0.38, "up", "Protective antibody"),
                create_feature_importance("Lactoferrin", 0.30, "up", "Antimicrobial protein"),
                create_feature_importance("Lysozyme", 0.15, "up", "Enzyme protection")
            ],
            evidence_chain=["High protective proteins detected", "Immune system functioning well"],
            model_metadata={"version": "v1.0.0", "threshold": 0.5}
        )
    ]

    return opinions


def create_conflicting_scenario_2():
    """
    Scenario 2: Borderline conflict - two agree, one borderline disagrees
    Expected: Borderline conflict detected
    """
    print("\n" + "="*70)
    print("SCENARIO 2: Borderline Conflict - Two Agree, One Borderline")
    print("="*70)

    opinions = [
        ExpertOpinion(
            expert_name="microbiome_expert",
            omics_type="microbiome",
            diagnosis="Periodontitis",
            probability=0.88,
            confidence=0.92,
            biological_explanation="HIGH levels of P.gingivalis (22x) and T.denticola (20x) detected",
            top_features=[
                create_feature_importance("Porphyromonas_gingivalis", 0.40, "up", "Pathogenic bacteria"),
                create_feature_importance("Treponema_denticola", 0.32, "up", "Periodontal pathogen"),
                create_feature_importance("Prevotella_intermedia", 0.15, "up", "Opportunistic pathogen")
            ],
            evidence_chain=["Strong pathogenic signal", "Consistent with periodontal disease"],
            model_metadata={"version": "v1.0.0", "threshold": 0.5}
        ),
        ExpertOpinion(
            expert_name="metabolome_expert",
            omics_type="metabolome",
            diagnosis="Periodontitis",
            probability=0.86,
            confidence=0.90,
            biological_explanation="HIGH butyrate (28x) and propionate (25x) levels",
            top_features=[
                create_feature_importance("Butyrate", 0.38, "up", "Inflammatory metabolite"),
                create_feature_importance("Propionate", 0.35, "up", "Bacterial fermentation product"),
                create_feature_importance("Acetate", 0.12, "up", "Short-chain fatty acid")
            ],
            evidence_chain=["Elevated inflammatory metabolites", "Supports periodontal diagnosis"],
            model_metadata={"version": "v1.0.0", "threshold": 0.5}
        ),
        ExpertOpinion(
            expert_name="proteome_expert",
            omics_type="proteome",
            diagnosis="Healthy",
            probability=0.55,  # Borderline - just above threshold
            confidence=0.60,
            biological_explanation="IgA levels slightly elevated but MMP9 also present",
            top_features=[
                create_feature_importance("IgA", 0.30, "up", "Protective antibody"),
                create_feature_importance("MMP9", 0.28, "up", "Matrix metalloproteinase"),
                create_feature_importance("IL6", 0.25, "up", "Inflammatory cytokine")
            ],
            evidence_chain=["Mixed signals detected", "Borderline classification"],
            model_metadata={"version": "v1.0.0", "threshold": 0.5, "borderline": True}
        )
    ]

    return opinions


def create_no_conflict_scenario():
    """
    Scenario 3: No conflict - all experts agree
    Expected: No conflict, quick consensus
    """
    print("\n" + "="*70)
    print("SCENARIO 3: No Conflict - All Experts Agree")
    print("="*70)

    opinions = [
        ExpertOpinion(
            expert_name="microbiome_expert",
            omics_type="microbiome",
            diagnosis="Periodontitis",
            probability=0.92,
            confidence=0.95,
            biological_explanation="EXTREMELY HIGH levels of P.gingivalis (24x) and T.denticola (22x)",
            top_features=[
                create_feature_importance("Porphyromonas_gingivalis", 0.42, "up", "Major periodontal pathogen"),
                create_feature_importance("Treponema_denticola", 0.35, "up", "Periodontal spirochete"),
                create_feature_importance("Tannerella_forsythia", 0.15, "up", "Red complex bacteria")
            ],
            evidence_chain=["All red complex bacteria elevated", "Strong periodontal signal"],
            model_metadata={"version": "v1.0.0", "threshold": 0.5}
        ),
        ExpertOpinion(
            expert_name="metabolome_expert",
            omics_type="metabolome",
            diagnosis="Periodontitis",
            probability=0.90,
            confidence=0.93,
            biological_explanation="EXTREMELY HIGH butyrate (30x) and propionate (28x) from bacterial fermentation",
            top_features=[
                create_feature_importance("Butyrate", 0.40, "up", "Bacterial metabolite"),
                create_feature_importance("Propionate", 0.38, "up", "Fermentation product"),
                create_feature_importance("Indole", 0.15, "up", "Protein degradation marker")
            ],
            evidence_chain=["High fermentation products", "Consistent with periodontal disease"],
            model_metadata={"version": "v1.0.0", "threshold": 0.5}
        ),
        ExpertOpinion(
            expert_name="proteome_expert",
            omics_type="proteome",
            diagnosis="Periodontitis",
            probability=0.88,
            confidence=0.91,
            biological_explanation="HIGH inflammatory markers MMP9 (25x) and IL6 (23x)",
            top_features=[
                create_feature_importance("MMP9", 0.45, "up", "Tissue destruction enzyme"),
                create_feature_importance("IL6", 0.38, "up", "Pro-inflammatory cytokine"),
                create_feature_importance("TNF", 0.12, "up", "Inflammatory mediator")
            ],
            evidence_chain=["Severe inflammation detected", "Supports periodontal diagnosis"],
            model_metadata={"version": "v1.0.0", "threshold": 0.5}
        )
    ]

    return opinions


def print_opinions(opinions):
    """Print expert opinions in a readable format."""
    print("\nExpert Opinions:")
    for i, opinion in enumerate(opinions, 1):
        print(f"\n  {i}. {opinion.expert_name} ({opinion.omics_type}):")
        print(f"     Diagnosis: {opinion.diagnosis}")
        print(f"     Probability: {opinion.probability:.2f}")
        print(f"     Confidence: {opinion.confidence:.2f}")
        print(f"     Top Features: {', '.join([f.feature_name for f in opinion.top_features[:3]])}")


def test_scenario(opinions):
    """Test a scenario with conflict detection."""
    # Print opinions
    print_opinions(opinions)

    # Detect conflicts
    print("\n--- Conflict Detection ---")
    resolver = ConflictResolver()
    conflict_analysis = resolver.detect_conflict(opinions)

    print(f"Has conflict: {conflict_analysis.has_conflict}")
    print(f"Conflict types: {[ct.value for ct in conflict_analysis.conflict_types]}")
    print(f"Average confidence: {conflict_analysis.avg_confidence:.2f}")
    print(f"Requires debate: {conflict_analysis.requires_debate}")
    print(f"Requires RAG: {conflict_analysis.requires_rag}")
    print(f"Requires CAG: {conflict_analysis.requires_cag}")
    if conflict_analysis.metadata:
        print(f"Metadata: {conflict_analysis.metadata}")

    # Show diagnosis distribution
    print(f"\nDiagnosis distribution: {conflict_analysis.diagnosis_distribution}")

    return conflict_analysis


async def main():
    """Run all test scenarios."""
    print("\n" + "#"*70)
    print("# DEBATE SYSTEM DEMONSTRATION")
    print("#"*70)
    print("\nThis demonstrates the conflict detection mechanism with:")
    print("- Conflict detection (multiple types)")
    print("- Expert opinion analysis")
    print("- Diagnosis distribution")
    print("#"*70)

    # Scenario 1: Strong conflict
    print("\n" + "="*70)
    print("SCENARIO 1: Strong Conflict - All Experts Disagree")
    print("="*70)
    scenario1_opinions = create_conflicting_scenario_1()
    test_scenario(scenario1_opinions)

    # Scenario 2: Borderline conflict
    print("\n" + "="*70)
    print("SCENARIO 2: Borderline Conflict - Two Agree, One Borderline")
    print("="*70)
    scenario2_opinions = create_conflicting_scenario_2()
    test_scenario(scenario2_opinions)

    # Scenario 3: No conflict
    print("\n" + "="*70)
    print("SCENARIO 3: No Conflict - All Experts Agree")
    print("="*70)
    scenario3_opinions = create_no_conflict_scenario()
    test_scenario(scenario3_opinions)

    print("\n" + "="*70)
    print("✓ All scenarios tested successfully!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())

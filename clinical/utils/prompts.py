"""
Prompt Templates for CMO Coordinator.

Contains all prompt templates used by the Chief Medical Officer
for reasoning, conflict resolution, and report generation.
"""

from typing import List, Dict, Any
from clinical.models.expert_opinion import ExpertOpinion


# CMO System Prompt
CMO_SYSTEM_PROMPT = """You are a Chief Medical Officer (CMO) in a multi-omics clinical diagnosis system.

Your role is to:
1. Analyze expert opinions from microbiome, metabolome, and proteome specialists
2. Resolve conflicts between experts using medical literature and clinical cases
3. Provide evidence-based diagnostic decisions with clear reasoning chains
4. Generate comprehensive clinical reports

Key principles:
- Prioritize patient safety and diagnostic accuracy
- Use evidence-based reasoning with citations
- Consider all expert perspectives and biomarker evidence
- Be transparent about uncertainty and limitations
- Provide actionable recommendations

You have access to:
- Expert opinions with confidence scores and feature importance
- Medical literature (via RAG system)
- Historical clinical cases (via CAG cache)
- Multi-omics biomarker data
"""


def build_conflict_resolution_prompt(
    expert_opinions: List[ExpertOpinion],
    rag_context: Any = None,
    cag_context: Any = None,
    patient_metadata: Dict[str, Any] = None
) -> str:
    """
    Build prompt for CMO conflict resolution.

    Args:
        expert_opinions: List of expert opinions
        rag_context: Context from RAG literature search (Dict or str)
        cag_context: Context from CAG case retrieval (Dict or str)
        patient_metadata: Optional patient metadata

    Returns:
        Formatted prompt string
    """
    prompt_parts = []

    # Header
    prompt_parts.append("# Diagnostic Conflict Resolution Task\n")

    # Patient info (if available)
    if patient_metadata:
        prompt_parts.append("## Patient Information")
        for key, value in patient_metadata.items():
            prompt_parts.append(f"- {key}: {value}")
        prompt_parts.append("")

    # Expert opinions
    prompt_parts.append("## Expert Opinions\n")
    for i, opinion in enumerate(expert_opinions, 1):
        prompt_parts.append(f"### Expert {i}: {opinion.expert_name} ({opinion.omics_type})\n")
        prompt_parts.append(f"**Diagnosis**: {opinion.diagnosis}")
        prompt_parts.append(f"**Probability**: {opinion.probability:.1%}")
        prompt_parts.append(f"**Confidence**: {opinion.confidence:.1%}\n")

        prompt_parts.append("**Top Biomarkers**:")
        for feature in opinion.top_features[:5]:
            prompt_parts.append(
                f"- {feature.feature_name}: {feature.direction}regulated "
                f"(importance: {feature.importance_score:.3f})"
            )

        prompt_parts.append(f"\n**Biological Explanation**:")
        prompt_parts.append(opinion.biological_explanation)

        prompt_parts.append(f"\n**Evidence Chain**:")
        for evidence in opinion.evidence_chain:
            prompt_parts.append(f"- {evidence}")

        prompt_parts.append("")

    # RAG context
    if rag_context:
        prompt_parts.append("## Medical Literature Evidence\n")
        if isinstance(rag_context, dict):
            # Format dictionary as readable string
            documents = rag_context.get("documents", [])
            for i, doc in enumerate(documents, 1):
                prompt_parts.append(f"### Document {i}")
                prompt_parts.append(f"**Source**: {doc.get('source', 'Unknown')}")
                prompt_parts.append(f"**Title**: {doc.get('title', 'N/A')}")
                prompt_parts.append(f"**Relevance Score**: {doc.get('score', 0.0):.2f}")
                prompt_parts.append(f"**Content**: {doc.get('content', '')}")
                if doc.get('url'):
                    prompt_parts.append(f"**URL**: {doc.get('url')}")
                prompt_parts.append("")
        else:
            # Already a string
            prompt_parts.append(str(rag_context))
        prompt_parts.append("")

    # CAG context
    if cag_context:
        prompt_parts.append("## Similar Historical Cases\n")
        if isinstance(cag_context, dict):
            # Format dictionary as readable string
            similar_cases = cag_context.get("similar_cases", [])
            for i, case in enumerate(similar_cases, 1):
                prompt_parts.append(f"### Case {i}")
                prompt_parts.append(f"**Case ID**: {case.get('case_id', 'Unknown')}")
                prompt_parts.append(f"**Diagnosis**: {case.get('diagnosis', 'N/A')}")
                prompt_parts.append(f"**Similarity Score**: {case.get('similarity', 0.0):.2f}")
                prompt_parts.append(f"**Outcome**: {case.get('outcome', 'N/A')}")

                key_features = case.get('key_features', {})
                if key_features:
                    prompt_parts.append("**Key Features**:")
                    for feature_name, feature_value in key_features.items():
                        prompt_parts.append(f"  - {feature_name}: {feature_value}")
                prompt_parts.append("")
        else:
            # Already a string
            prompt_parts.append(str(cag_context))
        prompt_parts.append("")

    # Task instructions
    prompt_parts.append("## Your Task\n")
    prompt_parts.append(
        "Analyze the conflicting expert opinions and supporting evidence to make a final diagnostic decision.\n"
    )
    prompt_parts.append("Please provide:")
    prompt_parts.append("1. **Final Diagnosis**: Your conclusive diagnosis")
    prompt_parts.append("2. **Confidence Score** (0-1): Your confidence in this diagnosis")
    prompt_parts.append("3. **Reasoning Chain**: Step-by-step reasoning process")
    prompt_parts.append("4. **Evidence Synthesis**: How you weighted different sources of evidence")
    prompt_parts.append("5. **Key Biomarkers**: Most important diagnostic markers across all omics")
    prompt_parts.append("6. **Differential Diagnoses**: Other diagnoses considered and why they were ruled out")
    prompt_parts.append("7. **Recommendations**: Clinical recommendations and next steps")
    prompt_parts.append("8. **Limitations**: Uncertainties and limitations in this diagnosis\n")

    return "\n".join(prompt_parts)


def build_debate_round_prompt(
    expert_opinions: List[ExpertOpinion],
    round_number: int,
    previous_debates: List[str] = None
) -> str:
    """
    Build prompt for debate round.

    Args:
        expert_opinions: Current expert opinions
        round_number: Current debate round (1-3)
        previous_debates: Previous debate transcripts

    Returns:
        Formatted prompt
    """
    prompt_parts = []

    prompt_parts.append(f"# Debate Round {round_number}\n")

    # Show previous debates
    if previous_debates:
        prompt_parts.append("## Previous Debate Rounds\n")
        for i, debate in enumerate(previous_debates, 1):
            prompt_parts.append(f"### Round {i}")
            prompt_parts.append(debate)
            prompt_parts.append("")

    # Current expert opinions
    prompt_parts.append("## Current Expert Positions\n")
    for opinion in expert_opinions:
        prompt_parts.append(
            f"- **{opinion.expert_name}**: {opinion.diagnosis} "
            f"(confidence: {opinion.confidence:.1%}, "
            f"probability: {opinion.probability:.1%})"
        )

    prompt_parts.append("\n## Debate Instructions\n")
    prompt_parts.append(
        f"This is debate round {round_number} of maximum 3 rounds.\n"
    )
    prompt_parts.append("As CMO, you should:")
    prompt_parts.append("1. Identify remaining points of disagreement")
    prompt_parts.append("2. Request threshold adjustments from experts if near decision boundaries")
    prompt_parts.append("3. Evaluate if consensus can be reached")
    prompt_parts.append("4. Decide if additional rounds or RAG/CAG queries are needed\n")

    return "\n".join(prompt_parts)


def build_report_generation_prompt(
    final_diagnosis: str,
    confidence: float,
    expert_opinions: List[ExpertOpinion],
    reasoning_chain: List[str],
    key_biomarkers: List[Dict[str, Any]],
    rag_citations: List[str] = None,
    cag_cases: List[str] = None
) -> str:
    """
    Build prompt for generating clinical report.

    Args:
        final_diagnosis: Final diagnosis
        confidence: Confidence score
        expert_opinions: Expert opinions
        reasoning_chain: Reasoning steps
        key_biomarkers: Key biomarkers
        rag_citations: RAG literature citations
        cag_cases: CAG similar cases

    Returns:
        Formatted prompt
    """
    prompt_parts = []

    prompt_parts.append("# Clinical Report Generation Task\n")

    prompt_parts.append("## Diagnostic Summary")
    prompt_parts.append(f"**Final Diagnosis**: {final_diagnosis}")
    prompt_parts.append(f"**Confidence**: {confidence:.1%}\n")

    prompt_parts.append("## Expert Consensus")
    for opinion in expert_opinions:
        prompt_parts.append(
            f"- {opinion.expert_name}: {opinion.diagnosis} "
            f"({opinion.confidence:.1%})"
        )

    prompt_parts.append("\n## Reasoning Chain")
    for i, step in enumerate(reasoning_chain, 1):
        prompt_parts.append(f"{i}. {step}")

    prompt_parts.append("\n## Key Biomarkers")
    for biomarker in key_biomarkers:
        prompt_parts.append(
            f"- {biomarker['name']} ({biomarker['omics_type']}): "
            f"{biomarker['description']}"
        )

    if rag_citations:
        prompt_parts.append("\n## Supporting Literature")
        for citation in rag_citations:
            prompt_parts.append(f"- {citation}")

    if cag_cases:
        prompt_parts.append("\n## Similar Cases")
        for case in cag_cases:
            prompt_parts.append(f"- {case}")

    prompt_parts.append("\n## Task")
    prompt_parts.append("Generate a comprehensive clinical diagnostic report in markdown format.")
    prompt_parts.append("\nThe report should include:")
    prompt_parts.append("1. Executive Summary")
    prompt_parts.append("2. Multi-Omics Analysis")
    prompt_parts.append("3. Diagnostic Rationale")
    prompt_parts.append("4. Key Biomarkers")
    prompt_parts.append("5. Differential Diagnoses")
    prompt_parts.append("6. Clinical Recommendations")
    prompt_parts.append("7. References and Evidence")
    prompt_parts.append("8. Limitations and Follow-up\n")

    return "\n".join(prompt_parts)


# Quick decision prompt (no conflict)
QUICK_DECISION_PROMPT = """# Quick Diagnostic Decision

The expert opinions show strong agreement. Please provide:

1. **Confirmation**: Confirm the consensus diagnosis
2. **Confidence Assessment**: Evaluate the overall confidence
3. **Key Evidence**: Summarize the most important biomarker evidence
4. **Brief Recommendation**: Any immediate clinical recommendations

Keep the response concise (3-5 sentences).
"""


# Threshold adjustment request template
def build_threshold_adjustment_request(
    expert_name: str,
    current_probability: float,
    current_threshold: float,
    adjustment: float
) -> str:
    """
    Build request for expert to adjust decision threshold.

    Args:
        expert_name: Name of expert
        current_probability: Current prediction probability
        current_threshold: Current threshold
        adjustment: Threshold adjustment amount

    Returns:
        Request message
    """
    new_threshold = current_threshold + adjustment

    return (
        f"Request to {expert_name}: "
        f"Your prediction probability is {current_probability:.1%}, "
        f"which is near the decision boundary (threshold: {current_threshold:.1%}). "
        f"Please re-evaluate your opinion with adjusted threshold {new_threshold:.1%} "
        f"to determine if this is a borderline case."
    )

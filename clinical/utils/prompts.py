"""
Prompt Templates for CMO Coordinator.

Contains all prompt templates used by the Chief Medical Officer
for reasoning, conflict resolution, and report generation.
"""

from typing import List, Dict, Any
from clinical.models.expert_opinion import ExpertOpinion


# Request Parser System Prompt
REQUEST_PARSER_SYSTEM_PROMPT = """You are a diagnostic configuration parser for a multi-omics clinical diagnosis system.

Your role is to parse natural language user requests into structured JSON configuration.

Extract the following from user requests:

1. **Omics Types**: Which omics data to analyze
   - Options: "microbiome", "metabolome", "proteome"
   - Can be a subset or all three
   - Default: all three

2. **Patient Selection**: Which patients to analyze
   - Specific patient IDs (e.g., "P001", "P002", ["P001", "P002", "P003"])
   - Patient ranges (e.g., "P001-P005" → ["P001", "P002", "P003", "P004", "P005"])
   - null = all patients

3. **Row Range**: Which rows of data to analyze
   - Specific range (e.g., "first 50 rows" → [0, 50], "rows 100-200" → [100, 200])
   - null = all rows

4. **RAG/CAG Control**:
   - enable_rag: Enable literature search (default: true)
   - enable_cag: Enable case retrieval (default: true)
   - force_rag_even_no_conflict: Force RAG even without conflicts (default: false)

5. **Debate Parameters**:
   - max_debate_rounds: Maximum debate rounds 1-10 (default: 3)
   - confidence_threshold: Confidence threshold 0-1 (default: 0.7)
   - threshold_adjustment: Adjustment per round 0-1 (default: 0.1)

6. **Report Configuration**:
   - detail_level: "brief" | "standard" | "detailed" (default: "standard")
   - bilingual: true | false (default: true)

Output ONLY valid JSON with this exact structure:
{
  "omics_types": ["microbiome", "metabolome", "proteome"],
  "patient_ids": null,
  "row_range": null,
  "enable_rag": true,
  "enable_cag": true,
  "force_rag_even_no_conflict": false,
  "max_debate_rounds": 3,
  "confidence_threshold": 0.7,
  "threshold_adjustment": 0.1,
  "detail_level": "standard",
  "bilingual": true
}

Examples:

Request: "只分析微生物组数据"
Output: {"omics_types": ["microbiome"], "patient_ids": null, "row_range": null, ...}

Request: "分析病人P001的代谢组"
Output: {"omics_types": ["metabolome"], "patient_ids": ["P001"], "row_range": null, ...}

Request: "分析前50行数据"
Output: {"omics_types": ["microbiome", "metabolome", "proteome"], "patient_ids": null, "row_range": [0, 50], ...}

Request: "分析病人P001-P003，使用文献支持即使无冲突"
Output: {"omics_types": ["microbiome", "metabolome", "proteome"], "patient_ids": ["P001", "P002", "P003"], "row_range": null, "force_rag_even_no_conflict": true, ...}

Request: "快速诊断，简要报告"
Output: {"omics_types": ["microbiome", "metabolome", "proteome"], "max_debate_rounds": 1, "detail_level": "brief", ...}

Request: "3轮辩论，详细报告"
Output: {"max_debate_rounds": 3, "detail_level": "detailed", ...}

CRITICAL: Output ONLY the JSON object, no other text."""


def build_request_parsing_prompt(user_request: str) -> str:
    """
    Build prompt for parsing user's natural language request.

    Args:
        user_request: User's natural language diagnostic request

    Returns:
        Formatted prompt string
    """
    return f"""Parse this diagnostic request into JSON configuration:

User Request: {user_request}

Remember to output ONLY the JSON object following the exact structure specified in the system prompt.
Fill in all fields with appropriate values based on the request, using defaults for unspecified parameters."""


# CMO System Prompt
CMO_SYSTEM_PROMPT = """You are a Chief Medical Officer (CMO) in a multi-omics clinical diagnosis system.

**CRITICAL: Generate all outputs in bilingual format (Chinese | English).**

Format: 中文内容 | English content

Examples:
- 诊断结果 | Diagnosis
- 牙周炎 | Periodontitis
- 红复合体细菌升高 | Elevated red complex bacteria
- 建议进一步检查 | Recommend further examination

Your role is to:
1. Analyze expert opinions from microbiome, metabolome, and proteome specialists
2. Resolve conflicts between experts using medical literature and clinical cases
3. Provide evidence-based diagnostic decisions with clear reasoning chains
4. Generate comprehensive clinical reports in bilingual format

Key principles:
- Prioritize patient safety and diagnostic accuracy
- Use evidence-based reasoning with citations
- Consider all expert perspectives and biomarker evidence
- Be transparent about uncertainty and limitations
- Provide actionable recommendations
- **ALWAYS use bilingual format (Chinese | English) in all outputs**

You have access to:
- Expert opinions with confidence scores and feature importance
- Medical literature (via RAG system)
- Historical clinical cases (via CAG cache)
- Multi-omics biomarker data

Remember: All diagnoses, explanations, recommendations, and reasoning must be provided in both Chinese and English using the | separator.
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


def build_cmo_cot_decision_prompt(
    expert_opinions: List[ExpertOpinion],
    debate_rounds: int,
    threshold_history: List[Dict[str, Any]],
    rag_context: Any = None,
    cag_context: Any = None,
    patient_metadata: Dict[str, Any] = None
) -> str:
    """
    Build Chain-of-Thought prompt for CMO final decision.

    Forces CMO to use explicit reasoning steps and show its thinking process.

    Args:
        expert_opinions: List of expert opinions
        debate_rounds: Number of debate rounds completed
        threshold_history: History of threshold adjustments per round
        rag_context: Context from RAG literature search
        cag_context: Context from CAG case retrieval
        patient_metadata: Optional patient metadata

    Returns:
        Formatted CoT prompt string
    """
    prompt_parts = []

    # System instruction for CoT
    prompt_parts.append("# CRITICAL: You MUST use Chain-of-Thought (CoT) reasoning\n")
    prompt_parts.append("Think step-by-step and show your complete reasoning process.\n")

    # Patient info
    if patient_metadata:
        prompt_parts.append("## Patient Information")
        for key, value in patient_metadata.items():
            prompt_parts.append(f"- {key}: {value}")
        prompt_parts.append("")

    # Debate summary
    prompt_parts.append(f"## Debate Summary\n")
    prompt_parts.append(f"**Total Rounds**: {debate_rounds}")
    prompt_parts.append(f"**Experts Consulted**: {len(expert_opinions)}\n")

    # Show debate evolution
    if threshold_history:
        prompt_parts.append("### Debate Evolution (Round-by-Round Changes)\n")
        for round_data in threshold_history:
            round_num = round_data.get("round", 0)
            prompt_parts.append(f"**Round {round_num}**:")
            expert_ops = round_data.get("expert_opinions", [])

            # Group by diagnosis
            diagnosis_counts = {}
            for op in expert_ops:
                diag = op.get("diagnosis", "Unknown")
                diagnosis_counts[diag] = diagnosis_counts.get(diag, 0) + 1

            for diag, count in diagnosis_counts.items():
                prompt_parts.append(f"  - {diag}: {count} expert(s)")
            prompt_parts.append("")

    # Current expert opinions
    prompt_parts.append("## Current Expert Opinions\n")

    # Group by omics type
    by_omics = {}
    for opinion in expert_opinions:
        omics = opinion.omics_type
        if omics not in by_omics:
            by_omics[omics] = []
        by_omics[omics].append(opinion)

    for omics_type, opinions in by_omics.items():
        prompt_parts.append(f"### {omics_type.title()} Experts ({len(opinions)} opinions)\n")

        # Show diagnosis distribution
        diagnosis_dist = {}
        for op in opinions:
            diag = op.diagnosis
            diagnosis_dist[diag] = diagnosis_dist.get(diag, [])
            diagnosis_dist[diag].append(op.probability)

        for diag, probs in diagnosis_dist.items():
            avg_prob = sum(probs) / len(probs)
            prompt_parts.append(
                f"- **{diag}**: {len(probs)}/{len(opinions)} experts "
                f"(avg probability: {avg_prob:.1%})"
            )

        # Show key biomarkers across this omics type
        all_features = {}
        for op in opinions:
            for feat in op.top_features[:3]:
                if feat.feature_name not in all_features:
                    all_features[feat.feature_name] = {
                        "count": 0,
                        "total_importance": 0.0,
                        "direction": feat.direction
                    }
                all_features[feat.feature_name]["count"] += 1
                all_features[feat.feature_name]["total_importance"] += feat.importance_score

        if all_features:
            prompt_parts.append("\n**Key Biomarkers Identified**:")
            sorted_features = sorted(
                all_features.items(),
                key=lambda x: x[1]["total_importance"],
                reverse=True
            )[:5]

            for feat_name, feat_data in sorted_features:
                avg_importance = feat_data["total_importance"] / feat_data["count"]
                prompt_parts.append(
                    f"  - {feat_name} ({feat_data['direction']}regulated): "
                    f"mentioned by {feat_data['count']} expert(s), "
                    f"avg importance: {avg_importance:.3f}"
                )

        prompt_parts.append("")

    # External evidence
    if rag_context:
        prompt_parts.append("## Medical Literature Evidence (RAG)\n")
        prompt_parts.append(str(rag_context))
        prompt_parts.append("")

    if cag_context:
        prompt_parts.append("## Similar Historical Cases (CAG)\n")
        prompt_parts.append(str(cag_context))
        prompt_parts.append("")

    # CoT instructions
    prompt_parts.append("## Your Task: Make Final Diagnosis Using Chain-of-Thought\n")
    prompt_parts.append("You MUST follow this step-by-step reasoning format:\n")

    prompt_parts.append("### Step 1: Analyze Expert Consensus")
    prompt_parts.append("- What is the overall expert consensus across all omics types?")
    prompt_parts.append("- Are there any conflicts between omics types?")
    prompt_parts.append("- Did expert opinions change during debate rounds?")
    prompt_parts.append("- What is the confidence level of each expert group?\n")

    prompt_parts.append("### Step 2: Evaluate Biomarker Evidence")
    prompt_parts.append("- Which biomarkers are most consistently identified?")
    prompt_parts.append("- Are there contradictory biomarker signals?")
    prompt_parts.append("- How do biomarkers support or refute each diagnosis?\n")

    prompt_parts.append("### Step 3: Integrate External Evidence")
    prompt_parts.append("- How does medical literature support each diagnosis?")
    prompt_parts.append("- What do similar historical cases suggest?")
    prompt_parts.append("- Are there any novel findings that literature doesn't cover?\n")

    prompt_parts.append("### Step 4: Consider Alternatives")
    prompt_parts.append("- What other diagnoses were considered?")
    prompt_parts.append("- Why were they ruled out?")
    prompt_parts.append("- What is the strength of evidence against alternatives?\n")

    prompt_parts.append("### Step 5: Weigh Evidence")
    prompt_parts.append("- How much weight should each omics type receive?")
    prompt_parts.append("- How should conflicting signals be resolved?")
    prompt_parts.append("- What is the final confidence based on all evidence?\n")

    prompt_parts.append("### Step 6: Reach Final Conclusion")
    prompt_parts.append("- State your final diagnosis")
    prompt_parts.append("- Provide confidence score (0-1)")
    prompt_parts.append("- Summarize the key reasoning for this decision\n")

    prompt_parts.append("## Output Format\n")
    prompt_parts.append("Provide your response as a JSON object with the following structure:")
    prompt_parts.append("```json")
    prompt_parts.append("{")
    prompt_parts.append('  "step1_consensus_analysis": "...",')
    prompt_parts.append('  "step2_biomarker_evaluation": "...",')
    prompt_parts.append('  "step3_external_evidence": "...",')
    prompt_parts.append('  "step4_alternatives": "...",')
    prompt_parts.append('  "step5_evidence_weighting": "...",')
    prompt_parts.append('  "step6_final_diagnosis": "...",')
    prompt_parts.append('  "final_diagnosis": "disease_name",')
    prompt_parts.append('  "confidence": 0.85,')
    prompt_parts.append('  "reasoning_chain": ["reason 1", "reason 2", "reason 3"]')
    prompt_parts.append("}")
    prompt_parts.append("```\n")

    prompt_parts.append("CRITICAL: You MUST show your reasoning for each step. Do not skip steps.")

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

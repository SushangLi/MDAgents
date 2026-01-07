"""
CMO (Chief Medical Officer) Coordinator.

Coordinates final diagnostic decision using LLM reasoning with
expert opinions, RAG/CAG evidence, and debate outcomes.
"""

from typing import List, Dict, Any, Optional, Callable, Awaitable
import json
from dataclasses import asdict

from clinical.models.expert_opinion import ExpertOpinion
from clinical.models.diagnosis_result import DiagnosisResult, ConflictResolution
from clinical.decision.conflict_resolver import ConflictAnalysis
from clinical.utils.prompts import (
    CMO_SYSTEM_PROMPT,
    build_conflict_resolution_prompt,
    QUICK_DECISION_PROMPT
)


# Type alias for LLM call function
LLMCallFunc = Callable[[List[Dict[str, Any]], float], Awaitable[Dict[str, Any]]]


class CMOCoordinator:
    """
    Chief Medical Officer coordinator for final diagnostic decisions.

    Uses LLM reasoning to synthesize expert opinions, medical literature,
    and historical cases into final diagnostic recommendations.
    """

    def __init__(
        self,
        llm_call_func: Optional[LLMCallFunc] = None,
        temperature: float = 0.3,
        use_mcp: bool = False
    ):
        """
        Initialize CMO coordinator.

        Args:
            llm_call_func: Function to call LLM
                          Signature: async (messages, temperature) -> response_dict
                          If None, requires manual prompt generation
            temperature: LLM temperature for reasoning
            use_mcp: Whether using MCP integration

        Example usage:
            from clinical.decision.llm_wrapper import create_llm_wrapper

            # Auto mode (uses real LLM if API key available, else mock)
            wrapper = create_llm_wrapper(use_mock=False)
            cmo = CMOCoordinator(llm_call_func=wrapper.call, temperature=0.3)

            # Force mock mode (for testing without API calls)
            wrapper = create_llm_wrapper(use_mock=True)
            cmo = CMOCoordinator(llm_call_func=wrapper.call)

            # No LLM mode (uses fallback voting)
            cmo = CMOCoordinator()  # llm_call_func=None
        """
        self.llm_call_func = llm_call_func
        self.temperature = temperature
        self.use_mcp = use_mcp

        print("✓ CMO Coordinator initialized")
        if use_mcp:
            print("  Mode: MCP integration")
        else:
            print("  Mode: Standalone")

    async def make_quick_decision(
        self,
        expert_opinions: List[ExpertOpinion],
        conflict_analysis: ConflictAnalysis
    ) -> DiagnosisResult:
        """
        Make quick decision without full debate (no conflict).

        Args:
            expert_opinions: List of expert opinions
            conflict_analysis: Conflict analysis result

        Returns:
            Diagnosis result
        """
        print("\n[CMO] Making quick decision (no conflict)...")

        # Get consensus diagnosis
        diagnoses = [op.diagnosis for op in expert_opinions]
        consensus_diagnosis = max(set(diagnoses), key=diagnoses.count)

        # Calculate confidence
        avg_confidence = conflict_analysis.avg_confidence

        # Build quick reasoning
        reasoning_chain = [
            f"All {len(expert_opinions)} experts agree on diagnosis: {consensus_diagnosis}",
            f"Average confidence: {avg_confidence:.1%}",
            "No conflicts detected - proceeding with consensus decision"
        ]

        # Extract key biomarkers across all experts
        key_biomarkers = []
        for opinion in expert_opinions:
            for feature in opinion.top_features[:3]:
                key_biomarkers.append({
                    "name": feature.feature_name,
                    "omics_type": opinion.omics_type,
                    "importance": feature.importance_score,
                    "direction": feature.direction,
                    "description": f"{feature.direction}regulated in {opinion.omics_type}"
                })

        # Sort by importance
        key_biomarkers = sorted(
            key_biomarkers,
            key=lambda x: x["importance"],
            reverse=True
        )[:10]

        # Create result
        result = DiagnosisResult(
            patient_id="",  # Will be set by caller
            diagnosis=consensus_diagnosis,
            confidence=avg_confidence,
            expert_opinions=expert_opinions,
            conflict_resolution=None,  # No conflict
            key_biomarkers=key_biomarkers,
            clinical_recommendations=[
                f"Confirmed diagnosis: {consensus_diagnosis}",
                "Monitor patient progress",
                "Follow standard treatment protocols"
            ],
            explanation=" ".join(reasoning_chain),
            references=[],
            metadata={
                "decision_type": "quick_consensus",
                "conflict_detected": False,
                "n_experts": len(expert_opinions),
                "reasoning_chain": reasoning_chain
            }
        )

        print(f"  Quick decision: {consensus_diagnosis} ({avg_confidence:.1%})")

        return result

    async def make_conflict_resolution(
        self,
        expert_opinions: List[ExpertOpinion],
        conflict_analysis: ConflictAnalysis,
        rag_context: Optional[str] = None,
        cag_context: Optional[str] = None,
        debate_history: Optional[List[str]] = None,
        patient_metadata: Optional[Dict[str, Any]] = None
    ) -> DiagnosisResult:
        """
        Make decision after conflict resolution using LLM reasoning.

        Args:
            expert_opinions: List of expert opinions
            conflict_analysis: Conflict analysis
            rag_context: Medical literature context
            cag_context: Similar cases context
            debate_history: Debate transcripts
            patient_metadata: Patient information

        Returns:
            Diagnosis result
        """
        print("\n[CMO] Resolving conflict with LLM reasoning...")

        # Build conflict resolution prompt
        prompt = build_conflict_resolution_prompt(
            expert_opinions=expert_opinions,
            rag_context=rag_context or "",
            cag_context=cag_context or "",
            patient_metadata=patient_metadata
        )

        # Call LLM
        if self.llm_call_func:
            llm_response = await self._call_llm(prompt)
        else:
            # If no LLM function, return prompt for manual processing
            print("  ⚠ No LLM function provided - returning prompt for manual processing")
            llm_response = {
                "content": "MANUAL_PROCESSING_REQUIRED",
                "prompt": prompt
            }

        # Parse LLM response
        diagnosis_data = self._parse_llm_response(
            llm_response,
            expert_opinions,
            conflict_analysis
        )

        # Build conflict resolution record
        conflict_resolution = ConflictResolution(
            conflicts_detected=[ct.value for ct in conflict_analysis.conflict_types],
            resolution_method="cmo_llm_reasoning" if llm_response.get("content") != "MANUAL_PROCESSING_REQUIRED" else "fallback_voting",
            rag_evidence=self._format_rag_evidence(rag_context) if rag_context else [],
            cag_cases=self._format_cag_cases(cag_context) if cag_context else [],
            cmo_reasoning=diagnosis_data.get("reasoning", ""),
            confidence_score=diagnosis_data.get("confidence", 0.0)
        )

        # Create diagnosis result
        result = DiagnosisResult(
            patient_id="",  # Will be set by caller
            diagnosis=diagnosis_data["diagnosis"],
            confidence=diagnosis_data["confidence"],
            expert_opinions=expert_opinions,
            conflict_resolution=conflict_resolution,
            key_biomarkers=self._extract_key_biomarkers(expert_opinions),
            clinical_recommendations=diagnosis_data.get("recommendations", []),
            explanation=diagnosis_data.get("explanation", "") or self._generate_default_explanation(
                diagnosis_data["diagnosis"],
                expert_opinions,
                conflict_resolution
            ),
            references=self._extract_references(rag_context),
            metadata={
                "decision_type": "conflict_resolution",
                "conflict_detected": True,
                "debate_rounds": len(debate_history) if debate_history else 0,
                "rag_used": rag_context is not None,
                "cag_used": cag_context is not None,
                "llm_provider": llm_response.get("provider"),
                "llm_model": llm_response.get("model")
            }
        )

        print(f"  Final decision: {result.diagnosis} ({result.confidence:.1%})")

        return result

    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM with prompt.

        Args:
            prompt: User prompt

        Returns:
            LLM response dictionary
        """
        # Build messages
        messages = [
            {
                "role": "system",
                "content": CMO_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Call LLM function
        response = await self.llm_call_func(messages, self.temperature)

        return response

    def _parse_llm_response(
        self,
        llm_response: Dict[str, Any],
        expert_opinions: List[ExpertOpinion],
        conflict_analysis: ConflictAnalysis
    ) -> Dict[str, Any]:
        """
        Parse LLM response into structured diagnosis data.

        Args:
            llm_response: LLM response
            expert_opinions: Expert opinions
            conflict_analysis: Conflict analysis

        Returns:
            Parsed diagnosis data
        """
        content = llm_response.get("content", "")

        # If manual processing required, use fallback
        if content == "MANUAL_PROCESSING_REQUIRED":
            return self._fallback_decision(expert_opinions, conflict_analysis)

        # Try to extract structured information from LLM response
        # This is a simplified parser - in production, use more robust parsing

        diagnosis_data = {
            "diagnosis": "",
            "confidence": 0.0,
            "reasoning_chain": [],
            "key_biomarkers": [],
            "differential_diagnoses": [],
            "recommendations": []
        }

        # Extract diagnosis (simple heuristic)
        lines = content.split("\n")
        for line in lines:
            if "final diagnosis" in line.lower():
                # Extract after colon
                if ":" in line:
                    diagnosis_data["diagnosis"] = line.split(":", 1)[1].strip()
                    break

        # If no diagnosis extracted, use weighted consensus
        if not diagnosis_data["diagnosis"]:
            # Weighted vote
            diagnosis_weights = {}
            for opinion in expert_opinions:
                diagnosis_weights[opinion.diagnosis] = \
                    diagnosis_weights.get(opinion.diagnosis, 0.0) + opinion.confidence

            diagnosis_data["diagnosis"] = max(
                diagnosis_weights.keys(),
                key=lambda k: diagnosis_weights[k]
            )

        # Extract confidence (simple heuristic)
        for line in lines:
            if "confidence" in line.lower() and ":" in line:
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    # Extract percentage or decimal
                    conf_str = conf_str.replace("%", "").strip()
                    confidence = float(conf_str)
                    if confidence > 1:  # Percentage
                        confidence /= 100
                    diagnosis_data["confidence"] = confidence
                    break
                except:
                    pass

        # If no confidence extracted, use average
        if diagnosis_data["confidence"] == 0.0:
            diagnosis_data["confidence"] = conflict_analysis.avg_confidence

        # Extract reasoning (lines as chain)
        in_reasoning = False
        for line in lines:
            if "reasoning" in line.lower() and ":" in line:
                in_reasoning = True
                continue

            if in_reasoning and line.strip():
                if line.startswith(("#", "**")):  # New section
                    break
                diagnosis_data["reasoning_chain"].append(line.strip(" -•"))

        # If no reasoning extracted, build from content
        if not diagnosis_data["reasoning_chain"]:
            diagnosis_data["reasoning_chain"] = [
                f"LLM analysis completed with {len(expert_opinions)} expert opinions",
                f"Final diagnosis: {diagnosis_data['diagnosis']}",
                content[:200] + "..." if len(content) > 200 else content
            ]

        # Collect key biomarkers from experts
        key_biomarkers = []
        for opinion in expert_opinions:
            for feature in opinion.top_features[:3]:
                key_biomarkers.append({
                    "name": feature.feature_name,
                    "omics_type": opinion.omics_type,
                    "importance": feature.importance_score,
                    "direction": feature.direction,
                    "description": f"{feature.direction}regulated in {opinion.omics_type}"
                })

        diagnosis_data["key_biomarkers"] = sorted(
            key_biomarkers,
            key=lambda x: x["importance"],
            reverse=True
        )[:10]

        # Default recommendations
        diagnosis_data["recommendations"] = [
            f"Confirmed diagnosis: {diagnosis_data['diagnosis']}",
            "Initiate appropriate treatment protocol",
            "Monitor patient response",
            "Consider follow-up multi-omics analysis"
        ]

        return diagnosis_data

    def _fallback_decision(
        self,
        expert_opinions: List[ExpertOpinion],
        conflict_analysis: ConflictAnalysis
    ) -> Dict[str, Any]:
        """
        Fallback decision when LLM not available.

        Args:
            expert_opinions: Expert opinions
            conflict_analysis: Conflict analysis

        Returns:
            Basic diagnosis data
        """
        # Weighted vote
        diagnosis_weights = {}
        for opinion in expert_opinions:
            diagnosis_weights[opinion.diagnosis] = \
                diagnosis_weights.get(opinion.diagnosis, 0.0) + opinion.confidence

        final_diagnosis = max(
            diagnosis_weights.keys(),
            key=lambda k: diagnosis_weights[k]
        )

        return {
            "diagnosis": final_diagnosis,
            "confidence": conflict_analysis.avg_confidence,
            "reasoning_chain": [
                "LLM unavailable - using weighted consensus",
                f"Final diagnosis: {final_diagnosis}",
                f"Based on {len(expert_opinions)} expert opinions"
            ],
            "key_biomarkers": [],
            "differential_diagnoses": [],
            "recommendations": [
                f"Diagnosis: {final_diagnosis} (consensus-based)",
                "LLM reasoning not available",
                "Manual review recommended"
            ]
        }

    def _format_rag_evidence(self, rag_context: Optional[Dict]) -> List[Dict[str, Any]]:
        """格式化RAG证据为ConflictResolution所需格式"""
        if not rag_context:
            return []

        # Handle both string (formatted context) and dict (raw data) types
        if isinstance(rag_context, str):
            # Formatted markdown string from debate system - no structured data to extract
            return []

        documents = rag_context.get("documents", [])
        return [{
            "source": doc.get("source", "Unknown"),
            "content": doc.get("content", ""),
            "relevance_score": doc.get("score", 0.0)
        } for doc in documents]

    def _format_cag_cases(self, cag_context: Optional[Dict]) -> List[Dict[str, Any]]:
        """格式化CAG案例为ConflictResolution所需格式"""
        if not cag_context:
            return []

        # Handle both string (formatted context) and dict (raw data) types
        if isinstance(cag_context, str):
            # Formatted markdown string from debate system - no structured data to extract
            return []

        similar_cases = cag_context.get("similar_cases", [])
        return [{
            "case_id": case.get("case_id", "Unknown"),
            "diagnosis": case.get("diagnosis", ""),
            "similarity_score": case.get("similarity", 0.0),
            "outcome": case.get("outcome", "")
        } for case in similar_cases]

    def _generate_default_explanation(
        self,
        diagnosis: str,
        expert_opinions: List[ExpertOpinion],
        conflict_resolution: Optional[ConflictResolution]
    ) -> str:
        """生成默认诊断解释（当LLM不可用时）"""
        explanations = []

        # 专家共识
        diagnoses = [op.diagnosis for op in expert_opinions]
        diagnosis_counts = {d: diagnoses.count(d) for d in set(diagnoses)}

        if len(diagnosis_counts) == 1:
            explanations.append(f"所有{len(expert_opinions)}个专家一致诊断为{diagnosis}。")
        else:
            explanations.append(f"{diagnosis_counts.get(diagnosis, 0)}/{len(expert_opinions)}个专家支持{diagnosis}诊断。")

        # 置信度
        avg_confidence = sum(op.confidence for op in expert_opinions) / len(expert_opinions)
        explanations.append(f"平均置信度: {avg_confidence:.1%}")

        # 冲突解决
        if conflict_resolution:
            explanations.append(f"通过{conflict_resolution.resolution_method}解决了专家意见分歧。")

        return " ".join(explanations)

    def _extract_key_biomarkers(self, expert_opinions: List[ExpertOpinion]) -> List[Dict[str, Any]]:
        """从专家意见中提取关键生物标志物"""
        biomarkers = []
        for opinion in expert_opinions:
            for feature in opinion.top_features[:3]:  # 每个专家取前3个特征
                biomarkers.append({
                    "name": feature.feature_name,
                    "importance_score": feature.importance_score,
                    "direction": feature.direction,
                    "omics_type": opinion.omics_type,
                    "biological_meaning": feature.biological_meaning
                })

        # 按重要性排序并去重
        biomarkers.sort(key=lambda x: x["importance_score"], reverse=True)
        unique_biomarkers = []
        seen_names = set()
        for bm in biomarkers:
            if bm["name"] not in seen_names:
                unique_biomarkers.append(bm)
                seen_names.add(bm["name"])

        return unique_biomarkers[:10]  # 返回前10个

    def _extract_references(self, rag_context: Optional[Dict]) -> List[Dict[str, Any]]:
        """从RAG上下文提取参考文献"""
        if not rag_context:
            return []

        # Handle both string (formatted context) and dict (raw data) types
        if isinstance(rag_context, str):
            # Formatted markdown string from debate system - no structured data to extract
            return []

        documents = rag_context.get("documents", [])
        return [{
            "title": doc.get("title", "Unknown"),
            "source": doc.get("source", "Unknown"),
            "url": doc.get("url", ""),
            "relevance": doc.get("score", 0.0)
        } for doc in documents]

    def _extract_rag_citations(self, rag_context: str) -> List[str]:
        """Extract citations from RAG context."""
        citations = []
        # Simple extraction - look for DOI or title patterns
        lines = rag_context.split("\n")
        for line in lines:
            if "**Title**:" in line or "**DOI**:" in line:
                citations.append(line.strip())

        return citations[:5]  # Top 5

    def _extract_cag_cases(self, cag_context: str) -> List[str]:
        """Extract similar cases from CAG context."""
        cases = []
        # Simple extraction - look for case IDs or diagnoses
        lines = cag_context.split("\n")
        for line in lines:
            if "**Diagnosis**:" in line:
                cases.append(line.strip())

        return cases[:3]  # Top 3

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CMOCoordinator("
            f"temperature={self.temperature}, "
            f"use_mcp={self.use_mcp})"
        )

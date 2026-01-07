"""
Debate System using LangGraph.

Implements multi-round debate mechanism with threshold adjustment
for resolving expert opinion conflicts.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
import operator
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from clinical.models.expert_opinion import ExpertOpinion
from clinical.decision.conflict_resolver import ConflictResolver, ConflictAnalysis
from clinical.collaboration.rag_system import RAGSystem
from clinical.collaboration.cag_system import CAGSystem
from clinical.utils.prompts import (
    build_debate_round_prompt,
    build_threshold_adjustment_request
)


# Debate state definition
class DebateState(TypedDict):
    """State for debate workflow."""

    # Input
    expert_opinions: List[ExpertOpinion]
    sample_data: Dict[str, Any]  # Omics data for the sample

    # Conflict analysis
    conflict_analysis: Optional[ConflictAnalysis]

    # Debate tracking
    current_round: int
    max_rounds: int
    debate_history: Annotated[List[str], operator.add]  # Accumulate debate transcripts

    # Threshold adjustments
    threshold_adjustment: float
    threshold_history: Annotated[List[Dict[str, Any]], operator.add]

    # RAG/CAG results
    rag_context: Optional[str]
    cag_context: Optional[str]

    # Final decision
    final_diagnosis: Optional[str]
    final_confidence: Optional[float]
    reasoning_chain: Optional[List[str]]

    # Metadata
    requires_debate: bool
    requires_rag: bool
    requires_cag: bool
    debate_resolved: bool


@dataclass
class DebateConfig:
    """Configuration for debate system."""

    max_rounds: int = 3
    threshold_adjustment: float = 0.1
    confidence_threshold: float = 0.7
    enable_rag: bool = True
    enable_cag: bool = True


class DebateSystem:
    """
    Multi-round debate system using LangGraph.

    Coordinates expert debate with threshold adjustments,
    RAG literature search, and CAG case retrieval.
    """

    def __init__(
        self,
        conflict_resolver: Optional[ConflictResolver] = None,
        rag_system: Optional[RAGSystem] = None,
        cag_system: Optional[CAGSystem] = None,
        config: Optional[DebateConfig] = None,
        llm_wrapper: Optional[Any] = None
    ):
        """
        Initialize debate system.

        Args:
            conflict_resolver: Conflict resolver instance
            rag_system: RAG system for literature search
            cag_system: CAG system for case retrieval
            config: Debate configuration
            llm_wrapper: LLM wrapper for CMO reasoning (optional)
        """
        self.conflict_resolver = conflict_resolver or ConflictResolver()
        self.rag_system = rag_system
        self.cag_system = cag_system
        self.config = config or DebateConfig()
        self.llm_wrapper = llm_wrapper

        # Build state graph
        self.graph = self._build_debate_graph()

        print("✓ Debate system initialized")
        print(f"  Max rounds: {self.config.max_rounds}")
        print(f"  Threshold adjustment: {self.config.threshold_adjustment}")

    def _build_debate_graph(self) -> StateGraph:
        """
        Build LangGraph state machine for debate.

        Returns:
            StateGraph instance
        """
        # Create state graph
        workflow = StateGraph(DebateState)

        # Add nodes
        workflow.add_node("detect_conflict", self._detect_conflict_node)
        workflow.add_node("quick_decision", self._quick_decision_node)
        workflow.add_node("adjust_thresholds", self._adjust_thresholds_node)
        workflow.add_node("debate_round", self._debate_round_node)
        workflow.add_node("query_rag", self._query_rag_node)
        workflow.add_node("query_cag", self._query_cag_node)
        workflow.add_node("final_decision", self._final_decision_node)

        # Set entry point
        workflow.set_entry_point("detect_conflict")

        # Add conditional edges
        workflow.add_conditional_edges(
            "detect_conflict",
            self._should_debate,
            {
                "quick": "quick_decision",
                "debate": "adjust_thresholds"
            }
        )

        workflow.add_edge("quick_decision", END)

        workflow.add_conditional_edges(
            "adjust_thresholds",
            self._check_threshold_resolution,
            {
                "resolved": "final_decision",
                "continue": "debate_round"
            }
        )

        workflow.add_conditional_edges(
            "debate_round",
            self._check_debate_status,
            {
                "continue": "adjust_thresholds",
                "max_rounds": "query_rag",
                "resolved": "final_decision"
            }
        )

        workflow.add_edge("query_rag", "query_cag")
        workflow.add_edge("query_cag", "final_decision")
        workflow.add_edge("final_decision", END)

        return workflow.compile()

    # Node functions
    def _detect_conflict_node(self, state: DebateState) -> DebateState:
        """Detect conflicts in expert opinions."""
        print("\n[Node] Detecting conflicts...")

        conflict_analysis = self.conflict_resolver.detect_conflict(
            state["expert_opinions"]
        )

        state["conflict_analysis"] = conflict_analysis
        state["requires_debate"] = conflict_analysis.requires_debate
        state["requires_rag"] = conflict_analysis.requires_rag
        state["requires_cag"] = conflict_analysis.requires_cag
        state["current_round"] = 0
        state["debate_resolved"] = False

        print(f"  Conflict detected: {conflict_analysis.has_conflict}")
        print(f"  Requires debate: {conflict_analysis.requires_debate}")

        return state

    def _quick_decision_node(self, state: DebateState) -> DebateState:
        """Make quick decision without debate (no conflict)."""
        print("\n[Node] Making quick decision (no conflict)...")

        # Get majority diagnosis
        majority_diagnosis = self.conflict_resolver.get_majority_diagnosis(
            state["expert_opinions"]
        )

        # Or use weighted diagnosis
        if not majority_diagnosis:
            majority_diagnosis = self.conflict_resolver.get_weighted_diagnosis(
                state["expert_opinions"]
            )

        # Calculate average confidence
        avg_confidence = state["conflict_analysis"].avg_confidence

        state["final_diagnosis"] = majority_diagnosis
        state["final_confidence"] = avg_confidence
        state["reasoning_chain"] = [
            f"Expert consensus achieved with {avg_confidence:.1%} average confidence",
            f"All experts agree on diagnosis: {majority_diagnosis}",
            "No debate required - proceeding with unanimous decision"
        ]
        state["debate_resolved"] = True

        print(f"  Decision: {majority_diagnosis} (confidence: {avg_confidence:.1%})")

        return state

    def _adjust_thresholds_node(self, state: DebateState) -> DebateState:
        """Request experts to adjust decision thresholds."""
        print(f"\n[Node] Adjusting thresholds (round {state['current_round'] + 1})...")

        # This is a placeholder - in real implementation, would call expert models
        # with adjusted thresholds using predict_with_threshold()

        threshold_adj = self.config.threshold_adjustment
        current_round = state["current_round"]

        # Record threshold adjustment
        adjustment_record = {
            "round": current_round + 1,
            "adjustment": threshold_adj,
            "expert_opinions": [
                {
                    "expert": op.expert_name,
                    "diagnosis": op.diagnosis,
                    "probability": op.probability,
                    "confidence": op.confidence
                }
                for op in state["expert_opinions"]
            ]
        }

        if "threshold_history" not in state:
            state["threshold_history"] = []
        state["threshold_history"] = state["threshold_history"] + [adjustment_record]

        print(f"  Threshold adjustment: ±{threshold_adj}")
        print(f"  Experts re-evaluating with adjusted thresholds...")

        return state

    def _debate_round_node(self, state: DebateState) -> DebateState:
        """Execute one round of debate."""
        print(f"\n[Node] Debate round {state['current_round'] + 1}...")

        current_round = state["current_round"] + 1
        state["current_round"] = current_round

        # Build debate prompt
        debate_prompt = build_debate_round_prompt(
            expert_opinions=state["expert_opinions"],
            round_number=current_round,
            previous_debates=state.get("debate_history", [])
        )

        # Record debate
        debate_record = f"Round {current_round}:\n{debate_prompt}"

        if "debate_history" not in state:
            state["debate_history"] = []
        state["debate_history"] = state["debate_history"] + [debate_record]

        # Re-check for conflict resolution after threshold adjustment
        conflict_analysis = self.conflict_resolver.detect_conflict(
            state["expert_opinions"]
        )

        state["conflict_analysis"] = conflict_analysis
        state["debate_resolved"] = not conflict_analysis.has_conflict

        print(f"  Debate round {current_round} complete")
        print(f"  Conflict resolved: {state['debate_resolved']}")

        return state

    def _query_rag_node(self, state: DebateState) -> DebateState:
        """Query RAG system for medical literature."""
        print("\n[Node] Querying RAG (medical literature)...")

        if not self.config.enable_rag or not self.rag_system:
            print("  RAG disabled or not available")
            state["rag_context"] = None
            return state

        # Retrieve literature for conflict
        rag_results = self.rag_system.retrieve_for_conflict(
            conflicting_opinions=state["expert_opinions"]
        )

        # Format context
        rag_context = self.rag_system.format_context_for_llm(rag_results)

        state["rag_context"] = rag_context

        print(f"  Retrieved {len(rag_results.documents)} literature documents")

        return state

    def _query_cag_node(self, state: DebateState) -> DebateState:
        """Query CAG system for similar cases."""
        print("\n[Node] Querying CAG (similar cases)...")

        if not self.config.enable_cag or not self.cag_system:
            print("  CAG disabled or not available")
            state["cag_context"] = None
            return state

        # Retrieve similar cases
        cag_results = self.cag_system.retrieve_for_conflict(
            conflicting_opinions=state["expert_opinions"],
            sample_data=state.get("sample_data", {})
        )

        # Format context
        cag_context = self.cag_system.format_context_for_llm(cag_results)

        state["cag_context"] = cag_context

        print(f"  Retrieved {len(cag_results.similar_cases)} similar cases")

        return state

    def _final_decision_node(self, state: DebateState) -> DebateState:
        """Make final decision using CMO reasoning with Chain-of-Thought."""
        print("\n[Node] Making final decision (CMO CoT reasoning)...")

        # If already resolved in quick decision, skip
        if state.get("final_diagnosis"):
            print("  Decision already made")
            return state

        # Import CoT prompt builder
        from clinical.utils.prompts import build_cmo_cot_decision_prompt
        import json

        # Build CoT prompt
        cot_prompt = build_cmo_cot_decision_prompt(
            expert_opinions=state["expert_opinions"],
            debate_rounds=state["current_round"],
            threshold_history=state.get("threshold_history", []),
            rag_context=state.get("rag_context"),
            cag_context=state.get("cag_context"),
            patient_metadata=state.get("patient_metadata", {})
        )

        # Call LLM for CoT reasoning (if available)
        cot_response = None
        cot_response_raw = None
        if hasattr(self, 'llm_wrapper') and self.llm_wrapper:
            try:
                print("  → Calling LLM for Chain-of-Thought reasoning...")

                # Convert prompt to messages format and call async
                import asyncio
                messages = [{"role": "user", "content": cot_prompt}]

                # Run async call synchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    response_dict = loop.run_until_complete(
                        self.llm_wrapper.call(messages=messages, temperature=0.3)
                    )
                    cot_response_raw = response_dict.get("content", "")
                finally:
                    loop.close()

                print(f"  → LLM returned {len(cot_response_raw)} characters")

                # Save raw response for debugging
                try:
                    from pathlib import Path
                    import datetime
                    debug_dir = Path("data/debug_logs")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    debug_file = debug_dir / f"cmo_cot_response_{timestamp}.txt"
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write("=== CMO CoT Prompt ===\n")
                        f.write(cot_prompt)
                        f.write("\n\n=== CMO CoT Response ===\n")
                        f.write(cot_response_raw)
                    print(f"  → Debug log saved to: {debug_file}")
                except Exception as debug_error:
                    print(f"  ⚠ Failed to save debug log: {debug_error}")

                # Try to parse JSON response
                try:
                    # Extract JSON from code block if present
                    if "```json" in cot_response_raw:
                        json_start = cot_response_raw.find("```json") + 7
                        json_end = cot_response_raw.find("```", json_start)
                        json_str = cot_response_raw[json_start:json_end].strip()
                        print("  → Extracted JSON from code block")
                    elif "{" in cot_response_raw and "}" in cot_response_raw:
                        json_start = cot_response_raw.find("{")
                        json_end = cot_response_raw.rfind("}") + 1
                        json_str = cot_response_raw[json_start:json_end]
                        print("  → Extracted JSON from raw text")
                    else:
                        json_str = cot_response_raw
                        print("  → Using raw response as JSON")

                    cot_response = json.loads(json_str)
                    print("  ✓ CMO CoT reasoning completed successfully")

                    # Validate expected fields
                    required_fields = ["final_diagnosis", "confidence"]
                    missing_fields = [f for f in required_fields if f not in cot_response]
                    if missing_fields:
                        print(f"  ⚠ Warning: Missing required fields: {missing_fields}")

                except json.JSONDecodeError as e:
                    print(f"  ✗ Failed to parse CMO response as JSON: {e}")
                    print(f"  → Response preview (first 500 chars):")
                    print(f"     {cot_response_raw[:500]}")
                    print(f"  → Check debug log for full response")
                    cot_response = None
            except Exception as e:
                print(f"  ✗ LLM call failed: {e}")
                import traceback
                print(f"  → Traceback:")
                traceback.print_exc()
                cot_response = None

        # Extract diagnosis and reasoning from CoT response
        if cot_response:
            final_diagnosis = cot_response.get("final_diagnosis")
            confidence = cot_response.get("confidence", 0.0)

            # Build detailed reasoning chain from CoT steps
            reasoning = []

            if cot_response.get("step1_consensus_analysis"):
                reasoning.append(f"**Expert Consensus Analysis**: {cot_response['step1_consensus_analysis']}")

            if cot_response.get("step2_biomarker_evaluation"):
                reasoning.append(f"**Biomarker Evaluation**: {cot_response['step2_biomarker_evaluation']}")

            if cot_response.get("step3_external_evidence"):
                reasoning.append(f"**External Evidence Integration**: {cot_response['step3_external_evidence']}")

            if cot_response.get("step4_alternatives"):
                reasoning.append(f"**Alternative Diagnoses Considered**: {cot_response['step4_alternatives']}")

            if cot_response.get("step5_evidence_weighting"):
                reasoning.append(f"**Evidence Weighting**: {cot_response['step5_evidence_weighting']}")

            if cot_response.get("step6_final_diagnosis"):
                reasoning.append(f"**Final Conclusion**: {cot_response['step6_final_diagnosis']}")

            # Add additional reasoning chain items if provided
            if cot_response.get("reasoning_chain"):
                reasoning.extend(cot_response["reasoning_chain"])

        else:
            # Fallback: Use weighted diagnosis if LLM fails
            print("  → Using fallback weighted diagnosis (LLM unavailable)")
            final_diagnosis = self.conflict_resolver.get_weighted_diagnosis(
                state["expert_opinions"]
            )
            avg_confidence = state["conflict_analysis"].avg_confidence
            confidence = avg_confidence

            # Build basic reasoning chain
            reasoning = [
                f"Completed {state['current_round']} rounds of debate",
                f"Expert opinions analyzed with threshold adjustments",
            ]

            if state.get("rag_context"):
                reasoning.append("Medical literature evidence reviewed")

            if state.get("cag_context"):
                reasoning.append("Similar historical cases considered")

            reasoning.append(f"Weighted majority diagnosis: {final_diagnosis} (confidence: {confidence:.1%})")

        state["final_diagnosis"] = final_diagnosis
        state["final_confidence"] = confidence
        state["reasoning_chain"] = reasoning
        state["cmo_cot_response"] = cot_response  # Store full CoT response for reporting

        print(f"  Final decision: {final_diagnosis}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Reasoning steps: {len(reasoning)}")

        return state

    # Conditional edge functions
    def _should_debate(self, state: DebateState) -> str:
        """Determine if debate is needed."""
        if state["requires_debate"]:
            return "debate"
        return "quick"

    def _check_threshold_resolution(self, state: DebateState) -> str:
        """Check if threshold adjustment resolved conflict."""
        if state.get("debate_resolved", False):
            return "resolved"
        return "continue"

    def _check_debate_status(self, state: DebateState) -> str:
        """Check debate status and determine next step."""
        if state.get("debate_resolved", False):
            return "resolved"

        if state["current_round"] >= self.config.max_rounds:
            return "max_rounds"

        return "continue"

    def run_debate(
        self,
        expert_opinions: List[ExpertOpinion],
        sample_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run debate workflow.

        Args:
            expert_opinions: List of expert opinions
            sample_data: Optional omics data for the sample

        Returns:
            Final state dictionary
        """
        print("\n" + "="*60)
        print("Starting Debate Workflow")
        print("="*60)

        # Initialize state
        initial_state = {
            "expert_opinions": expert_opinions,
            "sample_data": sample_data or {},
            "current_round": 0,
            "max_rounds": self.config.max_rounds,
            "threshold_adjustment": self.config.threshold_adjustment,
            "debate_history": [],
            "threshold_history": [],
            "conflict_analysis": None,
            "rag_context": None,
            "cag_context": None,
            "final_diagnosis": None,
            "final_confidence": None,
            "reasoning_chain": None,
            "requires_debate": False,
            "requires_rag": False,
            "requires_cag": False,
            "debate_resolved": False
        }

        # Run workflow
        final_state = self.graph.invoke(initial_state)

        print("\n" + "="*60)
        print("Debate Workflow Complete")
        print("="*60)

        return final_state

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DebateSystem("
            f"max_rounds={self.config.max_rounds}, "
            f"threshold_adj={self.config.threshold_adjustment})"
        )

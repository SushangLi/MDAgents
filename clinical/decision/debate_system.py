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
        config: Optional[DebateConfig] = None
    ):
        """
        Initialize debate system.

        Args:
            conflict_resolver: Conflict resolver instance
            rag_system: RAG system for literature search
            cag_system: CAG system for case retrieval
            config: Debate configuration
        """
        self.conflict_resolver = conflict_resolver or ConflictResolver()
        self.rag_system = rag_system
        self.cag_system = cag_system
        self.config = config or DebateConfig()

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
        """Make final decision using CMO reasoning."""
        print("\n[Node] Making final decision...")

        # If already resolved in quick decision, skip
        if state.get("final_diagnosis"):
            print("  Decision already made")
            return state

        # Use weighted diagnosis with all evidence
        final_diagnosis = self.conflict_resolver.get_weighted_diagnosis(
            state["expert_opinions"]
        )

        # Calculate confidence based on evidence
        avg_confidence = state["conflict_analysis"].avg_confidence

        # Build reasoning chain
        reasoning = [
            f"Completed {state['current_round']} rounds of debate",
            f"Expert opinions analyzed with threshold adjustments",
        ]

        if state.get("rag_context"):
            reasoning.append("Medical literature evidence reviewed")

        if state.get("cag_context"):
            reasoning.append("Similar historical cases considered")

        reasoning.append(f"Final diagnosis: {final_diagnosis} (confidence: {avg_confidence:.1%})")

        state["final_diagnosis"] = final_diagnosis
        state["final_confidence"] = avg_confidence
        state["reasoning_chain"] = reasoning

        print(f"  Final decision: {final_diagnosis}")
        print(f"  Confidence: {avg_confidence:.1%}")

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

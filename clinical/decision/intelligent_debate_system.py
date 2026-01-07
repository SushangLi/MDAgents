"""
Intelligent Debate System with Natural Language Request Support.

Extends DebateSystem to support intelligent scheduling based on
user's natural language requests, data filtering, and bilingual reporting.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated, Tuple
import operator
import pandas as pd
import numpy as np

from langgraph.graph import StateGraph, END

from clinical.decision.debate_system import DebateSystem, DebateConfig, DebateState
from clinical.decision.request_parser import RequestParser
from clinical.decision.bilingual_report_generator import BilingualReportGenerator
from clinical.models.diagnosis_config import DiagnosisConfig
from clinical.models.expert_opinion import ExpertOpinion
from clinical.models.diagnosis_result import DiagnosisResult


# Extended state for intelligent diagnosis
class IntelligentDiagnosisState(TypedDict):
    """Extended state for intelligent diagnosis workflow."""

    # User request
    user_request: str
    parsed_config: Optional[DiagnosisConfig]

    # Data selection
    selected_omics_types: List[str]
    selected_patient_ids: Optional[List[str]]
    selected_row_range: Optional[Tuple[int, int]]
    available_data: Dict[str, Any]
    filtered_data: Dict[str, Any]

    # Preprocessed data
    preprocessed_data: Dict[str, Any]

    # Expert opinions
    expert_opinions: List[ExpertOpinion]
    sample_data: Dict[str, Any]

    # Inherit from DebateState
    conflict_analysis: Any
    current_round: int
    max_rounds: int
    debate_history: Annotated[List[str], operator.add]
    threshold_adjustment: float
    threshold_history: Annotated[List[Dict[str, Any]], operator.add]
    rag_context: Optional[str]
    cag_context: Optional[str]
    final_diagnosis: Optional[str]
    final_confidence: Optional[float]
    reasoning_chain: Optional[List[str]]
    requires_debate: bool
    requires_rag: bool
    requires_cag: bool
    debate_resolved: bool
    force_rag_even_no_conflict: bool

    # Report configuration
    report_detail_level: str
    bilingual: bool
    bilingual_report: Optional[str]

    # Metadata
    patient_metadata: Optional[Dict[str, Any]]


class IntelligentDebateSystem(DebateSystem):
    """
    Intelligent debate system with natural language request support.

    Extends DebateSystem to add:
    - Natural language request parsing
    - Data filtering (patient IDs, row ranges)
    - Dynamic omics selection
    - Bilingual report generation
    """

    def __init__(
        self,
        request_parser: Optional[RequestParser] = None,
        preprocessors: Optional[Dict] = None,
        experts: Optional[Dict] = None,
        bilingual_generator: Optional[BilingualReportGenerator] = None,
        **kwargs
    ):
        """
        Initialize intelligent debate system.

        Args:
            request_parser: RequestParser for NL requests
            preprocessors: Dict of preprocessors {omics_type: preprocessor}
            experts: Dict of expert models {omics_type: expert}
            bilingual_generator: Bilingual report generator
            **kwargs: Arguments for parent DebateSystem
        """
        super().__init__(**kwargs)

        self.request_parser = request_parser
        self.preprocessors = preprocessors or {}
        self.experts = experts or {}
        self.bilingual_generator = bilingual_generator or BilingualReportGenerator()

        # Rebuild graph with intelligent nodes
        self.graph = self._build_intelligent_graph()

        print("✓ Intelligent Debate System initialized")
        print(f"  Request Parser: {'Yes' if request_parser else 'No'}")
        print(f"  Preprocessors: {list(self.preprocessors.keys())}")
        print(f"  Experts: {list(self.experts.keys())}")

    def _build_intelligent_graph(self) -> StateGraph:
        """Build extended LangGraph with intelligent scheduling."""
        workflow = StateGraph(IntelligentDiagnosisState)

        # New nodes for intelligent scheduling
        workflow.add_node("parse_request", self._parse_request_node)
        workflow.add_node("filter_data", self._filter_data_node)
        workflow.add_node("select_omics", self._select_omics_node)
        workflow.add_node("preprocess_data", self._preprocess_data_node)
        workflow.add_node("get_expert_opinions", self._get_expert_opinions_node)

        # Inherited nodes from parent
        workflow.add_node("detect_conflict", self._detect_conflict_node)
        workflow.add_node("quick_decision", self._quick_decision_node)
        workflow.add_node("adjust_thresholds", self._adjust_thresholds_node)
        workflow.add_node("debate_round", self._debate_round_node)
        workflow.add_node("query_rag", self._query_rag_conditional_node)
        workflow.add_node("query_cag", self._query_cag_node)
        workflow.add_node("final_decision", self._final_decision_node)

        # New node for bilingual reporting
        workflow.add_node("generate_bilingual_report", self._generate_bilingual_report_node)

        # Entry point
        workflow.set_entry_point("parse_request")

        # Linear flow for data preparation
        workflow.add_edge("parse_request", "filter_data")
        workflow.add_edge("filter_data", "select_omics")
        workflow.add_edge("select_omics", "preprocess_data")
        workflow.add_edge("preprocess_data", "get_expert_opinions")
        workflow.add_edge("get_expert_opinions", "detect_conflict")

        # Conditional branching for debate/RAG
        workflow.add_conditional_edges(
            "detect_conflict",
            self._should_debate_or_rag,
            {
                "quick": "quick_decision",
                "quick_with_rag": "query_rag",  # New: force RAG even without conflict
                "debate": "adjust_thresholds"
            }
        )

        # Quick decision flows
        workflow.add_edge("quick_decision", "generate_bilingual_report")

        # Debate flow (inherited from parent)
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

        # RAG/CAG flow
        workflow.add_edge("query_rag", "query_cag")
        workflow.add_edge("query_cag", "final_decision")
        workflow.add_edge("final_decision", "generate_bilingual_report")

        # Exit
        workflow.add_edge("generate_bilingual_report", END)

        return workflow.compile()

    # New intelligent nodes

    async def _parse_request_node(self, state: IntelligentDiagnosisState) -> IntelligentDiagnosisState:
        """Parse natural language request into config."""
        print("\n[Intelligent Node] Parsing user request...")

        # Skip if config already provided
        if state.get("parsed_config"):
            print("  ✓ Using pre-parsed config")
            config = state["parsed_config"]
            state["selected_omics_types"] = config.omics_types
            state["selected_patient_ids"] = config.patient_ids
            state["selected_row_range"] = config.row_range
            state["force_rag_even_no_conflict"] = config.force_rag_even_no_conflict
            state["max_rounds"] = config.max_debate_rounds
            state["report_detail_level"] = config.detail_level
            state["bilingual"] = config.bilingual
            return state

        if self.request_parser and state.get("user_request"):
            try:
                config = await self.request_parser.parse_request(state["user_request"])
                state["parsed_config"] = config
                state["selected_omics_types"] = config.omics_types
                state["selected_patient_ids"] = config.patient_ids
                state["selected_row_range"] = config.row_range
                state["force_rag_even_no_conflict"] = config.force_rag_even_no_conflict
                state["max_rounds"] = config.max_debate_rounds
                state["report_detail_level"] = config.detail_level
                state["bilingual"] = config.bilingual

                print(f"  ✓ Parsed: {config}")
            except Exception as e:
                print(f"  ⚠ Parsing failed: {e}, using defaults")
                # Use default config
                config = DiagnosisConfig.get_default()
                state["parsed_config"] = config
                state["selected_omics_types"] = config.omics_types
        else:
            print("  No request parser or user request, using defaults")
            config = DiagnosisConfig.get_default()
            state["parsed_config"] = config
            state["selected_omics_types"] = state.get("selected_omics_types", config.omics_types)

        return state

    def _filter_data_node(self, state: IntelligentDiagnosisState) -> IntelligentDiagnosisState:
        """Filter data by patient IDs and row range."""
        print("\n[Intelligent Node] Filtering data...")

        available_data = state.get("available_data", {})
        filtered_data = {}

        # If no filtering needed, pass through
        if not state.get("selected_patient_ids") and not state.get("selected_row_range"):
            state["filtered_data"] = available_data
            print("  No filtering needed, using all data")
            return state

        # Filter by patient IDs or row range
        for omics_type, data in available_data.items():
            filtered = data

            # Filter by patient IDs
            if state.get("selected_patient_ids") and isinstance(data, (pd.DataFrame, dict)):
                if isinstance(data, pd.DataFrame) and "patient_id" in data.columns:
                    filtered = data[data["patient_id"].isin(state["selected_patient_ids"])]
                    print(f"  ✓ Filtered {omics_type} by patient IDs: {len(filtered)} rows")

            # Filter by row range
            if state.get("selected_row_range") and isinstance(filtered, pd.DataFrame):
                start, end = state["selected_row_range"]
                filtered = filtered.iloc[start:end]
                print(f"  ✓ Filtered {omics_type} by row range [{start}:{end}]: {len(filtered)} rows")

            filtered_data[omics_type] = filtered

        state["filtered_data"] = filtered_data
        return state

    def _select_omics_node(self, state: IntelligentDiagnosisState) -> IntelligentDiagnosisState:
        """Select which omics data to analyze."""
        print("\n[Intelligent Node] Selecting omics data...")

        selected_omics = state.get("selected_omics_types", ["microbiome", "metabolome", "proteome"])
        filtered_data = state.get("filtered_data", state.get("available_data", {}))

        # Select only requested omics types
        selected_data = {}
        for omics_type in selected_omics:
            if omics_type in filtered_data:
                selected_data[omics_type] = filtered_data[omics_type]
                print(f"  ✓ Selected: {omics_type}")
            else:
                print(f"  ⚠ {omics_type} not available in data")

        state["sample_data"] = selected_data
        return state

    def _preprocess_data_node(self, state: IntelligentDiagnosisState) -> IntelligentDiagnosisState:
        """Preprocess selected omics data."""
        print("\n[Intelligent Node] Preprocessing data...")

        sample_data = state.get("sample_data", {})
        preprocessed = {}

        # Check if preprocessors are available
        if not self.preprocessors:
            print("  ⚠ No preprocessors configured, using data as-is (assuming already preprocessed)")
            state["preprocessed_data"] = sample_data
            return state

        for omics_type, data in sample_data.items():
            if omics_type in self.preprocessors:
                try:
                    preprocessor = self.preprocessors[omics_type]
                    # Assuming preprocessor has transform or fit_transform method
                    if hasattr(preprocessor, 'transform'):
                        result = preprocessor.transform(data)
                    elif hasattr(preprocessor, 'fit_transform'):
                        result = preprocessor.fit_transform(data)
                    else:
                        result = data

                    preprocessed[omics_type] = result
                    print(f"  ✓ Preprocessed: {omics_type}")
                except Exception as e:
                    print(f"  ⚠ Preprocessing {omics_type} failed: {e}")
                    preprocessed[omics_type] = data
            else:
                print(f"  ⚠ No preprocessor for {omics_type}, using data as-is (assuming already preprocessed)")
                preprocessed[omics_type] = data

        state["preprocessed_data"] = preprocessed
        return state

    def _get_expert_opinions_node(self, state: IntelligentDiagnosisState) -> IntelligentDiagnosisState:
        """Get expert opinions from selected omics."""
        print("\n[Intelligent Node] Getting expert opinions...")

        preprocessed_data = state.get("preprocessed_data", {})
        expert_opinions = []

        for omics_type, data in preprocessed_data.items():
            if omics_type in self.experts:
                try:
                    expert = self.experts[omics_type]

                    # Extract raw data for prediction
                    if hasattr(data, 'data'):
                        prediction_data = data.data
                    elif isinstance(data, pd.DataFrame):
                        prediction_data = data
                    elif isinstance(data, np.ndarray):
                        prediction_data = data
                    else:
                        prediction_data = data

                    # Get prediction
                    opinions = expert.predict(prediction_data)

                    if opinions:
                        expert_opinions.extend(opinions if isinstance(opinions, list) else [opinions])
                        conf_str = f"{opinions[0].confidence:.1%}" if opinions else "N/A"
                        print(f"  ✓ {omics_type}: {opinions[0].diagnosis if opinions else 'N/A'} ({conf_str})")
                except Exception as e:
                    print(f"  ⚠ Expert {omics_type} failed: {e}")
            else:
                print(f"  ⚠ No expert for {omics_type}")

        state["expert_opinions"] = expert_opinions
        return state

    def _should_debate_or_rag(self, state: IntelligentDiagnosisState) -> str:
        """Decide whether to debate, force RAG, or quick decision."""
        force_rag = state.get("force_rag_even_no_conflict", False)
        requires_debate = state.get("requires_debate", False)
        enable_rag = state.get("parsed_config", {}).enable_rag if state.get("parsed_config") else True

        if force_rag and enable_rag and not requires_debate:
            print("  → Forcing RAG query (no conflict)")
            return "quick_with_rag"
        elif requires_debate:
            print("  → Starting debate (conflict detected)")
            return "debate"
        else:
            print("  → Quick decision (no conflict, no forced RAG)")
            return "quick"

    def _query_rag_conditional_node(self, state: IntelligentDiagnosisState) -> IntelligentDiagnosisState:
        """Query RAG conditionally (supports no-conflict scenarios)."""
        print("\n[Node] Querying RAG...")

        enable_rag = state.get("parsed_config", {}).enable_rag if state.get("parsed_config") else True

        if not enable_rag or not self.rag_system:
            print("  RAG disabled or unavailable")
            state["rag_context"] = None
            return state

        # Check if forced RAG (no conflict)
        if state.get("force_rag_even_no_conflict") and not state.get("requires_debate"):
            print("  Forced RAG query (no conflict)")
            # Use diagnosis as query
            if state.get("final_diagnosis"):
                query = f"Medical evidence for {state['final_diagnosis']}"
            elif state.get("expert_opinions"):
                query = f"Medical evidence for {state['expert_opinions'][0].diagnosis}"
            else:
                query = "General periodontal diagnosis"

            rag_results = self.rag_system.search(query, top_k=3)
        else:
            # Conflict-based query
            rag_results = self.rag_system.retrieve_for_conflict(state["expert_opinions"])

        rag_context = self.rag_system.format_context_for_llm(rag_results)
        state["rag_context"] = rag_context

        print(f"  ✓ Retrieved {len(rag_results.documents) if hasattr(rag_results, 'documents') else 0} documents")
        return state

    def _generate_bilingual_report_node(self, state: IntelligentDiagnosisState) -> IntelligentDiagnosisState:
        """Generate bilingual report."""
        print("\n[Intelligent Node] Generating bilingual report...")

        # Construct DiagnosisResult from state
        diagnosis_result = DiagnosisResult(
            patient_id=state.get("patient_metadata", {}).get("patient_id", "Unknown"),
            diagnosis=state.get("final_diagnosis", "N/A"),
            confidence=state.get("final_confidence", 0.0),
            expert_opinions=state.get("expert_opinions", []),
            conflict_resolution={
                "method": "debate" if state.get("requires_debate") else "consensus",
                "debate_rounds": state.get("current_round", 0),
                "threshold_history": state.get("threshold_history", []),  # Include debate evolution
                "debate_resolved": state.get("debate_resolved", False),
                "rag_used": state.get("rag_context") is not None,
                "cag_used": state.get("cag_context") is not None,
                "cmo_cot_response": state.get("cmo_cot_response")  # Include CMO CoT reasoning
            } if state.get("requires_debate") or state.get("cmo_cot_response") else None,
            key_biomarkers=[],  # Would extract from expert opinions
            clinical_recommendations=[],  # Would extract from LLM
            explanation="\n\n".join(state.get("reasoning_chain", ["Based on expert analysis"])),  # Full reasoning chain
            references=[],
            metadata={}
        )

        # Generate report
        if state.get("bilingual", True):
            report = self.bilingual_generator.generate_report(
                diagnosis_result,
                patient_metadata=state.get("patient_metadata")
            )
        else:
            # Use standard generator
            from clinical.decision.report_generator import ReportGenerator
            generator = ReportGenerator()
            report = generator.generate_report(
                diagnosis_result,
                patient_metadata=state.get("patient_metadata")
            )

        state["bilingual_report"] = report
        print(f"  ✓ Report generated ({len(report)} characters)")

        return state

    async def run_intelligent_diagnosis(
        self,
        user_request: str = None,
        available_data: Dict[str, Any] = None,
        patient_metadata: Dict[str, Any] = None,
        parsed_config: DiagnosisConfig = None
    ) -> Dict[str, Any]:
        """
        Run intelligent diagnosis workflow.

        Args:
            user_request: Natural language request (optional if parsed_config provided)
            available_data: Available omics data
            patient_metadata: Patient metadata
            parsed_config: Pre-parsed config (skips NL parsing if provided)

        Returns:
            Final state dict with bilingual report
        """
        print("\n" + "="*70)
        print("Starting Intelligent Diagnosis Workflow")
        print("="*70)

        # Initialize state
        initial_state = {
            "user_request": user_request or "",
            "parsed_config": parsed_config,
            "available_data": available_data or {},
            "patient_metadata": patient_metadata or {},
            "current_round": 0,
            "max_rounds": 3,
            "threshold_adjustment": 0.1,
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
            "debate_resolved": False,
            "force_rag_even_no_conflict": False,
            "report_detail_level": "standard",
            "bilingual": True,
            "bilingual_report": None
        }

        # Run workflow asynchronously
        final_state = await self.graph.ainvoke(initial_state)

        print("\n" + "="*70)
        print("Intelligent Diagnosis Complete")
        print("="*70)

        return final_state

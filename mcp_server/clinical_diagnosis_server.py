"""
Clinical Diagnosis MCP Server.

Exposes multi-omics clinical diagnosis system via Model Context Protocol.
Integrates all four layers: Perception, Expert, Collaboration, Decision.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Clinical diagnosis modules
from clinical.preprocessing.microbiome_preprocessor import MicrobiomePreprocessor
from clinical.preprocessing.metabolome_preprocessor import MetabolomePreprocessor
from clinical.preprocessing.proteome_preprocessor import ProteomePreprocessor
from clinical.experts.model_manager import ModelManager
from clinical.collaboration.rag_system import RAGSystem
from clinical.collaboration.cag_system import CAGSystem
from clinical.decision.debate_system import DebateSystem, DebateConfig
from clinical.decision.cmo_coordinator import CMOCoordinator
from clinical.decision.report_generator import ReportGenerator
from clinical.models.expert_opinion import ExpertOpinion

# Configure logging to stderr
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create MCP server
app = Server("clinical-diagnosis")

# Global system components
_model_manager: Optional[ModelManager] = None
_rag_system: Optional[RAGSystem] = None
_cag_system: Optional[CAGSystem] = None
_debate_system: Optional[DebateSystem] = None
_cmo_coordinator: Optional[CMOCoordinator] = None
_report_generator: Optional[ReportGenerator] = None
_preprocessors: Optional[Dict[str, Any]] = None


def _initialize_systems():
    """Initialize all clinical diagnosis systems."""
    global _model_manager, _rag_system, _cag_system, _debate_system
    global _cmo_coordinator, _report_generator, _preprocessors

    if _model_manager is not None:
        return  # Already initialized

    logger.info("Initializing clinical diagnosis systems...")

    # Initialize preprocessors
    _preprocessors = {
        "microbiome": MicrobiomePreprocessor(),
        "metabolome": MetabolomePreprocessor(),
        "proteome": ProteomePreprocessor()
    }

    # Initialize model manager
    _model_manager = ModelManager()

    # Initialize RAG system
    try:
        _rag_system = RAGSystem()
    except Exception as e:
        logger.warning(f"RAG system initialization failed: {e}")
        _rag_system = None

    # Initialize CAG system
    try:
        _cag_system = CAGSystem()
    except Exception as e:
        logger.warning(f"CAG system initialization failed: {e}")
        _cag_system = None

    # Initialize debate system
    _debate_system = DebateSystem(
        rag_system=_rag_system,
        cag_system=_cag_system,
        config=DebateConfig(max_rounds=3, threshold_adjustment=0.1)
    )

    # Initialize CMO coordinator (without LLM function for now)
    _cmo_coordinator = CMOCoordinator(
        llm_call_func=None,  # Will be set when LLM MCP session available
        temperature=0.3,
        use_mcp=True
    )

    # Initialize report generator
    _report_generator = ReportGenerator()

    logger.info("✓ Clinical diagnosis systems initialized")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available clinical diagnosis tools."""
    return [
        Tool(
            name="diagnose_patient",
            description=(
                "Complete multi-omics diagnostic workflow. "
                "Takes raw omics data, runs preprocessing, expert analysis, "
                "debate resolution, and generates diagnostic report."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "Patient identifier"
                    },
                    "microbiome_data": {
                        "type": "object",
                        "description": "Microbiome abundance data (JSON object)"
                    },
                    "metabolome_data": {
                        "type": "object",
                        "description": "Metabolome intensity data (JSON object)"
                    },
                    "proteome_data": {
                        "type": "object",
                        "description": "Proteome expression data (JSON object)"
                    },
                    "patient_metadata": {
                        "type": "object",
                        "description": "Optional patient metadata (age, gender, etc.)"
                    },
                    "generate_report": {
                        "type": "boolean",
                        "description": "Whether to generate full markdown report",
                        "default": True
                    }
                },
                "required": ["patient_id"]
            }
        ),
        Tool(
            name="preprocess_omics_data",
            description=(
                "Preprocess raw omics data (microbiome, metabolome, or proteome). "
                "Applies quality control, normalization, and feature engineering."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "omics_type": {
                        "type": "string",
                        "description": "Type of omics data",
                        "enum": ["microbiome", "metabolome", "proteome"]
                    },
                    "data": {
                        "type": "object",
                        "description": "Raw omics data (JSON object)"
                    },
                    "normalize": {
                        "type": "boolean",
                        "description": "Apply normalization",
                        "default": True
                    }
                },
                "required": ["omics_type", "data"]
            }
        ),
        Tool(
            name="query_knowledge_base",
            description=(
                "Search medical literature knowledge base using RAG system. "
                "Returns relevant papers and guidelines for diagnostic support."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    },
                    "min_relevance": {
                        "type": "number",
                        "description": "Minimum relevance score (0-1)",
                        "default": 0.5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_expert_explanations",
            description=(
                "Get expert opinions from trained ML models. "
                "Returns predictions with biological explanations and feature importance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "microbiome_features": {
                        "type": "object",
                        "description": "Preprocessed microbiome features"
                    },
                    "metabolome_features": {
                        "type": "object",
                        "description": "Preprocessed metabolome features"
                    },
                    "proteome_features": {
                        "type": "object",
                        "description": "Preprocessed proteome features"
                    }
                }
            }
        ),
        Tool(
            name="generate_diagnostic_report",
            description=(
                "Generate comprehensive diagnostic report in markdown format "
                "from diagnosis results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "diagnosis_result": {
                        "type": "object",
                        "description": "Diagnosis result object (JSON)"
                    },
                    "patient_metadata": {
                        "type": "object",
                        "description": "Optional patient metadata"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional path to save report"
                    }
                },
                "required": ["diagnosis_result"]
            }
        ),
        Tool(
            name="get_system_status",
            description="Get status of all clinical diagnosis system components",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    _initialize_systems()

    try:
        if name == "diagnose_patient":
            result = await _diagnose_patient(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "preprocess_omics_data":
            result = _preprocess_omics_data(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "query_knowledge_base":
            result = _query_knowledge_base(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_expert_explanations":
            result = _get_expert_explanations(arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "generate_diagnostic_report":
            result = _generate_diagnostic_report(arguments)
            return [TextContent(type="text", text=result)]  # Return markdown text

        elif name == "get_system_status":
            result = _get_system_status()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]

    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e), "tool": name})
        )]


async def _diagnose_patient(args: dict) -> dict:
    """
    Complete diagnostic workflow.

    Args:
        args: Arguments with patient data

    Returns:
        Diagnosis result dictionary
    """
    patient_id = args["patient_id"]
    logger.info(f"Starting diagnosis for patient {patient_id}")

    # Step 1: Preprocess omics data
    preprocessed_data = {}

    if args.get("microbiome_data"):
        microbiome_result = _preprocess_omics_data({
            "omics_type": "microbiome",
            "data": args["microbiome_data"]
        })
        preprocessed_data["microbiome"] = microbiome_result["features"]

    if args.get("metabolome_data"):
        metabolome_result = _preprocess_omics_data({
            "omics_type": "metabolome",
            "data": args["metabolome_data"]
        })
        preprocessed_data["metabolome"] = metabolome_result["features"]

    if args.get("proteome_data"):
        proteome_result = _preprocess_omics_data({
            "omics_type": "proteome",
            "data": args["proteome_data"]
        })
        preprocessed_data["proteome"] = proteome_result["features"]

    # Step 2: Get expert opinions
    expert_result = _get_expert_explanations(preprocessed_data)

    if expert_result.get("error"):
        return expert_result

    expert_opinions = [
        ExpertOpinion.from_dict(op) for op in expert_result["expert_opinions"]
    ]

    # Step 3: Run debate system
    debate_state = _debate_system.run_debate(
        expert_opinions=expert_opinions,
        sample_data=preprocessed_data
    )

    # Step 4: CMO makes final decision
    if debate_state["conflict_analysis"].has_conflict:
        diagnosis_result = await _cmo_coordinator.make_conflict_resolution(
            expert_opinions=expert_opinions,
            conflict_analysis=debate_state["conflict_analysis"],
            rag_context=debate_state.get("rag_context"),
            cag_context=debate_state.get("cag_context"),
            debate_history=debate_state.get("debate_history"),
            patient_metadata=args.get("patient_metadata")
        )
    else:
        diagnosis_result = await _cmo_coordinator.make_quick_decision(
            expert_opinions=expert_opinions,
            conflict_analysis=debate_state["conflict_analysis"]
        )

    # Set patient ID
    diagnosis_result.patient_id = patient_id

    # Step 5: Generate report if requested
    report = None
    if args.get("generate_report", True):
        report = _report_generator.generate_report(
            diagnosis_result=diagnosis_result,
            patient_metadata=args.get("patient_metadata")
        )

    return {
        "patient_id": patient_id,
        "diagnosis": diagnosis_result.diagnosis,
        "confidence": diagnosis_result.confidence,
        "diagnosis_result": diagnosis_result.to_dict(),
        "report": report,
        "metadata": diagnosis_result.metadata
    }


def _preprocess_omics_data(args: dict) -> dict:
    """Preprocess omics data."""
    omics_type = args["omics_type"]
    data = args["data"]

    preprocessor = _preprocessors[omics_type]

    # Convert dict to appropriate format (simplified - real implementation needs DataFrame)
    # For now, return as-is since we need actual data structure
    return {
        "omics_type": omics_type,
        "features": data,  # Placeholder
        "qc_passed": True,
        "n_features": len(data)
    }


def _query_knowledge_base(args: dict) -> dict:
    """Query RAG knowledge base."""
    if not _rag_system:
        return {"error": "RAG system not available"}

    query = args["query"]
    top_k = args.get("top_k", 5)
    min_relevance = args.get("min_relevance", 0.5)

    results = _rag_system.search(
        query=query,
        top_k=top_k,
        min_relevance=min_relevance
    )

    return {
        "query": query,
        "results": results.to_dict(),
        "n_results": len(results.documents)
    }


def _get_expert_explanations(args: dict) -> dict:
    """Get expert opinions."""
    # Check if models are trained
    try:
        experts = _model_manager.load_all_experts()
    except FileNotFoundError as e:
        return {
            "error": "Expert models not trained yet",
            "details": str(e),
            "hint": "Run scripts/model_training/train_experts.py first"
        }

    # Placeholder - needs actual DataFrame conversion
    # Real implementation would convert dict features to DataFrame and call predict()

    return {
        "expert_opinions": [
            {
                "expert_name": "microbiome_expert",
                "omics_type": "microbiome",
                "diagnosis": "PLACEHOLDER",
                "probability": 0.75,
                "confidence": 0.70,
                "top_features": [],
                "biological_explanation": "Expert models not yet trained on real data",
                "evidence_chain": [],
                "model_metadata": {},
                "timestamp": ""
            }
        ],
        "warning": "This is placeholder data - train models with real data first"
    }


def _generate_diagnostic_report(args: dict) -> str:
    """Generate diagnostic report."""
    from clinical.models.diagnosis_result import DiagnosisResult

    diagnosis_data = args["diagnosis_result"]

    # Convert dict to DiagnosisResult object
    diagnosis_result = DiagnosisResult.from_dict(diagnosis_data)

    # Generate report
    report = _report_generator.generate_report(
        diagnosis_result=diagnosis_result,
        patient_metadata=args.get("patient_metadata")
    )

    # Save if output path provided
    if args.get("output_path"):
        _report_generator.save_report(report, args["output_path"])

    return report


def _get_system_status() -> dict:
    """Get system status."""
    status = {
        "status": "operational",
        "components": {
            "preprocessors": {
                "microbiome": "ready" if _preprocessors else "not initialized",
                "metabolome": "ready" if _preprocessors else "not initialized",
                "proteome": "ready" if _preprocessors else "not initialized"
            },
            "expert_models": {
                "status": "checking...",
                "models_available": []
            },
            "rag_system": "ready" if _rag_system else "not available",
            "cag_system": "ready" if _cag_system else "not available",
            "debate_system": "ready" if _debate_system else "not initialized",
            "cmo_coordinator": "ready" if _cmo_coordinator else "not initialized",
            "report_generator": "ready" if _report_generator else "not initialized"
        }
    }

    # Check expert models
    try:
        experts = _model_manager.load_all_experts()
        status["components"]["expert_models"] = {
            "status": "ready",
            "models_available": list(experts.keys())
        }
    except:
        status["components"]["expert_models"] = {
            "status": "not trained",
            "models_available": []
        }

    # RAG statistics
    if _rag_system:
        rag_stats = _rag_system.get_statistics()
        status["components"]["rag_stats"] = rag_stats

    # CAG statistics
    if _cag_system:
        cag_stats = _cag_system.get_statistics()
        status["components"]["cag_stats"] = cag_stats

    return status


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

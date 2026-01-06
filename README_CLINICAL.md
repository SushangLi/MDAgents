# Clinical Diagnosis System - Quick Start Guide

## Overview

This is a multi-omics clinical diagnosis system with 4-layer architecture:
1. **Perception Layer**: Data preprocessing (microbiome, metabolome, proteome)
2. **Expert Layer**: 3 ML expert agents (RandomForest/XGBoost + SHAP)
3. **Collaboration Layer**: RAG (medical literature) + CAG (case cache)
4. **Decision Layer**: CMO coordinator with LangGraph debate mechanism

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Check System Status

```bash
python main_clinical.py status
```

### 3. Generate Test Data

```bash
python main_clinical.py generate-data
```

This creates:
- `data/test/` - 100 synthetic multi-omics samples (4 disease classes)
- `data/labeled/annotations.json` - Training labels
- Train/Val/Test splits (70/15/15)

### 4. Initialize Vector Database

```bash
python main_clinical.py init-vectordb
```

Initializes ChromaDB with 5 sample medical literature documents for RAG testing.

### 5. Run Tests

```bash
python main_clinical.py test
```

Runs all unit and integration tests:
- Preprocessing modules
- RAG/CAG systems
- Conflict detection & debate
- End-to-end diagnosis flow

### 6. Run Demo Diagnosis

```bash
python main_clinical.py demo
```

Runs a complete diagnosis workflow with mock expert opinions and generates a report.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               MCP Orchestrator (编排层)                      │
│   Routes messages between MCP servers                        │
└──────────┬──────────────┬──────────────┬────────────────────┘
           │              │              │
           ↓              ↓              ↓
   ┌──────────┐    ┌──────────┐   ┌─────────────────────┐
   │ LLM MCP  │    │ Tools    │   │ Clinical Diagnosis  │
   │ Server   │    │ MCP      │   │ MCP Server          │
   └──────────┘    │ Server   │   └──────────┬──────────┘
                   └──────────┘              │
                                             │
              ┌──────────────────────────────┴────────────────┐
              │                                                │
              ↓                                                ↓
   ┌──────────────────┐                          ┌──────────────────┐
   │ 1. Perception    │                          │ 2. Expert Layer  │
   │ - Preprocessing  │                          │ - 3 ML Experts   │
   │ - QC & Filtering │                          │ - SHAP Explain   │
   └──────────────────┘                          └──────────────────┘
              ↓                                                ↓
   ┌──────────────────┐                          ┌──────────────────┐
   │ 3. Collaboration │  ←─────────────────────→ │ 4. Decision      │
   │ - RAG (PubMed)   │      Debate Triggers     │ - CMO (LLM)      │
   │ - CAG (Cache)    │                          │ - LangGraph      │
   └──────────────────┘                          └──────────────────┘
```

## MCP Tools Exposed

The Clinical Diagnosis MCP Server exposes 6 tools:

1. **diagnose_patient**: Complete diagnostic workflow
2. **preprocess_omics_data**: Data preprocessing
3. **query_knowledge_base**: RAG literature search
4. **get_expert_explanations**: Expert ML predictions
5. **generate_diagnostic_report**: Report generation
6. **get_system_status**: System health check

## Training Expert Models

To train real expert models (requires labeled data):

```bash
python main_clinical.py train
```

Or manually:

```bash
python scripts/model_training/train_experts.py --grid-search
```

Models saved to `data/models/`.

## Testing Individual Modules

### Test RAG System
```bash
pytest tests/test_rag.py -v -s
```

### Test CAG System
```bash
pytest tests/test_cag.py -v -s
```

### Test Preprocessing
```bash
pytest tests/test_preprocessing.py -v -s
```

### Test Conflict Resolution
```bash
pytest tests/test_conflict_resolver.py -v -s
```

### Test Complete Workflow
```bash
pytest tests/test_diagnosis_flow.py -v -s
```

## Key Features

### 1. Multi-Round Debate with Threshold Adjustment

When experts disagree, the system:
1. Detects conflict (diagnosis disagreement, low confidence, etc.)
2. Triggers debate mechanism (max 3 rounds)
3. Adjusts expert decision thresholds (default ±0.1)
4. If unresolved → queries RAG (literature) and CAG (cases)
5. CMO makes final decision with reasoning chain

### 2. RAG (Retrieval-Augmented Generation)

- **Vector DB**: ChromaDB
- **Embeddings**: PubMedBERT (biomedical domain)
- **Usage**: Retrieves medical literature when experts conflict

Initialize with sample docs:
```bash
python scripts/knowledge_base/build_vector_db.py
```

Add your own literature:
```bash
python scripts/knowledge_base/ingest_literature.py path/to/pdfs --pattern "*.pdf"
```

### 3. CAG (Cache-Augmented Generation)

Caches diagnosed cases for similarity matching:
- Omics feature similarity (cosine distance)
- Clinical notes similarity (semantic)
- Diagnosis distribution in similar cases

### 4. Interpretable Reports

Generated reports include:
- Executive summary with confidence level
- Multi-omics analysis (all 3 experts)
- Diagnostic rationale with reasoning chain
- Key biomarkers table
- Differential diagnoses
- Clinical recommendations
- Literature references (if RAG used)
- Similar cases (if CAG used)

Example report: `data/test/test_report.md`

## File Structure

```
MDAgents/
├── clinical/                    # Core diagnosis system
│   ├── preprocessing/          # Perception layer (6 files)
│   ├── experts/                # Expert layer (7 files)
│   ├── collaboration/          # RAG + CAG (6 files)
│   ├── decision/               # CMO + Debate (4 files)
│   ├── models/                 # Data models (3 files)
│   └── utils/                  # Prompts, validators
│
├── mcp_server/
│   ├── clinical_diagnosis_server.py  # NEW MCP server
│   ├── llm_mcp_server.py            # Existing
│   ├── unified_server.py            # Existing
│   └── agents_mcp_server.py         # Existing
│
├── data/
│   ├── test/                   # Test data (100 samples)
│   ├── labeled/                # Annotations + splits
│   ├── models/                 # Trained models
│   └── knowledge_base/         # RAG vector DB + CAG cases
│
├── scripts/
│   ├── generate_test_data.py       # Synthetic data generator
│   ├── data_annotation/            # Streamlit GUI
│   ├── model_training/             # Training scripts
│   └── knowledge_base/             # RAG setup scripts
│
├── tests/                      # All test files
│   ├── test_rag.py
│   ├── test_cag.py
│   ├── test_preprocessing.py
│   ├── test_conflict_resolver.py
│   └── test_diagnosis_flow.py
│
└── main_clinical.py            # CLI entry point
```

## Interactive Menu

For easier navigation:

```bash
python main_clinical.py
```

This shows an interactive menu with all options.

## Next Steps

1. **Generate real labeled data**: Use `scripts/data_annotation/annotation_gui.py`
2. **Train models**: `python main_clinical.py train`
3. **Add medical literature**: Ingest PDFs to RAG system
4. **Build CAG cache**: Add diagnosed cases
5. **Integrate with MCP**: Run full orchestrator

## Notes

- Test data is synthetic for demonstration
- Expert models need training on real labeled data
- RAG works with sample documents (5 papers)
- CAG starts empty (add cases via API)
- LangGraph debate system fully functional
- CMO needs LLM API key for reasoning (optional)

## Troubleshooting

### "No test data found"
Run: `python main_clinical.py generate-data`

### "RAG system not available"
Run: `python main_clinical.py init-vectordb`

### "Expert models not trained"
This is expected. Use mock predictions or train with:
`python main_clinical.py train`

### Dependencies issues
Ensure all packages installed:
```bash
pip install -r requirements.txt
```

## Support

For issues or questions, check:
- Test files in `tests/` for usage examples
- Module docstrings for API details
- Sample reports in `data/test/`

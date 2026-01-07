# MDAgents - Multi-Agent System with Clinical Diagnosis

A sophisticated multi-agent system integrating:
1. **Multi-Agent Scientific Computing** - Data analysis, visualization, and scientific tools
2. **Clinical Diagnosis System** - Multi-omics diagnostic system with 4-layer architecture

## 🏥 Clinical Diagnosis System (NEW)

A complete oral multi-omics clinical diagnosis system with AI-powered decision support.

### Quick Start

```bash
# Check system status
python main_clinical.py status

# Generate training data
python main_clinical.py generate-data

# Train expert models
python main_clinical.py train

# Run demo diagnosis
python main_clinical.py demo

# Run tests
python main_clinical.py test
```

### Architecture (4 Layers)

1. **Perception Layer** - Multi-omics data preprocessing (microbiome, metabolome, proteome)
2. **Expert Layer** - 3 ML expert agents with SHAP interpretability
3. **Collaboration Layer** - RAG (medical literature) + CAG (case cache)
4. **Decision Layer** - LangGraph debate system + CMO coordinator

**See [`README_CLINICAL.md`](README_CLINICAL.md) for complete documentation.**

---

## 🔬 Scientific Computing Features

- **LLM Cascade**: Automatic fallback (DeepSeek → Gemini → GPT → Claude)
- **Data Analysis**: Pandas, NumPy, SciPy for data processing
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Scientific Illustrations**: Nanobanana agent for cover images
- **MCP Protocol**: All components communicate via Model Context Protocol

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│               MCP Orchestrator (Router)                  │
└──────┬──────────┬──────────┬──────────────────┬─────────┘
       │          │          │                  │
       ↓          ↓          ↓                  ↓
   [LLM MCP]  [Tools]   [Agents]    [Clinical Diagnosis]
   [Server]   [Server]  [Server]    [Server - NEW]
                                          │
                            ┌─────────────┴─────────────┐
                            ↓                           ↓
                    [Perception + Expert]    [Collaboration + Decision]
                    [ML Models + SHAP]       [RAG + CAG + LangGraph]
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd MDAgents

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Scientific Computing Mode

```bash
python main.py
```

### Clinical Diagnosis Mode

```bash
python main_clinical.py
```

Or use interactive menu:
```bash
python main_clinical.py
# Select from: status, generate-data, train, test, demo
```

## Project Structure

```
MDAgents/
├── clinical/              # Clinical diagnosis system (NEW)
│   ├── preprocessing/    # Perception layer
│   ├── experts/          # Expert agents (ML models)
│   ├── collaboration/    # RAG + CAG systems
│   └── decision/         # Debate + CMO coordinator
│
├── mcp_server/           # MCP servers
│   ├── llm_mcp_server.py
│   ├── unified_server.py
│   ├── agents_mcp_server.py
│   └── clinical_diagnosis_server.py  # NEW
│
├── core/                 # Core orchestration
│   └── mcp_orchestrator.py  # Routes between servers
│
├── tests/                # Test suites
└── main_clinical.py      # Clinical system CLI
```

## Key Features

### Clinical Diagnosis System

- ✅ Multi-omics integration (microbiome, metabolome, proteome)
- ✅ ML expert agents with threshold adjustment
- ✅ LangGraph-based debate mechanism (3 rounds)
- ✅ RAG for medical literature retrieval
- ✅ CAG for historical case matching
- ✅ Explainable AI (SHAP + reasoning chains)
- ✅ Markdown diagnostic reports

### Scientific Tools

- ✅ File operations (read, write, search)
- ✅ Data analysis (pandas, numpy, scipy)
- ✅ Plotting (matplotlib, plotly, seaborn)
- ✅ Image generation (nanobanana agent)
- ✅ Persistent conversation history

## Documentation

- [`README_CLINICAL.md`](README_CLINICAL.md) - Clinical system guide
- [`MCP_ARCHITECTURE.md`](MCP_ARCHITECTURE.md) - MCP architecture details
- [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) - Implementation details
- [`PROJECT_COMPLETION_REPORT.md`](PROJECT_COMPLETION_REPORT.md) - Statistics

## Testing

### Clinical System Tests
```bash
python main_clinical.py test
```

### Run specific test suites
```bash
pytest tests/test_rag.py -v
pytest tests/test_diagnosis_flow.py -v
```

## Contributing

This is a research project. For issues or questions, see the documentation files.

## License

See LICENSE file for details.

## Credits

Developed with Claude Sonnet 4.5

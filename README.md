# Multi-Agent Scientific Computing System

A sophisticated multi-agent system for scientific computing, data analysis, and visualization, powered by a cascading LLM architecture and specialized expert agents.

## Features

- **LLM Cascade**: Automatic fallback across 4 providers (DeepSeek → Gemini → GPT-5 → Claude)
- **Data Analysis**: Pandas and NumPy for powerful data processing
- **Visualization**: Matplotlib, Plotly, and Seaborn for publication-quality plots
- **Scientific Illustrations**: Nanobanana agent for creating cover images and graphical abstracts
- **Persistent Memory**: Redis-based conversation history
- **Interactive CLI**: Rich console interface with streaming responses

## Prerequisites

- Python 3.14
- API keys for:
  - DeepSeek
  - Google Gemini
  - OpenAI
  - Anthropic Claude

## Quick Start

### 1. Set Up API Keys

Edit `.env` file and add your API keys:

```bash
# Replace with your actual API keys
DEEPSEEK_API_KEY=sk-your-deepseek-key
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

### 2. Install Dependencies

Dependencies are already installed in your virtual environment:

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate

# Verify installation
python -c "import pandas, numpy, matplotlib; print('✓ All dependencies installed')"
```

### 3. Run the System

```bash
python main.py
```

## Usage Examples

### Data Analysis

```
You: Load sales_data.csv and show me the first 10 rows
Assistant: [Loads data and displays preview]

You: Analyze the data and compute statistics for the revenue column
Assistant: [Provides comprehensive statistics]

You: Create a bar plot of monthly revenue and save it to ./plots/revenue.png
Assistant: [Creates and saves plot]
```

### Scientific Illustrations

```
You: Create a cover image for my neural networks research paper
Assistant: [Generates cover image using nanobanana]

You: Generate a graphical abstract showing the deep learning pipeline
Assistant: [Creates graphical abstract]
```

### File Operations

```
You: Search for all CSV files in the data directory
Assistant: [Lists all CSV files]

You: Read analysis_results.txt
Assistant: [Displays file contents]
```

## CLI Commands

- `/help` - Show help message
- `/clear` - Clear conversation history
- `/new` - Start new session
- `/stats` - Show usage statistics
- `/exit` - Exit application

## Architecture

```
User (Console) → main.py
                    ↓
              [Orchestrator] ← [Conversation Memory (Redis)]
                    ↓
         [LLM Client with Cascade]
         (DeepSeek → Gemini → GPT-5 → Claude)
                    ↓
         ┌──────────┴──────────┐
         ↓                     ↓
    [MCP Tools]          [nanobanana Agent]
    - File ops           - Image generation
    - Data analysis      - Cover figures
    - Plotting           - Illustrations
```

## Project Structure

```
MultiAgents/
├── config/          # Configuration management
├── core/            # Core components (LLM, orchestrator)
├── agents/          # Expert agents (nanobanana)
├── mcp_server/      # MCP server for tools
├── tools/           # Tool implementations
├── memory/          # Redis memory management
├── utils/           # Utilities (logging, prompts)
├── main.py          # Interactive CLI entry point
├── .env             # API keys and configuration
└── requirements.txt # Dependencies
```

## Configuration

Edit `.env` to customize:

```bash
# LLM Cascade Order
CASCADE_ORDER=deepseek,gemini,gpt5,claude

# Redis Settings
REDIS_USE_FAKEREDIS=true  # Set to false for real Redis

# Application Settings
LOG_LEVEL=INFO
MAX_CONVERSATION_LENGTH=50
OUTPUT_DIR=./output
PLOTS_DIR=./plots
```

## Troubleshooting

### API Key Errors

Make sure all API keys in `.env` are valid and properly formatted.

### Import Errors

Ensure virtual environment is activated:
```bash
source .venv/bin/activate
```

### Redis Connection Issues

The system uses FakeRedis by default (no Redis server needed). To use real Redis:
1. Install and start Redis server
2. Set `REDIS_USE_FAKEREDIS=false` in `.env`

### Plot Display Issues

Plots are saved to `./plots/` directory. Check there for generated visualizations.

## Development

### Running MCP Server Separately

```bash
python mcp_server/unified_server.py
```

### Testing Individual Components

```python
# Test LLM cascade
python -c "from core.llm_client import *; print('✓ LLM client OK')"

# Test data tools
python -c "from tools.data_tools import *; print('✓ Data tools OK')"

# Test plotting
python -c "from tools.plot_tools import *; print('✓ Plot tools OK')"
```

### Debug Mode

Set `LOG_LEVEL=DEBUG` in `.env` for detailed logging.

## Capabilities

### Data Formats Supported
- CSV files
- Excel files (.xlsx, .xls)
- JSON data
- Plain text files

### Plot Types Available
- Line plots
- Scatter plots
- Bar charts
- Histograms
- Box plots
- Violin plots
- Heatmaps
- Interactive plots (Plotly)

### Image Generation
- Article cover images
- Graphical abstracts
- Scientific illustrations
- Technical diagrams

## Next Steps

After getting started:

1. Try analyzing your own datasets
2. Create custom visualizations
3. Generate scientific illustrations for papers
4. Explore multi-turn conversations with context
5. Experiment with different plot types

## Contributing

See `ReadMe_Claude.md` for implementation details and architecture notes.

## License

MIT License - see LICENSE file for details.

## Support

For issues or questions:
- Check the `/help` command in the CLI
- Review error messages for troubleshooting hints
- Ensure all API keys are properly configured

---

Built with ❤️ for scientific computing and research.

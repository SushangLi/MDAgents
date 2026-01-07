# MCP-Native Architecture Guide

## 🎯 **True MCP Implementation**

This system now uses **100% MCP protocol** for all communications. Everything is an MCP server:

```
┌─────────────────────────────────────┐
│   MCP Orchestrator (Router)         │
│   - Thin MCP client                 │
│   - Routes messages between servers │
└──────────┬──────────────────────────┘
           │ (MCP Protocol)
    ┌──────┴────────┬─────────┬─────────┐
    ↓               ↓         ↓         ↓
[LLM MCP        [Tools    [Agents   [Future
 Server]         MCP       MCP       MCP
                Server]   Server]   Servers]
```

## 📁 **New Files Created**

### **MCP Servers:**
1. **`mcp_server/llm_mcp_server.py`** - LLM cascade as MCP server
2. **`mcp_server/agents_mcp_server.py`** - Expert agents as MCP server
3. **`mcp_server/unified_server.py`** - Tools as MCP server (already existed)

### **MCP Orchestrator:**
4. **`core/mcp_orchestrator.py`** - Pure MCP router/client

### **New Entry Point:**
5. **`main_mcp.py`** - MCP-native CLI

## 🚀 **How to Run**

### **Start the MCP System:**

```bash
python main_mcp.py
```

That's it! The orchestrator will automatically:
1. Start LLM MCP server
2. Start Tools MCP server
3. Start Agents MCP server
4. Connect to all servers via MCP protocol
5. Route messages between them

## 🔄 **How It Works**

### **Message Flow:**

```
1. User: "analyze iris data in ./data"
   ↓
2. Orchestrator → LLM MCP Server
   generate_completion(messages, tools)
   ↓
3. LLM returns: tool_calls = [
     {name: "list_directory", args: {dir_path: "./data"}},
     {name: "load_csv", args: {file_path: "./data/iris.csv"}},
     {name: "analyze_dataframe", args: {df_id: "..."}}
   ]
   ↓
4. Orchestrator routes each tool call:
   - list_directory → Tools MCP Server
   - load_csv → Tools MCP Server
   - analyze_dataframe → Tools MCP Server
   ↓
5. Orchestrator sends results back to LLM MCP Server
   ↓
6. LLM generates final response
   ↓
7. User receives answer
```

### **Key Differences from Old Architecture:**

| Old (Direct Calls) | New (MCP-Native) |
|-------------------|------------------|
| `import file_tools` | MCP client → Tools MCP Server |
| `file_tools.read_file()` | `tools_session.call_tool("read_file")` |
| Direct Python function | MCP protocol message |
| Single process | Multiple MCP server processes |
| Orchestrator decides flow | **LLM decides flow via tool calls** |

## 🛠️ **MCP Server Details**

### **1. LLM MCP Server**

**File:** `mcp_server/llm_mcp_server.py`

**Exposes:**
- `generate_completion(messages, tools)` - Generate LLM response with tool calling
- `get_llm_stats()` - Get usage statistics
- `list_available_providers()` - List cascade providers

**How it works:**
- Wraps the LLM cascade (DeepSeek → Gemini → GPT-5 → Claude)
- Exposes as MCP tool
- Returns responses with tool_calls that orchestrator can route

**Start standalone:**
```bash
python mcp_server/llm_mcp_server.py
```

### **2. Tools MCP Server**

**File:** `mcp_server/unified_server.py`

**Exposes:**
- File operations: `read_file`, `write_file`, `list_directory`, `search_files`
- Data analysis: `load_csv`, `analyze_dataframe`, `compute_statistics`
- Visualization: `create_matplotlib_plot`, `create_seaborn_plot`, `save_figure`

**Start standalone:**
```bash
python mcp_server/unified_server.py
```

### **3. Agents MCP Server**

**File:** `mcp_server/agents_mcp_server.py`

**Exposes:**
- `create_cover_image(prompt, style)` - Generate article cover
- `create_abstract_figure(description, style)` - Create graphical abstract
- `process_image(image_path, instruction)` - Process existing image
- `list_available_agents()` - List available agents

**Start standalone:**
```bash
python mcp_server/agents_mcp_server.py
```

## 🎮 **CLI Commands**

Run `main_mcp.py` and use:

- `/help` - Show help
- `/clear` - Clear conversation history
- `/new` - Start new session
- `/stats` - Show MCP server connection status
- `/mcp` - Show detailed MCP architecture info
- `/exit` - Exit (closes all MCP connections)

## 🔍 **LLM-Driven Flow**

**The key difference:** The LLM now drives the entire workflow!

### **Example:**

```
User: "analyze iris data and create a scatter plot"

LLM MCP Server receives messages and returns:
{
  "content": "I'll analyze the iris data for you.",
  "tool_calls": [
    {
      "name": "search_files",
      "arguments": {"root_dir": "./data", "pattern": "*.csv"}
    }
  ]
}

Orchestrator routes to Tools MCP Server →
Returns: ["./data/iris.csv"]

Orchestrator sends result back to LLM MCP Server →

LLM returns:
{
  "tool_calls": [
    {
      "name": "load_csv",
      "arguments": {"file_path": "./data/iris.csv"}
    }
  ]
}

... and so on until LLM is satisfied and returns final response.
```

## 📊 **Benefits of MCP Architecture**

✅ **True Separation:** Each component is independent MCP server
✅ **LLM-Driven:** LLM decides what to do next, not hardcoded logic
✅ **Scalable:** Can run MCP servers on different machines
✅ **Extensible:** Add new MCP servers without changing orchestrator
✅ **Standard Protocol:** Uses industry-standard MCP
✅ **Distributable:** MCP servers can be anywhere
✅ **Tool Discovery:** Orchestrator queries servers for available tools
✅ **Isolation:** Each server runs in its own process

## 🔧 **Debugging**

### **Check MCP Server Status:**

Use `/stats` command in CLI to see connection status.

### **Run Servers Manually:**

For debugging, run each server separately:

```bash
# Terminal 1
python mcp_server/llm_mcp_server.py

# Terminal 2
python mcp_server/unified_server.py

# Terminal 3
python mcp_server/agents_mcp_server.py

# Terminal 4
python main_mcp.py
```

### **Check Logs:**

Set `LOG_LEVEL=DEBUG` in `.env` for detailed MCP communication logs.

## 🎯 **Next Steps**

Now that you have true MCP architecture:

1. ✅ **Test the system:** Run `python main_mcp.py`
2. ✅ **Try data analysis:** "analyze iris data in ./data"
3. ✅ **Try visualization:** "create a scatter plot"
4. ✅ **Try image generation:** "create a cover image"
5. ✅ **Explore:** Use `/mcp` to see architecture details

## 🆚 **Old vs New**

### **Old (`main.py`):**
- Direct Python imports
- Orchestrator calls functions directly
- Single process
- Fast but not MCP-compliant

### **New (`main_mcp.py`):**
- MCP protocol everywhere
- LLM drives the flow
- Multiple server processes
- True MCP implementation

Both are available - use `main_mcp.py` for true MCP architecture!

## 🎉 **You Now Have:**

✅ **100% MCP-native architecture**
✅ **LLM-driven workflow** (LLM decides what to do)
✅ **Distributed capable** (servers can run anywhere)
✅ **Standard protocol** (industry-standard MCP)
✅ **Fully extensible** (add MCP servers easily)

**This is what you asked for - everything via MCP! 🚀**

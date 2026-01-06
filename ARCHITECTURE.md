# Architecture Overview

## Current Implementation (MVP)

### Communication Flow

```
User Input
    ↓
[main.py - CLI]
    ↓
[Orchestrator]
    ↓
[LLM Client (Cascade)]
    ↓
┌─────────┴─────────┐
↓                   ↓
[Direct Imports]    [Agent.handle()]
    ↓                   ↓
[tools/file_tools.py]   [nanobanana_agent.py]
[tools/data_tools.py]
[tools/plot_tools.py]
```

### Current State

**✅ Implemented:**
- Direct function calls to tools (NOT using MCP protocol)
- Tool imports via Python `import` statements
- LLM cascade with proper tool calling
- Conversation memory via Redis

**❌ Not Yet Implemented:**
- MCP client/server communication
- MCP protocol for tool calling
- Separate MCP server process

### Why Direct Calls for MVP?

1. **Simpler**: No need to manage MCP server process
2. **Faster**: Direct Python calls are faster than protocol overhead
3. **Easier to debug**: Standard Python stack traces
4. **Good enough**: Works well for single-machine deployment

## Path to Full MCP Implementation

### Phase 1: Current MVP ✅
- Direct tool calls
- Everything in one process
- Good for development and testing

### Phase 2: MCP Integration (Future)

**Steps to implement:**

1. **Run MCP Server**
   ```bash
   python mcp_server/unified_server.py
   ```

2. **Implement MCP Client**
   ```python
   # In orchestrator
   from mcp import Client

   class Orchestrator:
       async def init_mcp_client(self):
           self.mcp_client = await Client.connect_stdio(
               command="python",
               args=["mcp_server/unified_server.py"]
           )
           self.mcp_tools = await self.mcp_client.list_tools()
   ```

3. **Replace Direct Calls**
   ```python
   # Replace:
   from tools import file_tools
   result = file_tools.read_file(**arguments)

   # With:
   result = await self.mcp_client.call_tool(tool_name, arguments)
   ```

**Benefits of Full MCP:**
- ✅ Tools run in separate process (isolation)
- ✅ Can distribute tools across machines
- ✅ Standard protocol for tool discovery
- ✅ Better for production deployments
- ✅ Can integrate 3rd-party MCP servers

## Tool Calling Flow

### Current Flow (Direct)
```
1. LLM returns tool_calls
2. Orchestrator parses tool_calls
3. Orchestrator imports tool module
4. Orchestrator calls function directly
5. Result returned to LLM
```

### Future Flow (MCP)
```
1. LLM returns tool_calls
2. Orchestrator parses tool_calls
3. Orchestrator sends MCP request
4. MCP server executes tool
5. MCP server returns result
6. Result forwarded to LLM
```

## Message Format

### Tool Call Format (OpenAI/DeepSeek)
```python
{
    "role": "assistant",
    "content": "I'll help with that...",
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": "{\"path\": \"./data/iris.csv\"}"
            }
        }
    ]
}
```

### Tool Result Format
```python
{
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": "sepal_length,sepal_width,petal_length..."
}
```

## LLM Provider Compatibility

### DeepSeek (OpenAI-compatible)
- Requires `type: "function"` in tool_calls
- Arguments must be JSON string
- Tool results must have `tool_call_id`

### Gemini
- Different format - converted by adapter
- Uses `function_call` instead of `tool_calls`
- Arguments can be dict

### OpenAI GPT-4/GPT-5
- Standard OpenAI format
- Same as DeepSeek

### Claude
- Uses `tool_use` blocks
- Different format - converted by adapter

## Current Limitations

1. **No MCP Protocol**: Tools called directly, not via MCP
2. **Single Process**: Everything runs in one Python process
3. **No Tool Discovery**: Tools hardcoded in orchestrator
4. **No Remote Tools**: Can't use tools on other machines

## Recommended Next Steps

For production deployment:

1. **Implement MCP Client**: Connect to MCP server
2. **Run MCP Server Separately**: Use process manager (systemd, supervisor)
3. **Tool Discovery**: Query MCP server for available tools
4. **Error Handling**: Add retry logic for MCP communication
5. **Monitoring**: Add logging for MCP requests/responses
6. **Load Balancing**: Run multiple MCP servers if needed

## Benefits of Current Design

Despite not using MCP protocol yet:

✅ **MCP-Ready**: Tools already use MCP-compatible schemas
✅ **Easy Migration**: Can switch to MCP client with minimal changes
✅ **Works Well**: Direct calls are reliable and fast
✅ **Simpler Deployment**: No need to manage MCP server process
✅ **Good for Development**: Easier to debug and test

The current design is a solid foundation that can easily evolve to use full MCP when needed!

"""
MCP-Native Orchestrator - Routes messages between MCP servers.

This orchestrator is a thin MCP client that:
1. Receives user input
2. Sends to LLM MCP server
3. Routes tool calls to appropriate MCP servers (tools/agents)
4. Returns results back to LLM
5. Delivers final response to user

Everything communicates via MCP protocol over stdio.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.conversation import ConversationManager
from utils.logging import get_logger
from utils.prompts import get_system_prompt

logger = get_logger(__name__)


class MCPOrchestrator:
    """
    MCP-native orchestrator that routes between MCP servers.

    Uses stdio transport with native MCP Python SDK servers.
    """

    def __init__(
        self,
        conversation_manager: ConversationManager,
    ):
        """
        Initialize MCP orchestrator.

        Args:
            conversation_manager: Conversation manager for history
        """
        self.conversation = conversation_manager
        self.llm_session: Optional[ClientSession] = None
        self.tools_session: Optional[ClientSession] = None
        self.agents_session: Optional[ClientSession] = None
        self.clinical_session: Optional[ClientSession] = None

        # Context managers to keep connections alive
        self.llm_stdio_context = None
        self.llm_session_context = None

        self.tools_stdio_context = None
        self.tools_session_context = None

        self.agents_stdio_context = None
        self.agents_session_context = None

        self.clinical_stdio_context = None
        self.clinical_session_context = None

    async def initialize(self):
        """Initialize connections to all MCP servers."""
        logger.info("Initializing MCP connections...")

        # Start LLM MCP server
        llm_params = StdioServerParameters(
            command="python",
            args=["mcp_server/llm_mcp_server.py"],
            env=None,
        )

        # Start Tools MCP server
        tools_params = StdioServerParameters(
            command="python",
            args=["mcp_server/unified_server.py"],
            env=None,
        )

        # Start Agents MCP server
        agents_params = StdioServerParameters(
            command="python",
            args=["mcp_server/agents_mcp_server.py"],
            env=None,
        )

        # Start Clinical Diagnosis MCP server
        clinical_params = StdioServerParameters(
            command="python",
            args=["mcp_server/clinical_diagnosis_server.py"],
            env=None,
        )

        try:
            # Connect to LLM server
            logger.info("Connecting to LLM MCP server...")
            self.llm_stdio_context = stdio_client(llm_params)
            read, write = await self.llm_stdio_context.__aenter__()

            # Create and enter ClientSession context
            session = ClientSession(read, write)
            self.llm_session_context = session
            self.llm_session = await session.__aenter__()
            await self.llm_session.initialize()
            logger.info("✓ Connected to LLM MCP server")

            # Connect to Tools server
            logger.info("Connecting to Tools MCP server...")
            self.tools_stdio_context = stdio_client(tools_params)
            read, write = await self.tools_stdio_context.__aenter__()

            session = ClientSession(read, write)
            self.tools_session_context = session
            self.tools_session = await session.__aenter__()
            await self.tools_session.initialize()
            logger.info("✓ Connected to Tools MCP server")

            # Connect to Agents server
            logger.info("Connecting to Agents MCP server...")
            self.agents_stdio_context = stdio_client(agents_params)
            read, write = await self.agents_stdio_context.__aenter__()

            session = ClientSession(read, write)
            self.agents_session_context = session
            self.agents_session = await session.__aenter__()
            await self.agents_session.initialize()
            logger.info("✓ Connected to Agents MCP server")

            # Connect to Clinical Diagnosis server
            logger.info("Connecting to Clinical Diagnosis MCP server...")
            self.clinical_stdio_context = stdio_client(clinical_params)
            read, write = await self.clinical_stdio_context.__aenter__()

            session = ClientSession(read, write)
            self.clinical_session_context = session
            self.clinical_session = await session.__aenter__()
            await self.clinical_session.initialize()
            logger.info("✓ Connected to Clinical Diagnosis MCP server")

            logger.info("✓ All MCP connections established")

        except Exception as e:
            logger.error(f"✗ Failed to initialize MCP connections: {e}")
            raise

    async def process_message(
        self,
        user_message: str,
        session_id: str,
    ) -> str:
        """
        Process user message using MCP servers.

        Flow:
        1. Add user message to history
        2. Get conversation context
        3. Call LLM MCP server
        4. If LLM returns tool calls, route to appropriate MCP servers
        5. Send results back to LLM
        6. Return final response

        Args:
            user_message: User's message
            session_id: Session identifier

        Returns:
            Assistant's response
        """
        logger.info(f"Processing message for session {session_id}")

        # Add user message to history
        self.conversation.add_user_message(session_id, user_message)

        # Get conversation context
        messages = self.conversation.get_full_context(
            session_id=session_id,
            system_prompt=get_system_prompt(),
        )

        # Ensure all tool messages have the required 'type' field
        messages = self._sanitize_messages(messages)

        # Get available tools from all MCP servers
        tools = await self._get_all_tools()

        # Call LLM via MCP
        max_iterations = 100  # Allow extensive multi-step workflows
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Call LLM MCP server
            llm_response = await self._call_llm(messages, tools)

            if llm_response.get("error"):
                error_msg = f"LLM error: {llm_response['error']}"
                logger.error(error_msg)
                return error_msg

            content = llm_response.get("content", "")
            tool_calls = llm_response.get("tool_calls")

            # If no tool calls, we're done
            if not tool_calls:
                # Save final response
                self.conversation.add_assistant_message(
                    session_id=session_id,
                    content=content,
                    metadata={
                        "provider": llm_response.get("provider"),
                        "model": llm_response.get("model"),
                    },
                )
                return content

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": self._format_tool_calls(tool_calls),
            })

            # Save to history
            self.conversation.add_assistant_message(
                session_id=session_id,
                content=content or "",
                tool_calls=tool_calls,
                metadata={
                    "provider": llm_response.get("provider"),
                    "model": llm_response.get("model"),
                },
            )

            # Execute tools via MCP
            tool_results = await self._execute_tools(tool_calls, session_id)

            # Add tool results to messages
            for result in tool_results:
                messages.append({
                    "role": "tool",
                    "type": "function",
                    "tool_call_id": result["id"],
                    "content": result["content"],
                })

        # Max iterations reached
        return "Maximum iterations reached. Please try a simpler request."

    async def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Call LLM via MCP server."""
        try:
            result = await self.llm_session.call_tool(
                "generate_completion",
                arguments={
                    "messages": messages,
                    "tools": tools,
                    "temperature": 0.7,
                },
            )
            # Parse JSON response
            content_str = result.content[0].text
            return json.loads(content_str)
        except Exception as e:
            logger.error(f"LLM MCP call failed: {e}")
            return {"error": str(e)}

    async def _get_all_tools(self) -> List[Dict[str, Any]]:
        """Get tool schemas from all MCP servers."""
        tools = []

        try:
            # Get tools from Tools server
            tools_list = await self.tools_session.list_tools()
            tools.extend([{
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            } for tool in tools_list.tools])

            # Get tools from Agents server
            agents_list = await self.agents_session.list_tools()
            tools.extend([{
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            } for tool in agents_list.tools])

            # Get tools from Clinical Diagnosis server
            clinical_list = await self.clinical_session.list_tools()
            tools.extend([{
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            } for tool in clinical_list.tools])

        except Exception as e:
            logger.error(f"Failed to get tools: {e}")

        logger.debug(f"Loaded {len(tools)} tools from MCP servers")
        return tools

    async def _execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """Execute tool calls by routing to appropriate MCP servers."""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_id = tool_call.get("id")
            arguments = tool_call.get("arguments", {})

            # Parse arguments if string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            logger.info(f"Executing tool via MCP: {tool_name}")

            try:
                # Route to appropriate MCP server
                if tool_name in ["create_cover_image", "create_abstract_figure", "process_image", "list_available_agents"]:
                    # Call Agents MCP server
                    result = await self.agents_session.call_tool(tool_name, arguments)
                elif tool_name in ["diagnose_patient", "preprocess_omics_data", "query_knowledge_base",
                                   "get_expert_explanations", "generate_diagnostic_report", "get_system_status"]:
                    # Call Clinical Diagnosis MCP server
                    result = await self.clinical_session.call_tool(tool_name, arguments)
                else:
                    # Call Tools MCP server
                    result = await self.tools_session.call_tool(tool_name, arguments)

                # Extract content from result
                content = result.content[0].text

                # Save to history
                self.conversation.add_tool_message(
                    session_id=session_id,
                    tool_name=tool_name,
                    tool_result=content,
                )

                results.append({
                    "id": tool_id,
                    "content": content,
                })

                logger.info(f"✓ Tool {tool_name} completed via MCP")

            except Exception as e:
                logger.error(f"✗ Tool {tool_name} failed: {e}")
                results.append({
                    "id": tool_id,
                    "content": json.dumps({"error": str(e)}),
                })

        return results

    def _format_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tool calls for LLM."""
        formatted = []
        for tc in tool_calls:
            formatted.append({
                "id": tc.get("id"),
                "type": "function",
                "function": {
                    "name": tc.get("name"),
                    "arguments": tc.get("arguments") if isinstance(tc.get("arguments"), str)
                                else json.dumps(tc.get("arguments", {}))
                }
            })
        return formatted

    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure all messages have required fields for LLM compatibility.

        Fixes legacy messages that may be missing the 'type' field.

        Args:
            messages: List of message dictionaries

        Returns:
            Sanitized message list
        """
        sanitized = []
        for msg in messages:
            # Create a copy to avoid modifying the original
            sanitized_msg = msg.copy()

            # Tool messages must have 'type' field for DeepSeek compatibility
            if msg.get("role") == "tool" and "type" not in msg:
                sanitized_msg["type"] = "function"

            sanitized.append(sanitized_msg)

        return sanitized

    async def close(self):
        """Close all MCP connections."""
        try:
            # Exit ClientSession contexts first
            if self.llm_session_context:
                await self.llm_session_context.__aexit__(None, None, None)
            if self.tools_session_context:
                await self.tools_session_context.__aexit__(None, None, None)
            if self.agents_session_context:
                await self.agents_session_context.__aexit__(None, None, None)

            # Then exit stdio contexts
            if self.llm_stdio_context:
                await self.llm_stdio_context.__aexit__(None, None, None)
            if self.tools_stdio_context:
                await self.tools_stdio_context.__aexit__(None, None, None)
            if self.agents_stdio_context:
                await self.agents_stdio_context.__aexit__(None, None, None)

            logger.info("MCP connections closed")
        except Exception as e:
            logger.error(f"Error closing MCP connections: {e}")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "mcp_servers": {
                "llm": "connected" if self.llm_session else "disconnected",
                "tools": "connected" if self.tools_session else "disconnected",
                "agents": "connected" if self.agents_session else "disconnected",
            }
        }

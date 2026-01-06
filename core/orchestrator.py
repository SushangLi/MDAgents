"""
Main orchestrator that coordinates LLM, tools, agents, and memory.

The orchestrator is the central controller that:
- Manages conversation flow
- Routes tool calls to MCP or agents
- Maintains context and history
- Handles multi-turn interactions
"""

import json
from typing import Any, Dict, List, Optional

from core.conversation import ConversationManager
from core.llm_client import CascadeLLMClient
from agents.nanobanana_agent import NanobananaAgent
from utils.logging import get_logger
from utils.prompts import get_system_prompt

logger = get_logger(__name__)


class Orchestrator:
    """
    Main orchestrator for the multi-agent system.

    Coordinates between LLM, MCP tools, expert agents, and memory.
    """

    def __init__(
        self,
        llm_client: CascadeLLMClient,
        conversation_manager: ConversationManager,
        agents: Dict[str, Any],
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            llm_client: Cascade LLM client
            conversation_manager: Conversation manager
            agents: Dict of expert agents (e.g., {"nanobanana": NanobananaAgent()})
            mcp_tools: List of MCP tool schemas
        """
        self.llm = llm_client
        self.conversation = conversation_manager
        self.agents = agents
        self.mcp_tools = mcp_tools or []
        self.tool_schemas = self._build_tool_schemas()

        logger.info(f"Orchestrator initialized with {len(self.tool_schemas)} tools")

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Build complete tool schemas from MCP and agents.

        Returns:
            List of tool schemas for LLM
        """
        schemas = []

        # Add MCP tool schemas
        schemas.extend(self.mcp_tools)

        # Add nanobanana agent tools
        agent_tools = [
            {
                "name": "nanobanana_create_cover_image",
                "description": "Generate a scientific article cover image or illustration using AI",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Description of the image to generate",
                        },
                        "style": {
                            "type": "string",
                            "description": "Visual style: 'scientific', 'abstract', 'technical'",
                            "default": "scientific",
                        },
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "nanobanana_create_abstract_figure",
                "description": "Create a graphical abstract for a scientific paper",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Description of the abstract content and key points",
                        },
                        "style": {
                            "type": "string",
                            "description": "Visual style preference",
                            "default": "clean",
                        },
                    },
                    "required": ["description"],
                },
            },
        ]
        schemas.extend(agent_tools)

        logger.debug(f"Built {len(schemas)} tool schemas")
        return schemas

    def _format_messages_for_llm(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format messages to ensure compatibility with LLM providers.

        Ensures tool_calls have proper structure with 'type' field.

        Args:
            messages: Raw messages from conversation history

        Returns:
            Formatted messages compatible with LLM APIs
        """
        formatted = []

        for msg in messages:
            formatted_msg = {
                "role": msg["role"],
                "content": msg.get("content", ""),
            }

            # Format tool_calls if present (for assistant messages)
            if "tool_calls" in msg and msg["tool_calls"]:
                formatted_tool_calls = []
                for tc in msg["tool_calls"]:
                    # Ensure proper OpenAI/DeepSeek format
                    if isinstance(tc, dict):
                        formatted_tc = {
                            "id": tc.get("id", f"call_{len(formatted_tool_calls)}"),
                            "type": "function",
                            "function": {
                                "name": tc.get("name") or tc.get("function", {}).get("name"),
                                "arguments": tc.get("arguments") if isinstance(tc.get("arguments"), str)
                                           else json.dumps(tc.get("arguments", {}))
                            }
                        }
                        formatted_tool_calls.append(formatted_tc)

                if formatted_tool_calls:
                    formatted_msg["tool_calls"] = formatted_tool_calls

            # Ensure tool_call_id for tool messages
            if msg["role"] == "tool" and "tool_call_id" not in formatted_msg:
                # Try to get it from metadata or generate one
                formatted_msg["tool_call_id"] = msg.get("metadata", {}).get("tool_call_id", "call_unknown")
            elif msg["role"] == "tool" and "tool_call_id" in msg:
                formatted_msg["tool_call_id"] = msg["tool_call_id"]

            formatted.append(formatted_msg)

        return formatted

    async def process_message(
        self,
        user_message: str,
        session_id: str,
    ) -> str:
        """
        Process user message and generate response.

        Args:
            user_message: User's message
            session_id: Session identifier

        Returns:
            Assistant's response
        """
        logger.info(f"Processing message for session {session_id}")

        # Add user message to history
        self.conversation.add_user_message(session_id, user_message)

        # Get conversation context with system prompt
        messages = self.conversation.get_full_context(
            session_id=session_id,
            system_prompt=get_system_prompt(),
        )

        # Format all messages to ensure compatibility with LLM providers
        messages = self._format_messages_for_llm(messages)

        # Initial LLM call
        response = await self.llm.complete(
            messages=messages,
            tools=self.tool_schemas,
        )

        # Handle tool calling loop
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while response.has_tool_calls() and iteration < max_iterations:
            iteration += 1
            logger.debug(f"Tool calling iteration {iteration}")

            # Format tool_calls properly for different LLM providers
            formatted_tool_calls = []
            for tc in response.tool_calls:
                # OpenAI/DeepSeek format
                formatted_tc = {
                    "id": tc.get("id", f"call_{iteration}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name"),
                        "arguments": tc.get("arguments") if isinstance(tc.get("arguments"), str) else json.dumps(tc.get("arguments", {}))
                    }
                }
                formatted_tool_calls.append(formatted_tc)

            # Add assistant message with tool calls to messages (for LLM context)
            messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": formatted_tool_calls,
            })

            # Also save to conversation history
            self.conversation.add_assistant_message(
                session_id=session_id,
                content=response.content or "",
                tool_calls=response.tool_calls,  # Save original format
                metadata={"provider": response.provider, "model": response.model},
            )

            # Execute tools
            tool_results = await self._execute_tools(response.tool_calls, session_id)

            # Add tool results to messages for LLM
            for tool_result in tool_results:
                result_content = tool_result["result"]
                if not isinstance(result_content, str):
                    result_content = json.dumps(result_content)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result.get("id"),
                    "content": result_content,
                })

            # Get next response from LLM
            response = await self.llm.complete(
                messages=messages,
                tools=self.tool_schemas,
            )

        # Add final assistant message
        self.conversation.add_assistant_message(
            session_id=session_id,
            content=response.content,
            metadata={"provider": response.provider, "model": response.model},
        )

        logger.info(f"✓ Completed message processing (used {response.provider})")
        return response.content

    async def _execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls by routing to MCP or agents.

        Args:
            tool_calls: List of tool calls from LLM
            session_id: Session identifier

        Returns:
            List of tool results
        """
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_id = tool_call.get("id")

            # Parse arguments
            if isinstance(tool_call.get("arguments"), str):
                try:
                    arguments = json.loads(tool_call["arguments"])
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = tool_call.get("arguments", {})

            logger.info(f"Executing tool: {tool_name}")

            try:
                # Route to appropriate handler
                if tool_name.startswith("nanobanana_"):
                    # Delegate to nanobanana agent
                    result = await self._call_agent("nanobanana", tool_call)
                else:
                    # Call MCP tool (placeholder - would use actual MCP client)
                    result = await self._call_mcp_tool(tool_name, arguments)

                # Add tool result to conversation
                self.conversation.add_tool_message(
                    session_id=session_id,
                    tool_name=tool_name,
                    tool_result=result,
                )

                results.append({
                    "id": tool_id,
                    "tool_name": tool_name,
                    "result": result,
                })

                logger.info(f"✓ Tool {tool_name} completed successfully")

            except Exception as e:
                logger.error(f"✗ Tool {tool_name} failed: {e}")
                error_result = {"error": str(e), "success": False}
                results.append({
                    "id": tool_id,
                    "tool_name": tool_name,
                    "result": error_result,
                })

        return results

    async def _call_agent(
        self,
        agent_name: str,
        tool_call: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call expert agent.

        Args:
            agent_name: Name of the agent
            tool_call: Tool call dict

        Returns:
            Agent result
        """
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")

        agent = self.agents[agent_name]
        result = await agent.handle(tool_call)
        return result

    async def _call_mcp_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """
        Call MCP tool.

        Args:
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result

        Note: This is a placeholder. In a real implementation,
        you would use an MCP client to call the actual server.
        """
        # Import tools dynamically
        if tool_name == "read_file":
            from tools import file_tools
            return file_tools.read_file(**arguments)
        elif tool_name == "write_file":
            from tools import file_tools
            return file_tools.write_file(**arguments)
        elif tool_name == "list_directory":
            from tools import file_tools
            return file_tools.list_directory(**arguments)
        elif tool_name == "search_files":
            from tools import file_tools
            return file_tools.search_files(**arguments)
        elif tool_name == "load_csv":
            from tools import data_tools
            return data_tools.load_csv(**arguments)
        elif tool_name == "load_excel":
            from tools import data_tools
            return data_tools.load_excel(**arguments)
        elif tool_name == "analyze_dataframe":
            from tools import data_tools
            return data_tools.analyze_dataframe(**arguments)
        elif tool_name == "get_dataframe_head":
            from tools import data_tools
            return data_tools.get_dataframe_head(**arguments)
        elif tool_name == "query_dataframe":
            from tools import data_tools
            return data_tools.query_dataframe(**arguments)
        elif tool_name == "compute_statistics":
            from tools import data_tools
            return data_tools.compute_statistics(**arguments)
        elif tool_name == "create_matplotlib_plot":
            from tools import plot_tools
            return plot_tools.create_matplotlib_plot(**arguments)
        elif tool_name == "create_plotly_plot":
            from tools import plot_tools
            return plot_tools.create_plotly_plot(**arguments)
        elif tool_name == "create_seaborn_plot":
            from tools import plot_tools
            return plot_tools.create_seaborn_plot(**arguments)
        elif tool_name == "save_figure":
            from tools import plot_tools
            return plot_tools.save_figure(**arguments)
        else:
            raise ValueError(f"Unknown MCP tool: {tool_name}")

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dict with usage stats
        """
        return {
            "llm_usage": self.llm.get_usage_stats(),
            "tools_available": len(self.tool_schemas),
            "agents_available": list(self.agents.keys()),
        }

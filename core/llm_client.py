"""
LLM Client with cascade fallback across multiple providers.

Supports:
- DeepSeek (OpenAI-compatible API)
- Google Gemini
- OpenAI GPT-4/GPT-5
- Anthropic Claude

Features:
- Automatic fallback on errors (rate limits, timeouts, service errors)
- Tool/function calling normalization
- Retry with exponential backoff
- Token usage tracking
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import anthropic
from google import genai
from google.genai import types
import openai

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response."""

    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    model: str = ""
    provider: str = ""
    tokens_used: int = 0
    finish_reason: str = ""
    raw_response: Any = None

    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0

    @property
    def message(self) -> Dict[str, Any]:
        """Get message dict for conversation history."""
        msg = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class LLMError(Exception):
    """Base exception for LLM errors."""

    pass


class RateLimitError(LLMError):
    """Rate limit exceeded."""

    pass


class ServiceError(LLMError):
    """Service unavailable or error."""

    pass


class TimeoutError(LLMError):
    """Request timeout."""

    pass


class AllLLMsFailedError(LLMError):
    """All LLMs in cascade failed."""

    pass


class BaseLLMAdapter(ABC):
    """Abstract base class for LLM adapters."""

    def __init__(self, model: str, api_key: str, timeout: int = 60):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.provider_name = self.__class__.__name__.replace("Adapter", "").lower()

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate completion from messages.

        Args:
            messages: Conversation messages
            tools: Optional tool schemas
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse object

        Raises:
            RateLimitError: Rate limit exceeded
            ServiceError: Service error
            TimeoutError: Request timeout
            LLMError: Other errors
        """
        pass

    def _handle_error(self, error: Exception) -> None:
        """Convert provider-specific errors to standardized exceptions."""
        error_msg = str(error).lower()

        if "rate" in error_msg or "429" in error_msg:
            raise RateLimitError(f"{self.provider_name}: {error}")
        elif "timeout" in error_msg or "timed out" in error_msg:
            raise TimeoutError(f"{self.provider_name}: {error}")
        elif any(code in error_msg for code in ["500", "502", "503", "504"]):
            raise ServiceError(f"{self.provider_name}: {error}")
        else:
            raise LLMError(f"{self.provider_name}: {error}")


class DeepSeekAdapter(BaseLLMAdapter):
    """DeepSeek adapter using OpenAI-compatible API."""

    def __init__(self, model: str, api_key: str, base_url: str, timeout: int = 60):
        super().__init__(model, api_key, timeout)
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        try:
            # Convert tools to OpenAI format if needed
            tool_params = {}
            if tools:
                tool_params["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        },
                    }
                    for tool in tools
                ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                **tool_params,
                **kwargs,
            )

            message = response.choices[0].message
            tool_calls = None

            if message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in message.tool_calls
                ]

            return LLMResponse(
                content=message.content or "",
                tool_calls=tool_calls,
                model=self.model,
                provider="deepseek",
                tokens_used=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response,
            )
        except Exception as e:
            self._handle_error(e)


class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini adapter using new google-genai SDK."""

    def __init__(self, model: str, api_key: str, timeout: int = 60):
        super().__init__(model, api_key, timeout)
        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        try:
            # Convert messages to Gemini format
            gemini_contents = self._convert_messages(messages)

            # Convert tools to Gemini format
            gemini_tools = None
            if tools:
                gemini_tools = [self._convert_tool(tool) for tool in tools]

            # Generate content
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=kwargs.get("max_tokens", 4096),
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE"
                    ),
                ],
                tools=gemini_tools,
            )

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=gemini_contents,
                config=config,
            )

            content = ""
            tool_calls = None

            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            content += part.text
                        elif hasattr(part, "function_call") and part.function_call:
                            if tool_calls is None:
                                tool_calls = []
                            tool_calls.append(
                                {
                                    "id": f"call_{int(time.time())}",
                                    "name": part.function_call.name,
                                    "arguments": dict(part.function_call.args),
                                }
                            )

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                model=self.model,
                provider="gemini",
                tokens_used=0,  # Gemini doesn't provide token usage in response
                finish_reason="stop",
                raw_response=response,
            )
        except Exception as e:
            self._handle_error(e)

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List:
        """Convert standard messages to Gemini format."""
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Gemini doesn't have system role, convert to user message
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                parts = []
                # Handle text content
                if msg.get("content"):
                    parts.append(msg["content"])
                # Handle tool calls
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        function_call = types.FunctionCall(
                            name=tc["function"]["name"],
                            args=json.loads(tc["function"]["arguments"])
                                if isinstance(tc["function"]["arguments"], str)
                                else tc["function"]["arguments"]
                        )
                        parts.append(function_call)
                gemini_messages.append({"role": "model", "parts": parts if parts else [""]})
            elif msg["role"] == "tool":
                # Convert tool response to function response
                function_response = types.FunctionResponse(
                    name=msg.get("tool_call_id", "unknown"),
                    response={"result": msg["content"]}
                )
                gemini_messages.append({"role": "user", "parts": [function_response]})
        return gemini_messages

    def _convert_tool(self, tool: Dict[str, Any]) -> Any:
        """Convert tool schema to Gemini format."""
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=tool.get("parameters", {}),
                )
            ]
        )


class GPT5Adapter(BaseLLMAdapter):
    """OpenAI GPT-4/GPT-5 adapter."""

    def __init__(self, model: str, api_key: str, timeout: int = 60):
        super().__init__(model, api_key, timeout)
        self.client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout)

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        try:
            # Convert tools to OpenAI format
            tool_params = {}
            if tools:
                tool_params["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        },
                    }
                    for tool in tools
                ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                **tool_params,
                **kwargs,
            )

            message = response.choices[0].message
            tool_calls = None

            if message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in message.tool_calls
                ]

            return LLMResponse(
                content=message.content or "",
                tool_calls=tool_calls,
                model=self.model,
                provider="gpt5",
                tokens_used=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response,
            )
        except Exception as e:
            self._handle_error(e)


class ClaudeAdapter(BaseLLMAdapter):
    """Anthropic Claude adapter."""

    def __init__(self, model: str, api_key: str, timeout: int = 60):
        super().__init__(model, api_key, timeout)
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        try:
            # Extract system message if present
            system_message = ""
            filtered_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    filtered_messages.append(msg)

            # Convert tools to Claude format
            tool_params = {}
            if tools:
                tool_params["tools"] = [
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("parameters", {}),
                    }
                    for tool in tools
                ]

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 4096),
                messages=filtered_messages,
                system=system_message if system_message else None,
                temperature=temperature,
                **tool_params,
            )

            content = ""
            tool_calls = None

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "arguments": block.input,
                        }
                    )

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                model=self.model,
                provider="claude",
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                finish_reason=response.stop_reason,
                raw_response=response,
            )
        except Exception as e:
            self._handle_error(e)


class CascadeLLMClient:
    """
    LLM client with automatic cascade fallback.

    Tries LLMs in order until one succeeds. Automatically retries on errors.
    """

    def __init__(
        self,
        adapters: List[BaseLLMAdapter],
        max_retries: int = 3,
    ):
        """
        Initialize cascade client.

        Args:
            adapters: List of LLM adapters in priority order
            max_retries: Max retries per adapter
        """
        self.adapters = adapters
        self.max_retries = max_retries
        self.usage_stats = {adapter.provider_name: 0 for adapter in adapters}

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate completion with automatic fallback.

        Tries each adapter in order. On error, tries next adapter.

        Args:
            messages: Conversation messages
            tools: Optional tool schemas
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            LLMResponse from first successful adapter

        Raises:
            AllLLMsFailedError: If all adapters fail
        """
        last_error = None

        for adapter in self.adapters:
            for attempt in range(self.max_retries):
                try:
                    logger.info(
                        f"Trying {adapter.provider_name} (attempt {attempt + 1}/{self.max_retries})"
                    )

                    response = await adapter.complete(
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        **kwargs,
                    )

                    # Success!
                    self.usage_stats[adapter.provider_name] += 1
                    logger.info(f"✓ Success with {adapter.provider_name}")
                    return response

                except (RateLimitError, ServiceError, TimeoutError) as e:
                    logger.warning(f"✗ {adapter.provider_name} failed: {e}")
                    last_error = e

                    # Exponential backoff before retry
                    if attempt < self.max_retries - 1:
                        wait_time = 2**attempt
                        logger.info(f"Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(
                            f"{adapter.provider_name} exhausted all {self.max_retries} attempts"
                        )
                        break  # Try next adapter

                except LLMError as e:
                    # Non-retryable error, skip to next adapter
                    logger.error(f"✗ {adapter.provider_name} non-retryable error: {e}")
                    last_error = e
                    break

        # All adapters failed
        raise AllLLMsFailedError(
            f"All LLMs failed. Last error: {last_error}. Usage: {self.usage_stats}"
        )

    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for each provider."""
        return self.usage_stats.copy()

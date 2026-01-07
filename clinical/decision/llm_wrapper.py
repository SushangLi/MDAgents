"""
LLM Call Wrapper for CMOCoordinator.

Wraps CascadeLLMClient to match CMOCoordinator's expected LLM function signature,
with automatic fallback to mock mode when API keys are unavailable.
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from core.llm_client import CascadeLLMClient, AllLLMsFailedError

# Load environment variables from .env.local
env_local = Path(__file__).parent.parent.parent / ".env.local"
if env_local.exists():
    load_dotenv(env_local)
    print(f"✓ Loaded environment from {env_local}")


class LLMCallWrapper:
    """
    Wrapper that adapts CascadeLLMClient to CMOCoordinator's expected signature.

    The CMOCoordinator expects:
        async (messages: List[Dict], temperature: float) -> Dict[str, Any]

    Returns:
        {
            "content": str,        # LLM response text
            "model": str,          # Model name used
            "provider": str,       # Provider name (deepseek/gemini/openai/anthropic/mock)
            "tokens_used": int     # Token count
        }
    """

    def __init__(self, use_mock: bool = False):
        """
        Initialize LLM wrapper.

        Args:
            use_mock: Force mock mode (for testing without API calls)
        """
        if use_mock:
            self.mode = "mock"
            self.client = None
            print("ℹ️  LLMCallWrapper initialized in MOCK mode (no API calls)")
        elif not self._has_api_key():
            self.mode = "mock"
            self.client = None
            print("⚠️  No LLM API keys found - using MOCK mode")
        else:
            self.mode = "real"
            self.client = self._initialize_cascade_client()
            print("✓ LLMCallWrapper initialized with real LLM cascade")

    def _has_api_key(self) -> bool:
        """Check if any LLM API key is available."""
        keys = [
            "DEEPSEEK_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY"
        ]
        return any(os.getenv(key) for key in keys)

    def _initialize_cascade_client(self) -> CascadeLLMClient:
        """Initialize cascade LLM client with available providers."""
        from core.llm_client import DeepSeekAdapter, ClaudeAdapter, GeminiAdapter, GPT5Adapter

        # Build list of adapters based on available API keys
        adapters = []

        # Priority order: DeepSeek → Gemini → GPT5 → Claude
        if os.getenv("DEEPSEEK_API_KEY"):
            try:
                adapter = DeepSeekAdapter(
                    model="deepseek-chat",
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",
                    timeout=60
                )
                adapters.append(adapter)
                print(f"  ✓ DeepSeek adapter initialized")
            except Exception as e:
                print(f"  ⚠ Failed to initialize DeepSeek: {e}")

        if os.getenv("GEMINI_API_KEY"):
            try:
                adapter = GeminiAdapter(
                    model="gemini-2.0-flash-exp",
                    api_key=os.getenv("GEMINI_API_KEY"),
                    timeout=60
                )
                adapters.append(adapter)
                print(f"  ✓ Gemini adapter initialized")
            except Exception as e:
                print(f"  ⚠ Failed to initialize Gemini: {e}")

        if os.getenv("OPENAI_API_KEY"):
            try:
                adapter = GPT5Adapter(
                    model="gpt-4-turbo",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    timeout=60
                )
                adapters.append(adapter)
                print(f"  ✓ GPT5 adapter initialized")
            except Exception as e:
                print(f"  ⚠ Failed to initialize GPT5: {e}")

        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                adapter = ClaudeAdapter(
                    model="claude-3-5-sonnet-20241022",
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    timeout=60
                )
                adapters.append(adapter)
                print(f"  ✓ Claude adapter initialized")
            except Exception as e:
                print(f"  ⚠ Failed to initialize Claude: {e}")

        if not adapters:
            raise ValueError("No LLM adapters could be initialized")

        print(f"  Cascade configured with {len(adapters)} provider(s)")
        return CascadeLLMClient(adapters=adapters, max_retries=3)

    async def call(
        self,
        messages: List[Dict[str, Any]],
        temperature: float
    ) -> Dict[str, Any]:
        """
        Call LLM with given messages and temperature.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Temperature for sampling (0.0-1.0)

        Returns:
            Response dictionary with content, model, provider, tokens_used
        """
        if self.mode == "mock":
            return self._generate_mock_response(messages)

        try:
            # Call cascade client
            response = await self.client.complete(
                messages=messages,
                temperature=temperature
            )

            return {
                "content": response.content,
                "model": response.model,
                "provider": response.provider,
                "tokens_used": response.tokens_used
            }

        except AllLLMsFailedError as e:
            # All LLMs failed - fall back to mock
            print(f"⚠️  All LLMs failed: {e}")
            print("  Falling back to MOCK mode")
            return self._generate_mock_response(messages)

        except Exception as e:
            # Unexpected error - fall back to mock
            print(f"⚠️  Unexpected LLM error: {e}")
            print("  Falling back to MOCK mode")
            return self._generate_mock_response(messages)

    def _generate_mock_response(self, messages: List[Dict]) -> Dict:
        """
        Generate mock LLM response based on prompt analysis.

        Args:
            messages: Message list (extracts last user message as prompt)

        Returns:
            Mock response dictionary
        """
        # Extract user prompt
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        # Extract diagnoses mentioned in prompt
        diagnoses = self._extract_diagnoses_from_prompt(prompt)

        if not diagnoses:
            diagnoses = ["Periodontitis"]  # Default

        # Determine confidence based on prompt keywords
        confidence = 0.82
        if "conflict" in prompt.lower() or "disagree" in prompt.lower():
            confidence = 0.78
        elif "consensus" in prompt.lower() or "agree" in prompt.lower():
            confidence = 0.90

        # Generate structured response
        response = f"""Final Diagnosis: {diagnoses[0]}
Confidence: {confidence:.2f}

Reasoning:
- Multi-omics evidence synthesis completed
- Expert consensus weighted by confidence scores
- RAG/CAG evidence considered where available
- Final decision: {diagnoses[0]} based on predominant evidence

Key Biomarkers:
- Porphyromonas gingivalis (microbiome): Primary periodontal pathogen, 3.5x elevated
- MMP-9 (proteome): Tissue degradation marker, 2.5x elevated
- IL-6 (metabolome): Inflammatory cytokine, 2.5x elevated
- TNF-alpha (metabolome): Pro-inflammatory marker, 2.0x elevated

Differential Diagnoses:
"""
        # Add other diagnoses if present
        for i, dx in enumerate(diagnoses[1:3], 1):
            prob = confidence - (i * 0.15)
            response += f"- {dx}: Probability {prob:.2f}, less likely based on biomarker profile\n"

        response += """
Recommendations:
1. Initiate appropriate treatment protocol based on diagnosis
2. Monitor key biomarker response over 4-6 weeks
3. Follow-up multi-omics analysis to track treatment efficacy
4. Consider additional diagnostics if symptoms persist or worsen
5. Patient education on disease management and prevention

Clinical Notes:
- Diagnosis confidence is {conf_level} ({confidence:.1%})
- Multi-omics integration provides comprehensive disease assessment
- Treatment should be individualized based on patient history
- Regular monitoring recommended for chronic conditions
""".replace("{conf_level}", "high" if confidence >= 0.8 else "moderate").replace("{confidence:.1%}", f"{confidence:.1%}")

        return {
            "content": response,
            "model": "mock-gpt-4",
            "provider": "mock",
            "tokens_used": 500
        }

    def _extract_diagnoses_from_prompt(self, prompt: str) -> List[str]:
        """
        Extract diagnosis names from prompt text.

        Args:
            prompt: User prompt text

        Returns:
            List of diagnosis names found
        """
        # Common diagnosis patterns
        diagnosis_keywords = [
            "Periodontitis",
            "Gingivitis",
            "Diabetes",
            "Healthy",
            "Oral Cancer",
            "Candidiasis"
        ]

        found = []
        prompt_lower = prompt.lower()

        for dx in diagnosis_keywords:
            if dx.lower() in prompt_lower:
                found.append(dx)

        # Also try to extract from structured expert opinion blocks
        # Pattern: "diagnosis": "Name" or Diagnosis: Name
        pattern = r'(?:diagnosis["\']?\s*:\s*["\']?)(\w+)'
        matches = re.findall(pattern, prompt, re.IGNORECASE)
        for match in matches:
            capitalized = match.capitalize()
            if capitalized in diagnosis_keywords and capitalized not in found:
                found.append(capitalized)

        return found if found else ["Periodontitis"]

    def __repr__(self) -> str:
        """String representation."""
        return f"LLMCallWrapper(mode={self.mode})"


def create_llm_wrapper(use_mock: bool = False) -> LLMCallWrapper:
    """
    Factory function to create LLM wrapper.

    Args:
        use_mock: Force mock mode (default: auto-detect based on API keys)

    Returns:
        Configured LLMCallWrapper instance

    Example:
        # Auto mode - uses real LLM if keys available, else mock
        wrapper = create_llm_wrapper()
        cmo = CMOCoordinator(llm_call_func=wrapper.call)

        # Force mock mode (for testing)
        wrapper = create_llm_wrapper(use_mock=True)
        cmo = CMOCoordinator(llm_call_func=wrapper.call)
    """
    return LLMCallWrapper(use_mock=use_mock)

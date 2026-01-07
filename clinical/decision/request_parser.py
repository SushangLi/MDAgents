"""
Request Parser for Natural Language Diagnostic Requests.

Parses user's natural language requests into structured DiagnosisConfig
using LLM-based interpretation.
"""

import json
import re
from typing import Callable, Awaitable, List, Dict, Any

from clinical.models.diagnosis_config import DiagnosisConfig
from clinical.utils.prompts import REQUEST_PARSER_SYSTEM_PROMPT, build_request_parsing_prompt


# Type alias for LLM call function
LLMCallFunc = Callable[[List[Dict[str, Any]], float], Awaitable[Dict[str, Any]]]


class RequestParser:
    """
    Parses natural language diagnostic requests into structured configuration.

    Uses LLM to extract diagnostic parameters from user requests like:
    - "只分析微生物组数据，使用文献支持"
    - "分析病人P001的代谢组"
    - "分析前50行数据的微生物组和蛋白质组"
    - "全面分析病人P002、P003、P005，3轮辩论，详细报告"
    """

    def __init__(self, llm_call_func: Callable):
        """
        Initialize request parser.

        Args:
            llm_call_func: Async function for calling LLM
                          Signature: (messages: List[Dict], temperature: float) -> Dict
        """
        self.llm_call_func = llm_call_func
        print("✓ Request Parser initialized")

    async def parse_request(self, user_request: str) -> DiagnosisConfig:
        """
        Parse natural language request into DiagnosisConfig.

        Args:
            user_request: User's natural language diagnostic request

        Returns:
            DiagnosisConfig instance

        Examples:
            >>> parser = RequestParser(llm_call_func)
            >>> config = await parser.parse_request("只分析微生物组数据")
            >>> config.omics_types
            ['microbiome']

            >>> config = await parser.parse_request("分析病人P001-P003")
            >>> config.patient_ids
            ['P001', 'P002', 'P003']
        """
        print(f"\n[RequestParser] Parsing request: \"{user_request}\"")

        try:
            # Build messages for LLM
            messages = [
                {"role": "system", "content": REQUEST_PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": build_request_parsing_prompt(user_request)}
            ]

            # Call LLM with low temperature for consistency
            response = await self.llm_call_func(messages, temperature=0.1)

            # Extract JSON from response
            config_dict = self._extract_json_from_response(response["content"])

            # Validate and create config
            config = DiagnosisConfig.from_dict(config_dict)

            print(f"[RequestParser] ✓ Parsed successfully: {config}")
            return config

        except Exception as e:
            print(f"[RequestParser] ⚠ Parsing failed: {e}")
            print(f"[RequestParser] Using default configuration")

            # Return default configuration on failure
            return DiagnosisConfig.get_default()

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON object from LLM response.

        Handles cases where LLM includes additional text besides JSON.

        Args:
            response_text: LLM response text

        Returns:
            Parsed JSON dictionary

        Raises:
            json.JSONDecodeError: If no valid JSON found
        """
        # Try direct JSON parsing first
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in response
        # Look for {...} pattern
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # If all else fails, try to extract from code block
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # Raise error if no valid JSON found
        raise json.JSONDecodeError(
            "Could not extract valid JSON from LLM response",
            response_text,
            0
        )

    def parse_patient_range(self, range_str: str) -> List[str]:
        """
        Parse patient range string into list of patient IDs.

        Args:
            range_str: Range string like "P001-P005" or "P001,P002,P003"

        Returns:
            List of patient IDs

        Examples:
            >>> parser.parse_patient_range("P001-P005")
            ['P001', 'P002', 'P003', 'P004', 'P005']

            >>> parser.parse_patient_range("P001,P002,P003")
            ['P001', 'P002', 'P003']
        """
        # Handle comma-separated list
        if ',' in range_str:
            return [pid.strip() for pid in range_str.split(',')]

        # Handle range (e.g., "P001-P005")
        match = re.match(r'([A-Za-z]+)(\d+)-([A-Za-z]+)?(\d+)', range_str)
        if match:
            prefix = match.group(1)
            start_num = int(match.group(2))
            end_prefix = match.group(3) or prefix
            end_num = int(match.group(4))

            if prefix != end_prefix:
                raise ValueError(f"Patient ID prefixes must match: {prefix} vs {end_prefix}")

            # Generate range
            num_width = len(match.group(2))  # Preserve leading zeros
            return [f"{prefix}{str(i).zfill(num_width)}" for i in range(start_num, end_num + 1)]

        # Single patient ID
        return [range_str.strip()]

    def validate_config(self, config: DiagnosisConfig) -> bool:
        """
        Validate parsed configuration.

        Args:
            config: Diagnosis configuration

        Returns:
            True if valid, False otherwise
        """
        try:
            # Validation is done in DiagnosisConfig.__post_init__
            # If we got here, it's valid
            return True
        except Exception as e:
            print(f"[RequestParser] ⚠ Validation failed: {e}")
            return False

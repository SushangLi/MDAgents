"""
Diagnosis Configuration Data Model.

This module defines the configuration structure for intelligent diagnosis,
parsed from natural language user requests or provided as structured input.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any
import json


@dataclass
class DiagnosisConfig:
    """
    Configuration for intelligent diagnosis workflow.

    Parsed from natural language requests or provided directly.
    Controls which data to analyze, how to analyze it, and report format.

    Attributes:
        omics_types: Which omics types to analyze (subset or all)
        patient_ids: Specific patient IDs to analyze (None = all patients)
        row_range: Specific row range to analyze (None = all rows)
        enable_rag: Whether to enable RAG (literature search)
        enable_cag: Whether to enable CAG (case retrieval)
        force_rag_even_no_conflict: Force RAG even without expert conflicts
        max_debate_rounds: Maximum debate rounds (1-10)
        confidence_threshold: Confidence threshold for decisions (0-1)
        threshold_adjustment: Threshold adjustment per round (0-1)
        detail_level: Report detail level ("brief", "standard", "detailed")
        bilingual: Whether to generate bilingual reports (Chinese | English)
    """

    # Data selection
    omics_types: List[str] = field(default_factory=lambda: ["microbiome", "metabolome", "proteome"])
    patient_ids: Optional[List[str]] = None
    row_range: Optional[Tuple[int, int]] = None

    # RAG/CAG control
    enable_rag: bool = True
    enable_cag: bool = True
    force_rag_even_no_conflict: bool = False

    # Debate parameters
    max_debate_rounds: int = 3
    confidence_threshold: float = 0.7
    threshold_adjustment: float = 0.1

    # Report configuration
    detail_level: str = "standard"  # "brief" | "standard" | "detailed"
    bilingual: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate omics_types
        valid_omics = {"microbiome", "metabolome", "proteome"}
        for omics in self.omics_types:
            if omics not in valid_omics:
                raise ValueError(
                    f"Invalid omics type '{omics}'. "
                    f"Must be one of: {', '.join(valid_omics)}"
                )

        # Validate max_debate_rounds
        if not 1 <= self.max_debate_rounds <= 10:
            raise ValueError(
                f"max_debate_rounds must be between 1 and 10, got {self.max_debate_rounds}"
            )

        # Validate confidence_threshold
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                f"confidence_threshold must be between 0 and 1, got {self.confidence_threshold}"
            )

        # Validate threshold_adjustment
        if not 0 <= self.threshold_adjustment <= 1:
            raise ValueError(
                f"threshold_adjustment must be between 0 and 1, got {self.threshold_adjustment}"
            )

        # Validate detail_level
        valid_levels = {"brief", "standard", "detailed"}
        if self.detail_level not in valid_levels:
            raise ValueError(
                f"Invalid detail_level '{self.detail_level}'. "
                f"Must be one of: {', '.join(valid_levels)}"
            )

        # Validate row_range
        if self.row_range is not None:
            if len(self.row_range) != 2:
                raise ValueError(
                    f"row_range must be a tuple of (start, end), got {self.row_range}"
                )
            start, end = self.row_range
            if start < 0 or end < 0:
                raise ValueError(
                    f"row_range values must be non-negative, got ({start}, {end})"
                )
            if start >= end:
                raise ValueError(
                    f"row_range start must be less than end, got ({start}, {end})"
                )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of configuration
        """
        result = {
            "omics_types": self.omics_types,
            "patient_ids": self.patient_ids,
            "row_range": list(self.row_range) if self.row_range else None,
            "enable_rag": self.enable_rag,
            "enable_cag": self.enable_cag,
            "force_rag_even_no_conflict": self.force_rag_even_no_conflict,
            "max_debate_rounds": self.max_debate_rounds,
            "confidence_threshold": self.confidence_threshold,
            "threshold_adjustment": self.threshold_adjustment,
            "detail_level": self.detail_level,
            "bilingual": self.bilingual
        }
        return result

    def to_json(self) -> str:
        """
        Convert to JSON string.

        Returns:
            JSON representation of configuration
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosisConfig":
        """
        Create from dictionary.

        Args:
            data: Dictionary containing configuration data

        Returns:
            DiagnosisConfig instance
        """
        # Convert row_range from list to tuple if present
        if "row_range" in data and data["row_range"] is not None:
            data["row_range"] = tuple(data["row_range"])

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "DiagnosisConfig":
        """
        Create from JSON string.

        Args:
            json_str: JSON string containing configuration

        Returns:
            DiagnosisConfig instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def get_default(cls) -> "DiagnosisConfig":
        """
        Get default configuration (all omics, full analysis).

        Returns:
            Default DiagnosisConfig instance
        """
        return cls(
            omics_types=["microbiome", "metabolome", "proteome"],
            patient_ids=None,
            row_range=None,
            enable_rag=True,
            enable_cag=True,
            force_rag_even_no_conflict=False,
            max_debate_rounds=3,
            confidence_threshold=0.7,
            threshold_adjustment=0.1,
            detail_level="standard",
            bilingual=True
        )

    def __repr__(self) -> str:
        """String representation of configuration."""
        parts = []
        parts.append(f"omics={','.join(self.omics_types)}")

        if self.patient_ids:
            patient_str = ','.join(self.patient_ids[:3])
            if len(self.patient_ids) > 3:
                patient_str += f",... ({len(self.patient_ids)} total)"
            parts.append(f"patients={patient_str}")

        if self.row_range:
            parts.append(f"rows={self.row_range[0]}-{self.row_range[1]}")

        parts.append(f"RAG={'on' if self.enable_rag else 'off'}")
        parts.append(f"CAG={'on' if self.enable_cag else 'off'}")

        if self.force_rag_even_no_conflict:
            parts.append("force_RAG=on")

        parts.append(f"rounds={self.max_debate_rounds}")
        parts.append(f"detail={self.detail_level}")
        parts.append(f"bilingual={'yes' if self.bilingual else 'no'}")

        return f"DiagnosisConfig({', '.join(parts)})"

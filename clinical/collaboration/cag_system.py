"""
CAG (Cache-Augmented Generation) System.

Provides caching and retrieval of historical clinical cases to support
expert opinion conflict resolution and diagnostic validation.

By caching diagnosed cases with their omics features and final diagnoses,
the system can retrieve similar cases to augment LLM reasoning and reduce
redundant inference costs.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from clinical.collaboration.embeddings import BiomedicalEmbeddings
from clinical.models.expert_opinion import ExpertOpinion


@dataclass
class ClinicalCase:
    """Historical clinical case with omics data and diagnosis."""

    case_id: str
    patient_id: str
    diagnosis: str
    microbiome_features: Optional[Dict[str, float]] = None
    metabolome_features: Optional[Dict[str, float]] = None
    proteome_features: Optional[Dict[str, float]] = None
    clinical_notes: str = ""
    diagnosis_date: Optional[str] = None
    severity: Optional[str] = None
    treatment_outcome: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "diagnosis": self.diagnosis,
            "microbiome_features": self.microbiome_features,
            "metabolome_features": self.metabolome_features,
            "proteome_features": self.proteome_features,
            "clinical_notes": self.clinical_notes,
            "diagnosis_date": self.diagnosis_date,
            "severity": self.severity,
            "treatment_outcome": self.treatment_outcome,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClinicalCase":
        """Create from dictionary."""
        return cls(
            case_id=data["case_id"],
            patient_id=data["patient_id"],
            diagnosis=data["diagnosis"],
            microbiome_features=data.get("microbiome_features"),
            metabolome_features=data.get("metabolome_features"),
            proteome_features=data.get("proteome_features"),
            clinical_notes=data.get("clinical_notes", ""),
            diagnosis_date=data.get("diagnosis_date"),
            severity=data.get("severity"),
            treatment_outcome=data.get("treatment_outcome"),
            metadata=data.get("metadata", {})
        )


@dataclass
class CAGResult:
    """Result from CAG case retrieval."""

    query_features: Dict[str, Any]
    similar_cases: List[ClinicalCase]
    similarity_scores: List[float]
    diagnosis_distribution: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_features": self.query_features,
            "similar_cases": [case.to_dict() for case in self.similar_cases],
            "similarity_scores": self.similarity_scores,
            "diagnosis_distribution": self.diagnosis_distribution
        }

    def get_top_k(self, k: int = 3) -> "CAGResult":
        """Get top-k similar cases."""
        return CAGResult(
            query_features=self.query_features,
            similar_cases=self.similar_cases[:k],
            similarity_scores=self.similarity_scores[:k],
            diagnosis_distribution=self._recalculate_distribution(k)
        )

    def _recalculate_distribution(self, k: int) -> Dict[str, int]:
        """Recalculate diagnosis distribution for top-k."""
        dist = {}
        for case in self.similar_cases[:k]:
            dist[case.diagnosis] = dist.get(case.diagnosis, 0) + 1
        return dist


class CAGSystem:
    """
    Cache-Augmented Generation system for case-based reasoning.

    Caches and retrieves similar historical cases based on omics feature
    similarity to augment LLM reasoning and support diagnostic validation.
    Reduces redundant LLM inference by reusing cached diagnostic patterns.
    """

    def __init__(
        self,
        case_database_path: str = "data/knowledge_base/clinical_cases.json",
        embedding_model: Optional[BiomedicalEmbeddings] = None
    ):
        """
        Initialize CAG system.

        Args:
            case_database_path: Path to clinical case database (JSON)
            embedding_model: Embeddings model for clinical notes similarity
        """
        self.case_database_path = Path(case_database_path)
        self.cases: List[ClinicalCase] = []

        # Initialize embedding model for clinical notes
        if embedding_model is None:
            from clinical.collaboration.embeddings import get_default_embeddings
            self.embedding_model = get_default_embeddings()
        else:
            self.embedding_model = embedding_model

        # Load existing cases if available
        if self.case_database_path.exists():
            self.load_cases()
        else:
            print(f"⚠ CAG database not found at {case_database_path}")
            print(f"  Creating empty database")
            self.case_database_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_cases()

        print(f"✓ CAG system initialized with {len(self.cases)} cases")

    def add_case(
        self,
        patient_id: str,
        diagnosis: str,
        microbiome_features: Optional[Dict[str, float]] = None,
        metabolome_features: Optional[Dict[str, float]] = None,
        proteome_features: Optional[Dict[str, float]] = None,
        clinical_notes: str = "",
        severity: Optional[str] = None,
        treatment_outcome: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new clinical case to the database.

        Args:
            patient_id: Patient identifier
            diagnosis: Confirmed diagnosis
            microbiome_features: Microbiome feature dict
            metabolome_features: Metabolome feature dict
            proteome_features: Proteome feature dict
            clinical_notes: Clinical notes/description
            severity: Disease severity
            treatment_outcome: Treatment outcome description
            metadata: Additional metadata

        Returns:
            Case ID
        """
        # Generate case ID
        case_id = f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.cases)}"

        # Create case
        case = ClinicalCase(
            case_id=case_id,
            patient_id=patient_id,
            diagnosis=diagnosis,
            microbiome_features=microbiome_features,
            metabolome_features=metabolome_features,
            proteome_features=proteome_features,
            clinical_notes=clinical_notes,
            diagnosis_date=datetime.now().isoformat(),
            severity=severity,
            treatment_outcome=treatment_outcome,
            metadata=metadata or {}
        )

        # Add to database
        self.cases.append(case)
        self.save_cases()

        print(f"✓ Added case {case_id} (diagnosis: {diagnosis})")
        return case_id

    def search_similar_cases(
        self,
        microbiome_features: Optional[Dict[str, float]] = None,
        metabolome_features: Optional[Dict[str, float]] = None,
        proteome_features: Optional[Dict[str, float]] = None,
        clinical_notes: Optional[str] = None,
        top_k: int = 5,
        diagnosis_filter: Optional[List[str]] = None,
        min_similarity: float = 0.6
    ) -> CAGResult:
        """
        Search for similar historical cases.

        Args:
            microbiome_features: Query microbiome features
            metabolome_features: Query metabolome features
            proteome_features: Query proteome features
            clinical_notes: Query clinical notes
            top_k: Number of top similar cases to return
            diagnosis_filter: Filter by diagnosis list
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            CAG results with similar cases
        """
        if not self.cases:
            print("⚠ No cases in database")
            return CAGResult(
                query_features={
                    "microbiome": microbiome_features,
                    "metabolome": metabolome_features,
                    "proteome": proteome_features
                },
                similar_cases=[],
                similarity_scores=[],
                diagnosis_distribution={}
            )

        # Calculate similarities
        similarities = []
        for case in self.cases:
            # Filter by diagnosis if specified
            if diagnosis_filter and case.diagnosis not in diagnosis_filter:
                similarities.append(0.0)
                continue

            # Calculate multi-omics similarity
            sim = self._calculate_case_similarity(
                case,
                microbiome_features,
                metabolome_features,
                proteome_features,
                clinical_notes
            )
            similarities.append(sim)

        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]

        # Filter by minimum similarity
        filtered_indices = [
            i for i in sorted_indices
            if similarities[i] >= min_similarity
        ][:top_k]

        # Get top-k cases
        similar_cases = [self.cases[i] for i in filtered_indices]
        similarity_scores = [similarities[i] for i in filtered_indices]

        # Calculate diagnosis distribution
        diagnosis_dist = {}
        for case in similar_cases:
            diagnosis_dist[case.diagnosis] = diagnosis_dist.get(case.diagnosis, 0) + 1

        return CAGResult(
            query_features={
                "microbiome": microbiome_features,
                "metabolome": metabolome_features,
                "proteome": proteome_features,
                "clinical_notes": clinical_notes
            },
            similar_cases=similar_cases,
            similarity_scores=similarity_scores,
            diagnosis_distribution=diagnosis_dist
        )

    def retrieve_for_conflict(
        self,
        conflicting_opinions: List[ExpertOpinion],
        sample_data: Dict[str, pd.Series],
        top_k: int = 5
    ) -> CAGResult:
        """
        Retrieve similar cases for resolving expert conflicts.

        Args:
            conflicting_opinions: List of conflicting expert opinions
            sample_data: Dictionary with omics data for the query sample
                        {"microbiome": Series, "metabolome": Series, "proteome": Series}
            top_k: Number of cases to retrieve

        Returns:
            CAG results with similar cases
        """
        # Extract features from sample data
        microbiome_features = None
        metabolome_features = None
        proteome_features = None

        if "microbiome" in sample_data:
            data = sample_data["microbiome"]
            # If DataFrame (multiple rows), use first row only
            if isinstance(data, pd.DataFrame):
                if len(data) > 0:
                    microbiome_features = data.iloc[0].to_dict()
            else:
                # Series (single row)
                microbiome_features = data.to_dict()

        if "metabolome" in sample_data:
            data = sample_data["metabolome"]
            # If DataFrame (multiple rows), use first row only
            if isinstance(data, pd.DataFrame):
                if len(data) > 0:
                    metabolome_features = data.iloc[0].to_dict()
            else:
                # Series (single row)
                metabolome_features = data.to_dict()

        if "proteome" in sample_data:
            data = sample_data["proteome"]
            # If DataFrame (multiple rows), use first row only
            if isinstance(data, pd.DataFrame):
                if len(data) > 0:
                    proteome_features = data.iloc[0].to_dict()
            else:
                # Series (single row)
                proteome_features = data.to_dict()

        # Build clinical notes from expert opinions
        clinical_notes = self._build_notes_from_opinions(conflicting_opinions)

        # Get diagnoses from conflicting opinions for filtering
        diagnoses = list(set([op.diagnosis for op in conflicting_opinions]))

        # Search for similar cases
        results = self.search_similar_cases(
            microbiome_features=microbiome_features,
            metabolome_features=metabolome_features,
            proteome_features=proteome_features,
            clinical_notes=clinical_notes,
            top_k=top_k,
            diagnosis_filter=diagnoses,
            min_similarity=0.5
        )

        return results

    def _calculate_case_similarity(
        self,
        case: ClinicalCase,
        microbiome_features: Optional[Dict[str, float]],
        metabolome_features: Optional[Dict[str, float]],
        proteome_features: Optional[Dict[str, float]],
        clinical_notes: Optional[str]
    ) -> float:
        """
        Calculate similarity between query and a case.

        Uses weighted combination of:
        - Omics feature similarity (cosine)
        - Clinical notes similarity (semantic)

        Args:
            case: Historical case
            microbiome_features: Query microbiome features
            metabolome_features: Query metabolome features
            proteome_features: Query proteome features
            clinical_notes: Query clinical notes

        Returns:
            Similarity score (0-1)
        """
        similarities = []
        weights = []

        # Microbiome similarity
        if microbiome_features and case.microbiome_features:
            sim = self._cosine_similarity_dict(
                microbiome_features,
                case.microbiome_features
            )
            similarities.append(sim)
            weights.append(1.0)

        # Metabolome similarity
        if metabolome_features and case.metabolome_features:
            sim = self._cosine_similarity_dict(
                metabolome_features,
                case.metabolome_features
            )
            similarities.append(sim)
            weights.append(1.0)

        # Proteome similarity
        if proteome_features and case.proteome_features:
            sim = self._cosine_similarity_dict(
                proteome_features,
                case.proteome_features
            )
            similarities.append(sim)
            weights.append(1.0)

        # Clinical notes similarity (semantic)
        if clinical_notes and case.clinical_notes:
            sim = self.embedding_model.compute_similarity(
                clinical_notes,
                case.clinical_notes
            )
            similarities.append(sim)
            weights.append(0.5)  # Lower weight for clinical notes

        # Weighted average
        if not similarities:
            return 0.0

        total_weight = sum(weights)
        weighted_sim = sum(s * w for s, w in zip(similarities, weights)) / total_weight

        return weighted_sim

    def _cosine_similarity_dict(
        self,
        features1: Dict[str, float],
        features2: Dict[str, float]
    ) -> float:
        """
        Calculate cosine similarity between two feature dictionaries.

        Args:
            features1: First feature dict
            features2: Second feature dict

        Returns:
            Cosine similarity (0-1)
        """
        # Get common features
        common_features = set(features1.keys()) & set(features2.keys())

        if not common_features:
            return 0.0

        # Build vectors
        vec1 = np.array([features1[f] for f in common_features])
        vec2 = np.array([features2[f] for f in common_features])

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Ensure in [0, 1] range
        return max(0.0, min(1.0, similarity))

    def _build_notes_from_opinions(
        self,
        opinions: List[ExpertOpinion]
    ) -> str:
        """
        Build clinical notes query from expert opinions.

        Args:
            opinions: List of expert opinions

        Returns:
            Clinical notes string
        """
        notes_parts = []

        for opinion in opinions:
            notes_parts.append(
                f"{opinion.omics_type} expert suggests {opinion.diagnosis} "
                f"with {opinion.confidence:.1%} confidence. "
                f"{opinion.biological_explanation}"
            )

        return " ".join(notes_parts)

    def format_context_for_llm(
        self,
        cag_result: CAGResult,
        max_context_length: int = 1500
    ) -> str:
        """
        Format CAG results as context for LLM.

        Args:
            cag_result: CAG retrieval results
            max_context_length: Maximum context length in characters

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        context_parts.append("## Similar Historical Cases\n")

        for i, (case, score) in enumerate(zip(
            cag_result.similar_cases,
            cag_result.similarity_scores
        )):
            # Format case
            case_str = f"\n### Case {i+1} (Similarity: {score:.2%})\n"
            case_str += f"**Diagnosis**: {case.diagnosis}\n"

            if case.severity:
                case_str += f"**Severity**: {case.severity}\n"

            if case.diagnosis_date:
                case_str += f"**Date**: {case.diagnosis_date}\n"

            if case.treatment_outcome:
                case_str += f"**Outcome**: {case.treatment_outcome}\n"

            if case.clinical_notes:
                case_str += f"\n{case.clinical_notes}\n"

            # Check length
            if current_length + len(case_str) > max_context_length:
                break

            context_parts.append(case_str)
            current_length += len(case_str)

        # Add diagnosis distribution
        if cag_result.diagnosis_distribution:
            dist_str = "\n### Diagnosis Distribution in Similar Cases\n"
            for diagnosis, count in sorted(
                cag_result.diagnosis_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                dist_str += f"- {diagnosis}: {count} cases\n"

            context_parts.append(dist_str)

        return "".join(context_parts)

    def load_cases(self):
        """Load cases from JSON file."""
        with open(self.case_database_path, "r") as f:
            data = json.load(f)

        self.cases = [ClinicalCase.from_dict(case_data) for case_data in data]
        print(f"✓ Loaded {len(self.cases)} cases from {self.case_database_path}")

    def save_cases(self):
        """Save cases to JSON file."""
        self.case_database_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.case_database_path, "w") as f:
            json.dump(
                [case.to_dict() for case in self.cases],
                f,
                indent=2
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get CAG system statistics.

        Returns:
            Dictionary with statistics
        """
        # Calculate diagnosis distribution
        diagnosis_dist = {}
        for case in self.cases:
            diagnosis_dist[case.diagnosis] = diagnosis_dist.get(case.diagnosis, 0) + 1

        return {
            "total_cases": len(self.cases),
            "diagnosis_distribution": diagnosis_dist,
            "database_path": str(self.case_database_path)
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"CAGSystem(cases={len(self.cases)})"

"""
Test CAG (Cache-Augmented Generation) System.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clinical.collaboration.cag_system import CAGSystem, ClinicalCase


def test_cag_initialization():
    """Test CAG system initialization."""
    cag = CAGSystem()
    assert cag is not None
    assert cag.cases is not None


def test_add_case():
    """Test adding a clinical case."""
    cag = CAGSystem(case_database_path="data/test/test_cag.json")

    # Add a test case
    case_id = cag.add_case(
        patient_id="TEST001",
        diagnosis="Periodontitis",
        microbiome_features={"Porphyromonas_gingivalis": 0.15, "Streptococcus_mutans": 0.08},
        metabolome_features={"Butyrate": 1500, "Propionate": 800},
        proteome_features={"MMP9": 450, "IL6": 120},
        clinical_notes="Severe periodontitis with elevated inflammatory markers",
        severity="Severe",
        treatment_outcome="Improved with scaling and antibiotics"
    )

    assert case_id is not None
    assert len(cag.cases) > 0

    print(f"\nAdded case: {case_id}")


def test_search_similar_cases():
    """Test searching for similar cases."""
    cag = CAGSystem(case_database_path="data/test/test_cag.json")

    # Add multiple cases
    cag.add_case(
        patient_id="TEST001",
        diagnosis="Periodontitis",
        microbiome_features={"Porphyromonas_gingivalis": 0.15},
        metabolome_features={"Butyrate": 1500}
    )

    cag.add_case(
        patient_id="TEST002",
        diagnosis="Healthy",
        microbiome_features={"Streptococcus_salivarius": 0.20},
        metabolome_features={"Acetate": 800}
    )

    # Search for similar cases
    results = cag.search_similar_cases(
        microbiome_features={"Porphyromonas_gingivalis": 0.14},
        metabolome_features={"Butyrate": 1400},
        top_k=2,
        min_similarity=0.3
    )

    assert results is not None
    assert hasattr(results, 'similar_cases')
    assert hasattr(results, 'similarity_scores')

    print(f"\nSearch Results:")
    print(f"  Found {len(results.similar_cases)} similar cases")

    for i, (case, score) in enumerate(zip(results.similar_cases, results.similarity_scores)):
        print(f"\n  Case {i+1} (similarity: {score:.2%}):")
        print(f"    Patient: {case.patient_id}")
        print(f"    Diagnosis: {case.diagnosis}")


def test_diagnosis_distribution():
    """Test diagnosis distribution in similar cases."""
    cag = CAGSystem(case_database_path="data/test/test_cag.json")

    # Add cases with different diagnoses
    for i in range(3):
        cag.add_case(
            patient_id=f"PERIO{i+1}",
            diagnosis="Periodontitis",
            microbiome_features={"Porphyromonas_gingivalis": 0.15 + i*0.01}
        )

    for i in range(2):
        cag.add_case(
            patient_id=f"HEALTHY{i+1}",
            diagnosis="Healthy",
            microbiome_features={"Streptococcus_salivarius": 0.20 + i*0.01}
        )

    # Search
    results = cag.search_similar_cases(
        microbiome_features={"Porphyromonas_gingivalis": 0.16},
        top_k=5
    )

    assert results.diagnosis_distribution is not None

    print(f"\nDiagnosis Distribution:")
    for diagnosis, count in results.diagnosis_distribution.items():
        print(f"  {diagnosis}: {count} cases")


def test_cag_format_context():
    """Test CAG context formatting."""
    cag = CAGSystem(case_database_path="data/test/test_cag.json")

    # Add a case
    cag.add_case(
        patient_id="TEST001",
        diagnosis="Periodontitis",
        severity="Severe",
        treatment_outcome="Improved",
        clinical_notes="Patient responded well to treatment"
    )

    # Search and format
    results = cag.search_similar_cases(
        microbiome_features={"Porphyromonas_gingivalis": 0.15},
        top_k=1
    )

    context = cag.format_context_for_llm(results)

    assert isinstance(context, str)
    assert len(context) > 0
    assert "Similar Historical Cases" in context

    print(f"\nFormatted Context:\n{context}")


def test_cag_statistics():
    """Test CAG statistics."""
    cag = CAGSystem(case_database_path="data/test/test_cag.json")

    # Add some cases
    for i in range(5):
        cag.add_case(
            patient_id=f"TEST{i+1}",
            diagnosis="Periodontitis" if i < 3 else "Healthy"
        )

    stats = cag.get_statistics()

    assert "total_cases" in stats
    assert "diagnosis_distribution" in stats

    print(f"\nCAG Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

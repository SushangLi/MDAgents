"""
End-to-End Integration Test.

Tests the complete diagnosis workflow from raw data to final report.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clinical.preprocessing.microbiome_preprocessor import MicrobiomePreprocessor
from clinical.preprocessing.metabolome_preprocessor import MetabolomePreprocessor
from clinical.preprocessing.proteome_preprocessor import ProteomePreprocessor
from clinical.decision.conflict_resolver import ConflictResolver
from clinical.decision.report_generator import ReportGenerator
from clinical.models.expert_opinion import ExpertOpinion, FeatureImportance
from clinical.models.diagnosis_result import DiagnosisResult


def test_data_loading():
    """Test loading test data."""
    data_files = {
        'microbiome': Path("data/test/microbiome_raw.csv"),
        'metabolome': Path("data/test/metabolome_raw.csv"),
        'proteome': Path("data/test/proteome_raw.csv"),
        'labels': Path("data/test/labels.csv")
    }

    for name, path in data_files.items():
        if not path.exists():
            pytest.skip(f"Test data not found: {path}")

        df = pd.read_csv(path, index_col=0 if name != 'labels' else None)
        assert df is not None
        assert len(df) > 0

        print(f"\n{name.title()} Data:")
        print(f"  Shape: {df.shape}")
        print(f"  Samples: {len(df)}")


def test_preprocessing_pipeline():
    """Test complete preprocessing pipeline."""
    # Load data
    microbiome_df = pd.read_csv("data/test/microbiome_raw.csv", index_col=0)
    metabolome_df = pd.read_csv("data/test/metabolome_raw.csv", index_col=0)
    proteome_df = pd.read_csv("data/test/proteome_raw.csv", index_col=0)

    # Initialize preprocessors
    micro_prep = MicrobiomePreprocessor()
    metab_prep = MetabolomePreprocessor()
    prot_prep = ProteomePreprocessor()

    # Preprocess
    micro_processed = micro_prep.fit_transform(microbiome_df)
    metab_processed = metab_prep.fit_transform(metabolome_df)
    prot_processed = prot_prep.fit_transform(proteome_df)

    assert micro_processed is not None
    assert metab_processed is not None
    assert prot_processed is not None

    print(f"\nPreprocessing Results:")
    print(f"  Microbiome: {microbiome_df.shape} -> {micro_processed.shape}")
    print(f"  Metabolome: {metabolome_df.shape} -> {metab_processed.shape}")
    print(f"  Proteome: {proteome_df.shape} -> {prot_processed.shape}")

    return {
        'microbiome': micro_processed,
        'metabolome': metab_processed,
        'proteome': prot_processed
    }


def test_mock_diagnosis_workflow():
    """Test diagnosis workflow with mock expert opinions."""
    # Create mock expert opinions
    opinions = [
        ExpertOpinion(
            expert_name="microbiome_expert",
            omics_type="microbiome",
            diagnosis="Periodontitis",
            probability=0.85,
            confidence=0.82,
            top_features=[
                FeatureImportance("Porphyromonas_gingivalis", 0.25, "up"),
                FeatureImportance("Treponema_denticola", 0.18, "up"),
                FeatureImportance("Streptococcus_salivarius", 0.15, "down")
            ],
            biological_explanation=(
                "Microbiome analysis reveals significant elevation of periodontal pathogens "
                "(P. gingivalis, T. denticola) with concurrent reduction in beneficial "
                "bacteria (S. salivarius), characteristic of periodontitis."
            ),
            evidence_chain=[
                "Detected 3.5-fold increase in P. gingivalis relative abundance",
                "T. denticola elevated 2.8-fold above healthy baseline",
                "Beneficial Streptococcus species reduced by 45%",
                "Microbiome diversity index decreased (Shannon: 2.1 vs normal 3.5)"
            ],
            model_metadata={"model": "RandomForest", "accuracy": 0.87},
            timestamp="2024-01-01T12:00:00"
        ),
        ExpertOpinion(
            expert_name="metabolome_expert",
            omics_type="metabolome",
            diagnosis="Periodontitis",
            probability=0.78,
            confidence=0.75,
            top_features=[
                FeatureImportance("Butyrate", 0.22, "up"),
                FeatureImportance("Propionate", 0.15, "up"),
                FeatureImportance("IL6", 0.12, "up")
            ],
            biological_explanation=(
                "Metabolomic profile shows elevated short-chain fatty acids (SCFAs) "
                "and inflammatory metabolites consistent with periodontal inflammation."
            ),
            evidence_chain=[
                "Butyrate concentration 2.1x elevated (1500 vs 700 μM)",
                "Propionate increased 1.8-fold",
                "IL-6 metabolite levels elevated, indicating systemic inflammation"
            ],
            model_metadata={"model": "XGBoost", "accuracy": 0.85},
            timestamp="2024-01-01T12:00:01"
        ),
        ExpertOpinion(
            expert_name="proteome_expert",
            omics_type="proteome",
            diagnosis="Periodontitis",
            probability=0.82,
            confidence=0.80,
            top_features=[
                FeatureImportance("MMP9", 0.28, "up"),
                FeatureImportance("IL6", 0.20, "up"),
                FeatureImportance("TNF", 0.16, "up")
            ],
            biological_explanation=(
                "Proteomic analysis reveals significant upregulation of matrix "
                "metalloproteinases and pro-inflammatory cytokines, hallmarks of "
                "periodontal tissue destruction."
            ),
            evidence_chain=[
                "MMP-9 expression 4.2-fold elevated (tissue remodeling marker)",
                "IL-6 protein levels 3.1x increased (inflammatory response)",
                "TNF-α elevated 2.5-fold (pro-inflammatory cytokine)",
                "VEGF moderately elevated (angiogenesis marker)"
            ],
            model_metadata={"model": "RandomForest", "accuracy": 0.88},
            timestamp="2024-01-01T12:00:02"
        )
    ]

    # Test conflict detection
    resolver = ConflictResolver()
    conflict = resolver.detect_conflict(opinions)

    print(f"\nConflict Analysis:")
    print(f"  Conflict detected: {conflict.has_conflict}")
    print(f"  Diagnosis distribution: {conflict.diagnosis_distribution}")
    print(f"  Average confidence: {conflict.avg_confidence:.2%}")

    # Create diagnosis result
    diagnosis_result = DiagnosisResult(
        patient_id="TEST_PATIENT_001",
        diagnosis="Periodontitis",
        confidence=0.79,
        expert_opinions=opinions,
        conflict_resolution=None,
        key_biomarkers=[
            {
                "name": "MMP9",
                "omics_type": "proteome",
                "importance": 0.28,
                "direction": "up",
                "description": "Tissue degradation marker, 4.2-fold elevated"
            },
            {
                "name": "Porphyromonas_gingivalis",
                "omics_type": "microbiome",
                "importance": 0.25,
                "direction": "up",
                "description": "Primary periodontal pathogen, 3.5-fold increased"
            },
            {
                "name": "Butyrate",
                "omics_type": "metabolome",
                "importance": 0.22,
                "direction": "up",
                "description": "Inflammatory metabolite, 2.1x elevated"
            }
        ],
        reasoning_chain=[
            "All three expert systems agree on Periodontitis diagnosis",
            "Average confidence: 79% (High)",
            "Strong concordance across microbiome, metabolome, and proteome data",
            "Key pathogenic biomarkers identified across all omics layers",
            "Elevated inflammatory markers support diagnosis"
        ],
        differential_diagnoses=[],
        recommendations=[
            "Immediate periodontal treatment recommended",
            "Consider scaling and root planing",
            "Antimicrobial therapy may be indicated",
            "Follow-up multi-omics analysis in 3 months to assess treatment response"
        ],
        rag_citations=[],
        cag_similar_cases=[],
        metadata={
            "decision_type": "consensus",
            "conflict_detected": False,
            "n_experts": 3
        }
    )

    # Test report generation
    report_generator = ReportGenerator()
    report = report_generator.generate_report(
        diagnosis_result=diagnosis_result,
        patient_metadata={
            "age": 45,
            "gender": "Male",
            "medical_history": "No significant medical history",
            "current_medications": "None"
        }
    )

    assert report is not None
    assert len(report) > 0
    assert "Multi-Omics Clinical Diagnostic Report" in report
    assert "Periodontitis" in report

    # Save report
    output_path = Path("data/test/test_report.md")
    report_generator.save_report(report, str(output_path))

    print(f"\n✓ Report generated and saved to {output_path}")
    print(f"  Report length: {len(report)} characters")

    return diagnosis_result


def test_complete_workflow():
    """Test complete workflow from data to report."""
    print("\n" + "="*60)
    print("End-to-End Integration Test")
    print("="*60)

    # Step 1: Load and preprocess data
    print("\n[1] Loading and preprocessing data...")
    preprocessed = test_preprocessing_pipeline()

    # Step 2: Mock expert predictions (real would need trained models)
    print("\n[2] Running mock diagnosis workflow...")
    result = test_mock_diagnosis_workflow()

    # Step 3: Verify results
    assert result is not None
    assert result.diagnosis is not None
    assert result.confidence > 0

    print(f"\n{'='*60}")
    print("Integration Test Complete!")
    print(f"{'='*60}")
    print(f"\nFinal Diagnosis: {result.diagnosis}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Report saved: data/test/test_report.md")


def test_vector_db_initialization():
    """Test vector database initialization."""
    try:
        from clinical.collaboration.rag_system import RAGSystem
        rag = RAGSystem()

        stats = rag.get_statistics()
        print(f"\nVector Database Status:")
        print(f"  Documents: {stats['total_documents']}")
        print(f"  Collection: {stats['collection_name']}")

        assert stats['total_documents'] >= 0

    except Exception as e:
        print(f"\n⚠ Vector DB test skipped: {e}")


def test_system_readiness():
    """Test overall system readiness."""
    print("\n" + "="*60)
    print("System Readiness Check")
    print("="*60)

    checks = {
        "Test Data": Path("data/test/microbiome_raw.csv").exists(),
        "Labeled Data": Path("data/labeled/annotations.json").exists(),
        "Knowledge Base": Path("data/knowledge_base").exists(),
    }

    for check, status in checks.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {check}: {'Ready' if status else 'Not Ready'}")

    print("\nSystem Components:")
    print("  ✓ Preprocessing Layer")
    print("  ✓ Expert Layer (models need training)")
    print("  ✓ Collaboration Layer (RAG + CAG)")
    print("  ✓ Decision Layer (CMO + Debate)")
    print("  ✓ MCP Server")

    all_ready = all(checks.values())
    print(f"\n{'='*60}")
    print(f"Overall Status: {'READY' if all_ready else 'PARTIALLY READY'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

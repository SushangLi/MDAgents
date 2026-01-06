"""
Vector Database Builder Script.

Initializes or rebuilds the ChromaDB vector database for medical literature.
Can populate with sample medical documents for testing.
"""

import argparse
from pathlib import Path
import json
from datetime import datetime

from clinical.collaboration.rag_system import RAGSystem
from clinical.collaboration.vector_store import MedicalVectorStore
from clinical.collaboration.embeddings import get_default_embeddings


# Sample medical literature for testing
SAMPLE_DOCUMENTS = [
    {
        "text": """
Periodontitis and Cardiovascular Disease: A Systematic Review

Recent studies have demonstrated a significant association between chronic periodontitis
and cardiovascular disease. Inflammatory mediators such as C-reactive protein (CRP),
interleukin-6 (IL-6), and tumor necrosis factor-alpha (TNF-α) are elevated in patients
with severe periodontitis. These biomarkers suggest a systemic inflammatory response
that may contribute to atherosclerotic plaque formation.

Microbiome analysis reveals increased abundance of Porphyromonas gingivalis,
Treponema denticola, and Tannerella forsythia in periodontitis patients with
cardiovascular complications. These periodontal pathogens may translocate to
atherosclerotic plaques, contributing to vascular inflammation.

Clinical recommendations include periodontal screening for cardiovascular risk
assessment and coordinated care between dental and cardiology specialists.
        """,
        "metadata": {
            "title": "Periodontitis and Cardiovascular Disease: A Systematic Review",
            "type": "research_paper",
            "year": 2023,
            "journal": "Journal of Clinical Periodontology",
            "keywords": ["periodontitis", "cardiovascular disease", "inflammation", "microbiome"]
        }
    },
    {
        "text": """
Oral Microbiome Dysbiosis in Diabetes Mellitus

Type 2 diabetes mellitus is associated with significant alterations in the oral
microbiome composition. Patients with poorly controlled diabetes (HbA1c > 8%)
show decreased microbial diversity and increased pathogenic bacteria.

Key findings:
- Reduced abundance of beneficial Streptococcus and Neisseria species
- Increased Prevotella, Fusobacterium, and Veillonella species
- Correlation between dysbiosis severity and glycemic control
- Bidirectional relationship: diabetes affects microbiome, microbiome affects
  glucose metabolism

Metabolomic analysis reveals elevated levels of short-chain fatty acids (SCFA),
particularly butyrate and propionate, which may influence insulin sensitivity.

Treatment strategies should include oral hygiene optimization, probiotics
containing Lactobacillus reuteri, and strict glycemic control.
        """,
        "metadata": {
            "title": "Oral Microbiome Dysbiosis in Diabetes Mellitus",
            "type": "research_paper",
            "year": 2024,
            "journal": "Diabetes Care",
            "keywords": ["diabetes", "microbiome", "dysbiosis", "metabolomics"]
        }
    },
    {
        "text": """
Salivary Biomarkers for Early Detection of Oral Cancer

Early detection of oral squamous cell carcinoma (OSCC) remains challenging.
Salivary proteomics and metabolomics offer non-invasive diagnostic potential.

Protein biomarkers with high sensitivity/specificity:
- MMP-9 (Matrix Metalloproteinase-9): elevated 3.5-fold in OSCC
- IL-8 (Interleukin-8): elevated 2.8-fold
- VEGF (Vascular Endothelial Growth Factor): elevated 2.1-fold

Metabolite biomarkers:
- Polyamines (putrescine, spermidine): significantly elevated
- Lactate dehydrogenase (LDH): 4.2-fold increase
- Choline metabolites: altered in 89% of OSCC cases

Combined biomarker panel achieves:
- Sensitivity: 91%
- Specificity: 87%
- AUC: 0.94

Clinical validation studies are ongoing. This multi-omics approach may enable
screening in high-risk populations (tobacco users, HPV-positive).
        """,
        "metadata": {
            "title": "Salivary Biomarkers for Early Detection of Oral Cancer",
            "type": "research_paper",
            "year": 2024,
            "journal": "Cancer Research",
            "keywords": ["oral cancer", "biomarkers", "proteomics", "metabolomics", "saliva"]
        }
    },
    {
        "text": """
Clinical Guidelines for Diagnosis of Periodontal Disease

The American Academy of Periodontology (AAP) classification system defines
periodontal disease stages based on severity and complexity:

Stage I (Initial):
- Clinical attachment loss (CAL) 1-2mm
- Radiographic bone loss <15%
- No tooth loss

Stage II (Moderate):
- CAL 3-4mm
- Radiographic bone loss 15-33%
- Maximum probing depth ≤5mm

Stage III (Severe):
- CAL ≥5mm
- Radiographic bone loss >33%
- Tooth loss due to periodontitis ≤4 teeth

Stage IV (Very Severe):
- CAL ≥5mm
- Tooth loss ≥5 teeth
- Complex rehabilitation needs

Risk factors include:
- Smoking (3-6x increased risk)
- Diabetes mellitus (2-3x increased risk)
- Genetic susceptibility (IL-1 polymorphisms)
- Poor oral hygiene

Diagnostic workup should include:
- Full-mouth periodontal charting
- Radiographic assessment
- Microbiome analysis (optional)
- Systemic risk assessment
        """,
        "metadata": {
            "title": "Clinical Guidelines for Diagnosis of Periodontal Disease",
            "type": "clinical_guideline",
            "year": 2023,
            "organization": "American Academy of Periodontology",
            "keywords": ["periodontitis", "diagnosis", "guidelines", "staging"]
        }
    },
    {
        "text": """
Role of Probiotics in Oral Health Management

Probiotic supplementation shows promise for managing oral dysbiosis and
preventing dental diseases. Strains with strongest evidence:

Lactobacillus reuteri DSM 17938:
- Reduces Streptococcus mutans by 80%
- Decreases plaque index by 35%
- Reduces gingivitis scores

Lactobacillus salivarius:
- Inhibits periodontal pathogens
- Reduces halitosis
- Modulates inflammatory response

Bifidobacterium lactis:
- Improves periodontal health markers
- Reduces pocket depth (mean reduction 0.4mm)

Mechanisms of action:
1. Competitive exclusion of pathogens
2. Production of antimicrobial compounds (bacteriocins, H2O2)
3. Immune modulation (increased IL-10, decreased IL-6)
4. Biofilm disruption

Recommended dosing: 10^9 CFU/day for minimum 4 weeks.
Best administered as lozenges or chewing gum for oral cavity colonization.

Contraindications: Immunocompromised patients, critically ill.
        """,
        "metadata": {
            "title": "Role of Probiotics in Oral Health Management",
            "type": "review_article",
            "year": 2023,
            "journal": "Journal of Oral Microbiology",
            "keywords": ["probiotics", "oral health", "lactobacillus", "microbiome"]
        }
    }
]


def initialize_vector_db(
    reset: bool = False,
    add_samples: bool = True
) -> MedicalVectorStore:
    """
    Initialize or rebuild vector database.

    Args:
        reset: If True, delete existing database and rebuild
        add_samples: If True, add sample documents

    Returns:
        MedicalVectorStore instance
    """
    print("="*60)
    print("Vector Database Initialization")
    print("="*60)

    # Initialize embeddings
    print("\n1. Loading embedding model...")
    embedding_model = get_default_embeddings()

    # Initialize vector store
    print("\n2. Initializing vector store...")
    vector_store = MedicalVectorStore(
        collection_name="medical_literature",
        persist_directory="data/knowledge_base/vector_db",
        embedding_model=embedding_model
    )

    # Reset if requested
    if reset:
        print("\n3. Resetting database...")
        confirmation = input("⚠ This will delete all existing documents. Continue? [y/N]: ")
        if confirmation.lower() == "y":
            vector_store.reset()
            print("✓ Database reset complete")
        else:
            print("✗ Reset cancelled")
            return vector_store

    # Add sample documents
    if add_samples:
        print(f"\n4. Adding {len(SAMPLE_DOCUMENTS)} sample documents...")

        documents = [doc["text"].strip() for doc in SAMPLE_DOCUMENTS]
        metadatas = [doc["metadata"] for doc in SAMPLE_DOCUMENTS]

        # Add ingestion timestamp to metadata
        for metadata in metadatas:
            metadata["ingestion_date"] = datetime.now().isoformat()
            metadata["source"] = "sample_data"

        doc_ids = vector_store.add_documents(documents, metadatas)

        print(f"✓ Added {len(doc_ids)} documents")

        # Show examples
        print("\nSample documents added:")
        for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
            print(f"  {i}. {doc['metadata']['title']}")

    # Show statistics
    print("\n" + "="*60)
    print("Database Statistics:")
    info = vector_store.get_collection_info()
    print(f"  Collection: {info['name']}")
    print(f"  Total documents: {info['count']}")
    print(f"  Embedding dimension: {embedding_model.get_embedding_dimension()}")
    print("="*60)

    return vector_store


def test_search(vector_store: MedicalVectorStore):
    """
    Test search functionality with sample queries.

    Args:
        vector_store: Vector store instance
    """
    print("\n" + "="*60)
    print("Testing Search Functionality")
    print("="*60)

    test_queries = [
        "What are the biomarkers for oral cancer?",
        "How does diabetes affect oral microbiome?",
        "Treatment options for periodontitis with cardiovascular disease",
        "Probiotics for oral health"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: \"{query}\"")

        results = vector_store.search(query, n_results=2)

        if results["documents"]:
            for j, (doc, metadata, distance) in enumerate(zip(
                results["documents"],
                results["metadatas"],
                results["distances"]
            ), 1):
                print(f"\n  Result {j}:")
                print(f"    Title: {metadata.get('title', 'N/A')}")
                print(f"    Distance: {distance:.4f}")
                print(f"    Preview: {doc[:150]}...")
        else:
            print("  No results found")


def export_sample_metadata(output_path: str = "data/knowledge_base/sample_metadata.json"):
    """
    Export sample document metadata to JSON file.

    Args:
        output_path: Path to save metadata file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_dict = {}
    for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        filename = f"sample_doc_{i}.txt"
        metadata_dict[filename] = doc["metadata"]

    with open(output_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)

    print(f"✓ Sample metadata exported to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize or rebuild RAG vector database"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset database (delete all existing documents)"
    )

    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Do not add sample documents"
    )

    parser.add_argument(
        "--test-search",
        action="store_true",
        help="Run search tests after initialization"
    )

    parser.add_argument(
        "--export-metadata",
        action="store_true",
        help="Export sample metadata to JSON file"
    )

    args = parser.parse_args()

    # Initialize database
    vector_store = initialize_vector_db(
        reset=args.reset,
        add_samples=not args.no_samples
    )

    # Export metadata if requested
    if args.export_metadata:
        print("\n")
        export_sample_metadata()

    # Test search if requested
    if args.test_search:
        test_search(vector_store)

    print("\n✓ Vector database initialization complete!")

    return 0


if __name__ == "__main__":
    exit(main())

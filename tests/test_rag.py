"""
Test RAG System.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clinical.collaboration.rag_system import RAGSystem
from clinical.collaboration.embeddings import get_default_embeddings


def test_rag_initialization():
    """Test RAG system initialization."""
    rag = RAGSystem()
    assert rag is not None
    assert rag.vector_store is not None
    assert rag.embedding_model is not None


def test_rag_search():
    """Test RAG search functionality."""
    rag = RAGSystem()

    # Search for a query
    results = rag.search(
        query="What are the biomarkers for oral cancer?",
        top_k=3,
        min_relevance=0.3
    )

    assert results is not None
    assert hasattr(results, 'documents')
    assert hasattr(results, 'relevance_scores')

    print(f"\nSearch Results:")
    print(f"  Query: What are the biomarkers for oral cancer?")
    print(f"  Retrieved: {len(results.documents)} documents")

    if results.documents:
        for i, (doc, score) in enumerate(zip(results.documents, results.relevance_scores)):
            print(f"\n  Result {i+1} (score: {score:.2%}):")
            print(f"    {doc[:150]}...")


def test_rag_format_context():
    """Test RAG context formatting."""
    rag = RAGSystem()

    results = rag.search(
        query="periodontitis treatment",
        top_k=2
    )

    context = rag.format_context_for_llm(results)

    assert isinstance(context, str)
    assert len(context) > 0

    print(f"\nFormatted Context Length: {len(context)} characters")


def test_embeddings():
    """Test embeddings generation."""
    embeddings = get_default_embeddings()

    # Encode a query
    query = "Oral microbiome dysbiosis"
    embedding = embeddings.encode_query(query)

    assert embedding is not None
    assert len(embedding.shape) == 1
    assert embedding.shape[0] > 0

    print(f"\nEmbedding dimension: {embedding.shape[0]}")


def test_rag_statistics():
    """Test RAG statistics."""
    rag = RAGSystem()

    stats = rag.get_statistics()

    assert "total_documents" in stats
    assert "collection_name" in stats

    print(f"\nRAG Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

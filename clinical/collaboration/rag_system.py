"""
RAG (Retrieval-Augmented Generation) System.

Provides semantic search over medical literature to support
expert opinion conflict resolution.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from clinical.collaboration.vector_store import MedicalVectorStore
from clinical.collaboration.embeddings import BiomedicalEmbeddings
from clinical.models.expert_opinion import ExpertOpinion


@dataclass
class RAGResult:
    """Result from RAG retrieval."""

    query: str
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    relevance_scores: List[float]
    document_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "documents": self.documents,
            "metadatas": self.metadatas,
            "relevance_scores": self.relevance_scores,
            "document_ids": self.document_ids
        }

    def get_top_k(self, k: int = 3) -> "RAGResult":
        """Get top-k results."""
        return RAGResult(
            query=self.query,
            documents=self.documents[:k],
            metadatas=self.metadatas[:k],
            relevance_scores=self.relevance_scores[:k],
            document_ids=self.document_ids[:k]
        )


class RAGSystem:
    """
    Retrieval-Augmented Generation system for medical literature.

    Provides semantic search and context retrieval for clinical decision support.
    """

    def __init__(
        self,
        vector_store: Optional[MedicalVectorStore] = None,
        embedding_model: Optional[BiomedicalEmbeddings] = None
    ):
        """
        Initialize RAG system.

        Args:
            vector_store: Vector store instance
            embedding_model: Embeddings model
        """
        # Initialize components
        if embedding_model is None:
            from clinical.collaboration.embeddings import get_default_embeddings
            self.embedding_model = get_default_embeddings()
        else:
            self.embedding_model = embedding_model

        if vector_store is None:
            self.vector_store = MedicalVectorStore(
                embedding_model=self.embedding_model
            )
        else:
            self.vector_store = vector_store

        print(f"✓ RAG system initialized with {self.vector_store.count()} documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_relevance: float = 0.5
    ) -> RAGResult:
        """
        Search for relevant medical literature.

        Args:
            query: Search query
            top_k: Number of top results to return
            metadata_filter: Filter by metadata (e.g., {"type": "clinical_guideline"})
            min_relevance: Minimum relevance score (0-1)

        Returns:
            RAGResult with retrieved documents and metadata
        """
        # Search vector store
        results = self.vector_store.search(
            query=query,
            n_results=top_k,
            where=metadata_filter
        )

        # Convert distances to relevance scores (1 - normalized_distance)
        if results["distances"]:
            max_dist = max(results["distances"]) if results["distances"] else 1.0
            relevance_scores = [
                1.0 - (dist / (max_dist + 1e-8))
                for dist in results["distances"]
            ]
        else:
            relevance_scores = []

        # Filter by minimum relevance
        filtered_indices = [
            i for i, score in enumerate(relevance_scores)
            if score >= min_relevance
        ]

        return RAGResult(
            query=query,
            documents=[results["documents"][i] for i in filtered_indices],
            metadatas=[results["metadatas"][i] for i in filtered_indices],
            relevance_scores=[relevance_scores[i] for i in filtered_indices],
            document_ids=[results["ids"][i] for i in filtered_indices]
        )

    def build_conflict_query(
        self,
        conflicting_opinions: List[ExpertOpinion]
    ) -> str:
        """
        Build a query to resolve expert opinion conflicts.

        Args:
            conflicting_opinions: List of conflicting expert opinions

        Returns:
            Query string for RAG retrieval
        """
        # Extract key information from opinions
        diagnoses = [op.diagnosis for op in conflicting_opinions]
        omics_types = [op.omics_type for op in conflicting_opinions]

        # Build query
        query_parts = []

        # Add diagnosis conflict
        unique_diagnoses = list(set(diagnoses))
        if len(unique_diagnoses) > 1:
            query_parts.append(
                f"Differential diagnosis between {' and '.join(unique_diagnoses)}"
            )

        # Add omics context
        query_parts.append(
            f"based on {', '.join(set(omics_types))} analysis"
        )

        # Add key biomarkers from each expert
        for opinion in conflicting_opinions:
            if opinion.top_features:
                top_feature = opinion.top_features[0]
                query_parts.append(
                    f"{opinion.omics_type} showing {top_feature.direction}regulation of {top_feature.feature_name}"
                )

        query = ". ".join(query_parts)

        return query

    def retrieve_for_conflict(
        self,
        conflicting_opinions: List[ExpertOpinion],
        top_k: int = 5
    ) -> RAGResult:
        """
        Retrieve relevant literature for resolving expert conflicts.

        Args:
            conflicting_opinions: List of conflicting expert opinions
            top_k: Number of documents to retrieve

        Returns:
            RAG results with relevant literature
        """
        # Build query from conflicting opinions
        query = self.build_conflict_query(conflicting_opinions)

        # Search
        results = self.search(query, top_k=top_k)

        return results

    def format_context_for_llm(
        self,
        rag_result: RAGResult,
        max_context_length: int = 2000
    ) -> str:
        """
        Format RAG results as context for LLM.

        Args:
            rag_result: RAG retrieval results
            max_context_length: Maximum context length in characters

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        context_parts.append("## Relevant Medical Literature\n")

        for i, (doc, metadata, score) in enumerate(zip(
            rag_result.documents,
            rag_result.metadatas,
            rag_result.relevance_scores
        )):
            # Format document
            doc_str = f"\n### Document {i+1} (Relevance: {score:.2%})\n"

            # Add metadata if available
            if metadata:
                if "title" in metadata:
                    doc_str += f"**Title**: {metadata['title']}\n"
                if "year" in metadata:
                    doc_str += f"**Year**: {metadata['year']}\n"
                if "doi" in metadata:
                    doc_str += f"**DOI**: {metadata['doi']}\n"

            doc_str += f"\n{doc}\n"

            # Check length
            if current_length + len(doc_str) > max_context_length:
                break

            context_parts.append(doc_str)
            current_length += len(doc_str)

        return "".join(context_parts)

    def add_literature(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add medical literature to the knowledge base.

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document

        Returns:
            List of document IDs
        """
        return self.vector_store.add_documents(documents, metadatas)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get RAG system statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": self.vector_store.count(),
            "collection_name": self.vector_store.collection_name,
            "embedding_model": str(self.embedding_model)
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"RAGSystem(documents={self.vector_store.count()})"

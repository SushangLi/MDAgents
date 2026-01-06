"""
Vector Store Module.

Provides interface to ChromaDB for storing and retrieving medical literature embeddings.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
import uuid

from clinical.collaboration.embeddings import BiomedicalEmbeddings


class MedicalVectorStore:
    """
    Vector store for medical literature using ChromaDB.

    Stores document embeddings and supports semantic search.
    """

    def __init__(
        self,
        collection_name: str = "medical_literature",
        persist_directory: str = "data/knowledge_base/vector_db",
        embedding_model: Optional[BiomedicalEmbeddings] = None
    ):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_model: Embeddings model (default: PubMedBERT)
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        if embedding_model is None:
            from clinical.collaboration.embeddings import get_default_embeddings
            self.embedding_model = get_default_embeddings()
        else:
            self.embedding_model = embedding_model

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Medical literature for clinical diagnosis"}
        )

        print(f"✓ Vector store initialized: {collection_name}")
        print(f"  Documents: {self.collection.count()}")

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional document IDs (auto-generated if not provided)

        Returns:
            List of document IDs
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_model.encode_documents(documents)

        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas if metadatas else [{}] * len(documents),
            ids=ids
        )

        print(f"✓ Added {len(documents)} documents to vector store")
        return ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            n_results: Number of results to return
            where: Metadata filter (e.g., {"type": "research_paper"})
            where_document: Document content filter

        Returns:
            Dictionary with:
            - documents: List of document texts
            - metadatas: List of metadata dicts
            - distances: List of distance scores
            - ids: List of document IDs
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode_query(query)

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            where_document=where_document
        )

        # Format results
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Dictionary with document, metadata, and embedding
        """
        results = self.collection.get(
            ids=[doc_id],
            include=["documents", "metadatas", "embeddings"]
        )

        if not results["ids"]:
            return None

        return {
            "id": results["ids"][0],
            "document": results["documents"][0],
            "metadata": results["metadatas"][0],
            "embedding": results["embeddings"][0] if results["embeddings"] else None
        }

    def delete_documents(self, ids: List[str]):
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete
        """
        self.collection.delete(ids=ids)
        print(f"✓ Deleted {len(ids)} documents")

    def update_document(
        self,
        doc_id: str,
        document: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update a document.

        Args:
            doc_id: Document ID
            document: New document text (optional)
            metadata: New metadata (optional)
        """
        update_dict = {"ids": [doc_id]}

        if document is not None:
            # Regenerate embedding
            embedding = self.embedding_model.encode_query(document)
            update_dict["documents"] = [document]
            update_dict["embeddings"] = [embedding.tolist()]

        if metadata is not None:
            update_dict["metadatas"] = [metadata]

        self.collection.update(**update_dict)
        print(f"✓ Updated document {doc_id}")

    def count(self) -> int:
        """Get total number of documents."""
        return self.collection.count()

    def reset(self):
        """Delete all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Medical literature for clinical diagnosis"}
        )
        print(f"✓ Reset collection: {self.collection_name}")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection statistics
        """
        return {
            "name": self.collection_name,
            "count": self.count(),
            "metadata": self.collection.metadata
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"MedicalVectorStore(collection='{self.collection_name}', documents={self.count()})"

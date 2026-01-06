"""
Embeddings Module.

Provides embedding generation for medical literature using PubMedBERT
or other biomedical language models.
"""

from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class BiomedicalEmbeddings:
    """
    Biomedical text embeddings using PubMedBERT or similar models.

    This class provides a unified interface for generating embeddings
    from medical literature and clinical text.
    """

    def __init__(
        self,
        model_name: str = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
        device: Optional[str] = None
    ):
        """
        Initialize embeddings model.

        Args:
            model_name: Name of the sentence-transformers model
                       Default: PubMedBERT fine-tuned for semantic similarity
                       Alternatives:
                       - "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
                       - "dmis-lab/biobert-v1.1"
                       - "allenai/scibert_scivocab_uncased"
                       - "moka-ai/m3e-base" (for Chinese medical text)
            device: Device to run the model on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.device = device

        # Load model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✓ Model loaded successfully")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            normalize_embeddings: Whether to L2-normalize embeddings

        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )

        return embeddings

    def encode_query(
        self,
        query: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode a single query for retrieval.

        Args:
            query: Query text
            normalize: Whether to normalize the embedding

        Returns:
            Embedding vector
        """
        return self.encode(query, normalize_embeddings=normalize)[0]

    def encode_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode multiple documents for indexing.

        Args:
            documents: List of document texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings
        """
        return self.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )

    def compute_similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between two texts or embeddings.

        Args:
            text1: First text or embedding
            text2: Second text or embedding

        Returns:
            Cosine similarity score (0-1)
        """
        # Encode if needed
        if isinstance(text1, str):
            emb1 = self.encode_query(text1)
        else:
            emb1 = text1

        if isinstance(text2, str):
            emb2 = self.encode_query(text2)
        else:
            emb2 = text2

        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        """String representation."""
        return f"BiomedicalEmbeddings(model='{self.model_name}', dim={self.get_embedding_dimension()})"


# Convenience function for quick embeddings
def get_default_embeddings(device: Optional[str] = None) -> BiomedicalEmbeddings:
    """
    Get default biomedical embeddings model.

    Args:
        device: Device to run on

    Returns:
        BiomedicalEmbeddings instance
    """
    return BiomedicalEmbeddings(device=device)

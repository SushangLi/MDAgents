"""
Medical Literature Ingestion Script.

Processes medical literature (PDF, text) and ingests them into
the RAG vector database for semantic search.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime

# PDF processing
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    print("⚠ PyPDF2 not installed. PDF support disabled.")
    print("  Install: pip install PyPDF2")
    PDF_SUPPORT = False

from clinical.collaboration.rag_system import RAGSystem
from clinical.collaboration.vector_store import MedicalVectorStore
from clinical.collaboration.embeddings import get_default_embeddings


class LiteratureIngester:
    """
    Ingests medical literature into vector database.

    Supports:
    - PDF files
    - Text files
    - Metadata extraction
    - Document chunking
    """

    def __init__(
        self,
        rag_system: Optional[RAGSystem] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize literature ingester.

        Args:
            rag_system: RAG system instance (creates if None)
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize RAG system
        if rag_system is None:
            print("Initializing RAG system...")
            self.rag_system = RAGSystem()
        else:
            self.rag_system = rag_system

        print(f"✓ Literature ingester initialized")
        print(f"  Chunk size: {chunk_size} characters")
        print(f"  Chunk overlap: {chunk_overlap} characters")

    def ingest_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Ingest a single file.

        Args:
            file_path: Path to file
            metadata: Optional metadata (title, authors, year, doi, etc.)

        Returns:
            List of document IDs
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"\nProcessing: {file_path.name}")

        # Extract text
        if file_path.suffix.lower() == ".pdf":
            text = self._extract_pdf(file_path)
        elif file_path.suffix.lower() in [".txt", ".md"]:
            text = self._extract_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        if not text.strip():
            print(f"⚠ No text extracted from {file_path.name}")
            return []

        # Chunk text
        chunks = self._chunk_text(text)
        print(f"  Split into {len(chunks)} chunks")

        # Prepare metadata
        if metadata is None:
            metadata = {}

        # Add default metadata
        metadata.setdefault("source_file", str(file_path))
        metadata.setdefault("ingestion_date", datetime.now().isoformat())
        metadata.setdefault("type", "medical_literature")

        # Create metadata for each chunk
        chunk_metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = i
            chunk_meta["total_chunks"] = len(chunks)
            chunk_metadatas.append(chunk_meta)

        # Add to vector store
        doc_ids = self.rag_system.add_literature(chunks, chunk_metadatas)

        print(f"✓ Ingested {len(doc_ids)} chunks from {file_path.name}")
        return doc_ids

    def ingest_directory(
        self,
        directory_path: str,
        pattern: str = "*.pdf",
        metadata_file: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Ingest all files in a directory.

        Args:
            directory_path: Path to directory
            pattern: File glob pattern (e.g., "*.pdf", "*.txt")
            metadata_file: Optional JSON file with metadata per file

        Returns:
            Dictionary mapping filename to document IDs
        """
        directory_path = Path(directory_path)

        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        # Load metadata file if provided
        file_metadatas = {}
        if metadata_file:
            metadata_file = Path(metadata_file)
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    file_metadatas = json.load(f)
                print(f"✓ Loaded metadata for {len(file_metadatas)} files")

        # Find files
        files = list(directory_path.glob(pattern))
        print(f"\nFound {len(files)} files matching '{pattern}'")

        # Ingest each file
        results = {}
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]")

            # Get metadata for this file
            metadata = file_metadatas.get(file_path.name, {})

            try:
                doc_ids = self.ingest_file(file_path, metadata)
                results[file_path.name] = doc_ids
            except Exception as e:
                print(f"✗ Error processing {file_path.name}: {e}")
                results[file_path.name] = []

        # Summary
        total_chunks = sum(len(ids) for ids in results.values())
        successful = sum(1 for ids in results.values() if ids)
        failed = len(files) - successful

        print(f"\n{'='*60}")
        print(f"Ingestion Summary:")
        print(f"  Total files: {len(files)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total chunks: {total_chunks}")
        print(f"{'='*60}")

        return results

    def _extract_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text
        """
        if not PDF_SUPPORT:
            raise RuntimeError("PDF support not available. Install PyPDF2.")

        text_parts = []

        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                text_parts.append(text)

        full_text = "\n\n".join(text_parts)

        # Clean text
        full_text = self._clean_text(full_text)

        return full_text

    def _extract_text(self, file_path: Path) -> str:
        """
        Extract text from text file.

        Args:
            file_path: Path to text file

        Returns:
            File content
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Clean text
        text = self._clean_text(text)

        return text

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove excessive whitespace
        text = re.sub(r" {2,}", " ", text)

        # Remove page numbers (common pattern)
        text = re.sub(r"\n\d+\n", "\n", text)

        # Strip
        text = text.strip()

        return text

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Full text

        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk
                # Include overlap from previous chunk
                if chunks and self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest medical literature into RAG vector database"
    )

    parser.add_argument(
        "input",
        help="Path to file or directory to ingest"
    )

    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="File pattern for directory ingestion (default: *.pdf)"
    )

    parser.add_argument(
        "--metadata",
        help="Path to JSON file with metadata per file"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Maximum characters per chunk (default: 500)"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks (default: 50)"
    )

    parser.add_argument(
        "--title",
        help="Document title (for single file)"
    )

    parser.add_argument(
        "--authors",
        help="Authors (comma-separated)"
    )

    parser.add_argument(
        "--year",
        type=int,
        help="Publication year"
    )

    parser.add_argument(
        "--doi",
        help="DOI"
    )

    parser.add_argument(
        "--type",
        default="medical_literature",
        help="Document type (default: medical_literature)"
    )

    args = parser.parse_args()

    # Initialize ingester
    ingester = LiteratureIngester(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Prepare metadata for single file
    metadata = None
    if any([args.title, args.authors, args.year, args.doi]):
        metadata = {
            "type": args.type
        }
        if args.title:
            metadata["title"] = args.title
        if args.authors:
            metadata["authors"] = args.authors
        if args.year:
            metadata["year"] = args.year
        if args.doi:
            metadata["doi"] = args.doi

    # Ingest
    input_path = Path(args.input)

    if input_path.is_file():
        # Single file
        doc_ids = ingester.ingest_file(input_path, metadata)
        print(f"\n✓ Successfully ingested {len(doc_ids)} chunks")

    elif input_path.is_dir():
        # Directory
        results = ingester.ingest_directory(
            input_path,
            pattern=args.pattern,
            metadata_file=args.metadata
        )

        # Save results
        results_file = Path("data/knowledge_base/ingestion_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "input_directory": str(input_path),
                    "pattern": args.pattern,
                    "results": results
                },
                f,
                indent=2
            )

        print(f"\n✓ Results saved to {results_file}")

    else:
        print(f"✗ Invalid input: {input_path}")
        return 1

    # Show vector store statistics
    stats = ingester.rag_system.get_statistics()
    print(f"\nVector Store Statistics:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Collection: {stats['collection_name']}")

    return 0


if __name__ == "__main__":
    exit(main())

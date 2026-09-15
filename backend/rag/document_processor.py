"""
Document processor for chunking PDF files.

Provides simple, effective text chunking with overlap for RAG.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunk of text with metadata."""

    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "metadata": self.metadata
        }


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # Try to break at sentence boundary if possible
        if end < text_length:
            # Look for sentence endings near the chunk boundary
            search_start = max(start, end - 100)
            sentence_ends = [
                text.rfind('. ', search_start, end),
                text.rfind('.\n', search_start, end),
                text.rfind('! ', search_start, end),
                text.rfind('? ', search_start, end)
            ]
            best_end = max(sentence_ends)
            if best_end > start:
                end = best_end + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position with overlap
        start = end - chunk_overlap if end < text_length else text_length

    return chunks


def load_pdf(pdf_path: Path) -> List[DocumentChunk]:
    """
    Load a PDF file and extract text chunks.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of DocumentChunk objects
    """
    chunks = []

    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)

        logger.info(f"Processing {pdf_path.name} ({total_pages} pages)")

        for page_num, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text()

                if not text or not text.strip():
                    logger.debug(f"Page {page_num} is empty or could not be extracted")
                    continue

                # Chunk the page text
                page_chunks = chunk_text(text)

                # Create DocumentChunk objects with metadata
                for chunk_idx, text_chunk in enumerate(page_chunks):
                    metadata = {
                        "file_name": pdf_path.name,
                        "source_path": str(pdf_path),
                        "page_number": page_num,
                        "chunk_index": chunk_idx,
                        "total_pages": total_pages
                    }
                    chunks.append(DocumentChunk(text_chunk, metadata))

            except Exception as e:
                logger.error(f"Error processing page {page_num} of {pdf_path.name}: {e}")
                continue

        logger.info(f"Extracted {len(chunks)} chunks from {pdf_path.name}")

    except Exception as e:
        logger.error(f"Error loading PDF {pdf_path}: {e}")
        raise

    return chunks


def load_company_pdfs(company_dir: Path) -> List[DocumentChunk]:
    """
    Load all PDF files from a company directory.

    Args:
        company_dir: Path to company directory containing PDFs

    Returns:
        List of DocumentChunk objects from all PDFs
    """
    if not company_dir.exists():
        logger.warning(f"Company directory does not exist: {company_dir}")
        return []

    if not company_dir.is_dir():
        logger.warning(f"Path is not a directory: {company_dir}")
        return []

    pdf_files = list(company_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {company_dir.name}")

    all_chunks = []

    for pdf_path in pdf_files:
        try:
            chunks = load_pdf(pdf_path)

            # Add company metadata to each chunk
            for chunk in chunks:
                chunk.metadata["company"] = company_dir.name

            all_chunks.extend(chunks)

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            continue

    return all_chunks

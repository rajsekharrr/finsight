"""
RAG Indexer: PDF ingestion and FAISS storage.

Ingests company PDF filings from local data/filings/<COMPANY>/ directory,
chunks documents with metadata, and stores in FAISS with Google Gemini embeddings.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import os
import shutil

from backend.rag.document_processor import load_company_pdfs
from backend.rag.gemini_embedding import GeminiEmbedding
from backend.rag.vector_store import FAISSVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"


def get_company_pdf_paths(company: str) -> List[Path]:
    """
    Get all PDF file paths for a given company.

    Args:
        company: Company name (must match folder name in data/filings/)

    Returns:
        List of Path objects for PDF files, empty if none found
    """
    company_dir = FILINGS_DIR / company

    if not company_dir.exists():
        logger.warning(f"Company directory does not exist: {company_dir}")
        return []

    if not company_dir.is_dir():
        logger.warning(f"Path is not a directory: {company_dir}")
        return []

    pdf_files = list(company_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files for {company}")
    return pdf_files


def ingest_company_filings(
    company: str,
    force_reindex: bool = False
) -> Dict[str, Any]:
    """
    Ingest PDF filings for a single company into FAISS vector store.

    Args:
        company: Company name
        force_reindex: If True, delete existing index and rebuild

    Returns:
        Status dict with keys: success, company, files_processed,
        documents_count, error (if failed)
    """
    result = {
        "success": False,
        "company": company,
        "files_processed": 0,
        "documents_count": 0,
    }

    try:
        # Get PDF paths
        pdf_paths = get_company_pdf_paths(company)

        if not pdf_paths:
            result["success"] = True
            result["message"] = f"No PDFs found for {company}"
            logger.info(result["message"])
            return result

        # Setup storage directory
        company_store_path = VECTOR_STORE_DIR / company
        company_store_path.mkdir(parents=True, exist_ok=True)

        # Handle force_reindex
        if force_reindex and company_store_path.exists():
            shutil.rmtree(company_store_path)
            company_store_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Deleted existing index for {company}")

        # Check if index already exists
        if not force_reindex and (company_store_path / "index.faiss").exists():
            try:
                vector_store = FAISSVectorStore.load(company_store_path)
                doc_count = vector_store.count
                result["success"] = True
                result["documents_count"] = doc_count
                result["message"] = f"Using existing index for {company} ({doc_count} documents)"
                logger.info(result["message"])
                return result
            except Exception as e:
                logger.debug(f"Could not load existing index: {e}")

        # Initialize embedding model
        logger.info("Initializing Gemini embeddings...")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        embed_model = GeminiEmbedding(
            api_key=google_api_key,
            model_name="models/gemini-embedding-001"
        )

        # Load and process PDFs
        logger.info(f"Loading PDFs for {company}...")
        company_dir = FILINGS_DIR / company
        document_chunks = load_company_pdfs(company_dir)

        if not document_chunks:
            result["error"] = f"No documents extracted from PDFs"
            logger.warning(result["error"])
            return result

        # Extract texts and metadata
        texts = [chunk.text for chunk in document_chunks]
        metadatas = [chunk.metadata for chunk in document_chunks]

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = embed_model.embed_documents(texts)

        # Create vector store and add documents
        logger.info("Creating FAISS vector index...")
        vector_store = FAISSVectorStore(dimension=3072)  # Gemini embedding dimension
        vector_store.add_documents(texts, embeddings, metadatas)

        # Save the index
        vector_store.save(company_store_path)
        logger.info(f"Persisted index to {company_store_path}")

        # Count unique files
        unique_files = set(meta["file_name"] for meta in metadatas)
        files_processed = len(unique_files)

        result["success"] = True
        result["files_processed"] = files_processed
        result["documents_count"] = len(document_chunks)
        result["message"] = f"Successfully indexed {files_processed} files ({len(document_chunks)} chunks) for {company}"
        logger.info(result["message"])

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Failed to ingest company filings: {e}", exc_info=True)

    return result


def ingest_all_companies(force_reindex: bool = False) -> Dict[str, Any]:
    """
    Ingest PDF filings for all companies in data/filings/ directory.

    Args:
        force_reindex: If True, rebuild all indexes

    Returns:
        Status dict with keys: success, companies_processed,
        total_files, total_documents, results (list of per-company results)
    """
    result = {
        "success": False,
        "companies_processed": 0,
        "total_files": 0,
        "total_documents": 0,
        "results": []
    }

    try:
        if not FILINGS_DIR.exists():
            result["error"] = f"Filings directory does not exist: {FILINGS_DIR}"
            logger.warning(result["error"])
            result["success"] = True  # Not a fatal error
            return result

        # Get all company directories
        company_dirs = [d for d in FILINGS_DIR.iterdir() if d.is_dir()]

        if not company_dirs:
            result["message"] = "No company directories found in data/filings/"
            result["success"] = True
            logger.info(result["message"])
            return result

        logger.info(f"Found {len(company_dirs)} company directories")

        # Process each company
        for company_dir in company_dirs:
            company_name = company_dir.name
            logger.info(f"Processing company: {company_name}")

            company_result = ingest_company_filings(
                company=company_name,
                force_reindex=force_reindex
            )

            result["results"].append(company_result)

            if company_result["success"]:
                result["companies_processed"] += 1
                result["total_files"] += company_result["files_processed"]
                result["total_documents"] += company_result["documents_count"]

        result["success"] = True
        result["message"] = f"Processed {result['companies_processed']} companies, {result['total_files']} files, {result['total_documents']} documents"
        logger.info(result["message"])

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Failed to ingest all companies: {e}", exc_info=True)

    return result

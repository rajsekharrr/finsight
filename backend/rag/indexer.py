"""
RAG Indexer: PDF ingestion and ChromaDB storage.

Ingests company PDF filings from local data/filings/<COMPANY>/ directory,
chunks documents with metadata, and stores in ChromaDB with OpenAI embeddings.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import os

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"
CHROMA_STORE = DATA_DIR / "chroma_store"


def setup_llama_index() -> None:
    """Configure llama-index settings with OpenAI embeddings."""
    try:
        # Use OpenAI embeddings
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        Settings.embed_model = embed_model
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 200
        logger.info("Llama-index configured with OpenAI embeddings")
    except Exception as e:
        logger.error(f"Failed to setup llama-index: {e}")
        raise


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
    Ingest PDF filings for a single company into ChromaDB.

    Args:
        company: Company name
        force_reindex: If True, delete existing index and rebuild

    Returns:
        Status dict with keys: success, company, files_processed,
        documents_count, error (if failed)
    """
    setup_llama_index()

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

        # Setup ChromaDB
        CHROMA_STORE.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_STORE))
        collection_name = f"company_{company.lower().replace(' ', '_')}"

        # Handle force_reindex
        if force_reindex:
            try:
                chroma_client.delete_collection(name=collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except Exception as e:
                logger.debug(f"Collection {collection_name} did not exist: {e}")

        # Check if collection already exists
        try:
            existing_collections = [c.name for c in chroma_client.list_collections()]
            if collection_name in existing_collections and not force_reindex:
                collection = chroma_client.get_collection(name=collection_name)
                doc_count = collection.count()
                result["success"] = True
                result["documents_count"] = doc_count
                result["message"] = f"Using existing index for {company} ({doc_count} documents)"
                logger.info(result["message"])
                return result
        except Exception as e:
            logger.debug(f"Error checking existing collection: {e}")

        # Create new collection
        chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Load and process PDFs
        documents = []
        files_processed = 0

        for pdf_path in pdf_paths:
            try:
                logger.info(f"Processing: {pdf_path.name}")
                reader = SimpleDirectoryReader(
                    input_files=[str(pdf_path)]
                )
                file_docs = reader.load_data()

                # Add metadata to each document
                for i, doc in enumerate(file_docs):
                    doc.metadata.update({
                        "company": company,
                        "file_name": pdf_path.name,
                        "source_path": str(pdf_path),
                        "page_number": i + 1,
                    })

                documents.extend(file_docs)
                files_processed += 1
                logger.info(f"Loaded {len(file_docs)} pages from {pdf_path.name}")

            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue

        if not documents:
            result["error"] = f"No documents extracted from {files_processed} PDF(s)"
            logger.warning(result["error"])
            return result

        # Create index
        logger.info(f"Creating vector index for {len(documents)} documents...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )

        result["success"] = True
        result["files_processed"] = files_processed
        result["documents_count"] = len(documents)
        result["message"] = f"Successfully indexed {files_processed} files ({len(documents)} documents) for {company}"
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

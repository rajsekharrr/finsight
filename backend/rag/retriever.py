"""
RAG Retriever: Query and retrieve relevant chunks from ChromaDB.

Retrieves relevant document chunks for a company using vector similarity search
with optional BM25 fallback for hybrid retrieval.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import os

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root (reuse from indexer)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_STORE = DATA_DIR / "chroma_store"


def setup_embeddings() -> OpenAIEmbedding:
    """
    Configure OpenAI embeddings for retrieval.

    Returns:
        Configured OpenAIEmbedding instance

    Raises:
        Exception if embeddings cannot be configured
    """
    try:
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        Settings.embed_model = embed_model
        logger.info("Embeddings configured for retrieval")
        return embed_model
    except Exception as e:
        logger.error(f"Failed to setup embeddings: {e}")
        raise


def retrieve_for_query(
    company: str,
    query: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant document chunks for a company query.

    Args:
        company: Company name (must match indexed company)
        query: Search query text
        top_k: Number of top results to return (default: 5)

    Returns:
        List of dicts with keys:
        - text: The chunk text content
        - score: Relevance score (if available)
        - metadata: Dict with company, file_name, page_number, source_path

    Handles gracefully:
    - Missing/non-existent index for company
    - Empty query string
    - ChromaDB connection errors
    - Embedding API errors (with fallback to keyword search if possible)
    """
    result = []

    # Validate inputs
    if not query or not query.strip():
        logger.warning("Empty query provided")
        return result

    if not company or not company.strip():
        logger.warning("Empty company name provided")
        return result

    try:
        # Setup embeddings
        try:
            setup_embeddings()
            use_vector_search = True
        except Exception as e:
            logger.warning(f"Failed to setup embeddings, will try keyword fallback: {e}")
            use_vector_search = False

        # Connect to ChromaDB
        if not CHROMA_STORE.exists():
            logger.warning(f"ChromaDB store does not exist: {CHROMA_STORE}")
            return result

        chroma_client = chromadb.PersistentClient(path=str(CHROMA_STORE))
        collection_name = f"company_{company.lower().replace(' ', '_')}"

        # Check if collection exists
        existing_collections = [c.name for c in chroma_client.list_collections()]
        if collection_name not in existing_collections:
            logger.warning(f"No index found for company: {company}")
            return result

        # Get collection
        chroma_collection = chroma_client.get_collection(name=collection_name)

        if chroma_collection.count() == 0:
            logger.warning(f"Index for {company} is empty")
            return result

        logger.info(f"Querying {company} index with {chroma_collection.count()} documents")

        # Try vector search first
        if use_vector_search:
            try:
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                index = VectorStoreIndex.from_vector_store(vector_store)

                # Create retriever
                retriever = index.as_retriever(similarity_top_k=top_k)

                # Retrieve nodes
                nodes = retriever.retrieve(query)

                # Convert to JSON-serializable format
                for node in nodes:
                    chunk_dict = {
                        "text": node.text if hasattr(node, 'text') else node.get_content(),
                        "score": node.score if hasattr(node, 'score') else None,
                        "metadata": node.metadata if hasattr(node, 'metadata') else {}
                    }
                    result.append(chunk_dict)

                logger.info(f"Retrieved {len(result)} chunks via vector search")
                return result

            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                use_vector_search = False

        # Fallback to keyword/BM25 search
        if not use_vector_search:
            logger.info("Attempting keyword-based retrieval fallback")
            try:
                # Use ChromaDB's built-in query without embeddings
                # This searches metadata and text content
                query_result = chroma_collection.query(
                    query_texts=[query],
                    n_results=min(top_k, chroma_collection.count())
                )

                if query_result and query_result.get('documents'):
                    documents = query_result['documents'][0] if query_result['documents'] else []
                    metadatas = query_result['metadatas'][0] if query_result.get('metadatas') else []
                    distances = query_result['distances'][0] if query_result.get('distances') else []

                    for i, doc_text in enumerate(documents):
                        chunk_dict = {
                            "text": doc_text,
                            "score": 1.0 - distances[i] if i < len(distances) else None,
                            "metadata": metadatas[i] if i < len(metadatas) else {}
                        }
                        result.append(chunk_dict)

                    logger.info(f"Retrieved {len(result)} chunks via keyword fallback")

            except Exception as e:
                logger.error(f"Keyword fallback also failed: {e}")

    except Exception as e:
        logger.error(f"Failed to retrieve for query: {e}", exc_info=True)

    return result

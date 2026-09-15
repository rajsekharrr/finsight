"""
RAG module for FinSight.

Simplified RAG system using:
- google-generativeai for embeddings
- faiss-cpu for vector store
- Direct Python code for chunking and retrieval
"""

from backend.rag.indexer import ingest_company_filings, ingest_all_companies
from backend.rag.retriever import retrieve_for_query
from backend.rag.gemini_embedding import GeminiEmbedding
from backend.rag.vector_store import FAISSVectorStore
from backend.rag.document_processor import DocumentChunk, chunk_text, load_pdf, load_company_pdfs

__all__ = [
    "ingest_company_filings",
    "ingest_all_companies",
    "retrieve_for_query",
    "GeminiEmbedding",
    "FAISSVectorStore",
    "DocumentChunk",
    "chunk_text",
    "load_pdf",
    "load_company_pdfs",
]

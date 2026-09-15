"""
Google Gemini embedding wrapper for RAG.

Simple wrapper around google-generativeai for generating embeddings.
"""

import logging
import os
from typing import List, Union
import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiEmbedding:
    """Simple wrapper for Google Gemini embeddings."""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "models/gemini-embedding-001"
    ):
        """
        Initialize Gemini embedding model.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model_name: Embedding model name
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not provided and not found in environment")

        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        logger.info(f"Initialized Gemini embeddings with model: {model_name}")

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            text: Query text

        Returns:
            Embedding vector as list of floats
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of document texts

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i, text in enumerate(texts):
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])

                if (i + 1) % 10 == 0:
                    logger.info(f"Generated embeddings for {i + 1}/{len(texts)} documents")

            except Exception as e:
                logger.error(f"Error generating embedding for document {i}: {e}")
                # Return zero vector on failure to maintain index alignment
                embeddings.append([0.0] * 3072)  # Gemini embedding dimension (text-embedding-004: 3072)

        return embeddings

    def embed_document(self, text: str) -> List[float]:
        """
        Generate embedding for a single document.

        Args:
            text: Document text

        Returns:
            Embedding vector as list of floats
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating document embedding: {e}")
            raise

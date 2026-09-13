"""
Tests for RAG retriever: Query and retrieve relevant chunks from ChromaDB.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from backend.rag.retriever import (
    setup_embeddings,
    retrieve_for_query,
    CHROMA_STORE,
)


class TestSetupEmbeddings:
    """Tests for setup_embeddings function."""

    @patch("backend.rag.retriever.Settings")
    @patch("backend.rag.retriever.OpenAIEmbedding")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_successful_setup(self, mock_embedding, mock_settings):
        """Should configure OpenAI embeddings successfully."""
        mock_embed_instance = MagicMock()
        mock_embedding.return_value = mock_embed_instance

        result = setup_embeddings()

        mock_embedding.assert_called_once_with(
            model="text-embedding-3-small",
            api_key="test_key"
        )
        assert result == mock_embed_instance
        assert mock_settings.embed_model == mock_embed_instance

    @patch("backend.rag.retriever.OpenAIEmbedding")
    def test_setup_failure(self, mock_embedding):
        """Should raise exception on setup failure."""
        mock_embedding.side_effect = Exception("API key missing")

        with pytest.raises(Exception, match="API key missing"):
            setup_embeddings()


class TestRetrieveForQuery:
    """Tests for retrieve_for_query function."""

    def test_empty_query(self):
        """Should return empty list for empty query."""
        result = retrieve_for_query("TESTCO", "")
        assert result == []

        result = retrieve_for_query("TESTCO", "   ")
        assert result == []

    def test_empty_company(self):
        """Should return empty list for empty company name."""
        result = retrieve_for_query("", "test query")
        assert result == []

        result = retrieve_for_query("   ", "test query")
        assert result == []

    @patch("backend.rag.retriever.setup_embeddings")
    def test_chroma_store_does_not_exist(self, mock_setup, tmp_path, monkeypatch):
        """Should return empty list if ChromaDB store doesn't exist."""
        monkeypatch.setattr("backend.rag.retriever.CHROMA_STORE", tmp_path / "nonexistent")

        result = retrieve_for_query("TESTCO", "test query")

        assert result == []

    @patch("backend.rag.retriever.setup_embeddings")
    @patch("backend.rag.retriever.chromadb.PersistentClient")
    def test_collection_does_not_exist(
        self, mock_chroma_client, mock_setup, tmp_path, monkeypatch
    ):
        """Should return empty list if company collection doesn't exist."""
        monkeypatch.setattr("backend.rag.retriever.CHROMA_STORE", tmp_path)

        # Mock ChromaDB client with no collections
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_chroma_client.return_value = mock_client

        result = retrieve_for_query("TESTCO", "test query")

        assert result == []

    @patch("backend.rag.retriever.setup_embeddings")
    @patch("backend.rag.retriever.chromadb.PersistentClient")
    def test_empty_collection(
        self, mock_chroma_client, mock_setup, tmp_path, monkeypatch
    ):
        """Should return empty list if collection is empty."""
        monkeypatch.setattr("backend.rag.retriever.CHROMA_STORE", tmp_path)

        # Mock ChromaDB client with empty collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "company_testco"
        mock_collection.count.return_value = 0

        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        result = retrieve_for_query("TESTCO", "test query")

        assert result == []

    @patch("backend.rag.retriever.setup_embeddings")
    @patch("backend.rag.retriever.chromadb.PersistentClient")
    @patch("backend.rag.retriever.VectorStoreIndex.from_vector_store")
    def test_successful_vector_retrieval(
        self, mock_index, mock_chroma_client, mock_setup, tmp_path, monkeypatch
    ):
        """Should successfully retrieve chunks via vector search."""
        monkeypatch.setattr("backend.rag.retriever.CHROMA_STORE", tmp_path)

        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "company_testco"
        mock_collection.count.return_value = 10

        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        # Mock retriever and nodes
        mock_node1 = MagicMock()
        mock_node1.text = "This is test chunk 1"
        mock_node1.score = 0.95
        mock_node1.metadata = {
            "company": "TESTCO",
            "file_name": "report.pdf",
            "page_number": 1,
            "source_path": "/path/to/report.pdf"
        }

        mock_node2 = MagicMock()
        mock_node2.text = "This is test chunk 2"
        mock_node2.score = 0.88
        mock_node2.metadata = {
            "company": "TESTCO",
            "file_name": "report.pdf",
            "page_number": 2,
            "source_path": "/path/to/report.pdf"
        }

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node1, mock_node2]

        mock_index_instance = MagicMock()
        mock_index_instance.as_retriever.return_value = mock_retriever
        mock_index.return_value = mock_index_instance

        result = retrieve_for_query("TESTCO", "test query", top_k=5)

        assert len(result) == 2
        assert result[0]["text"] == "This is test chunk 1"
        assert result[0]["score"] == 0.95
        assert result[0]["metadata"]["company"] == "TESTCO"
        assert result[1]["text"] == "This is test chunk 2"
        assert result[1]["score"] == 0.88

    @patch("backend.rag.retriever.setup_embeddings")
    @patch("backend.rag.retriever.chromadb.PersistentClient")
    @patch("backend.rag.retriever.VectorStoreIndex.from_vector_store")
    def test_top_k_parameter(
        self, mock_index, mock_chroma_client, mock_setup, tmp_path, monkeypatch
    ):
        """Should respect top_k parameter."""
        monkeypatch.setattr("backend.rag.retriever.CHROMA_STORE", tmp_path)

        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "company_testco"
        mock_collection.count.return_value = 10

        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        mock_index_instance = MagicMock()
        mock_index_instance.as_retriever.return_value = mock_retriever
        mock_index.return_value = mock_index_instance

        retrieve_for_query("TESTCO", "test query", top_k=3)

        # Verify top_k was passed to retriever
        mock_index_instance.as_retriever.assert_called_once_with(similarity_top_k=3)

    @patch("backend.rag.retriever.setup_embeddings")
    @patch("backend.rag.retriever.chromadb.PersistentClient")
    def test_keyword_fallback_on_embedding_failure(
        self, mock_chroma_client, mock_setup, tmp_path, monkeypatch
    ):
        """Should fallback to keyword search when embeddings fail."""
        monkeypatch.setattr("backend.rag.retriever.CHROMA_STORE", tmp_path)

        # Setup embeddings to fail
        mock_setup.side_effect = Exception("OpenAI API error")

        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "company_testco"
        mock_collection.count.return_value = 5

        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection

        # Mock keyword query result
        mock_collection.query.return_value = {
            "documents": [["Keyword match text"]],
            "metadatas": [[{"company": "TESTCO", "file_name": "doc.pdf"}]],
            "distances": [[0.3]]
        }

        mock_chroma_client.return_value = mock_client

        result = retrieve_for_query("TESTCO", "test query")

        assert len(result) == 1
        assert result[0]["text"] == "Keyword match text"
        assert result[0]["score"] == 0.7  # 1.0 - 0.3
        assert result[0]["metadata"]["company"] == "TESTCO"

    @patch("backend.rag.retriever.setup_embeddings")
    @patch("backend.rag.retriever.chromadb.PersistentClient")
    @patch("backend.rag.retriever.VectorStoreIndex.from_vector_store")
    def test_keyword_fallback_on_vector_search_failure(
        self, mock_index, mock_chroma_client, mock_setup, tmp_path, monkeypatch
    ):
        """Should fallback to keyword search when vector search fails."""
        monkeypatch.setattr("backend.rag.retriever.CHROMA_STORE", tmp_path)

        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "company_testco"
        mock_collection.count.return_value = 5

        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection

        # Mock keyword query result
        mock_collection.query.return_value = {
            "documents": [["Fallback keyword result"]],
            "metadatas": [[{"company": "TESTCO"}]],
            "distances": [[0.2]]
        }

        mock_chroma_client.return_value = mock_client

        # Make vector search fail
        mock_index.side_effect = Exception("Vector search error")

        result = retrieve_for_query("TESTCO", "test query")

        assert len(result) == 1
        assert result[0]["text"] == "Fallback keyword result"
        assert result[0]["score"] == 0.8  # 1.0 - 0.2

    @patch("backend.rag.retriever.setup_embeddings")
    @patch("backend.rag.retriever.chromadb.PersistentClient")
    def test_json_serializable_output(
        self, mock_chroma_client, mock_setup, tmp_path, monkeypatch
    ):
        """Should return JSON-serializable output."""
        import json

        monkeypatch.setattr("backend.rag.retriever.CHROMA_STORE", tmp_path)

        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "company_testco"
        mock_collection.count.return_value = 5

        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection

        # Mock keyword query result
        mock_collection.query.return_value = {
            "documents": [["Test text"]],
            "metadatas": [[{"company": "TESTCO", "page_number": 1}]],
            "distances": [[0.1]]
        }

        mock_chroma_client.return_value = mock_client

        result = retrieve_for_query("TESTCO", "test query")

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        # Verify structure
        parsed = json.loads(json_str)
        assert len(parsed) == 1
        assert "text" in parsed[0]
        assert "score" in parsed[0]
        assert "metadata" in parsed[0]

    @patch("backend.rag.retriever.setup_embeddings")
    @patch("backend.rag.retriever.chromadb.PersistentClient")
    def test_complete_error_handling(
        self, mock_chroma_client, mock_setup, tmp_path, monkeypatch
    ):
        """Should handle all errors gracefully and return empty list."""
        monkeypatch.setattr("backend.rag.retriever.CHROMA_STORE", tmp_path)

        # Make everything fail
        mock_setup.side_effect = Exception("Setup failed")
        mock_chroma_client.side_effect = Exception("ChromaDB connection failed")

        result = retrieve_for_query("TESTCO", "test query")

        # Should return empty list, not crash
        assert result == []

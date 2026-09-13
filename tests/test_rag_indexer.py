"""
Tests for RAG indexer: PDF ingestion and ChromaDB storage.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from backend.rag.indexer import (
    get_company_pdf_paths,
    ingest_company_filings,
    ingest_all_companies,
    setup_llama_index,
    FILINGS_DIR,
    CHROMA_STORE,
)


class TestGetCompanyPdfPaths:
    """Tests for get_company_pdf_paths function."""

    def test_nonexistent_company_directory(self, tmp_path, monkeypatch):
        """Should return empty list if company directory doesn't exist."""
        monkeypatch.setattr("backend.rag.indexer.FILINGS_DIR", tmp_path)

        result = get_company_pdf_paths("NONEXISTENT")

        assert result == []

    def test_empty_company_directory(self, tmp_path, monkeypatch):
        """Should return empty list if no PDFs in company directory."""
        company_dir = tmp_path / "TESTCO"
        company_dir.mkdir()
        monkeypatch.setattr("backend.rag.indexer.FILINGS_DIR", tmp_path)

        result = get_company_pdf_paths("TESTCO")

        assert result == []

    def test_directory_with_pdfs(self, tmp_path, monkeypatch):
        """Should return list of PDF paths."""
        company_dir = tmp_path / "TESTCO"
        company_dir.mkdir()

        # Create test PDF files
        pdf1 = company_dir / "filing_2023.pdf"
        pdf2 = company_dir / "filing_2024.pdf"
        txt_file = company_dir / "notes.txt"

        pdf1.touch()
        pdf2.touch()
        txt_file.touch()

        monkeypatch.setattr("backend.rag.indexer.FILINGS_DIR", tmp_path)

        result = get_company_pdf_paths("TESTCO")

        assert len(result) == 2
        assert pdf1 in result
        assert pdf2 in result
        assert txt_file not in result

    def test_path_is_file_not_directory(self, tmp_path, monkeypatch):
        """Should return empty list if path is a file, not a directory."""
        file_path = tmp_path / "TESTCO"
        file_path.touch()

        monkeypatch.setattr("backend.rag.indexer.FILINGS_DIR", tmp_path)

        result = get_company_pdf_paths("TESTCO")

        assert result == []


class TestIngestCompanyFilings:
    """Tests for ingest_company_filings function."""

    @patch("backend.rag.indexer.setup_llama_index")
    @patch("backend.rag.indexer.get_company_pdf_paths")
    def test_no_pdfs_found(self, mock_get_paths, mock_setup):
        """Should return success with message when no PDFs found."""
        mock_get_paths.return_value = []

        result = ingest_company_filings("TESTCO")

        assert result["success"] is True
        assert result["company"] == "TESTCO"
        assert result["files_processed"] == 0
        assert result["documents_count"] == 0
        assert "No PDFs found" in result["message"]

    @patch("backend.rag.indexer.setup_llama_index")
    @patch("backend.rag.indexer.get_company_pdf_paths")
    @patch("backend.rag.indexer.chromadb.PersistentClient")
    def test_existing_index_not_force_reindex(
        self, mock_chroma_client, mock_get_paths, mock_setup, tmp_path, monkeypatch
    ):
        """Should use existing index when force_reindex=False and collection exists."""
        # Mock CHROMA_STORE path
        monkeypatch.setattr("backend.rag.indexer.CHROMA_STORE", tmp_path / "chroma")

        mock_get_paths.return_value = [Path("test.pdf")]

        # Mock ChromaDB client with existing collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 50
        mock_collection.name = "company_testco"

        # Return the collection in list_collections
        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        result = ingest_company_filings("TESTCO", force_reindex=False)

        assert result["success"] is True
        assert result["documents_count"] == 50
        assert "Using existing index" in result["message"]

        # Should not attempt to create new collection
        mock_client.get_or_create_collection.assert_not_called()

    @patch("backend.rag.indexer.setup_llama_index")
    @patch("backend.rag.indexer.get_company_pdf_paths")
    @patch("backend.rag.indexer.chromadb.PersistentClient")
    @patch("backend.rag.indexer.SimpleDirectoryReader")
    @patch("backend.rag.indexer.VectorStoreIndex.from_documents")
    def test_successful_ingestion(
        self,
        mock_index,
        mock_reader,
        mock_chroma_client,
        mock_get_paths,
        mock_setup,
        tmp_path,
    ):
        """Should successfully ingest PDFs and create index."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        mock_get_paths.return_value = [pdf_path]

        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        # Mock document reader
        mock_doc1 = Mock()
        mock_doc1.metadata = {}
        mock_doc2 = Mock()
        mock_doc2.metadata = {}

        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = [mock_doc1, mock_doc2]
        mock_reader.return_value = mock_reader_instance

        result = ingest_company_filings("TESTCO", force_reindex=True)

        assert result["success"] is True
        assert result["company"] == "TESTCO"
        assert result["files_processed"] == 1
        assert result["documents_count"] == 2
        assert "Successfully indexed" in result["message"]

        # Verify metadata was added
        assert mock_doc1.metadata["company"] == "TESTCO"
        assert mock_doc1.metadata["file_name"] == "test.pdf"
        assert "source_path" in mock_doc1.metadata
        assert "page_number" in mock_doc1.metadata

    @patch("backend.rag.indexer.setup_llama_index")
    @patch("backend.rag.indexer.get_company_pdf_paths")
    @patch("backend.rag.indexer.chromadb.PersistentClient")
    @patch("backend.rag.indexer.SimpleDirectoryReader")
    def test_pdf_processing_error(
        self,
        mock_reader,
        mock_chroma_client,
        mock_get_paths,
        mock_setup,
        tmp_path,
    ):
        """Should handle PDF processing errors gracefully."""
        pdf_path = tmp_path / "corrupt.pdf"
        pdf_path.touch()
        mock_get_paths.return_value = [pdf_path]

        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        # Mock reader to raise error
        mock_reader.side_effect = Exception("Invalid PDF")

        result = ingest_company_filings("TESTCO", force_reindex=True)

        assert result["success"] is False
        assert "error" in result

    @patch("backend.rag.indexer.setup_llama_index")
    @patch("backend.rag.indexer.get_company_pdf_paths")
    @patch("backend.rag.indexer.chromadb.PersistentClient")
    def test_force_reindex_deletes_collection(
        self, mock_chroma_client, mock_get_paths, mock_setup, tmp_path, monkeypatch
    ):
        """Should attempt to delete existing collection when force_reindex=True."""
        # Mock CHROMA_STORE path
        monkeypatch.setattr("backend.rag.indexer.CHROMA_STORE", tmp_path / "chroma")

        # Return at least one PDF so we get past the early return
        mock_get_paths.return_value = [Path("test.pdf")]

        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_client.delete_collection.return_value = None
        mock_client.list_collections.return_value = []
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        # Mock document processing to fail (so we don't need actual PDFs)
        with patch("backend.rag.indexer.SimpleDirectoryReader") as mock_reader:
            mock_reader.side_effect = Exception("Mocked failure")

            result = ingest_company_filings("TESTCO", force_reindex=True)

        # Should attempt to delete collection
        mock_client.delete_collection.assert_called_once_with(
            name="company_testco"
        )


class TestIngestAllCompanies:
    """Tests for ingest_all_companies function."""

    def test_filings_directory_does_not_exist(self, tmp_path, monkeypatch):
        """Should return success with message if filings dir doesn't exist."""
        non_existent = tmp_path / "nonexistent"
        monkeypatch.setattr("backend.rag.indexer.FILINGS_DIR", non_existent)

        result = ingest_all_companies()

        assert result["success"] is True
        assert result["companies_processed"] == 0
        assert "does not exist" in result["error"]

    def test_no_company_directories(self, tmp_path, monkeypatch):
        """Should return success with message if no companies found."""
        monkeypatch.setattr("backend.rag.indexer.FILINGS_DIR", tmp_path)

        result = ingest_all_companies()

        assert result["success"] is True
        assert result["companies_processed"] == 0
        assert "No company directories" in result["message"]

    @patch("backend.rag.indexer.ingest_company_filings")
    def test_successful_multi_company_ingestion(
        self, mock_ingest, tmp_path, monkeypatch
    ):
        """Should process all company directories."""
        # Create test company directories
        (tmp_path / "COMPANY_A").mkdir()
        (tmp_path / "COMPANY_B").mkdir()
        (tmp_path / "COMPANY_C").mkdir()

        monkeypatch.setattr("backend.rag.indexer.FILINGS_DIR", tmp_path)

        # Mock individual company ingestion
        mock_ingest.side_effect = [
            {
                "success": True,
                "files_processed": 2,
                "documents_count": 10,
            },
            {
                "success": True,
                "files_processed": 3,
                "documents_count": 15,
            },
            {
                "success": False,
                "files_processed": 0,
                "documents_count": 0,
                "error": "Test error",
            },
        ]

        result = ingest_all_companies(force_reindex=False)

        assert result["success"] is True
        assert result["companies_processed"] == 2  # Only successful ones
        assert result["total_files"] == 5
        assert result["total_documents"] == 25
        assert len(result["results"]) == 3
        assert mock_ingest.call_count == 3

    @patch("backend.rag.indexer.ingest_company_filings")
    def test_force_reindex_passed_through(
        self, mock_ingest, tmp_path, monkeypatch
    ):
        """Should pass force_reindex flag to individual company ingestion."""
        (tmp_path / "TESTCO").mkdir()
        monkeypatch.setattr("backend.rag.indexer.FILINGS_DIR", tmp_path)

        mock_ingest.return_value = {
            "success": True,
            "files_processed": 1,
            "documents_count": 5,
        }

        ingest_all_companies(force_reindex=True)

        mock_ingest.assert_called_once_with(
            company="TESTCO", force_reindex=True
        )


class TestSetupLlamaIndex:
    """Tests for setup_llama_index function."""

    @patch("backend.rag.indexer.Settings")
    @patch("backend.rag.indexer.OpenAIEmbedding")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_successful_setup(self, mock_embedding, mock_settings):
        """Should configure llama-index with OpenAI embeddings."""
        mock_embed_instance = MagicMock()
        mock_embedding.return_value = mock_embed_instance

        setup_llama_index()

        mock_embedding.assert_called_once_with(
            model="text-embedding-3-small",
            api_key="test_key"
        )
        # Verify Settings were configured
        assert mock_settings.embed_model == mock_embed_instance

    @patch("backend.rag.indexer.OpenAIEmbedding")
    def test_setup_failure(self, mock_embedding):
        """Should raise exception on setup failure."""
        mock_embedding.side_effect = Exception("API key missing")

        with pytest.raises(Exception, match="API key missing"):
            setup_llama_index()

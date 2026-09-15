"""
Simple test script to verify the simplified RAG system.

Tests:
1. Document chunking
2. Vector store operations
3. End-to-end indexing and retrieval (if sample PDFs exist)
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from backend.rag.document_processor import chunk_text, DocumentChunk
from backend.rag.gemini_embedding import GeminiEmbedding
from backend.rag.vector_store import FAISSVectorStore
from backend.rag import ingest_company_filings, retrieve_for_query


def test_chunking():
    """Test text chunking."""
    print("\n=== Testing Text Chunking ===")

    sample_text = "This is a test. " * 100  # Create a longer text
    chunks = chunk_text(sample_text, chunk_size=100, chunk_overlap=20)

    print(f"✓ Created {len(chunks)} chunks from {len(sample_text)} characters")
    print(f"  First chunk length: {len(chunks[0])}")
    print(f"  Last chunk length: {len(chunks[-1])}")

    return True


def test_embeddings():
    """Test Gemini embeddings."""
    print("\n=== Testing Gemini Embeddings ===")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("✗ GOOGLE_API_KEY not set - skipping embeddings test")
        return False

    try:
        embed_model = GeminiEmbedding(api_key=api_key)

        # Test single query embedding
        query_embedding = embed_model.embed_query("What is the revenue?")
        print(f"✓ Query embedding dimension: {len(query_embedding)}")

        # Test document embeddings
        docs = ["Document 1", "Document 2", "Document 3"]
        doc_embeddings = embed_model.embed_documents(docs)
        print(f"✓ Generated {len(doc_embeddings)} document embeddings")

        return True

    except Exception as e:
        print(f"✗ Embeddings test failed: {e}")
        return False


def test_vector_store():
    """Test FAISS vector store."""
    print("\n=== Testing FAISS Vector Store ===")

    try:
        # Create a vector store
        store = FAISSVectorStore(dimension=768)

        # Add some dummy documents
        texts = ["Revenue increased by 20%", "Profit margin is 15%", "Customer base grew significantly"]
        embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768]  # Dummy embeddings
        metadatas = [
            {"company": "TEST", "file_name": "test1.pdf", "page_number": 1},
            {"company": "TEST", "file_name": "test2.pdf", "page_number": 2},
            {"company": "TEST", "file_name": "test3.pdf", "page_number": 3},
        ]

        store.add_documents(texts, embeddings, metadatas)
        print(f"✓ Added {store.count} documents to vector store")

        # Test search
        query_embedding = [0.15] * 768  # Should be closer to first document
        results = store.search(query_embedding, top_k=2)
        print(f"✓ Retrieved {len(results)} results from search")

        # Test save/load
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            store.save(tmp_path)
            print(f"✓ Saved vector store to {tmp_path}")

            loaded_store = FAISSVectorStore.load(tmp_path)
            print(f"✓ Loaded vector store with {loaded_store.count} documents")

        return True

    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        return False


def test_end_to_end():
    """Test end-to-end indexing and retrieval."""
    print("\n=== Testing End-to-End RAG ===")

    # Check if we have sample data
    data_dir = Path(__file__).parent / "data" / "filings"

    if not data_dir.exists():
        print("ℹ No sample data found - skipping end-to-end test")
        return True

    companies = [d.name for d in data_dir.iterdir() if d.is_dir()]
    if not companies:
        print("ℹ No company directories found - skipping end-to-end test")
        return True

    test_company = companies[0]
    print(f"ℹ Testing with company: {test_company}")

    try:
        # Test ingestion
        result = ingest_company_filings(test_company, force_reindex=False)
        if result["success"]:
            print(f"✓ Ingestion successful: {result.get('message', 'OK')}")
        else:
            print(f"✗ Ingestion failed: {result.get('error', 'Unknown error')}")
            return False

        # Test retrieval
        if result["documents_count"] > 0:
            chunks = retrieve_for_query(test_company, "What is the revenue?", top_k=3)
            print(f"✓ Retrieved {len(chunks)} chunks")
            if chunks:
                print(f"  Top result score: {chunks[0]['score']:.4f}")

        return True

    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SIMPLIFIED RAG SYSTEM TEST")
    print("=" * 60)

    results = {
        "Chunking": test_chunking(),
        "Embeddings": test_embeddings(),
        "Vector Store": test_vector_store(),
        "End-to-End": test_end_to_end(),
    }

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")

    all_passed = all(results.values())
    print("\n" + ("🎉 ALL TESTS PASSED" if all_passed else "⚠️  SOME TESTS FAILED"))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

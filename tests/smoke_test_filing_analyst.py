"""
Smoke test for Filing Analyst agent with real indexed PDFs and OpenAI API.

This requires:
1. OPENAI_API_KEY environment variable set
2. ChromaDB indexes already created (run Phase 2A indexer first)
3. At least one company indexed in data/chroma_store

Run with: python -m pytest tests/smoke_test_filing_analyst.py -v -s
Or directly: python tests/smoke_test_filing_analyst.py
"""

import os
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.filing_analyst import run_filing_analyst
from backend.rag.retriever import CHROMA_STORE
import chromadb


def test_smoke_filing_analyst_basic():
    """Smoke test: Basic filing analysis with real indexed company."""
    print("\n=== Testing run_filing_analyst (basic) ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    if not CHROMA_STORE.exists():
        print(f"SKIP: ChromaDB store does not exist at {CHROMA_STORE}")
        print("Run Phase 2A indexer first: python tests/smoke_test_indexer.py")
        return

    # Find an indexed company
    try:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_STORE))
        collections = chroma_client.list_collections()

        if not collections:
            print("SKIP: No indexed companies found")
            print("Run Phase 2A indexer first: python tests/smoke_test_indexer.py")
            return

        # Get first company
        collection = collections[0]
        company_name = collection.name.replace("company_", "").replace("_", " ").upper()
        ticker = company_name  # Use company name as ticker for testing

        print(f"Testing with indexed company: {company_name}")
        print(f"Collection: {collection.name}, Documents: {collection.count()}")

        # Run filing analyst
        result = run_filing_analyst(ticker, company_name)

        print(f"\n=== Filing Analysis Result ===")
        print(f"Status: {result['status']}")
        print(f"Ticker: {result['ticker']}")
        print(f"Company: {result['company_name']}")
        print(f"\nFiling Summary:\n{result['filing_summary']}")
        print(f"\nKey Points ({len(result['key_points'])}):")
        for i, point in enumerate(result['key_points'], 1):
            print(f"  {i}. {point}")
        print(f"\nRisks ({len(result['risks'])}):")
        for i, risk in enumerate(result['risks'], 1):
            print(f"  {i}. {risk}")
        print(f"\nOutlook:\n{result['outlook']}")
        print(f"\nSources ({len(result['sources'])}):")
        for source in result['sources']:
            print(f"  - {source}")

        assert result["status"] in ["success", "partial_success"]
        assert result["ticker"] == ticker
        assert result["company_name"] == company_name
        assert len(result["filing_summary"]) > 0
        assert len(result["sources"]) > 0

        print("\n✓ PASSED - Filing analysis completed successfully")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_json_serializable():
    """Smoke test: Verify result is JSON serializable."""
    print("\n=== Testing JSON serialization ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    if not CHROMA_STORE.exists():
        print(f"SKIP: ChromaDB store does not exist")
        return

    try:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_STORE))
        collections = chroma_client.list_collections()

        if not collections:
            print("SKIP: No indexed companies found")
            return

        collection = collections[0]
        company_name = collection.name.replace("company_", "").replace("_", " ").upper()
        ticker = company_name

        result = run_filing_analyst(ticker, company_name)

        # Should be JSON serializable
        json_str = json.dumps(result, indent=2)
        assert json_str is not None

        # Verify structure
        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "ticker" in parsed
        assert "company_name" in parsed
        assert "filing_summary" in parsed
        assert "key_points" in parsed
        assert "risks" in parsed
        assert "outlook" in parsed
        assert "sources" in parsed

        print("✓ PASSED - Result is JSON serializable")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_multiple_companies():
    """Smoke test: Test filing analysis for multiple companies."""
    print("\n=== Testing multiple companies ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    if not CHROMA_STORE.exists():
        print(f"SKIP: ChromaDB store does not exist")
        return

    try:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_STORE))
        collections = chroma_client.list_collections()

        if not collections:
            print("SKIP: No indexed companies found")
            return

        # Test up to 3 companies
        num_to_test = min(3, len(collections))
        print(f"Testing {num_to_test} companies")

        for i, collection in enumerate(collections[:num_to_test], 1):
            company_name = collection.name.replace("company_", "").replace("_", " ").upper()
            ticker = company_name

            print(f"\n--- Company {i}: {company_name} ---")
            result = run_filing_analyst(ticker, company_name)

            print(f"Status: {result['status']}")
            print(f"Summary length: {len(result['filing_summary'])} chars")
            print(f"Key points: {len(result['key_points'])}")
            print(f"Risks: {len(result['risks'])}")
            print(f"Sources: {len(result['sources'])}")

            assert result["status"] in ["success", "partial_success"]

        print("\n✓ PASSED - Multiple companies analyzed successfully")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_missing_company():
    """Smoke test: Graceful handling of non-indexed company."""
    print("\n=== Testing missing company handling ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    try:
        result = run_filing_analyst("NONEXIST", "NONEXISTENT_COMPANY_XYZ")

        print(f"Status: {result['status']}")
        print(f"Summary: {result['filing_summary']}")
        print(f"Error: {result['error']}")

        assert result["status"] in ["partial_success", "error"]
        assert "no" in result["filing_summary"].lower() or result["error"] is not None

        print("✓ PASSED - Gracefully handled missing company")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_result_structure():
    """Smoke test: Verify all expected fields are present."""
    print("\n=== Testing result structure ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    if not CHROMA_STORE.exists():
        print(f"SKIP: ChromaDB store does not exist")
        return

    try:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_STORE))
        collections = chroma_client.list_collections()

        if not collections:
            print("SKIP: No indexed companies found")
            return

        collection = collections[0]
        company_name = collection.name.replace("company_", "").replace("_", " ").upper()
        ticker = company_name

        result = run_filing_analyst(ticker, company_name)

        # Verify all required fields
        required_fields = [
            "status", "ticker", "company_name", "filing_summary",
            "key_points", "risks", "outlook", "sources", "error"
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"
            print(f"✓ {field}: {type(result[field]).__name__}")

        # Verify types
        assert isinstance(result["status"], str)
        assert isinstance(result["ticker"], str)
        assert isinstance(result["company_name"], str)
        assert isinstance(result["filing_summary"], str)
        assert isinstance(result["key_points"], list)
        assert isinstance(result["risks"], list)
        assert isinstance(result["outlook"], str)
        assert isinstance(result["sources"], list)

        print("\n✓ PASSED - Result structure is correct")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


if __name__ == "__main__":
    """Run smoke tests directly."""
    print("=" * 60)
    print("FILING ANALYST SMOKE TESTS")
    print("=" * 60)

    try:
        test_smoke_filing_analyst_basic()
        test_smoke_json_serializable()
        test_smoke_multiple_companies()
        test_smoke_missing_company()
        test_smoke_result_structure()

        print("\n" + "=" * 60)
        print("ALL SMOKE TESTS PASSED ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

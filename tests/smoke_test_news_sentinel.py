"""
Smoke test for News Sentinel agent with real news data and OpenAI API.

This requires:
1. OPENAI_API_KEY environment variable set
2. Internet connection for Google News RSS feed

Run with: python -m pytest tests/smoke_test_news_sentinel.py -v -s
Or directly: python tests/smoke_test_news_sentinel.py
"""

import os
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.news_sentinel import run_news_sentinel


def test_smoke_news_sentinel_reliance():
    """Smoke test: Analyze RELIANCE news with real data."""
    print("\n=== Testing News Sentinel: RELIANCE ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    try:
        result = run_news_sentinel("RELIANCE", "Reliance Industries")

        print(f"\n=== News Sentiment Analysis Result ===")
        print(f"Status: {result['status']}")
        print(f"Ticker: {result['ticker']}")
        print(f"Company: {result['company_name']}")
        print(f"Article Count: {result['article_count']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Sentiment Score: {result['sentiment_score']:.2f}")

        if result['key_themes']:
            print(f"\n=== Key Themes ({len(result['key_themes'])}) ===")
            for i, theme in enumerate(result['key_themes'], 1):
                print(f"  {i}. {theme}")

        if result['positive_drivers']:
            print(f"\n=== Positive Drivers ({len(result['positive_drivers'])}) ===")
            for i, driver in enumerate(result['positive_drivers'], 1):
                print(f"  {i}. {driver}")

        if result['negative_drivers']:
            print(f"\n=== Negative Drivers ({len(result['negative_drivers'])}) ===")
            for i, driver in enumerate(result['negative_drivers'], 1):
                print(f"  {i}. {driver}")

        if result['notable_headlines']:
            print(f"\n=== Notable Headlines ({len(result['notable_headlines'])}) ===")
            for i, headline in enumerate(result['notable_headlines'], 1):
                print(f"  {i}. {headline}")

        print(f"\n=== Sources ===")
        print(f"Total sources: {len(result['sources'])}")
        if result['sources']:
            print(f"First source: {result['sources'][0]}")

        if result['error']:
            print(f"\n⚠ Warning: {result['error']}")

        assert result["status"] in ["success", "partial_success"]
        assert result["ticker"] == "RELIANCE"
        assert result["company_name"] == "Reliance Industries"
        assert result["sentiment"] in ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED", "UNKNOWN"]
        assert -1.0 <= result["sentiment_score"] <= 1.0

        print("\n✓ PASSED - News sentiment analysis completed successfully")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_news_sentinel_infy():
    """Smoke test: Analyze INFY news with real data."""
    print("\n=== Testing News Sentinel: INFY ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    try:
        result = run_news_sentinel("INFY", "Infosys")

        print(f"Status: {result['status']}")
        print(f"Company: {result['company_name']}")
        print(f"Articles: {result['article_count']}")
        print(f"Sentiment: {result['sentiment']} ({result['sentiment_score']:.2f})")
        print(f"Themes: {len(result['key_themes'])}")
        print(f"Sources: {len(result['sources'])}")

        assert result["status"] in ["success", "partial_success"]
        assert result["ticker"] == "INFY"

        print("✓ PASSED")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_json_serializable():
    """Smoke test: Verify result is JSON serializable."""
    print("\n=== Testing JSON serialization ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    try:
        result = run_news_sentinel("TCS", "Tata Consultancy Services")

        # Should be JSON serializable
        json_str = json.dumps(result, indent=2)
        assert json_str is not None

        # Verify structure
        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "sentiment" in parsed
        assert "sentiment_score" in parsed
        assert "key_themes" in parsed

        print("✓ PASSED - Result is JSON serializable")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_sentiment_bounds():
    """Smoke test: Verify sentiment score is within bounds."""
    print("\n=== Testing sentiment score bounds ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    try:
        result = run_news_sentinel("HDFCBANK", "HDFC Bank")

        print(f"Sentiment: {result['sentiment']}")
        print(f"Sentiment Score: {result['sentiment_score']}")

        assert -1.0 <= result["sentiment_score"] <= 1.0
        assert result["sentiment"] in ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED", "UNKNOWN"]

        print("✓ PASSED - Sentiment values within valid bounds")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_source_links():
    """Smoke test: Verify source links are preserved."""
    print("\n=== Testing source link preservation ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    try:
        result = run_news_sentinel("RELIANCE", "Reliance Industries")

        if result["article_count"] > 0:
            assert len(result["sources"]) > 0

            # Verify links are URLs
            for source in result["sources"]:
                assert source.startswith("http"), f"Invalid source link: {source}"

            print(f"Found {len(result['sources'])} valid source links")
            print("✓ PASSED - Source links preserved")
        else:
            print("SKIP: No articles found for testing")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_multiple_companies():
    """Smoke test: Analyze multiple companies."""
    print("\n=== Testing multiple companies ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    companies = [
        ("RELIANCE", "Reliance Industries"),
        ("TCS", "Tata Consultancy Services"),
        ("INFY", "Infosys"),
    ]

    try:
        for ticker, name in companies:
            print(f"\n--- Analyzing {ticker} ---")
            result = run_news_sentinel(ticker, name)

            print(f"Status: {result['status']}")
            print(f"Articles: {result['article_count']}")
            print(f"Sentiment: {result['sentiment']} ({result['sentiment_score']:.2f})")

            assert result["status"] in ["success", "partial_success", "error"]

        print("\n✓ PASSED - Multiple companies analyzed")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_no_articles_handling():
    """Smoke test: Handle case with potentially no articles gracefully."""
    print("\n=== Testing no articles handling ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    try:
        # Use an obscure company name unlikely to have news
        result = run_news_sentinel("OBSCURE", "Obscure Unknown Company XYZ")

        print(f"Status: {result['status']}")
        print(f"Article Count: {result['article_count']}")
        print(f"Sentiment: {result['sentiment']}")

        if result["article_count"] == 0:
            assert result["status"] in ["partial_success", "error"]
            assert result["sentiment"] == "UNKNOWN"
            print("✓ PASSED - Gracefully handled no articles")
        else:
            print(f"Note: Found {result['article_count']} articles unexpectedly")
            print("✓ PASSED - Analysis completed")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


if __name__ == "__main__":
    """Run smoke tests directly."""
    print("=" * 60)
    print("NEWS SENTINEL SMOKE TESTS")
    print("=" * 60)

    try:
        test_smoke_news_sentinel_reliance()
        test_smoke_news_sentinel_infy()
        test_smoke_json_serializable()
        test_smoke_sentiment_bounds()
        test_smoke_source_links()
        test_smoke_multiple_companies()
        test_smoke_no_articles_handling()

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

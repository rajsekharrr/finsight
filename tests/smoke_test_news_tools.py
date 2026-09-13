"""
Live smoke test for news_tools.py
Tests against real Google News RSS feed.
Run only when you have internet connection.
"""
import sys
import logging
from backend.tools.news_tools import fetch_google_news, fetch_economic_times_markets

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_live_google_news():
    """Test fetch_google_news with live data"""
    print("\n" + "="*70)
    print("TEST: fetch_google_news('Reliance Industries', 'RELIANCE')")
    print("="*70)

    articles = fetch_google_news("Reliance Industries", "RELIANCE", limit=10)

    if not articles:
        print("❌ FAILED: No articles returned")
        return False

    print(f"✓ Fetched {len(articles)} articles")
    print("\nFirst 3 articles:")
    for i, article in enumerate(articles[:3], 1):
        print(f"\n  Article {i}:")
        print(f"  Title: {article.get('title', 'N/A')[:80]}")
        print(f"  Source: {article.get('source', 'N/A')}")
        print(f"  Published: {article.get('published', 'N/A')[:30]}")
        print(f"  Link: {article.get('link', 'N/A')[:60]}...")

    return True


def test_live_et_markets():
    """Test fetch_economic_times_markets with live data"""
    print("\n" + "="*70)
    print("TEST: fetch_economic_times_markets()")
    print("="*70)

    articles = fetch_economic_times_markets(limit=5)

    if not articles:
        print("⚠️  WARNING: No articles returned (ET RSS may be down)")
        return True  # Non-fatal

    print(f"✓ Fetched {len(articles)} market news articles")
    print("\nFirst 2 articles:")
    for i, article in enumerate(articles[:2], 1):
        print(f"\n  Article {i}:")
        print(f"  Title: {article.get('title', 'N/A')[:80]}")
        print(f"  Source: {article.get('source', 'N/A')}")

    return True


def test_cache_functionality():
    """Test that caching works"""
    print("\n" + "="*70)
    print("TEST: Cache functionality")
    print("="*70)

    # First call - should fetch
    print("First call (should fetch from internet)...")
    articles1 = fetch_google_news("TCS", "TCS", limit=5)

    # Second call - should use cache
    print("Second call (should use cache)...")
    articles2 = fetch_google_news("TCS", "TCS", limit=5)

    if articles1 == articles2:
        print("✓ Cache working: Both calls returned same data")
        return True
    else:
        print("⚠️  Cache may not be working properly")
        return False


def main():
    """Run all live smoke tests"""
    print("\n" + "="*70)
    print("NEWS TOOLS - LIVE SMOKE TEST")
    print("Testing against real Google News RSS feeds")
    print("="*70)

    tests = [
        ("Google News - Reliance", test_live_google_news),
        ("Economic Times Markets", test_live_et_markets),
        ("Cache Functionality", test_cache_functionality),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} raised exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed_count}/{total_count} passed")

    if passed_count == total_count:
        print("\n🎉 All live smoke tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

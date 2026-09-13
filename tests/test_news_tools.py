"""
Tests for backend/tools/news_tools.py
Uses mocking to avoid dependence on live internet.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.tools.news_tools import fetch_google_news, fetch_economic_times_markets


# ============================================================================
# MOCK DATA
# ============================================================================

def _create_mock_entry(title, published, source, summary, link):
    """Create a properly mocked feed entry"""
    entry = Mock()
    # Set attributes directly
    entry.title = title
    entry.published = published
    entry.summary = summary
    entry.link = link
    entry.source = {"title": source}

    # Also mock .get() method for direct access
    def mock_get(key, default=""):
        return getattr(entry, key, default)
    entry.get = mock_get

    return entry


MOCK_FEED_ENTRIES = [
    _create_mock_entry(
        title="Reliance Industries Q4 results exceed expectations",
        published="Wed, 10 Jan 2024 14:30:00 GMT",
        source="Economic Times",
        summary="Reliance Industries reported strong Q4 results with revenue growth of 12%",
        link="https://example.com/news1"
    ),
    _create_mock_entry(
        title="Reliance to invest $10B in green energy",
        published="Tue, 09 Jan 2024 10:15:00 GMT",
        source="Business Standard",
        summary="Company announces major renewable energy expansion",
        link="https://example.com/news2"
    ),
    _create_mock_entry(
        title="Stock market update: Reliance shares up 3%",
        published="Mon, 08 Jan 2024 16:45:00 GMT",
        source="CNBC",
        summary="Reliance stock rallies on positive sentiment",
        link="https://example.com/news3"
    ),
]


# ============================================================================
# FETCH_GOOGLE_NEWS TESTS
# ============================================================================

class TestFetchGoogleNews:
    """Tests for fetch_google_news function"""

    @patch('backend.tools.news_tools.cache_get')
    @patch('backend.tools.news_tools.cache_set')
    @patch('backend.tools.news_tools.feedparser.parse')
    def test_fetch_google_news_success(self, mock_parse, mock_cache_set, mock_cache_get):
        """Test successful news fetch"""
        mock_cache_get.return_value = None

        # Mock feedparser response
        mock_feed = Mock()
        mock_feed.entries = MOCK_FEED_ENTRIES
        mock_parse.return_value = mock_feed

        result = fetch_google_news("Reliance Industries", "RELIANCE", limit=10)

        assert len(result) == 3
        assert result[0]["title"] == "Reliance Industries Q4 results exceed expectations"
        assert result[0]["source"] == "Economic Times"
        assert result[0]["link"] == "https://example.com/news1"
        assert "published" in result[0]
        assert "summary" in result[0]

        mock_cache_set.assert_called_once()

    @patch('backend.tools.news_tools.cache_get')
    @patch('backend.tools.news_tools.feedparser.parse')
    def test_fetch_google_news_cache_hit(self, mock_parse, mock_cache_get):
        """Test cache hit returns cached data"""
        cached_articles = [
            {"title": "Cached article", "source": "Test", "link": "http://test.com"}
        ]
        mock_cache_get.return_value = cached_articles

        result = fetch_google_news("Reliance Industries")

        assert result == cached_articles
        mock_parse.assert_not_called()

    @patch('backend.tools.news_tools.cache_get')
    @patch('backend.tools.news_tools.cache_set')
    @patch('backend.tools.news_tools.feedparser.parse')
    def test_fetch_google_news_no_results_fallback(self, mock_parse, mock_cache_set, mock_cache_get):
        """Test fallback to broader query when no results"""
        mock_cache_get.return_value = None

        # First call returns empty, second call returns results
        mock_feed_empty = Mock(entries=[])
        mock_feed_success = Mock(entries=MOCK_FEED_ENTRIES[:1])

        mock_parse.side_effect = [mock_feed_empty, mock_feed_success]

        result = fetch_google_news("Reliance Industries", "RELIANCE.NS")

        assert len(result) == 1
        assert mock_parse.call_count == 2  # Called twice due to fallback

    @patch('backend.tools.news_tools.cache_get')
    @patch('backend.tools.news_tools.cache_set')
    @patch('backend.tools.news_tools.feedparser.parse')
    def test_fetch_google_news_empty_result(self, mock_parse, mock_cache_set, mock_cache_get):
        """Test handling of empty feed"""
        mock_cache_get.return_value = None

        mock_feed = Mock(entries=[])
        mock_parse.return_value = mock_feed

        result = fetch_google_news("UnknownCompany")

        assert result == []
        assert isinstance(result, list)

    @patch('backend.tools.news_tools.cache_get')
    @patch('backend.tools.news_tools.feedparser.parse')
    def test_fetch_google_news_error_handling(self, mock_parse, mock_cache_get):
        """Test error handling returns empty list (non-fatal)"""
        mock_cache_get.return_value = None
        mock_parse.side_effect = Exception("Network error")

        result = fetch_google_news("Test Company")

        assert result == []
        assert isinstance(result, list)

    @patch('backend.tools.news_tools.cache_get')
    @patch('backend.tools.news_tools.cache_set')
    @patch('backend.tools.news_tools.feedparser.parse')
    def test_fetch_google_news_limit(self, mock_parse, mock_cache_set, mock_cache_get):
        """Test limit parameter works correctly"""
        mock_cache_get.return_value = None

        # Create 10 mock entries
        many_entries = [
            _create_mock_entry(
                title=f"Article {i}",
                published="2024-01-01",
                source="Source",
                summary=f"Summary {i}",
                link=f"http://example.com/{i}"
            )
            for i in range(10)
        ]

        mock_feed = Mock(entries=many_entries)
        mock_parse.return_value = mock_feed

        result = fetch_google_news("Test", limit=5)

        assert len(result) == 5
        assert result[0]["title"] == "Article 0"
        assert result[4]["title"] == "Article 4"

    @patch('backend.tools.news_tools.cache_get')
    @patch('backend.tools.news_tools.cache_set')
    @patch('backend.tools.news_tools.feedparser.parse')
    def test_fetch_google_news_missing_fields(self, mock_parse, mock_cache_set, mock_cache_get):
        """Test handling of entries with missing fields"""
        mock_cache_get.return_value = None

        incomplete_entry = _create_mock_entry(
            title="Test",
            published="",
            source="",
            summary="",
            link=""
        )
        incomplete_entry.source = {}  # Empty source dict

        mock_feed = Mock(entries=[incomplete_entry])
        mock_parse.return_value = mock_feed

        result = fetch_google_news("Test")

        assert len(result) == 1
        assert result[0]["title"] == "Test"
        assert result[0]["source"] == "Google News"  # Default value


# ============================================================================
# FETCH_ECONOMIC_TIMES_MARKETS TESTS
# ============================================================================

class TestFetchEconomicTimesMarkets:
    """Tests for fetch_economic_times_markets function"""

    @patch('backend.tools.news_tools.cache_get')
    @patch('backend.tools.news_tools.cache_set')
    @patch('backend.tools.news_tools.feedparser.parse')
    def test_fetch_et_markets_success(self, mock_parse, mock_cache_set, mock_cache_get):
        """Test successful Economic Times fetch"""
        mock_cache_get.return_value = None

        mock_feed = Mock()
        mock_feed.entries = MOCK_FEED_ENTRIES[:2]
        mock_parse.return_value = mock_feed

        result = fetch_economic_times_markets(limit=5)

        assert len(result) == 2
        assert result[0]["source"] == "Economic Times"

    @patch('backend.tools.news_tools.cache_get')
    @patch('backend.tools.news_tools.feedparser.parse')
    def test_fetch_et_markets_cache_hit(self, mock_parse, mock_cache_get):
        """Test cache hit"""
        cached = [{"title": "Cached ET news"}]
        mock_cache_get.return_value = cached

        result = fetch_economic_times_markets()

        assert result == cached
        mock_parse.assert_not_called()

    @patch('backend.tools.news_tools.cache_get')
    @patch('backend.tools.news_tools.feedparser.parse')
    def test_fetch_et_markets_error(self, mock_parse, mock_cache_get):
        """Test error handling"""
        mock_cache_get.return_value = None
        mock_parse.side_effect = Exception("RSS feed error")

        result = fetch_economic_times_markets()

        assert result == []


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestNewsIntegration:
    """Integration tests for news tools"""

    @patch('backend.tools.news_tools.cache_get')
    @patch('backend.tools.news_tools.cache_set')
    @patch('backend.tools.news_tools.feedparser.parse')
    def test_ticker_suffix_removal(self, mock_parse, mock_cache_set, mock_cache_get):
        """Test that .NS and .BO suffixes are removed from ticker in query"""
        mock_cache_get.return_value = None

        mock_feed = Mock(entries=[])
        mock_parse.return_value = mock_feed

        # Should work with both .NS and .BO
        fetch_google_news("Reliance", "RELIANCE.NS")
        fetch_google_news("Reliance", "RELIANCE.BO")

        # Should not crash, just returns empty list
        assert mock_parse.call_count == 4  # 2 initial + 2 fallback

    @patch('backend.tools.news_tools.cache_get')
    @patch('backend.tools.news_tools.cache_set')
    @patch('backend.tools.news_tools.feedparser.parse')
    def test_company_name_with_special_chars(self, mock_parse, mock_cache_set, mock_cache_get):
        """Test company names with special characters"""
        mock_cache_get.return_value = None

        mock_feed = Mock(entries=MOCK_FEED_ENTRIES[:1])
        mock_parse.return_value = mock_feed

        result = fetch_google_news("Larsen & Toubro", "L&T")

        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

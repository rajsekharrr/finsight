"""
Tests for News Sentinel agent.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from backend.agents.news_sentinel import run_news_sentinel


class TestNewsSentinel:
    """Tests for run_news_sentinel function."""

    @patch("backend.agents.news_sentinel.fetch_google_news")
    @patch("backend.agents.news_sentinel.ChatOpenAI")
    def test_successful_sentiment_analysis(self, mock_chat, mock_fetch_news):
        """Should successfully analyze news sentiment."""
        # Mock news articles
        mock_fetch_news.return_value = [
            {
                "title": "Company reports strong quarterly earnings",
                "published": "2024-01-15",
                "source": "Economic Times",
                "summary": "Strong performance in Q3",
                "link": "https://example.com/news1"
            },
            {
                "title": "Company announces expansion plans",
                "published": "2024-01-14",
                "source": "Business Standard",
                "summary": "New facility announced",
                "link": "https://example.com/news2"
            }
        ]

        # Mock LLM response
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "sentiment": "POSITIVE",
            "sentiment_score": 0.7,
            "key_themes": ["Earnings growth", "Expansion", "Market share"],
            "positive_drivers": ["Strong Q3 results", "New facility investment"],
            "negative_drivers": [],
            "notable_headlines": [
                "Company reports strong quarterly earnings",
                "Company announces expansion plans"
            ]
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        result = run_news_sentinel("TEST", "Test Company")

        assert result["status"] == "success"
        assert result["ticker"] == "TEST"
        assert result["company_name"] == "Test Company"
        assert result["article_count"] == 2
        assert result["sentiment"] == "POSITIVE"
        assert result["sentiment_score"] == 0.7
        assert len(result["key_themes"]) == 3
        assert len(result["positive_drivers"]) == 2
        assert len(result["negative_drivers"]) == 0
        assert len(result["notable_headlines"]) == 2
        assert len(result["sources"]) == 2
        assert "https://example.com/news1" in result["sources"]
        assert result["error"] is None

    @patch("backend.agents.news_sentinel.fetch_google_news")
    def test_no_articles_found(self, mock_fetch_news):
        """Should handle case when no articles are found."""
        mock_fetch_news.return_value = []

        result = run_news_sentinel("TEST", "Test Company")

        assert result["status"] == "partial_success"
        assert result["article_count"] == 0
        assert result["sentiment"] == "UNKNOWN"
        assert result["error"] == "No recent news articles found"
        assert result["sources"] == []

    @patch("backend.agents.news_sentinel.fetch_google_news")
    @patch("backend.agents.news_sentinel.ChatOpenAI")
    def test_malformed_llm_json(self, mock_chat, mock_fetch_news):
        """Should handle malformed JSON from LLM gracefully."""
        # Mock news articles
        mock_fetch_news.return_value = [
            {
                "title": "Test headline",
                "published": "2024-01-15",
                "source": "Test Source",
                "summary": "Test summary",
                "link": "https://example.com/news"
            }
        ]

        # Mock LLM with invalid JSON
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON {incomplete"
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        result = run_news_sentinel("TEST", "Test Company")

        assert result["status"] == "partial_success"
        assert result["sentiment"] == "UNKNOWN"
        assert "JSON parse error" in result["error"]
        assert result["article_count"] == 1
        # Should have fallback data
        assert len(result["notable_headlines"]) > 0

    @patch("backend.agents.news_sentinel.fetch_google_news")
    @patch("backend.agents.news_sentinel.ChatOpenAI")
    def test_llm_exception(self, mock_chat, mock_fetch_news):
        """Should handle LLM call exceptions gracefully."""
        # Mock news articles
        mock_fetch_news.return_value = [
            {
                "title": "Test headline",
                "published": "2024-01-15",
                "source": "Test Source",
                "summary": "Test summary",
                "link": "https://example.com/news"
            }
        ]

        # Mock LLM to raise exception
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = Exception("OpenAI API timeout")
        mock_chat.return_value = mock_llm_instance

        result = run_news_sentinel("TEST", "Test Company")

        assert result["status"] == "partial_success"
        assert result["sentiment"] == "UNKNOWN"
        assert "LLM analysis failed" in result["error"]
        assert result["article_count"] == 1
        # Should have fallback data
        assert len(result["key_themes"]) > 0

    @patch("backend.agents.news_sentinel.fetch_google_news")
    def test_feed_fetch_exception(self, mock_fetch_news):
        """Should handle news fetch exceptions gracefully."""
        mock_fetch_news.side_effect = Exception("Network error")

        result = run_news_sentinel("TEST", "Test Company")

        assert result["status"] == "error"
        assert "Network error" in result["error"]
        assert result["article_count"] == 0

    @patch("backend.agents.news_sentinel.fetch_google_news")
    @patch("backend.agents.news_sentinel.ChatOpenAI")
    def test_source_preservation(self, mock_chat, mock_fetch_news):
        """Should preserve article source links."""
        # Mock news articles with links
        mock_fetch_news.return_value = [
            {
                "title": "Article 1",
                "link": "https://example.com/article1"
            },
            {
                "title": "Article 2",
                "link": "https://example.com/article2"
            },
            {
                "title": "Article 3",
                "link": "https://example.com/article3"
            }
        ]

        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "sentiment": "NEUTRAL",
            "sentiment_score": 0.0,
            "key_themes": ["Theme"],
            "positive_drivers": [],
            "negative_drivers": [],
            "notable_headlines": ["Article 1"]
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        result = run_news_sentinel("TEST", "Test Company")

        assert len(result["sources"]) == 3
        assert "https://example.com/article1" in result["sources"]
        assert "https://example.com/article2" in result["sources"]
        assert "https://example.com/article3" in result["sources"]

    @patch("backend.agents.news_sentinel.fetch_google_news")
    @patch("backend.agents.news_sentinel.ChatOpenAI")
    def test_json_serializable_result(self, mock_chat, mock_fetch_news):
        """Should return JSON-serializable result."""
        mock_fetch_news.return_value = [
            {"title": "Test", "link": "https://example.com"}
        ]

        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "sentiment": "NEUTRAL",
            "sentiment_score": 0.0,
            "key_themes": ["Theme"],
            "positive_drivers": [],
            "negative_drivers": [],
            "notable_headlines": ["Test"]
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        result = run_news_sentinel("TEST", "Test Company")

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        # Verify structure
        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "sentiment" in parsed
        assert "sentiment_score" in parsed
        assert "key_themes" in parsed
        assert "positive_drivers" in parsed
        assert "negative_drivers" in parsed

    @patch("backend.agents.news_sentinel.fetch_google_news")
    @patch("backend.agents.news_sentinel.ChatOpenAI")
    def test_valid_sentiment_labels(self, mock_chat, mock_fetch_news):
        """Should only use valid sentiment labels."""
        mock_fetch_news.return_value = [
            {"title": "Test", "link": "https://example.com"}
        ]

        # Test with invalid sentiment
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "sentiment": "INVALID_SENTIMENT",  # Invalid
            "sentiment_score": 0.5,
            "key_themes": [],
            "positive_drivers": [],
            "negative_drivers": [],
            "notable_headlines": []
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        result = run_news_sentinel("TEST", "Test Company")

        # Should default to UNKNOWN for invalid sentiment
        assert result["sentiment"] == "UNKNOWN"

    @patch("backend.agents.news_sentinel.fetch_google_news")
    @patch("backend.agents.news_sentinel.ChatOpenAI")
    def test_sentiment_score_bounds(self, mock_chat, mock_fetch_news):
        """Should clamp sentiment_score to [-1, 1] range."""
        mock_fetch_news.return_value = [
            {"title": "Test", "link": "https://example.com"}
        ]

        # Test with out-of-bounds score
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "sentiment": "POSITIVE",
            "sentiment_score": 2.5,  # Out of bounds
            "key_themes": [],
            "positive_drivers": [],
            "negative_drivers": [],
            "notable_headlines": []
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        result = run_news_sentinel("TEST", "Test Company")

        # Should clamp to 1.0
        assert result["sentiment_score"] == 1.0
        assert -1.0 <= result["sentiment_score"] <= 1.0

    @patch("backend.agents.news_sentinel.fetch_google_news")
    @patch("backend.agents.news_sentinel.ChatOpenAI")
    def test_strips_markdown_code_fences(self, mock_chat, mock_fetch_news):
        """Should strip markdown code fences from LLM response."""
        mock_fetch_news.return_value = [
            {"title": "Test", "link": "https://example.com"}
        ]

        # Mock LLM with markdown fences
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """```json
{
  "sentiment": "POSITIVE",
  "sentiment_score": 0.8,
  "key_themes": ["Growth"],
  "positive_drivers": ["Strong results"],
  "negative_drivers": [],
  "notable_headlines": ["Test"]
}
```"""
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        result = run_news_sentinel("TEST", "Test Company")

        assert result["status"] == "success"
        assert result["sentiment"] == "POSITIVE"
        assert result["sentiment_score"] == 0.8

    @patch("backend.agents.news_sentinel.fetch_google_news")
    @patch("backend.agents.news_sentinel.ChatOpenAI")
    def test_all_sentiment_types(self, mock_chat, mock_fetch_news):
        """Should handle all valid sentiment types."""
        mock_fetch_news.return_value = [
            {"title": "Test", "link": "https://example.com"}
        ]

        sentiments_to_test = ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]

        for sentiment in sentiments_to_test:
            mock_llm_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.content = json.dumps({
                "sentiment": sentiment,
                "sentiment_score": 0.5,
                "key_themes": [],
                "positive_drivers": [],
                "negative_drivers": [],
                "notable_headlines": []
            })
            mock_llm_instance.invoke.return_value = mock_response
            mock_chat.return_value = mock_llm_instance

            result = run_news_sentinel("TEST", "Test Company")
            assert result["sentiment"] == sentiment

    @patch("backend.agents.news_sentinel.fetch_google_news")
    @patch("backend.agents.news_sentinel.ChatOpenAI")
    def test_llm_configuration(self, mock_chat, mock_fetch_news):
        """Should configure LLM with correct parameters."""
        mock_fetch_news.return_value = [
            {"title": "Test", "link": "https://example.com"}
        ]

        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "sentiment": "NEUTRAL",
            "sentiment_score": 0.0,
            "key_themes": [],
            "positive_drivers": [],
            "negative_drivers": [],
            "notable_headlines": []
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        run_news_sentinel("TEST", "Test Company")

        # Verify LLM was configured correctly
        mock_chat.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0,
            timeout=60
        )

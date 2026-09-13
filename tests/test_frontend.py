"""
Tests for Gradio frontend helper functions.

These tests verify formatting logic without requiring browser automation.
"""

import pytest
from frontend.app import (
    format_json,
    format_ratios_display,
    format_news_display,
    format_technical_display,
    format_risk_display,
    format_errors_display
)


class TestFrontendHelpers:
    """Tests for frontend formatting helper functions."""

    def test_format_json_with_dict(self):
        """Should format dictionary as pretty JSON."""
        data = {"key": "value", "number": 42}
        result = format_json(data)

        assert '"key": "value"' in result
        assert '"number": 42' in result

    def test_format_json_with_none(self):
        """Should handle None input."""
        result = format_json(None)
        assert result == "No data available"

    def test_format_ratios_display_success(self):
        """Should format successful ratio output."""
        ratio_output = {
            "status": "success",
            "ratios": {
                "roe_pct": 18.5,
                "roa_pct": 8.2,
                "net_margin_pct": 12.5,
                "debt_to_equity": 45.3,
                "current_ratio": 1.8,
                "pe_ratio": 28.5
            }
        }

        result = format_ratios_display(ratio_output)

        assert "Financial Ratios" in result
        assert "18.5" in result
        assert "ROE" in result
        assert "Profitability" in result
        assert "Valuation" in result

    def test_format_ratios_display_error(self):
        """Should handle error status."""
        ratio_output = {"status": "error", "error": "No data"}
        result = format_ratios_display(ratio_output)

        assert "error" in result.lower()

    def test_format_news_display_success(self):
        """Should format successful news output."""
        news_output = {
            "status": "success",
            "sentiment": "POSITIVE",
            "sentiment_score": 0.7,
            "article_count": 15,
            "key_themes": ["Growth", "Expansion"],
            "positive_drivers": ["Strong earnings"],
            "negative_drivers": ["Market volatility"]
        }

        result = format_news_display(news_output)

        assert "News Intelligence" in result
        assert "POSITIVE" in result
        assert "Growth" in result
        assert "Strong earnings" in result

    def test_format_technical_display_success(self):
        """Should format successful technical output."""
        technical_output = {
            "status": "success",
            "trend": "BULLISH",
            "momentum": "NEUTRAL",
            "technical_score": 68,
            "indicators": {
                "latest_price": 2485.50,
                "sma_20": 2420.00,
                "rsi_14": 58.5
            },
            "signals": ["Price above 20-day SMA"]
        }

        result = format_technical_display(technical_output)

        assert "Technical Analysis" in result
        assert "BULLISH" in result
        assert "2485.5" in result  # Python displays as 2485.5, not 2485.50
        assert "Price above 20-day SMA" in result

    def test_format_risk_display_success(self):
        """Should format successful risk output."""
        risk_output = {
            "status": "success",
            "risk_level": "MEDIUM",
            "risk_score": 42,
            "metrics": {
                "beta": 1.15,
                "volatility_pct": 28.5,
                "max_drawdown_pct": 18.5
            },
            "risk_factors": ["Moderate volatility", "Elevated valuation"]
        }

        result = format_risk_display(risk_output)

        assert "Risk Scorecard" in result
        assert "MEDIUM" in result
        assert "42" in result
        assert "Moderate volatility" in result

    def test_format_errors_display_no_errors(self):
        """Should format debug info with no errors."""
        result_data = {
            "ticker": "RELIANCE.NS",
            "company_name": "Reliance Industries",
            "sector": "Energy",
            "errors": [],
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:01:00",
            "filing_output": {"status": "success"},
            "ratio_output": {"status": "success"}
        }

        result = format_errors_display(result_data)

        assert "Debug Information" in result
        assert "No errors encountered" in result
        assert "RELIANCE.NS" in result

    def test_format_errors_display_with_errors(self):
        """Should format debug info with errors."""
        result_data = {
            "ticker": "TEST.NS",
            "company_name": "Test Company",
            "errors": ["filing_analyst error: No PDFs found", "news_sentinel error: API timeout"],
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:01:00",
            "filing_output": {"status": "error"},
            "ratio_output": {"status": "success"}
        }

        result = format_errors_display(result_data)

        assert "Errors (2)" in result
        assert "filing_analyst" in result
        assert "news_sentinel" in result

    def test_format_news_display_partial_success(self):
        """Should handle partial_success status."""
        news_output = {
            "status": "partial_success",
            "sentiment": "NEUTRAL",
            "sentiment_score": 0.0,
            "article_count": 3
        }

        result = format_news_display(news_output)

        assert "News Intelligence" in result
        assert "NEUTRAL" in result

    def test_format_risk_display_partial_success(self):
        """Should handle partial_success status."""
        risk_output = {
            "status": "partial_success",
            "risk_level": "MEDIUM",
            "risk_score": 50
        }

        result = format_risk_display(risk_output)

        assert "Risk Scorecard" in result
        assert "MEDIUM" in result

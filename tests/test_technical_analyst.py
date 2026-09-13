"""
Tests for Technical Analyst agent.
"""

import pytest
import json
from unittest.mock import Mock, patch
from backend.agents.technical_analyst import run_technical_analyst


class TestTechnicalAnalyst:
    """Tests for run_technical_analyst function."""

    @patch("backend.agents.technical_analyst.compute_technical_indicators")
    def test_successful_technical_analysis(self, mock_compute):
        """Should successfully analyze technical indicators."""
        # Mock technical indicators
        mock_compute.return_value = {
            "ticker": "RELIANCE.NS",
            "period": "1y",
            "current_price": 2500.0,
            "sma_20": 2450.0,
            "sma_50": 2400.0,
            "sma_200": 2300.0,
            "ema_12": 2480.0,
            "ema_26": 2420.0,
            "macd": 15.5,
            "macd_signal": 12.3,
            "macd_histogram": 3.2,
            "rsi": 58.5,
            "volume_avg_20d": 5000000,
            "latest_volume": 6000000,
            "support_20d": 2350.0,
            "resistance_20d": 2550.0,
            "52w_high": 2600.0,
            "52w_low": 2100.0,
            "price_return_pct": 15.5,
            "trend": "BULLISH",
            "momentum": "NEUTRAL",
            "technical_score": 65.0,
        }

        result = run_technical_analyst("RELIANCE", "Reliance Industries")

        assert result["status"] == "success"
        assert result["ticker"] == "RELIANCE"
        assert result["company_name"] == "Reliance Industries"
        assert result["trend"] == "BULLISH"
        assert result["momentum"] == "NEUTRAL"
        assert result["technical_score"] == 65.0
        assert result["indicators"]["latest_price"] == 2500.0
        assert result["indicators"]["rsi_14"] == 58.5
        assert len(result["signals"]) > 0
        assert result["support_resistance"]["support_20d"] == 2350.0
        assert len(result["summary"]) > 0
        assert result["error"] is None

    @patch("backend.agents.technical_analyst.compute_technical_indicators")
    def test_bullish_signal_generation(self, mock_compute):
        """Should generate bullish signals correctly."""
        mock_compute.return_value = {
            "ticker": "TEST.NS",
            "current_price": 1000.0,
            "sma_20": 950.0,
            "sma_50": 900.0,
            "sma_200": 850.0,
            "macd": 10.0,
            "macd_signal": 8.0,
            "rsi": 55.0,
            "trend": "BULLISH",
            "momentum": "NEUTRAL",
            "technical_score": 70.0,
            "support_20d": 950.0,
            "resistance_20d": 1050.0,
            "52w_high": 1100.0,
            "52w_low": 800.0,
            "price_return_pct": 20.0,
        }

        result = run_technical_analyst("TEST", "Test Company")

        signals_text = " ".join(result["signals"])
        assert "bullish" in signals_text.lower()
        assert "above" in signals_text.lower()
        assert result["trend"] == "BULLISH"

    @patch("backend.agents.technical_analyst.compute_technical_indicators")
    def test_bearish_signal_generation(self, mock_compute):
        """Should generate bearish signals correctly."""
        mock_compute.return_value = {
            "ticker": "TEST.NS",
            "current_price": 800.0,
            "sma_20": 850.0,
            "sma_50": 900.0,
            "sma_200": 950.0,
            "macd": -5.0,
            "macd_signal": -3.0,
            "rsi": 35.0,
            "trend": "BEARISH",
            "momentum": "NEUTRAL",
            "technical_score": 30.0,
            "support_20d": 750.0,
            "resistance_20d": 850.0,
            "52w_high": 1000.0,
            "52w_low": 700.0,
            "price_return_pct": -15.0,
        }

        result = run_technical_analyst("TEST", "Test Company")

        signals_text = " ".join(result["signals"])
        assert "bearish" in signals_text.lower() or "downtrend" in signals_text.lower()
        assert "below" in signals_text.lower()
        assert result["trend"] == "BEARISH"

    @patch("backend.agents.technical_analyst.compute_technical_indicators")
    def test_rsi_overbought_handling(self, mock_compute):
        """Should detect RSI overbought conditions."""
        mock_compute.return_value = {
            "ticker": "TEST.NS",
            "current_price": 1000.0,
            "sma_20": 950.0,
            "sma_50": 900.0,
            "macd": 5.0,
            "macd_signal": 4.0,
            "rsi": 75.0,  # Overbought
            "trend": "BULLISH",
            "momentum": "OVERBOUGHT",
            "technical_score": 60.0,
            "support_20d": 950.0,
            "resistance_20d": 1050.0,
            "52w_high": 1100.0,
            "52w_low": 800.0,
            "price_return_pct": 15.0,
        }

        result = run_technical_analyst("TEST", "Test Company")

        assert result["momentum"] == "OVERBOUGHT"
        signals_text = " ".join(result["signals"])
        assert "overbought" in signals_text.lower()
        assert "75.0" in signals_text or "75" in signals_text

    @patch("backend.agents.technical_analyst.compute_technical_indicators")
    def test_rsi_oversold_handling(self, mock_compute):
        """Should detect RSI oversold conditions."""
        mock_compute.return_value = {
            "ticker": "TEST.NS",
            "current_price": 800.0,
            "sma_20": 850.0,
            "sma_50": 900.0,
            "macd": -3.0,
            "macd_signal": -2.0,
            "rsi": 25.0,  # Oversold
            "trend": "BEARISH",
            "momentum": "OVERSOLD",
            "technical_score": 35.0,
            "support_20d": 750.0,
            "resistance_20d": 850.0,
            "52w_high": 1000.0,
            "52w_low": 700.0,
            "price_return_pct": -20.0,
        }

        result = run_technical_analyst("TEST", "Test Company")

        assert result["momentum"] == "OVERSOLD"
        signals_text = " ".join(result["signals"])
        assert "oversold" in signals_text.lower()
        assert "25.0" in signals_text or "25" in signals_text

    @patch("backend.agents.technical_analyst.compute_technical_indicators")
    def test_missing_optional_indicators(self, mock_compute):
        """Should handle missing optional indicators gracefully."""
        mock_compute.return_value = {
            "ticker": "TEST.NS",
            "current_price": 1000.0,
            "sma_20": 950.0,
            "sma_50": None,  # Missing
            "sma_200": None,  # Missing
            "macd": None,  # Missing
            "macd_signal": None,  # Missing
            "rsi": 50.0,
            "trend": "NEUTRAL",
            "momentum": "NEUTRAL",
            "technical_score": 50.0,
            "support_20d": None,
            "resistance_20d": None,
            "52w_high": 1100.0,
            "52w_low": 900.0,
            "price_return_pct": 5.0,
        }

        result = run_technical_analyst("TEST", "Test Company")

        assert result["status"] == "success"
        assert result["indicators"]["sma_50"] is None
        assert result["indicators"]["macd"] is None
        # Should still generate some signals
        assert len(result["signals"]) > 0

    @patch("backend.agents.technical_analyst.compute_technical_indicators")
    def test_tool_error_handling(self, mock_compute):
        """Should handle technical indicator computation errors."""
        mock_compute.return_value = {
            "error": "Price data unavailable",
            "ticker": "INVALID"
        }

        result = run_technical_analyst("INVALID", "Invalid Company")

        assert result["status"] == "error"
        assert "Price data unavailable" in result["error"]
        assert result["trend"] == "NEUTRAL"
        assert result["indicators"] == {}

    @patch("backend.agents.technical_analyst.compute_technical_indicators")
    def test_json_serializable_result(self, mock_compute):
        """Should return JSON-serializable result."""
        mock_compute.return_value = {
            "ticker": "TEST.NS",
            "current_price": 1000.0,
            "sma_20": 950.0,
            "sma_50": 900.0,
            "rsi": 55.0,
            "trend": "BULLISH",
            "momentum": "NEUTRAL",
            "technical_score": 65.0,
            "support_20d": 950.0,
            "resistance_20d": 1050.0,
            "52w_high": 1100.0,
            "52w_low": 800.0,
            "price_return_pct": 15.0,
        }

        result = run_technical_analyst("TEST", "Test Company")

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        # Verify structure
        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "trend" in parsed
        assert "momentum" in parsed
        assert "technical_score" in parsed
        assert "indicators" in parsed
        assert "signals" in parsed

    @patch("backend.agents.technical_analyst.compute_technical_indicators")
    def test_technical_score_bounds(self, mock_compute):
        """Should keep technical_score within bounds."""
        mock_compute.return_value = {
            "ticker": "TEST.NS",
            "current_price": 1000.0,
            "sma_20": 950.0,
            "sma_50": 900.0,
            "rsi": 55.0,
            "trend": "BULLISH",
            "momentum": "NEUTRAL",
            "technical_score": 75.0,
            "support_20d": 950.0,
            "resistance_20d": 1050.0,
            "52w_high": 1100.0,
            "52w_low": 800.0,
            "price_return_pct": 15.0,
        }

        result = run_technical_analyst("TEST", "Test Company")

        assert 0 <= result["technical_score"] <= 100

    @patch("backend.agents.technical_analyst.compute_technical_indicators")
    def test_golden_cross_detection(self, mock_compute):
        """Should detect golden cross (20-day SMA above 50-day SMA)."""
        mock_compute.return_value = {
            "ticker": "TEST.NS",
            "current_price": 1000.0,
            "sma_20": 980.0,  # Above 50-day
            "sma_50": 950.0,
            "macd": 5.0,
            "macd_signal": 4.0,
            "rsi": 60.0,
            "trend": "BULLISH",
            "momentum": "NEUTRAL",
            "technical_score": 70.0,
            "support_20d": 950.0,
            "resistance_20d": 1050.0,
            "52w_high": 1100.0,
            "52w_low": 800.0,
            "price_return_pct": 15.0,
        }

        result = run_technical_analyst("TEST", "Test Company")

        signals_text = " ".join(result["signals"])
        assert "golden cross" in signals_text.lower()

    @patch("backend.agents.technical_analyst.compute_technical_indicators")
    def test_death_cross_detection(self, mock_compute):
        """Should detect death cross (20-day SMA below 50-day SMA)."""
        mock_compute.return_value = {
            "ticker": "TEST.NS",
            "current_price": 900.0,
            "sma_20": 920.0,  # Below 50-day
            "sma_50": 950.0,
            "macd": -5.0,
            "macd_signal": -4.0,
            "rsi": 40.0,
            "trend": "BEARISH",
            "momentum": "NEUTRAL",
            "technical_score": 30.0,
            "support_20d": 850.0,
            "resistance_20d": 950.0,
            "52w_high": 1100.0,
            "52w_low": 800.0,
            "price_return_pct": -10.0,
        }

        result = run_technical_analyst("TEST", "Test Company")

        signals_text = " ".join(result["signals"])
        assert "death cross" in signals_text.lower()

    @patch("backend.agents.technical_analyst.compute_technical_indicators")
    def test_support_resistance_extraction(self, mock_compute):
        """Should extract support and resistance levels."""
        mock_compute.return_value = {
            "ticker": "TEST.NS",
            "current_price": 1000.0,
            "sma_20": 950.0,
            "sma_50": 900.0,
            "rsi": 55.0,
            "trend": "NEUTRAL",
            "momentum": "NEUTRAL",
            "technical_score": 50.0,
            "support_20d": 950.0,
            "resistance_20d": 1050.0,
            "52w_high": 1200.0,
            "52w_low": 800.0,
            "price_return_pct": 10.0,
        }

        result = run_technical_analyst("TEST", "Test Company")

        assert result["support_resistance"]["support_20d"] == 950.0
        assert result["support_resistance"]["resistance_20d"] == 1050.0
        assert result["support_resistance"]["52w_high"] == 1200.0
        assert result["support_resistance"]["52w_low"] == 800.0

    @patch("backend.agents.technical_analyst.compute_technical_indicators")
    def test_exception_handling(self, mock_compute):
        """Should handle unexpected exceptions gracefully."""
        mock_compute.side_effect = Exception("Unexpected error")

        result = run_technical_analyst("TEST", "Test Company")

        assert result["status"] == "error"
        assert "Unexpected error" in result["error"]

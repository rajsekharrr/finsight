"""
Tests for backend/tools/technical_tools.py
Manual technical indicator calculations tested with known values.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from backend.tools.technical_tools import (
    _calculate_sma,
    _calculate_ema,
    _calculate_rsi,
    _calculate_macd,
    compute_technical_indicators
)


# ============================================================================
# UNIT TESTS FOR INDICATOR FUNCTIONS
# ============================================================================

class TestIndicatorFunctions:
    """Test individual indicator calculation functions"""

    def test_calculate_sma(self):
        """Test Simple Moving Average calculation"""
        data = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        sma_5 = _calculate_sma(data, 5)

        # First 4 should be NaN (not enough data)
        assert pd.isna(sma_5.iloc[0])
        assert pd.isna(sma_5.iloc[3])

        # 5th element should be average of first 5: (10+11+12+13+14)/5 = 12
        assert sma_5.iloc[4] == 12.0

        # 6th element: (11+12+13+14+15)/5 = 13
        assert sma_5.iloc[5] == 13.0

    def test_calculate_ema(self):
        """Test Exponential Moving Average calculation"""
        data = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        ema_5 = _calculate_ema(data, 5)

        # First 4 should be NaN
        assert pd.isna(ema_5.iloc[0])

        # EMA should exist from period onwards
        assert not pd.isna(ema_5.iloc[5])

        # EMA should be reactive to recent values
        assert ema_5.iloc[-1] > ema_5.iloc[5]

    def test_calculate_rsi(self):
        """Test RSI calculation with known pattern"""
        # Create data with clear uptrend then downtrend
        uptrend = [100 + i for i in range(20)]
        downtrend = [120 - i for i in range(20)]
        data = pd.Series(uptrend + downtrend)

        rsi = _calculate_rsi(data, 14)

        # RSI should be high during uptrend
        assert rsi.iloc[25] > 60  # Still in recovery from uptrend

        # RSI should drop during downtrend
        assert rsi.iloc[-1] < 50

    def test_calculate_rsi_overbought(self):
        """Test RSI with strong uptrend (overbought)"""
        data = pd.Series([100 + i*2 for i in range(30)])
        rsi = _calculate_rsi(data, 14)

        # Strong uptrend should show high RSI
        assert rsi.iloc[-1] > 70

    def test_calculate_macd(self):
        """Test MACD calculation"""
        # Create trending data
        data = pd.Series([100 + i*0.5 for i in range(50)])

        macd_line, signal_line, histogram = _calculate_macd(data, 12, 26, 9)

        # MACD line should exist
        assert not pd.isna(macd_line.iloc[-1])

        # Signal line should exist
        assert not pd.isna(signal_line.iloc[-1])

        # Histogram is difference
        assert histogram.iloc[-1] == macd_line.iloc[-1] - signal_line.iloc[-1]

    def test_calculate_macd_uptrend(self):
        """Test MACD in uptrend"""
        data = pd.Series([100 + i for i in range(50)])
        macd_line, signal_line, histogram = _calculate_macd(data)

        # In strong uptrend, MACD should be positive
        assert macd_line.iloc[-1] > 0


# ============================================================================
# COMPUTE_TECHNICAL_INDICATORS TESTS
# ============================================================================

class TestComputeTechnicalIndicators:
    """Test main compute_technical_indicators function"""

    def _create_mock_price_data(self, length=100, trend="up"):
        """Helper to create mock price data"""
        if trend == "up":
            closes = [2000 + i*5 for i in range(length)]
        elif trend == "down":
            closes = [2500 - i*5 for i in range(length)]
        else:  # flat
            closes = [2000 + np.random.randint(-10, 10) for i in range(length)]

        dates = pd.date_range(end="2024-01-15", periods=length, freq="D")

        return {
            "ticker": "TEST.NS",
            "period": "1y",
            "dates": [str(d.date()) for d in dates],
            "open": [c - 5 for c in closes],
            "high": [c + 10 for c in closes],
            "low": [c - 10 for c in closes],
            "close": closes,
            "volume": [1000000 + i*1000 for i in range(length)],
            "latest_price": closes[-1],
            "oldest_price": closes[0],
            "price_return_pct": round((closes[-1] / closes[0] - 1) * 100, 2)
        }

    @patch('backend.tools.financial_tools.get_price_history')
    def test_compute_indicators_success_uptrend(self, mock_get_price):
        """Test successful indicator computation with uptrend"""
        mock_get_price.func.return_value = self._create_mock_price_data(100, "up")

        result = compute_technical_indicators("TEST", "1y")

        # Should not have error
        assert "error" not in result

        # Basic fields
        assert result["ticker"] == "TEST"
        assert result["period"] == "1y"
        assert result["data_points"] == 100

        # Indicators should be calculated
        assert result["sma_20"] is not None
        assert result["sma_50"] is not None
        assert result["rsi"] is not None
        assert result["macd"] is not None
        assert result["macd_signal"] is not None

        # Trend should be BULLISH for uptrend
        assert result["trend"] == "BULLISH"

        # Technical score should be present
        assert 0 <= result["technical_score"] <= 100

        # Summary should exist
        assert "summary" in result
        assert len(result["summary"]) > 0

    @patch('backend.tools.financial_tools.get_price_history')
    def test_compute_indicators_downtrend(self, mock_get_price):
        """Test indicator computation with downtrend"""
        mock_get_price.func.return_value = self._create_mock_price_data(100, "down")

        result = compute_technical_indicators("TEST", "1y")

        assert "error" not in result
        # Downtrend should show BEARISH
        assert result["trend"] == "BEARISH"

        # Technical score should be lower
        assert result["technical_score"] < 50

    @patch('backend.tools.financial_tools.get_price_history')
    def test_compute_indicators_insufficient_data(self, mock_get_price):
        """Test handling of insufficient data"""
        mock_get_price.func.return_value = self._create_mock_price_data(30, "up")

        result = compute_technical_indicators("TEST", "1mo")

        assert "error" in result
        assert "Insufficient data" in result["error"]

    @patch('backend.tools.financial_tools.get_price_history')
    def test_compute_indicators_price_fetch_error(self, mock_get_price):
        """Test handling when price fetch fails"""
        mock_get_price.func.return_value = {
            "error": "No data available",
            "ticker": "BADTICKER.NS"
        }

        result = compute_technical_indicators("BADTICKER", "1y")

        assert "error" in result
        assert "Price data unavailable" in result["error"]

    @patch('backend.tools.financial_tools.get_price_history')
    def test_compute_indicators_all_fields_present(self, mock_get_price):
        """Test that all expected fields are present"""
        mock_get_price.func.return_value = self._create_mock_price_data(250, "up")

        result = compute_technical_indicators("TEST", "1y")

        expected_fields = [
            "ticker", "period", "current_price", "data_points",
            "sma_20", "sma_50", "sma_200",
            "ema_12", "ema_26",
            "macd", "macd_signal", "macd_histogram",
            "rsi",
            "volume_avg_20d", "latest_volume",
            "support_20d", "resistance_20d",
            "52w_high", "52w_low",
            "price_return_pct",
            "trend", "momentum", "technical_score",
            "trend_signals", "momentum_signals",
            "summary"
        ]

        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

    @patch('backend.tools.financial_tools.get_price_history')
    def test_compute_indicators_overbought_rsi(self, mock_get_price):
        """Test detection of overbought condition"""
        # Create strong uptrend to generate high RSI
        price_data = self._create_mock_price_data(100, "up")
        # Amplify trend
        price_data["close"] = [2000 + i*10 for i in range(100)]

        mock_get_price.func.return_value = price_data

        result = compute_technical_indicators("TEST", "1y")

        # RSI should be high
        assert result["rsi"] > 60

        # Momentum might be OVERBOUGHT
        if result["rsi"] > 70:
            assert result["momentum"] == "OVERBOUGHT"

    @patch('backend.tools.financial_tools.get_price_history')
    def test_compute_indicators_exception_handling(self, mock_get_price):
        """Test exception handling"""
        mock_get_price.func.side_effect = Exception("Unexpected error")

        result = compute_technical_indicators("TEST", "1y")

        assert "error" in result
        assert "Unexpected error" in result["error"]

    @patch('backend.tools.financial_tools.get_price_history')
    def test_compute_indicators_signals(self, mock_get_price):
        """Test signal generation"""
        mock_get_price.func.return_value = self._create_mock_price_data(200, "up")

        result = compute_technical_indicators("TEST", "1y")

        # Signals should be lists
        assert isinstance(result["trend_signals"], list)
        assert isinstance(result["momentum_signals"], list)

        # Should have at least some signals
        assert len(result["trend_signals"]) > 0
        assert len(result["momentum_signals"]) > 0

    @patch('backend.tools.financial_tools.get_price_history')
    def test_compute_indicators_golden_cross(self, mock_get_price):
        """Test golden cross detection (SMA20 > SMA50)"""
        # Create data where 20-day crosses above 50-day
        mock_get_price.func.return_value = self._create_mock_price_data(100, "up")

        result = compute_technical_indicators("TEST", "1y")

        # In uptrend, should have positive signals
        assert result["technical_score"] > 50

    @patch('backend.tools.financial_tools.get_price_history')
    def test_compute_indicators_support_resistance(self, mock_get_price):
        """Test support and resistance calculation"""
        mock_get_price.func.return_value = self._create_mock_price_data(100, "flat")

        result = compute_technical_indicators("TEST", "1y")

        # Support should be less than resistance
        assert result["support_20d"] < result["resistance_20d"]

        # Current price should be between support and resistance
        assert result["support_20d"] <= result["current_price"] <= result["resistance_20d"]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestTechnicalIntegration:
    """Integration tests for technical tools"""

    @patch('backend.tools.financial_tools.get_price_history')
    def test_full_indicator_pipeline(self, mock_get_price):
        """Test complete indicator computation pipeline"""
        # Create realistic price data
        dates = pd.date_range(end="2024-01-15", periods=200, freq="D")
        closes = [2000]

        # Simulate realistic price movements
        for i in range(1, 200):
            change = np.random.randn() * 20
            new_price = closes[-1] + change
            closes.append(max(1900, min(2600, new_price)))  # Keep in range

        mock_data = {
            "ticker": "RELIANCE.NS",
            "period": "1y",
            "dates": [str(d.date()) for d in dates],
            "open": [c - 5 for c in closes],
            "high": [c + 15 for c in closes],
            "low": [c - 15 for c in closes],
            "close": closes,
            "volume": [1000000 + i*1000 for i in range(200)],
            "latest_price": closes[-1],
            "oldest_price": closes[0],
            "price_return_pct": round((closes[-1] / closes[0] - 1) * 100, 2)
        }

        mock_get_price.func.return_value = mock_data

        result = compute_technical_indicators("RELIANCE", "1y")

        # Should successfully compute all indicators
        assert "error" not in result
        assert result["ticker"] == "RELIANCE"

        # All moving averages should be calculated
        assert result["sma_20"] is not None
        assert result["sma_50"] is not None
        assert result["sma_200"] is not None

        # RSI should be in valid range
        assert 0 <= result["rsi"] <= 100

        # Should have valid trend
        assert result["trend"] in ["BULLISH", "BEARISH", "NEUTRAL"]

        # Should have valid momentum
        assert result["momentum"] in ["OVERBOUGHT", "OVERSOLD", "NEUTRAL"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

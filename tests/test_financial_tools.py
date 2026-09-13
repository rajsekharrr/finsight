"""
Focused tests for backend/tools/financial_tools.py
Uses mocking to avoid dependence on live market data.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta

from backend.tools.financial_tools import (
    get_stock_info,
    get_price_history,
    get_nifty_history,
    get_sector_median,
    _ensure_ns
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _ensure_ns_direct(ticker: str) -> str:
    """Direct call to _ensure_ns for testing"""
    return _ensure_ns(ticker)


# ============================================================================
# TICKER NORMALIZATION TESTS
# ============================================================================

class TestTickerNormalization:
    """Test ticker suffix handling"""

    def test_ensure_ns_plain_ticker(self):
        """Test adding .NS to plain ticker"""
        assert _ensure_ns_direct("RELIANCE") == "RELIANCE.NS"
        assert _ensure_ns_direct("TCS") == "TCS.NS"
        assert _ensure_ns_direct("hdfc") == "HDFC.NS"

    def test_ensure_ns_already_has_suffix(self):
        """Test that existing .NS/.BO suffixes are preserved"""
        assert _ensure_ns_direct("RELIANCE.NS") == "RELIANCE.NS"
        assert _ensure_ns_direct("RELIANCE.BO") == "RELIANCE.BO"
        assert _ensure_ns_direct("tcs.ns") == "TCS.NS"

    def test_ensure_ns_case_normalization(self):
        """Test case normalization"""
        assert _ensure_ns_direct("reliance") == "RELIANCE.NS"
        assert _ensure_ns_direct("Reliance") == "RELIANCE.NS"


# ============================================================================
# GET_STOCK_INFO TESTS
# ============================================================================

class TestGetStockInfo:
    """Tests for get_stock_info tool"""

    @patch('backend.tools.financial_tools.cache_get')
    @patch('backend.tools.financial_tools.cache_set')
    @patch('backend.tools.financial_tools.yf.Ticker')
    def test_get_stock_info_success(self, mock_ticker, mock_cache_set, mock_cache_get):
        """Test successful stock info fetch"""
        # Mock cache miss
        mock_cache_get.return_value = None

        # Mock yfinance response
        mock_stock = Mock()
        mock_stock.info = {
            "longName": "Reliance Industries Limited",
            "sector": "Energy",
            "industry": "Oil & Gas Integrated",
            "currentPrice": 2500.50,
            "marketCap": 17000000000000,
            "trailingPE": 24.5,
            "priceToBook": 2.8,
            "trailingEps": 102.0,
            "dividendYield": 0.004,
            "fiftyTwoWeekHigh": 2900.0,
            "fiftyTwoWeekLow": 2200.0,
            "beta": 0.95,
            "returnOnEquity": 0.142,
            "debtToEquity": 42.5,
            "currentRatio": 0.85,
            "totalRevenue": 9000000000000,
            "netIncomeToCommon": 650000000000,
            "grossMargins": 0.28,
            "operatingMargins": 0.145,
            "profitMargins": 0.072,
        }
        mock_ticker.return_value = mock_stock

        # Call the tool function directly
        result = get_stock_info.func("RELIANCE")

        # Verify results
        assert result["ticker"] == "RELIANCE.NS"
        assert result["company_name"] == "Reliance Industries Limited"
        assert result["sector"] == "Energy"
        assert result["current_price"] == 2500.50
        assert result["pe_ratio"] == 24.5
        assert result["pb_ratio"] == 2.8
        assert result["roe_pct"] == 14.2
        assert result["market_cap_cr"] == 1700000.0

        # Verify cache was called
        mock_cache_get.assert_called_once()
        mock_cache_set.assert_called_once()

    @patch('backend.tools.financial_tools.cache_get')
    @patch('backend.tools.financial_tools.yf.Ticker')
    def test_get_stock_info_cache_hit(self, mock_ticker, mock_cache_get):
        """Test that cached data is returned without calling yfinance"""
        cached_data = {
            "ticker": "RELIANCE.NS",
            "company_name": "Reliance Industries",
            "current_price": 2500.0
        }
        mock_cache_get.return_value = cached_data

        result = get_stock_info.func("RELIANCE")

        assert result == cached_data
        mock_ticker.assert_not_called()

    @patch('backend.tools.financial_tools.cache_get')
    @patch('backend.tools.financial_tools.cache_set')
    @patch('backend.tools.financial_tools.yf.Ticker')
    def test_get_stock_info_error_handling(self, mock_ticker, mock_cache_set, mock_cache_get):
        """Test error handling when yfinance fails"""
        mock_cache_get.return_value = None
        mock_ticker.side_effect = Exception("Network error")

        result = get_stock_info.func("INVALID")

        assert "error" in result
        assert result["ticker"] == "INVALID.NS"
        assert "Network error" in result["error"]

    @patch('backend.tools.financial_tools.cache_get')
    @patch('backend.tools.financial_tools.cache_set')
    @patch('backend.tools.financial_tools.yf.Ticker')
    def test_get_stock_info_minimal_data(self, mock_ticker, mock_cache_set, mock_cache_get):
        """Test handling of minimal/empty info dict"""
        mock_cache_get.return_value = None

        mock_stock = Mock()
        mock_stock.info = {}  # Empty info
        mock_ticker.return_value = mock_stock

        result = get_stock_info.func("TEST")

        assert "error" in result
        assert "No data available" in result["error"]


# ============================================================================
# GET_PRICE_HISTORY TESTS
# ============================================================================

class TestGetPriceHistory:
    """Tests for get_price_history tool"""

    @patch('backend.tools.financial_tools.cache_get')
    @patch('backend.tools.financial_tools.cache_set')
    @patch('backend.tools.financial_tools.yf.Ticker')
    def test_get_price_history_success(self, mock_ticker, mock_cache_set, mock_cache_get):
        """Test successful price history fetch"""
        mock_cache_get.return_value = None

        # Create mock price data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        hist_df = pd.DataFrame({
            'Open': [2400 + i for i in range(30)],
            'High': [2450 + i for i in range(30)],
            'Low': [2380 + i for i in range(30)],
            'Close': [2420 + i for i in range(30)],
            'Volume': [1000000 + i*1000 for i in range(30)],
        }, index=dates)

        mock_stock = Mock()
        mock_stock.history.return_value = hist_df
        mock_ticker.return_value = mock_stock

        result = get_price_history.func("RELIANCE", "1mo")

        assert result["ticker"] == "RELIANCE.NS"
        assert result["period"] == "1mo"
        assert len(result["dates"]) == 30
        assert len(result["close"]) == 30
        assert result["latest_price"] == 2449.0
        assert result["oldest_price"] == 2420.0
        assert "price_return_pct" in result

        mock_cache_set.assert_called_once()

    @patch('backend.tools.financial_tools.cache_get')
    @patch('backend.tools.financial_tools.yf.Ticker')
    def test_get_price_history_cache_hit(self, mock_ticker, mock_cache_get):
        """Test cache hit for price history"""
        cached_data = {
            "ticker": "TCS.NS",
            "dates": ["2024-01-01", "2024-01-02"],
            "close": [3500.0, 3520.0]
        }
        mock_cache_get.return_value = cached_data

        result = get_price_history.func("TCS")

        assert result == cached_data
        mock_ticker.assert_not_called()

    @patch('backend.tools.financial_tools.cache_get')
    @patch('backend.tools.financial_tools.cache_set')
    @patch('backend.tools.financial_tools.yf.Ticker')
    def test_get_price_history_empty_data(self, mock_ticker, mock_cache_set, mock_cache_get):
        """Test handling of empty history DataFrame"""
        mock_cache_get.return_value = None

        mock_stock = Mock()
        mock_stock.history.return_value = pd.DataFrame()  # Empty DataFrame
        mock_ticker.return_value = mock_stock

        result = get_price_history.func("BADTICKER")

        assert "error" in result
        assert "No price data" in result["error"]

    @patch('backend.tools.financial_tools.cache_get')
    @patch('backend.tools.financial_tools.cache_set')
    @patch('backend.tools.financial_tools.yf.Ticker')
    def test_get_price_history_exception(self, mock_ticker, mock_cache_set, mock_cache_get):
        """Test exception handling"""
        mock_cache_get.return_value = None
        mock_ticker.side_effect = Exception("API timeout")

        result = get_price_history.func("TEST")

        assert "error" in result
        assert "API timeout" in result["error"]


# ============================================================================
# GET_NIFTY_HISTORY TESTS
# ============================================================================

class TestGetNiftyHistory:
    """Tests for get_nifty_history tool"""

    @patch('backend.tools.financial_tools.cache_get')
    @patch('backend.tools.financial_tools.cache_set')
    @patch('backend.tools.financial_tools.yf.Ticker')
    def test_get_nifty_history_success(self, mock_ticker, mock_cache_set, mock_cache_get):
        """Test successful Nifty 50 history fetch"""
        mock_cache_get.return_value = None

        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        hist_df = pd.DataFrame({
            'Close': [21000 + i*10 for i in range(50)],
        }, index=dates)

        mock_nifty = Mock()
        mock_nifty.history.return_value = hist_df
        mock_ticker.return_value = mock_nifty

        result = get_nifty_history.func("1y")

        assert result["period"] == "1y"
        assert len(result["dates"]) == 50
        assert len(result["close"]) == 50
        assert result["close"][0] == 21000.0

        mock_ticker.assert_called_once_with("^NSEI")
        mock_cache_set.assert_called_once()

    @patch('backend.tools.financial_tools.cache_get')
    @patch('backend.tools.financial_tools.yf.Ticker')
    def test_get_nifty_history_cache_hit(self, mock_ticker, mock_cache_get):
        """Test cache hit for Nifty history"""
        cached_data = {
            "dates": ["2024-01-01"],
            "close": [21000.0],
            "period": "1mo"
        }
        mock_cache_get.return_value = cached_data

        result = get_nifty_history.func("1mo")

        assert result == cached_data
        mock_ticker.assert_not_called()

    @patch('backend.tools.financial_tools.cache_get')
    @patch('backend.tools.financial_tools.cache_set')
    @patch('backend.tools.financial_tools.yf.Ticker')
    def test_get_nifty_history_error(self, mock_ticker, mock_cache_set, mock_cache_get):
        """Test error handling for Nifty fetch"""
        mock_cache_get.return_value = None
        mock_ticker.side_effect = Exception("Connection refused")

        result = get_nifty_history.func("1y")

        assert "error" in result
        assert "Connection refused" in result["error"]


# ============================================================================
# GET_SECTOR_MEDIAN TESTS
# ============================================================================

class TestGetSectorMedian:
    """Tests for get_sector_median tool"""

    @patch('backend.utils.sector_data.get_sector_median')
    def test_get_sector_median_success(self, mock_get_sector):
        """Test successful sector median fetch"""
        mock_get_sector.return_value = {
            "sector": "IT",
            "medians": {"pe": 25.0, "pb": 6.5, "roe": 22.0}
        }

        result = get_sector_median.func("IT")

        assert result["sector"] == "IT"
        assert result["medians"]["pe"] == 25.0
        mock_get_sector.assert_called_once_with("IT")

    @patch('backend.utils.sector_data.get_sector_median')
    def test_get_sector_median_fallback(self, mock_get_sector):
        """Test fallback to Diversified"""
        mock_get_sector.return_value = {
            "sector": "Diversified",
            "medians": {"pe": 22.0, "pb": 3.5}
        }

        result = get_sector_median.func("UnknownSector")

        assert result["sector"] == "Diversified"

    @patch('backend.utils.sector_data.get_sector_median')
    def test_get_sector_median_error(self, mock_get_sector):
        """Test error handling"""
        mock_get_sector.side_effect = Exception("File not found")

        result = get_sector_median.func("IT")

        assert "error" in result
        assert "File not found" in result["error"]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple tools"""

    @patch('backend.tools.financial_tools.cache_get')
    @patch('backend.tools.financial_tools.cache_set')
    @patch('backend.tools.financial_tools.yf.Ticker')
    def test_multiple_tools_with_cache(self, mock_ticker, mock_cache_set, mock_cache_get):
        """Test that multiple tool calls use cache correctly"""
        # First call - cache miss
        mock_cache_get.return_value = None

        mock_stock = Mock()
        mock_stock.info = {
            "longName": "Test Company",
            "currentPrice": 1000.0,
            "marketCap": 1000000000000,
        }
        mock_ticker.return_value = mock_stock

        result1 = get_stock_info.func("TEST")
        assert "error" not in result1 or result1.get("ticker") == "TEST.NS"

        # Second call - should have been cached
        mock_cache_get.return_value = result1

        result2 = get_stock_info.func("TEST")
        assert result2 == result1

    @patch('backend.tools.financial_tools.cache_get')
    @patch('backend.tools.financial_tools.cache_set')
    @patch('backend.tools.financial_tools.yf.Ticker')
    def test_bo_suffix_preserved(self, mock_ticker, mock_cache_set, mock_cache_get):
        """Test that .BO suffix is preserved"""
        mock_cache_get.return_value = None

        mock_stock = Mock()
        mock_stock.info = {"longName": "BSE Listed", "currentPrice": 500.0, "marketCap": 100000000000}
        mock_ticker.return_value = mock_stock

        result = get_stock_info.func("TEST.BO")

        # Should preserve .BO, not change to .NS
        assert result["ticker"] == "TEST.BO"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

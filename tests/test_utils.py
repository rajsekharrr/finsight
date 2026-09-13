"""
Focused tests for utility modules: cache, sector_data, validators.
"""
import pytest
import json
import time
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from backend.utils.cache import cache_get, cache_set, cached, _cache_path, CACHE_DIR
from backend.utils.sector_data import get_sector_medians, get_sector_median, detect_sector_from_yfinance
from backend.utils.validators import (
    normalize_ticker,
    is_valid_ticker,
    add_ns_suffix,
    strip_suffix
)


# ============================================================================
# CACHE TESTS
# ============================================================================

class TestCache:
    """Tests for cache.py"""

    def setup_method(self):
        """Clean up cache before each test"""
        if CACHE_DIR.exists():
            for file in CACHE_DIR.glob("*.json"):
                file.unlink()

    def test_cache_set_and_get(self):
        """Test basic cache set and get"""
        key = "test_key"
        value = {"data": "test_value", "number": 42}

        cache_set(key, value)
        result = cache_get(key)

        assert result == value

    def test_cache_get_missing_key(self):
        """Test cache_get returns None for missing key"""
        result = cache_get("nonexistent_key_12345")
        assert result is None

    def test_cache_expiration(self):
        """Test that expired cache returns None"""
        key = "expiring_key"
        value = "test_data"

        # Set cache with mocked old timestamp
        cache_path = _cache_path(key)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        old_timestamp = time.time() - (7 * 3600)  # 7 hours ago (expired if TTL=6h)
        with open(cache_path, 'w') as f:
            json.dump({"_ts": old_timestamp, "value": value}, f)

        result = cache_get(key)
        assert result is None

    def test_cache_not_expired(self):
        """Test that non-expired cache returns value"""
        key = "fresh_key"
        value = "fresh_data"

        cache_set(key, value)
        result = cache_get(key)

        assert result == value

    def test_cache_corrupted_json(self):
        """Test that corrupted cache file returns None and doesn't crash"""
        key = "corrupted_key"
        cache_path = _cache_path(key)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Write invalid JSON
        with open(cache_path, 'w') as f:
            f.write("{ invalid json content }")

        result = cache_get(key)
        assert result is None  # Should not crash, returns None

    def test_cache_missing_timestamp(self):
        """Test cache file without _ts field returns None"""
        key = "no_timestamp"
        cache_path = _cache_path(key)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Write JSON without _ts
        with open(cache_path, 'w') as f:
            json.dump({"value": "test"}, f)

        result = cache_get(key)
        assert result is None

    def test_cache_path_sanitization(self):
        """Test that unsafe characters in keys are sanitized"""
        unsafe_key = "ticker/with.special:chars\\bad"
        path = _cache_path(unsafe_key)

        # Should not contain unsafe characters
        assert "/" not in path.name
        assert "\\" not in path.name
        assert ":" not in path.name
        assert path.name.endswith(".json")

    def test_cached_decorator(self):
        """Test cached decorator caches function results"""
        call_count = 0

        @cached("test_func")
        def expensive_function(arg):
            nonlocal call_count
            call_count += 1
            return f"result_{arg}"

        # First call - should execute
        result1 = expensive_function("A")
        assert result1 == "result_A"
        assert call_count == 1

        # Second call with same arg - should use cache
        result2 = expensive_function("A")
        assert result2 == "result_A"
        assert call_count == 1  # Not incremented

        # Different arg - should execute
        result3 = expensive_function("B")
        assert result3 == "result_B"
        assert call_count == 2


# ============================================================================
# SECTOR DATA TESTS
# ============================================================================

class TestSectorData:
    """Tests for sector_data.py"""

    def test_get_sector_medians_loads_data(self):
        """Test that get_sector_medians returns valid data"""
        medians = get_sector_medians()

        assert isinstance(medians, dict)
        assert len(medians) > 0
        assert "IT" in medians
        assert "BFSI" in medians
        assert "Diversified" in medians

    def test_sector_median_structure(self):
        """Test that sector data has expected structure"""
        medians = get_sector_medians()
        it_sector = medians["IT"]

        assert "pe" in it_sector
        assert "pb" in it_sector
        assert "roe" in it_sector
        assert "description" in it_sector

    def test_get_sector_median_exact_match(self):
        """Test get_sector_median with exact sector name"""
        result = get_sector_median("IT")

        assert result["sector"] == "IT"
        assert "medians" in result
        assert result["medians"]["pe"] == 25.0

    def test_get_sector_median_unknown_fallback(self):
        """Test fallback to Diversified for unknown sector"""
        result = get_sector_median("UnknownSector123")

        assert result["sector"] == "Diversified"
        assert "medians" in result

    def test_get_sector_median_case_insensitive(self):
        """Test case-insensitive sector matching"""
        result1 = get_sector_median("it")
        result2 = get_sector_median("IT")
        result3 = get_sector_median("It")

        assert result1["sector"] == "IT"
        assert result2["sector"] == "IT"
        assert result3["sector"] == "IT"

    def test_detect_sector_from_yfinance(self):
        """Test yfinance sector name mapping"""
        assert detect_sector_from_yfinance("Technology") == "IT"
        assert detect_sector_from_yfinance("Financial Services") == "BFSI"
        assert detect_sector_from_yfinance("Healthcare") == "Pharma"
        assert detect_sector_from_yfinance("Energy") == "Energy"
        assert detect_sector_from_yfinance("Unknown Sector") == "Diversified"

    def test_sector_data_missing_file(self, tmp_path, monkeypatch):
        """Test graceful handling when sector file is missing"""
        # Point to non-existent file
        fake_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr("backend.utils.sector_data._SECTOR_FILE", fake_path)
        monkeypatch.setattr("backend.utils.sector_data._cached_data", None)

        result = get_sector_medians()

        # Should return fallback data, not crash
        assert "Diversified" in result


# ============================================================================
# VALIDATOR TESTS
# ============================================================================

class TestValidators:
    """Tests for validators.py"""

    def test_normalize_ticker_basic(self):
        """Test basic ticker normalization"""
        assert normalize_ticker("reliance") == "RELIANCE"
        assert normalize_ticker("HDFC") == "HDFC"
        assert normalize_ticker("  TCS  ") == "TCS"

    def test_normalize_ticker_with_ns_suffix(self):
        """Test normalization with .NS suffix"""
        assert normalize_ticker("RELIANCE.NS") == "RELIANCE.NS"
        assert normalize_ticker("reliance.ns") == "RELIANCE.NS"
        assert normalize_ticker("TCS.ns") == "TCS.NS"

    def test_normalize_ticker_with_bo_suffix(self):
        """Test normalization with .BO suffix"""
        assert normalize_ticker("RELIANCE.BO") == "RELIANCE.BO"
        assert normalize_ticker("reliance.bo") == "RELIANCE.BO"

    def test_normalize_ticker_invalid_empty(self):
        """Test that empty strings return None"""
        assert normalize_ticker("") is None
        assert normalize_ticker("   ") is None
        assert normalize_ticker(None) is None

    def test_normalize_ticker_invalid_special_chars(self):
        """Test rejection of invalid special characters"""
        assert normalize_ticker("REL@ANCE") is None
        assert normalize_ticker("TCS#123") is None
        assert normalize_ticker("HDFC!") is None
        assert normalize_ticker("IN FY") is None  # Space in middle

    def test_normalize_ticker_too_long(self):
        """Test rejection of very long tickers"""
        long_ticker = "A" * 25
        assert normalize_ticker(long_ticker) is None

    def test_normalize_ticker_with_ampersand(self):
        """Test that ampersand is allowed (L&T)"""
        assert normalize_ticker("L&T") == "L&T"
        assert normalize_ticker("M&M") == "M&M"

    def test_normalize_ticker_with_numbers(self):
        """Test tickers with numbers"""
        assert normalize_ticker("IDEA") == "IDEA"
        assert normalize_ticker("52WEEK") == "52WEEK"

    def test_is_valid_ticker(self):
        """Test ticker validation"""
        assert is_valid_ticker("RELIANCE") is True
        assert is_valid_ticker("TCS.NS") is True
        assert is_valid_ticker("HDFC.BO") is True
        assert is_valid_ticker("L&T") is True

        assert is_valid_ticker("") is False
        assert is_valid_ticker("   ") is False
        assert is_valid_ticker("REL@NCE") is False
        assert is_valid_ticker("A" * 25) is False

    def test_add_ns_suffix(self):
        """Test adding .NS suffix"""
        assert add_ns_suffix("RELIANCE") == "RELIANCE.NS"
        assert add_ns_suffix("reliance") == "RELIANCE.NS"
        assert add_ns_suffix("RELIANCE.NS") == "RELIANCE.NS"
        assert add_ns_suffix("RELIANCE.BO") == "RELIANCE.BO"  # Don't change .BO

    def test_add_ns_suffix_invalid_ticker(self):
        """Test add_ns_suffix raises error for invalid ticker"""
        with pytest.raises(ValueError):
            add_ns_suffix("")

        with pytest.raises(ValueError):
            add_ns_suffix("INVAL!D")

    def test_strip_suffix(self):
        """Test removing suffix from ticker"""
        assert strip_suffix("RELIANCE.NS") == "RELIANCE"
        assert strip_suffix("RELIANCE.BO") == "RELIANCE"
        assert strip_suffix("RELIANCE") == "RELIANCE"
        assert strip_suffix("tcs.ns") == "TCS"

    def test_strip_suffix_invalid_ticker(self):
        """Test strip_suffix raises error for invalid ticker"""
        with pytest.raises(ValueError):
            strip_suffix("")

        with pytest.raises(ValueError):
            strip_suffix("INVAL!D")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests across multiple utilities"""

    def test_cache_with_sector_data(self):
        """Test caching sector data lookups"""
        # First call - loads from file
        result1 = get_sector_median("IT")

        # Second call - should use in-memory cache
        result2 = get_sector_median("IT")

        assert result1 == result2
        assert result1["sector"] == "IT"

    def test_validator_with_cache_key(self):
        """Test that normalized tickers work as cache keys"""
        ticker1 = "reliance"
        ticker2 = "RELIANCE"

        normalized1 = normalize_ticker(ticker1)
        normalized2 = normalize_ticker(ticker2)

        assert normalized1 == normalized2

        # Both should produce same cache key
        cache_set(f"ticker_{normalized1}", {"price": 2500})
        result = cache_get(f"ticker_{normalized2}")

        assert result == {"price": 2500}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

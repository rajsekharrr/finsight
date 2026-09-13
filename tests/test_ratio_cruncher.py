"""
Tests for Ratio Cruncher agent.
"""

import pytest
import json
from unittest.mock import Mock, patch
from backend.agents.ratio_cruncher import run_ratio_cruncher


class TestRatioCruncher:
    """Tests for run_ratio_cruncher function."""

    @patch("backend.agents.ratio_cruncher.get_sector_median")
    @patch("backend.agents.ratio_cruncher.get_stock_info")
    def test_successful_analysis(self, mock_stock_info, mock_sector_median):
        """Should successfully analyze ratios with sector benchmarks."""
        # Mock stock info
        mock_stock_info.return_value = {
            "ticker": "RELIANCE.NS",
            "company_name": "Reliance Industries",
            "sector": "Energy",
            "pe_ratio": 25.5,
            "forward_pe": 22.3,
            "pb_ratio": 2.8,
            "eps": 85.5,
            "roe_pct": 18.5,
            "roa_pct": 8.2,
            "debt_to_equity": 45.0,
            "current_ratio": 1.5,
            "dividend_yield_pct": 0.5,
            "operating_margins_pct": 12.5,
            "profit_margins_pct": 8.5,
            "market_cap_cr": 150000.0,
            "revenue_cr": 85000.0,
            "net_income_cr": 7200.0,
            "current_price": 2500.0,
            "52w_high": 2800.0,
            "52w_low": 2200.0,
            "beta": 1.1,
        }

        # Mock sector median
        mock_sector_median.return_value = {
            "sector": "Energy",
            "medians": {
                "pe_ratio": 28.0,
                "pb_ratio": 3.0,
                "roe_pct": 15.0,
                "roa_pct": 7.0,
                "debt_to_equity": 55.0,
                "operating_margins_pct": 10.0,
            }
        }

        result = run_ratio_cruncher("RELIANCE", "Reliance Industries", "Energy")

        assert result["status"] == "success"
        assert result["ticker"] == "RELIANCE"
        assert result["company_name"] == "Reliance Industries"
        assert result["sector"] == "Energy"
        assert result["ratios"]["pe_ratio"] == 25.5
        assert result["ratios"]["roe_pct"] == 18.5
        assert len(result["sector_benchmarks"]) > 0
        assert len(result["strengths"]) > 0
        assert len(result["concerns"]) > 0
        assert len(result["valuation_summary"]) > 0
        assert result["error"] is None

    @patch("backend.agents.ratio_cruncher.get_stock_info")
    def test_stock_info_error(self, mock_stock_info):
        """Should handle stock info fetch error gracefully."""
        mock_stock_info.return_value = {
            "error": "Stock not found",
            "ticker": "INVALID"
        }

        result = run_ratio_cruncher("INVALID", "Invalid Company")

        assert result["status"] == "error"
        assert "Stock data unavailable" in result["error"]
        assert result["ratios"] == {}

    @patch("backend.agents.ratio_cruncher.get_sector_median")
    @patch("backend.agents.ratio_cruncher.get_stock_info")
    def test_missing_optional_ratios(self, mock_stock_info, mock_sector_median):
        """Should handle missing optional ratios gracefully."""
        # Mock stock info with some None values
        mock_stock_info.return_value = {
            "ticker": "TEST.NS",
            "company_name": "Test Co",
            "sector": "IT",
            "pe_ratio": 30.0,
            "pb_ratio": 5.0,
            "roe_pct": None,  # Missing
            "roa_pct": None,  # Missing
            "debt_to_equity": 25.0,
            "current_ratio": None,  # Missing
            "dividend_yield_pct": 0.0,
            "operating_margins_pct": 15.0,
            "profit_margins_pct": None,  # Missing
            "market_cap_cr": 50000.0,
            "revenue_cr": None,  # Missing
            "net_income_cr": None,  # Missing
            "current_price": 1500.0,
            "52w_high": 1800.0,
            "52w_low": 1200.0,
            "beta": 1.2,
        }

        mock_sector_median.return_value = {
            "sector": "IT",
            "medians": {"pe_ratio": 25.0}
        }

        result = run_ratio_cruncher("TEST", "Test Co", "IT")

        assert result["status"] == "success"
        assert result["ratios"]["pe_ratio"] == 30.0
        assert result["ratios"]["roe_pct"] is None
        assert result["ratios"]["roa_pct"] is None
        # Should still have analysis despite missing data
        assert len(result["strengths"]) > 0
        assert len(result["valuation_summary"]) > 0

    @patch("backend.agents.ratio_cruncher.get_sector_median")
    @patch("backend.agents.ratio_cruncher.get_stock_info")
    def test_unknown_sector_fallback(self, mock_stock_info, mock_sector_median):
        """Should handle unknown sector gracefully."""
        mock_stock_info.return_value = {
            "ticker": "TEST.NS",
            "company_name": "Test Co",
            "sector": "",  # No sector
            "pe_ratio": 20.0,
            "pb_ratio": 2.0,
            "roe_pct": 12.0,
            "market_cap_cr": 10000.0,
        }

        # Sector median returns error
        mock_sector_median.return_value = {
            "error": "Unknown sector",
            "sector": "",
            "medians": {}
        }

        result = run_ratio_cruncher("TEST", "Test Co", "")

        assert result["status"] == "partial_success"
        assert "Sector benchmarks unavailable" in result["error"]
        assert result["ratios"]["pe_ratio"] == 20.0
        assert result["sector_benchmarks"] == {}
        # Should still analyze without benchmarks
        assert len(result["strengths"]) > 0
        assert len(result["valuation_summary"]) > 0

    @patch("backend.agents.ratio_cruncher.get_sector_median")
    @patch("backend.agents.ratio_cruncher.get_stock_info")
    def test_json_serializable_result(self, mock_stock_info, mock_sector_median):
        """Should return JSON-serializable result."""
        mock_stock_info.return_value = {
            "ticker": "TEST.NS",
            "company_name": "Test Co",
            "sector": "IT",
            "pe_ratio": 25.0,
            "pb_ratio": 4.0,
            "roe_pct": 15.0,
            "market_cap_cr": 30000.0,
        }

        mock_sector_median.return_value = {
            "sector": "IT",
            "medians": {"pe_ratio": 28.0}
        }

        result = run_ratio_cruncher("TEST", "Test Co", "IT")

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        # Verify structure
        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "ticker" in parsed
        assert "company_name" in parsed
        assert "sector" in parsed
        assert "ratios" in parsed
        assert "sector_benchmarks" in parsed
        assert "strengths" in parsed
        assert "concerns" in parsed
        assert "valuation_summary" in parsed

    @patch("backend.agents.ratio_cruncher.get_sector_median")
    @patch("backend.agents.ratio_cruncher.get_stock_info")
    def test_no_hallucinated_ratios(self, mock_stock_info, mock_sector_median):
        """Should only report ratios from input data, no hallucination."""
        # Provide limited input data
        mock_stock_info.return_value = {
            "ticker": "TEST.NS",
            "company_name": "Test Co",
            "sector": "IT",
            "pe_ratio": 30.0,
            "pb_ratio": None,  # Explicitly missing
            "roe_pct": None,  # Explicitly missing
        }

        mock_sector_median.return_value = {
            "sector": "IT",
            "medians": {}
        }

        result = run_ratio_cruncher("TEST", "Test Co", "IT")

        # Should only have PE ratio, others should be None
        assert result["ratios"]["pe_ratio"] == 30.0
        assert result["ratios"]["pb_ratio"] is None
        assert result["ratios"]["roe_pct"] is None

        # Verify no invented numbers in strengths/concerns
        all_text = " ".join(result["strengths"] + result["concerns"] + [result["valuation_summary"]])
        # Should mention P/E
        assert "P/E" in all_text or "30.0" in all_text or "Limited" in all_text

    @patch("backend.agents.ratio_cruncher.get_sector_median")
    @patch("backend.agents.ratio_cruncher.get_stock_info")
    def test_sector_detection(self, mock_stock_info, mock_sector_median):
        """Should detect sector from stock info if not provided."""
        mock_stock_info.return_value = {
            "ticker": "TCS.NS",
            "company_name": "TCS",
            "sector": "Information Technology",  # Should be auto-detected
            "pe_ratio": 28.0,
            "market_cap_cr": 120000.0,
        }

        mock_sector_median.return_value = {
            "sector": "Information Technology",
            "medians": {"pe_ratio": 30.0}
        }

        # Don't provide sector
        result = run_ratio_cruncher("TCS", "TCS", sector="")

        assert result["sector"] == "Information Technology"

    @patch("backend.agents.ratio_cruncher.get_sector_median")
    @patch("backend.agents.ratio_cruncher.get_stock_info")
    def test_strength_detection_logic(self, mock_stock_info, mock_sector_median):
        """Should detect strengths based on ratio thresholds."""
        mock_stock_info.return_value = {
            "ticker": "TEST.NS",
            "company_name": "Strong Co",
            "sector": "IT",
            "roe_pct": 25.0,  # Strong ROE
            "roa_pct": 15.0,  # Strong ROA
            "debt_to_equity": 30.0,  # Low debt
            "current_ratio": 2.0,  # Strong liquidity
            "operating_margins_pct": 25.0,  # Strong margins
            "profit_margins_pct": 18.0,  # Strong margins
            "dividend_yield_pct": 3.5,  # Good dividend
            "market_cap_cr": 50000.0,
        }

        mock_sector_median.return_value = {
            "sector": "IT",
            "medians": {
                "roe_pct": 15.0,
                "roa_pct": 8.0,
            }
        }

        result = run_ratio_cruncher("TEST", "Strong Co", "IT")

        # Should identify multiple strengths
        assert len(result["strengths"]) >= 5
        strengths_text = " ".join(result["strengths"])
        assert "ROE" in strengths_text or "25.0" in strengths_text
        assert "liquidity" in strengths_text.lower() or "current ratio" in strengths_text.lower()

    @patch("backend.agents.ratio_cruncher.get_sector_median")
    @patch("backend.agents.ratio_cruncher.get_stock_info")
    def test_concern_detection_logic(self, mock_stock_info, mock_sector_median):
        """Should detect concerns based on ratio thresholds."""
        mock_stock_info.return_value = {
            "ticker": "TEST.NS",
            "company_name": "Weak Co",
            "sector": "IT",
            "roe_pct": 5.0,  # Low ROE
            "roa_pct": 3.0,  # Low ROA
            "pe_ratio": 50.0,  # High P/E
            "debt_to_equity": 200.0,  # High debt
            "current_ratio": 0.8,  # Weak liquidity
            "operating_margins_pct": 3.0,  # Thin margins
            "profit_margins_pct": 2.0,  # Low margins
            "market_cap_cr": 10000.0,
        }

        mock_sector_median.return_value = {
            "sector": "IT",
            "medians": {
                "pe_ratio": 25.0,
                "roe_pct": 15.0,
            }
        }

        result = run_ratio_cruncher("TEST", "Weak Co", "IT")

        # Should identify multiple concerns
        assert len(result["concerns"]) >= 4
        concerns_text = " ".join(result["concerns"])
        assert "ROE" in concerns_text or "5.0" in concerns_text
        assert "liquidity" in concerns_text.lower() or "current ratio" in concerns_text.lower()

    @patch("backend.agents.ratio_cruncher.get_sector_median")
    @patch("backend.agents.ratio_cruncher.get_stock_info")
    def test_valuation_summary_generation(self, mock_stock_info, mock_sector_median):
        """Should generate meaningful valuation summary."""
        mock_stock_info.return_value = {
            "ticker": "TEST.NS",
            "company_name": "Test Company",
            "sector": "IT",
            "pe_ratio": 20.0,
            "pb_ratio": 3.0,
            "roe_pct": 18.0,
            "debt_to_equity": 40.0,
            "market_cap_cr": 120000.0,  # Large cap
        }

        mock_sector_median.return_value = {
            "sector": "IT",
            "medians": {
                "pe_ratio": 28.0,
                "pb_ratio": 4.0,
            }
        }

        result = run_ratio_cruncher("TEST", "Test Company", "IT")

        summary = result["valuation_summary"]
        assert len(summary) > 50  # Should be substantial
        assert "Test Company" in summary
        # Should mention key factors
        assert any(word in summary.lower() for word in ["cap", "discount", "premium", "profitability", "leverage"])

    @patch("backend.agents.ratio_cruncher.get_stock_info")
    def test_exception_handling(self, mock_stock_info):
        """Should handle unexpected exceptions gracefully."""
        mock_stock_info.side_effect = Exception("Unexpected error")

        result = run_ratio_cruncher("TEST", "Test Co")

        assert result["status"] == "error"
        assert "Unexpected error" in result["error"]

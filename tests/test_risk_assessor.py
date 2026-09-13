"""
Tests for Risk Assessor agent.
"""

import pytest
import json
from unittest.mock import Mock, patch
from backend.agents.risk_assessor import run_risk_assessor


class TestRiskAssessor:
    """Tests for run_risk_assessor function."""

    @patch("backend.agents.risk_assessor.get_stock_info")
    @patch("backend.agents.risk_assessor.get_nifty_history")
    @patch("backend.agents.risk_assessor.get_price_history")
    def test_successful_risk_assessment(self, mock_price, mock_nifty, mock_stock):
        """Should successfully assess risk with all data available."""
        # Mock price history
        mock_price.func = Mock(return_value={
            "ticker": "RELIANCE.NS",
            "close": [2400, 2450, 2420, 2480, 2500, 2460, 2490, 2510, 2480, 2520] * 10,
        })

        # Mock Nifty history
        mock_nifty.func = Mock(return_value={
            "close": [18000, 18100, 18050, 18150, 18200, 18180, 18220, 18250, 18230, 18270] * 10,
        })

        # Mock stock info
        mock_stock.func = Mock(return_value={
            "ticker": "RELIANCE.NS",
            "debt_to_equity": 45.0,
            "current_ratio": 1.5,
            "pe_ratio": 25.0,
            "pb_ratio": 2.8,
        })

        result = run_risk_assessor("RELIANCE", "Reliance Industries")

        assert result["status"] in ["success", "partial_success"]
        assert result["ticker"] == "RELIANCE"
        assert result["company_name"] == "Reliance Industries"
        assert 0 <= result["risk_score"] <= 100
        assert result["risk_level"] in ["LOW", "MEDIUM", "MEDIUM-HIGH", "HIGH", "VERY HIGH"]
        assert len(result["risk_factors"]) > 0
        assert len(result["summary"]) > 0

    @patch("backend.agents.risk_assessor.get_stock_info")
    @patch("backend.agents.risk_assessor.get_nifty_history")
    @patch("backend.agents.risk_assessor.get_price_history")
    def test_partial_success_one_source_fails(self, mock_price, mock_nifty, mock_stock):
        """Should handle partial success when one data source fails."""
        mock_price.func = Mock(return_value={"close": [1000] * 100})
        mock_nifty.func = Mock(return_value={"error": "Nifty data unavailable"})
        mock_stock.func = Mock(return_value={"debt_to_equity": 50.0, "current_ratio": 1.2})

        result = run_risk_assessor("TEST", "Test Company")

        assert result["status"] in ["success", "partial_success"]
        assert "beta" in result["unavailable_metrics"]

    @patch("backend.agents.risk_assessor.get_stock_info")
    @patch("backend.agents.risk_assessor.get_price_history")
    def test_all_data_unavailable(self, mock_price, mock_stock):
        """Should handle case when all data is unavailable."""
        mock_price.func = Mock(return_value={"error": "Price data unavailable"})
        mock_stock.func = Mock(return_value={"error": "Stock info unavailable"})

        result = run_risk_assessor("INVALID", "Invalid Company")

        assert result["status"] == "error"
        assert result["error"] is not None

    @patch("backend.agents.risk_assessor.get_stock_info")
    @patch("backend.agents.risk_assessor.get_nifty_history")
    @patch("backend.agents.risk_assessor.get_price_history")
    def test_json_serializable_result(self, mock_price, mock_nifty, mock_stock):
        """Should return JSON-serializable result."""
        mock_price.func = Mock(return_value={"close": [1000] * 50})
        mock_nifty.func = Mock(return_value={"close": [18000] * 50})
        mock_stock.func = Mock(return_value={"debt_to_equity": 50.0})

        result = run_risk_assessor("TEST", "Test Company")

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "risk_score" in parsed
        assert "risk_level" in parsed

    @patch("backend.agents.risk_assessor.get_stock_info")
    @patch("backend.agents.risk_assessor.get_price_history")
    def test_risk_score_bounds(self, mock_price, mock_stock):
        """Should keep risk_score within bounds [0, 100]."""
        mock_price.func = Mock(return_value={"close": [1000, 1010, 990, 1005, 995] * 20})
        mock_stock.func = Mock(return_value={
            "debt_to_equity": 250.0,
            "current_ratio": 0.8,
            "pe_ratio": 60.0,
        })

        result = run_risk_assessor("TEST", "Test Company")

        assert 0 <= result["risk_score"] <= 100

    @patch("backend.agents.risk_assessor.get_stock_info")
    @patch("backend.agents.risk_assessor.get_price_history")
    def test_high_risk_detection(self, mock_price, mock_stock):
        """Should detect high risk companies."""
        mock_price.func = Mock(return_value={
            "close": [1000, 1100, 900, 1050, 850, 1000, 950, 1100, 800, 1000] * 10
        })
        mock_stock.func = Mock(return_value={
            "debt_to_equity": 250.0,
            "current_ratio": 0.7,
            "pe_ratio": 65.0,
        })

        result = run_risk_assessor("HIGHRISK", "High Risk Company")

        assert result["risk_score"] >= 50
        assert result["risk_level"] in ["MEDIUM-HIGH", "HIGH", "VERY HIGH"]

    @patch("backend.agents.risk_assessor.get_stock_info")
    @patch("backend.agents.risk_assessor.get_price_history")
    def test_low_risk_detection(self, mock_price, mock_stock):
        """Should detect low risk companies."""
        mock_price.func = Mock(return_value={
            "close": [1000, 1005, 1002, 1008, 1006, 1004, 1007, 1009, 1003, 1006] * 10
        })
        mock_stock.func = Mock(return_value={
            "debt_to_equity": 30.0,
            "current_ratio": 2.0,
            "pe_ratio": 18.0,
        })

        result = run_risk_assessor("LOWRISK", "Low Risk Company")

        assert result["risk_score"] <= 50

    @patch("backend.agents.risk_assessor.get_stock_info")
    @patch("backend.agents.risk_assessor.get_price_history")
    def test_risk_flags_generation(self, mock_price, mock_stock):
        """Should generate appropriate risk flags."""
        mock_price.func = Mock(return_value={
            "close": [1000, 1150, 950, 1100, 850, 1050] * 20
        })
        mock_stock.func = Mock(return_value={
            "debt_to_equity": 220.0,
            "current_ratio": 0.9,
        })

        result = run_risk_assessor("TEST", "Test Company")

        # High volatility and leverage should generate flags
        assert isinstance(result["risk_flags"], list)

    @patch("backend.agents.risk_assessor.get_stock_info")
    @patch("backend.agents.risk_assessor.get_nifty_history")
    @patch("backend.agents.risk_assessor.get_price_history")
    def test_with_ratio_output(self, mock_price, mock_nifty, mock_stock):
        """Should incorporate ratio analysis output."""
        mock_price.func = Mock(return_value={"close": [1000] * 50})
        mock_nifty.func = Mock(return_value={"close": [18000] * 50})
        mock_stock.func = Mock(return_value={"debt_to_equity": 50.0})

        ratio_output = {
            "status": "success",
            "ratios": {"roe_pct": 8.0}
        }

        result = run_risk_assessor("TEST", "Test Company", ratio_output=ratio_output)

        # Low ROE should increase risk score
        assert result["status"] in ["success", "partial_success"]

    @patch("backend.agents.risk_assessor.get_stock_info")
    @patch("backend.agents.risk_assessor.get_price_history")
    def test_with_technical_output(self, mock_price, mock_stock):
        """Should incorporate technical analysis output."""
        mock_price.func = Mock(return_value={"close": [1000] * 50})
        mock_stock.func = Mock(return_value={"debt_to_equity": 50.0})

        technical_output = {
            "status": "success",
            "technical_score": 25.0
        }

        result = run_risk_assessor("TEST", "Test Company", technical_output=technical_output)

        assert result["status"] in ["success", "partial_success"]

    @patch("backend.agents.risk_assessor.get_stock_info")
    @patch("backend.agents.risk_assessor.get_price_history")
    def test_unavailable_metrics_tracking(self, mock_price, mock_stock):
        """Should track unavailable metrics."""
        mock_price.func = Mock(return_value={"close": [1000] * 50})
        mock_stock.func = Mock(return_value={"current_ratio": 1.5})

        result = run_risk_assessor("TEST", "Test Company")

        assert "promoter_pledge" in result["unavailable_metrics"]
        assert "altman_z_score" in result["unavailable_metrics"]

    @patch("backend.agents.risk_assessor.get_stock_info")
    @patch("backend.agents.risk_assessor.get_price_history")
    def test_exception_handling(self, mock_price, mock_stock):
        """Should handle unexpected exceptions gracefully."""
        mock_price.func = Mock(return_value={"close": [1000] * 50})
        mock_stock.func = Mock(side_effect=Exception("Unexpected error"))

        result = run_risk_assessor("TEST", "Test Company")

        assert result["status"] in ["error", "partial_success"]
        assert "risk_score" in result
        assert "risk_level" in result

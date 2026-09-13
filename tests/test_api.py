"""
Tests for FastAPI backend.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


class TestFastAPI:
    """Tests for FastAPI endpoints."""

    def test_root_endpoint(self):
        """Should return API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "FinSight AI"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data
        assert "health" in data["endpoints"]
        assert "analyze" in data["endpoints"]

    def test_health_endpoint(self):
        """Should return healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "FinSight AI"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data

    @patch("main.run_analysis")
    def test_successful_analyze_request(self, mock_run_analysis):
        """Should successfully analyze stock with full request."""
        # Mock workflow result
        mock_run_analysis.return_value = {
            "ticker": "RELIANCE.NS",
            "company_name": "Reliance Industries",
            "sector": "Energy",
            "filing_output": {"status": "success"},
            "ratio_output": {"status": "success"},
            "news_output": {"status": "success"},
            "technical_output": {"status": "success"},
            "risk_output": {"status": "success"},
            "final_report": {"status": "success", "confidence_score": 85},
            "errors": [],
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:01:00"
        }

        response = client.post(
            "/analyze",
            json={
                "ticker": "RELIANCE",
                "company_name": "Reliance Industries",
                "sector": "Energy"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "RELIANCE.NS"
        assert data["company_name"] == "Reliance Industries"
        assert data["sector"] == "Energy"
        assert data["final_report"]["status"] == "success"
        assert len(data["errors"]) == 0

        # Verify workflow was called correctly
        mock_run_analysis.assert_called_once_with(
            ticker="RELIANCE",
            company_name="Reliance Industries",
            sector="Energy"
        )

    @patch("main.run_analysis")
    def test_analyze_with_optional_fields_missing(self, mock_run_analysis):
        """Should handle analyze request with only ticker."""
        mock_run_analysis.return_value = {
            "ticker": "TCS.NS",
            "company_name": "TCS.NS",
            "sector": "",
            "filing_output": {"status": "success"},
            "ratio_output": {"status": "success"},
            "news_output": {"status": "success"},
            "technical_output": {"status": "success"},
            "risk_output": {"status": "success"},
            "final_report": {"status": "success", "confidence_score": 80},
            "errors": [],
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:01:00"
        }

        response = client.post(
            "/analyze",
            json={"ticker": "TCS"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "TCS.NS"

        # Verify workflow was called with empty strings for optional fields
        mock_run_analysis.assert_called_once_with(
            ticker="TCS",
            company_name="",
            sector=""
        )

    def test_invalid_ticker_empty(self):
        """Should return 400 for empty ticker."""
        response = client.post(
            "/analyze",
            json={"ticker": ""}
        )

        assert response.status_code == 400
        data = response.json()
        assert "required" in data["detail"].lower()

    def test_invalid_ticker_too_long(self):
        """Should return 400 for ticker too long."""
        response = client.post(
            "/analyze",
            json={"ticker": "A" * 25}
        )

        assert response.status_code == 400
        data = response.json()
        assert "too long" in data["detail"].lower()

    def test_invalid_ticker_missing(self):
        """Should return 422 for missing ticker field."""
        response = client.post(
            "/analyze",
            json={}
        )

        assert response.status_code == 422
        # FastAPI validation error

    @patch("main.run_analysis")
    def test_workflow_exception_returns_500(self, mock_run_analysis):
        """Should return 500 when workflow raises exception."""
        mock_run_analysis.side_effect = Exception("Unexpected workflow error")

        response = client.post(
            "/analyze",
            json={"ticker": "TEST"}
        )

        assert response.status_code == 500
        data = response.json()
        assert "failed" in data["detail"].lower()

    @patch("main.run_analysis")
    def test_workflow_value_error_returns_400(self, mock_run_analysis):
        """Should return 400 when workflow raises ValueError."""
        mock_run_analysis.side_effect = ValueError("Invalid ticker format")

        response = client.post(
            "/analyze",
            json={"ticker": "INVALID"}
        )

        assert response.status_code == 400
        data = response.json()
        assert "Invalid ticker format" in data["detail"]

    @patch("main.run_analysis")
    def test_json_serializable_response(self, mock_run_analysis):
        """Should return JSON-serializable response."""
        mock_run_analysis.return_value = {
            "ticker": "INFY.NS",
            "company_name": "Infosys",
            "sector": "IT",
            "filing_output": {"status": "success", "insights": {"key": "value"}},
            "ratio_output": {"status": "success", "ratios": {"roe_pct": 15.0}},
            "news_output": {"status": "success", "sentiment": "POSITIVE"},
            "technical_output": {"status": "success", "trend": "BULLISH"},
            "risk_output": {"status": "success", "risk_level": "MEDIUM"},
            "final_report": {
                "status": "success",
                "report_markdown": "# Report",
                "confidence_score": 85
            },
            "errors": [],
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:01:00"
        }

        response = client.post(
            "/analyze",
            json={"ticker": "INFY", "company_name": "Infosys", "sector": "IT"}
        )

        assert response.status_code == 200

        # Should be valid JSON
        import json
        json_str = json.dumps(response.json())
        assert json_str is not None

        parsed = json.loads(json_str)
        assert parsed["ticker"] == "INFY.NS"

    @patch("main.run_analysis")
    def test_ticker_whitespace_trimming(self, mock_run_analysis):
        """Should trim whitespace from ticker."""
        mock_run_analysis.return_value = {
            "ticker": "TCS.NS",
            "company_name": "TCS",
            "sector": "",
            "filing_output": None,
            "ratio_output": None,
            "news_output": None,
            "technical_output": None,
            "risk_output": None,
            "final_report": None,
            "errors": [],
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:01:00"
        }

        response = client.post(
            "/analyze",
            json={"ticker": "  TCS  "}
        )

        assert response.status_code == 200

        # Verify trimmed ticker was passed
        mock_run_analysis.assert_called_once_with(
            ticker="TCS",
            company_name="",
            sector=""
        )

    @patch("main.run_analysis")
    def test_partial_success_response(self, mock_run_analysis):
        """Should handle partial success with some agent failures."""
        mock_run_analysis.return_value = {
            "ticker": "TEST.NS",
            "company_name": "Test Company",
            "sector": "",
            "filing_output": {"status": "error", "error": "No PDFs found"},
            "ratio_output": {"status": "success", "ratios": {}},
            "news_output": {"status": "success", "sentiment": "NEUTRAL"},
            "technical_output": {"status": "success", "trend": "NEUTRAL"},
            "risk_output": {"status": "partial_success", "risk_level": "MEDIUM"},
            "final_report": {"status": "partial_success", "confidence_score": 60},
            "errors": ["filing_analyst error: No PDFs found"],
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:01:00"
        }

        response = client.post(
            "/analyze",
            json={"ticker": "TEST", "company_name": "Test Company"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["errors"]) > 0
        assert data["final_report"]["status"] == "partial_success"

    @patch("main.run_analysis")
    def test_cors_headers(self, mock_run_analysis):
        """Should include CORS headers for cross-origin requests."""
        mock_run_analysis.return_value = {
            "ticker": "TEST.NS",
            "company_name": "Test",
            "sector": "",
            "errors": [],
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:01:00"
        }

        # Make request with Origin header
        response = client.post(
            "/analyze",
            json={"ticker": "TEST"},
            headers={"Origin": "http://localhost:3000"}
        )

        assert response.status_code == 200
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers

    def test_openapi_docs_available(self):
        """Should serve OpenAPI documentation."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json_available(self):
        """Should serve OpenAPI JSON schema."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "FinSight AI"
        assert data["info"]["version"] == "0.1.0"

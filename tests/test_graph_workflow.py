"""
Tests for LangGraph workflow orchestration.
"""

import pytest
import json
from unittest.mock import Mock, patch
from backend.graph.workflow import run_analysis


class TestGraphWorkflow:
    """Tests for LangGraph workflow orchestration."""

    @patch("backend.graph.workflow.run_report_writer")
    @patch("backend.graph.workflow.run_risk_assessor")
    @patch("backend.graph.workflow.run_technical_analyst")
    @patch("backend.graph.workflow.run_news_sentinel")
    @patch("backend.graph.workflow.run_ratio_cruncher")
    @patch("backend.graph.workflow.run_filing_analyst")
    def test_successful_full_workflow(
        self, mock_filing, mock_ratio, mock_news, mock_tech, mock_risk, mock_report
    ):
        """Should successfully execute full workflow with all agents."""
        # Mock agent outputs
        mock_filing.return_value = {"status": "success", "insights": {"revenue": "10%"}}
        mock_ratio.return_value = {"status": "success", "ratios": {"roe_pct": 15.0}}
        mock_news.return_value = {"status": "success", "sentiment": "POSITIVE", "sentiment_score": 0.7, "article_count": 10}
        mock_tech.return_value = {"status": "success", "trend": "BULLISH", "technical_score": 65}
        mock_risk.return_value = {"status": "success", "risk_level": "MEDIUM", "risk_score": 40}
        mock_report.return_value = {
            "status": "success",
            "report_markdown": "# Report",
            "confidence_score": 85
        }

        result = run_analysis("RELIANCE", "Reliance Industries", "Energy")

        # Verify all agents were called
        mock_filing.assert_called_once()
        mock_ratio.assert_called_once()
        mock_news.assert_called_once()
        mock_tech.assert_called_once()
        mock_risk.assert_called_once()
        mock_report.assert_called_once()

        # Verify state structure
        assert result["ticker"] == "RELIANCE.NS"
        assert result["company_name"] == "Reliance Industries"
        assert result["sector"] == "Energy"
        assert result["filing_output"]["status"] == "success"
        assert result["ratio_output"]["status"] == "success"
        assert result["news_output"]["status"] == "success"
        assert result["technical_output"]["status"] == "success"
        assert result["risk_output"]["status"] == "success"
        assert result["final_report"]["status"] == "success"
        assert len(result["errors"]) == 0
        assert result["started_at"] != ""
        assert result["completed_at"] != ""

    @patch("backend.graph.workflow.run_report_writer")
    @patch("backend.graph.workflow.run_risk_assessor")
    @patch("backend.graph.workflow.run_technical_analyst")
    @patch("backend.graph.workflow.run_news_sentinel")
    @patch("backend.graph.workflow.run_ratio_cruncher")
    @patch("backend.graph.workflow.run_filing_analyst")
    def test_one_specialist_failure_still_produces_report(
        self, mock_filing, mock_ratio, mock_news, mock_tech, mock_risk, mock_report
    ):
        """Should continue workflow even if one specialist agent fails."""
        # Filing agent fails
        mock_filing.return_value = {"status": "error", "error": "No PDFs found"}
        # Others succeed
        mock_ratio.return_value = {"status": "success", "ratios": {"roe_pct": 15.0}}
        mock_news.return_value = {"status": "success", "sentiment": "NEUTRAL", "sentiment_score": 0.0, "article_count": 5}
        mock_tech.return_value = {"status": "success", "trend": "NEUTRAL", "technical_score": 50}
        mock_risk.return_value = {"status": "partial_success", "risk_level": "MEDIUM", "risk_score": 45}
        mock_report.return_value = {
            "status": "partial_success",
            "report_markdown": "# Partial Report",
            "confidence_score": 60
        }

        result = run_analysis("TEST", "Test Company")

        # All agents should still be called
        assert mock_filing.called
        assert mock_ratio.called
        assert mock_news.called
        assert mock_tech.called
        assert mock_risk.called
        assert mock_report.called

        # Error should be captured
        assert len(result["errors"]) > 0
        assert any("filing_analyst" in err for err in result["errors"])

        # Report should still be generated
        assert result["final_report"] is not None
        assert result["final_report"]["status"] in ["success", "partial_success"]

    @patch("backend.graph.workflow.run_report_writer")
    @patch("backend.graph.workflow.run_risk_assessor")
    @patch("backend.graph.workflow.run_technical_analyst")
    @patch("backend.graph.workflow.run_news_sentinel")
    @patch("backend.graph.workflow.run_ratio_cruncher")
    @patch("backend.graph.workflow.run_filing_analyst")
    def test_risk_assessor_receives_ratio_and_technical(
        self, mock_filing, mock_ratio, mock_news, mock_tech, mock_risk, mock_report
    ):
        """Should pass ratio_output and technical_output to risk_assessor."""
        ratio_data = {"status": "success", "ratios": {"roe_pct": 18.0}}
        tech_data = {"status": "success", "trend": "BULLISH", "technical_score": 70}

        mock_filing.return_value = {"status": "success", "insights": {}}
        mock_ratio.return_value = ratio_data
        mock_news.return_value = {"status": "success", "sentiment": "POSITIVE", "sentiment_score": 0.6, "article_count": 8}
        mock_tech.return_value = tech_data
        mock_risk.return_value = {"status": "success", "risk_level": "LOW", "risk_score": 25}
        mock_report.return_value = {"status": "success", "report_markdown": "# Report", "confidence_score": 90}

        run_analysis("TEST", "Test Company")

        # Verify risk_assessor received ratio and technical outputs
        mock_risk.assert_called_once()
        call_kwargs = mock_risk.call_args.kwargs
        assert call_kwargs["ratio_output"] == ratio_data
        assert call_kwargs["technical_output"] == tech_data

    @patch("backend.graph.workflow.run_report_writer")
    @patch("backend.graph.workflow.run_risk_assessor")
    @patch("backend.graph.workflow.run_technical_analyst")
    @patch("backend.graph.workflow.run_news_sentinel")
    @patch("backend.graph.workflow.run_ratio_cruncher")
    @patch("backend.graph.workflow.run_filing_analyst")
    def test_report_writer_receives_all_outputs(
        self, mock_filing, mock_ratio, mock_news, mock_tech, mock_risk, mock_report
    ):
        """Should pass all agent outputs to report_writer."""
        filing_data = {"status": "success", "insights": {"key": "value"}}
        ratio_data = {"status": "success", "ratios": {"roe_pct": 15.0}}
        news_data = {"status": "success", "sentiment": "NEUTRAL", "sentiment_score": 0.0, "article_count": 6}
        tech_data = {"status": "success", "trend": "NEUTRAL", "technical_score": 55}
        risk_data = {"status": "success", "risk_level": "MEDIUM", "risk_score": 48}

        mock_filing.return_value = filing_data
        mock_ratio.return_value = ratio_data
        mock_news.return_value = news_data
        mock_tech.return_value = tech_data
        mock_risk.return_value = risk_data
        mock_report.return_value = {"status": "success", "report_markdown": "# Report", "confidence_score": 75}

        run_analysis("TEST", "Test Company")

        # Verify report_writer received all outputs
        mock_report.assert_called_once()
        call_kwargs = mock_report.call_args.kwargs
        assert call_kwargs["filing_output"] == filing_data
        assert call_kwargs["ratio_output"] == ratio_data
        assert call_kwargs["news_output"] == news_data
        assert call_kwargs["technical_output"] == tech_data
        assert call_kwargs["risk_output"] == risk_data

    @patch("backend.graph.workflow.run_report_writer")
    @patch("backend.graph.workflow.run_risk_assessor")
    @patch("backend.graph.workflow.run_technical_analyst")
    @patch("backend.graph.workflow.run_news_sentinel")
    @patch("backend.graph.workflow.run_ratio_cruncher")
    @patch("backend.graph.workflow.run_filing_analyst")
    def test_invalid_ticker_handling(
        self, mock_filing, mock_ratio, mock_news, mock_tech, mock_risk, mock_report
    ):
        """Should handle invalid ticker gracefully."""
        # All agents return errors
        error_output = {"status": "error", "error": "Invalid ticker"}
        mock_filing.return_value = error_output
        mock_ratio.return_value = error_output
        mock_news.return_value = error_output
        mock_tech.return_value = error_output
        mock_risk.return_value = error_output
        mock_report.return_value = error_output

        result = run_analysis("INVALIDXYZ", "Invalid Company")

        # Should still return valid state
        assert result["ticker"] == "INVALIDXYZ.NS"
        assert result["company_name"] == "Invalid Company"
        assert len(result["errors"]) > 0
        assert result["completed_at"] != ""

    @patch("backend.graph.workflow.run_report_writer")
    @patch("backend.graph.workflow.run_risk_assessor")
    @patch("backend.graph.workflow.run_technical_analyst")
    @patch("backend.graph.workflow.run_news_sentinel")
    @patch("backend.graph.workflow.run_ratio_cruncher")
    @patch("backend.graph.workflow.run_filing_analyst")
    def test_errors_list_populated(
        self, mock_filing, mock_ratio, mock_news, mock_tech, mock_risk, mock_report
    ):
        """Should populate errors list with agent failures."""
        mock_filing.return_value = {"status": "error", "error": "Filing error"}
        mock_ratio.return_value = {"status": "success", "ratios": {}}
        mock_news.return_value = {"status": "error", "error": "News error"}
        mock_tech.return_value = {"status": "success", "trend": "NEUTRAL", "technical_score": 50}
        mock_risk.return_value = {"status": "error", "error": "Risk error"}
        mock_report.return_value = {"status": "success", "report_markdown": "# Report", "confidence_score": 40}

        result = run_analysis("TEST", "Test Company")

        # Should have 3 errors
        assert len(result["errors"]) == 3
        assert any("filing_analyst" in err for err in result["errors"])
        assert any("news_sentinel" in err for err in result["errors"])
        assert any("risk_assessor" in err for err in result["errors"])

    @patch("backend.graph.workflow.run_report_writer")
    @patch("backend.graph.workflow.run_risk_assessor")
    @patch("backend.graph.workflow.run_technical_analyst")
    @patch("backend.graph.workflow.run_news_sentinel")
    @patch("backend.graph.workflow.run_ratio_cruncher")
    @patch("backend.graph.workflow.run_filing_analyst")
    def test_final_state_json_serializable(
        self, mock_filing, mock_ratio, mock_news, mock_tech, mock_risk, mock_report
    ):
        """Should return JSON-serializable final state."""
        mock_filing.return_value = {"status": "success", "insights": {}}
        mock_ratio.return_value = {"status": "success", "ratios": {"roe_pct": 15.0}}
        mock_news.return_value = {"status": "success", "sentiment": "NEUTRAL", "sentiment_score": 0.0, "article_count": 5}
        mock_tech.return_value = {"status": "success", "trend": "NEUTRAL", "technical_score": 50}
        mock_risk.return_value = {"status": "success", "risk_level": "MEDIUM", "risk_score": 45}
        mock_report.return_value = {"status": "success", "report_markdown": "# Report", "confidence_score": 75}

        result = run_analysis("TEST", "Test Company")

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        parsed = json.loads(json_str)
        assert parsed["ticker"] == "TEST.NS"
        assert "started_at" in parsed
        assert "completed_at" in parsed

    @patch("backend.graph.workflow.run_report_writer")
    @patch("backend.graph.workflow.run_risk_assessor")
    @patch("backend.graph.workflow.run_technical_analyst")
    @patch("backend.graph.workflow.run_news_sentinel")
    @patch("backend.graph.workflow.run_ratio_cruncher")
    @patch("backend.graph.workflow.run_filing_analyst")
    def test_completed_at_is_set(
        self, mock_filing, mock_ratio, mock_news, mock_tech, mock_risk, mock_report
    ):
        """Should set completed_at timestamp."""
        mock_filing.return_value = {"status": "success", "insights": {}}
        mock_ratio.return_value = {"status": "success", "ratios": {}}
        mock_news.return_value = {"status": "success", "sentiment": "NEUTRAL", "sentiment_score": 0.0, "article_count": 3}
        mock_tech.return_value = {"status": "success", "trend": "NEUTRAL", "technical_score": 50}
        mock_risk.return_value = {"status": "success", "risk_level": "MEDIUM", "risk_score": 50}
        mock_report.return_value = {"status": "success", "report_markdown": "# Report", "confidence_score": 70}

        result = run_analysis("TEST", "Test Company")

        assert result["completed_at"] != ""
        assert result["started_at"] != ""
        # Completed should be after or equal to started
        assert result["completed_at"] >= result["started_at"]

    @patch("backend.graph.workflow.run_report_writer")
    @patch("backend.graph.workflow.run_risk_assessor")
    @patch("backend.graph.workflow.run_technical_analyst")
    @patch("backend.graph.workflow.run_news_sentinel")
    @patch("backend.graph.workflow.run_ratio_cruncher")
    @patch("backend.graph.workflow.run_filing_analyst")
    def test_ticker_normalization(
        self, mock_filing, mock_ratio, mock_news, mock_tech, mock_risk, mock_report
    ):
        """Should normalize ticker by adding .NS suffix."""
        mock_filing.return_value = {"status": "success", "insights": {}}
        mock_ratio.return_value = {"status": "success", "ratios": {}}
        mock_news.return_value = {"status": "success", "sentiment": "NEUTRAL", "sentiment_score": 0.0, "article_count": 4}
        mock_tech.return_value = {"status": "success", "trend": "NEUTRAL", "technical_score": 50}
        mock_risk.return_value = {"status": "success", "risk_level": "MEDIUM", "risk_score": 50}
        mock_report.return_value = {"status": "success", "report_markdown": "# Report", "confidence_score": 70}

        # Test various ticker formats
        result1 = run_analysis("reliance", "Reliance")
        assert result1["ticker"] == "RELIANCE.NS"

        result2 = run_analysis("TCS.NS", "TCS")
        assert result2["ticker"] == "TCS.NS"

        result3 = run_analysis("  INFY  ", "Infosys")
        assert result3["ticker"] == "INFY.NS"

    @patch("backend.graph.workflow.run_report_writer")
    @patch("backend.graph.workflow.run_risk_assessor")
    @patch("backend.graph.workflow.run_technical_analyst")
    @patch("backend.graph.workflow.run_news_sentinel")
    @patch("backend.graph.workflow.run_ratio_cruncher")
    @patch("backend.graph.workflow.run_filing_analyst")
    def test_agent_exception_handling(
        self, mock_filing, mock_ratio, mock_news, mock_tech, mock_risk, mock_report
    ):
        """Should handle agent exceptions gracefully."""
        # Filing agent raises exception
        mock_filing.side_effect = Exception("Unexpected error")
        mock_ratio.return_value = {"status": "success", "ratios": {}}
        mock_news.return_value = {"status": "success", "sentiment": "NEUTRAL", "sentiment_score": 0.0, "article_count": 4}
        mock_tech.return_value = {"status": "success", "trend": "NEUTRAL", "technical_score": 50}
        mock_risk.return_value = {"status": "success", "risk_level": "MEDIUM", "risk_score": 50}
        mock_report.return_value = {"status": "success", "report_markdown": "# Report", "confidence_score": 65}

        result = run_analysis("TEST", "Test Company")

        # Should capture exception in errors
        assert len(result["errors"]) > 0
        assert any("exception" in err.lower() for err in result["errors"])

        # Should still complete workflow
        assert result["final_report"] is not None

    @patch("backend.graph.workflow.run_report_writer")
    @patch("backend.graph.workflow.run_risk_assessor")
    @patch("backend.graph.workflow.run_technical_analyst")
    @patch("backend.graph.workflow.run_news_sentinel")
    @patch("backend.graph.workflow.run_ratio_cruncher")
    @patch("backend.graph.workflow.run_filing_analyst")
    def test_default_company_name(
        self, mock_filing, mock_ratio, mock_news, mock_tech, mock_risk, mock_report
    ):
        """Should use ticker as company_name if not provided."""
        mock_filing.return_value = {"status": "success", "insights": {}}
        mock_ratio.return_value = {"status": "success", "ratios": {}}
        mock_news.return_value = {"status": "success", "sentiment": "NEUTRAL", "sentiment_score": 0.0, "article_count": 5}
        mock_tech.return_value = {"status": "success", "trend": "NEUTRAL", "technical_score": 50}
        mock_risk.return_value = {"status": "success", "risk_level": "MEDIUM", "risk_score": 50}
        mock_report.return_value = {"status": "success", "report_markdown": "# Report", "confidence_score": 70}

        result = run_analysis("RELIANCE")

        # Should default to ticker
        assert result["company_name"] == "RELIANCE.NS"

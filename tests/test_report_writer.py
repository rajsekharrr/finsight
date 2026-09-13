"""
Tests for Report Writer agent.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from backend.agents.report_writer import run_report_writer


class TestReportWriter:
    """Tests for run_report_writer function."""

    @patch("backend.agents.report_writer.ChatOpenAI")
    def test_successful_full_report(self, mock_chat):
        """Should generate full report with all agent outputs."""
        # Mock LLM response
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "executive_summary": "Test company shows strong fundamentals with moderate risk.",
            "bull_case": "Strong profitability metrics including 18% ROE and healthy balance sheet.",
            "bear_case": "High valuation at 35x P/E and elevated market volatility present risks.",
            "key_risks": ["Valuation risk", "Market volatility", "Sector headwinds"],
            "confidence_score": 85,
            "report_markdown": "# Investment Research Report\n\nComplete report content here."
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        # Sample agent outputs
        filing_output = {"status": "success", "insights": {"revenue_growth": "15%"}}
        ratio_output = {"status": "success", "ratios": {"roe_pct": 18.0, "debt_to_equity": 45.0}}
        news_output = {"status": "success", "sentiment": "POSITIVE", "sentiment_score": 0.7, "article_count": 15}
        technical_output = {"status": "success", "trend": "BULLISH", "technical_score": 68}
        risk_output = {"status": "success", "risk_level": "MEDIUM", "risk_score": 42}

        result = run_report_writer(
            "TEST", "Test Company",
            filing_output, ratio_output, news_output, technical_output, risk_output
        )

        assert result["status"] == "success"
        assert result["ticker"] == "TEST"
        assert result["company_name"] == "Test Company"
        assert len(result["report_markdown"]) > 0
        assert len(result["executive_summary"]) > 0
        assert len(result["bull_case"]) > 0
        assert len(result["bear_case"]) > 0
        assert len(result["key_risks"]) > 0
        assert 0 <= result["confidence_score"] <= 100
        assert len(result["sources_used"]) == 5
        assert len(result["missing_sections"]) == 0
        assert result["error"] is None

    @patch("backend.agents.report_writer.ChatOpenAI")
    def test_partial_report_missing_agent_output(self, mock_chat):
        """Should generate partial report when some agent outputs missing."""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "executive_summary": "Partial analysis based on available data.",
            "bull_case": "Strong technical trend.",
            "bear_case": "Limited data availability increases uncertainty.",
            "key_risks": ["Data gaps", "Limited visibility"],
            "confidence_score": 45,
            "report_markdown": "# Partial Report\n\nBased on limited data."
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        # Only technical and news outputs
        technical_output = {"status": "success", "trend": "BULLISH", "technical_score": 65}
        news_output = {"status": "success", "sentiment": "NEUTRAL", "sentiment_score": 0.0, "article_count": 10}

        result = run_report_writer(
            "TEST", "Test Company",
            None, None, news_output, technical_output, None
        )

        assert result["status"] == "partial_success"
        assert len(result["sources_used"]) == 2
        assert len(result["missing_sections"]) == 3
        assert "missing" in result["error"].lower()
        assert result["confidence_score"] <= 100

    def test_all_outputs_missing(self):
        """Should return error when all agent outputs missing."""
        result = run_report_writer("TEST", "Test Company", None, None, None, None, None)

        assert result["status"] == "error"
        assert result["error"] is not None
        assert "no agent outputs" in result["error"].lower()
        assert len(result["sources_used"]) == 0

    @patch("backend.agents.report_writer.ChatOpenAI")
    def test_malformed_llm_json_fallback(self, mock_chat):
        """Should fall back to structured report when LLM returns malformed JSON."""
        # Mock LLM with invalid JSON
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON {incomplete"
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        ratio_output = {"status": "success", "ratios": {"roe_pct": 15.0}}
        news_output = {"status": "success", "sentiment": "POSITIVE", "sentiment_score": 0.6, "article_count": 8}

        result = run_report_writer("TEST", "Test Company", None, ratio_output, news_output, None, None)

        assert result["status"] == "partial_success"
        assert "fallback" in result["error"].lower()
        assert len(result["report_markdown"]) > 0
        assert len(result["executive_summary"]) > 0

    @patch("backend.agents.report_writer.ChatOpenAI")
    def test_llm_exception_fallback(self, mock_chat):
        """Should fall back to structured report when LLM raises exception."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = Exception("OpenAI API timeout")
        mock_chat.return_value = mock_llm_instance

        ratio_output = {"status": "success", "ratios": {"roe_pct": 15.0}}
        technical_output = {"status": "success", "trend": "NEUTRAL", "technical_score": 50}

        result = run_report_writer("TEST", "Test Company", None, ratio_output, None, technical_output, None)

        assert result["status"] == "partial_success"
        assert "error" in result["error"].lower() or "fallback" in result["error"].lower()
        assert len(result["report_markdown"]) > 0

    @patch("backend.agents.report_writer.ChatOpenAI")
    def test_confidence_score_bounds(self, mock_chat):
        """Should keep confidence_score within bounds [0, 100]."""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "executive_summary": "Test summary",
            "bull_case": "Test bull case",
            "bear_case": "Test bear case",
            "key_risks": ["Risk 1"],
            "confidence_score": 150,  # Out of bounds
            "report_markdown": "# Test Report"
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        ratio_output = {"status": "success", "ratios": {"roe_pct": 15.0}}

        result = run_report_writer("TEST", "Test Company", None, ratio_output, None, None, None)

        # Should clamp to 100
        assert 0 <= result["confidence_score"] <= 100

    @patch("backend.agents.report_writer.ChatOpenAI")
    def test_data_quality_correctness(self, mock_chat):
        """Should correctly assess data quality."""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "executive_summary": "Test",
            "bull_case": "Test",
            "bear_case": "Test",
            "key_risks": [],
            "confidence_score": 60,
            "report_markdown": "# Test"
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        # 3 out of 5 sources
        ratio_output = {"status": "success", "ratios": {}}
        news_output = {"status": "success", "sentiment": "NEUTRAL", "sentiment_score": 0, "article_count": 5}
        risk_output = {"status": "success", "risk_level": "LOW", "risk_score": 20}

        result = run_report_writer("TEST", "Test Company", None, ratio_output, news_output, None, risk_output)

        assert "3/5" in result["data_quality"]
        assert "60%" in result["data_quality"]

    @patch("backend.agents.report_writer.ChatOpenAI")
    def test_sources_used_collection(self, mock_chat):
        """Should correctly collect sources used."""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "executive_summary": "Test",
            "bull_case": "Test",
            "bear_case": "Test",
            "key_risks": [],
            "confidence_score": 50,
            "report_markdown": "# Test"
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        filing_output = {"status": "success", "insights": {}}
        technical_output = {"status": "success", "trend": "NEUTRAL", "technical_score": 50}

        result = run_report_writer("TEST", "Test Company", filing_output, None, None, technical_output, None)

        assert "filing_analysis" in result["sources_used"]
        assert "technical_analysis" in result["sources_used"]
        assert "ratio_analysis" not in result["sources_used"]
        assert "ratio_analysis" in result["missing_sections"]

    @patch("backend.agents.report_writer.ChatOpenAI")
    def test_json_serializable_result(self, mock_chat):
        """Should return JSON-serializable result."""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "executive_summary": "Test summary",
            "bull_case": "Bull case",
            "bear_case": "Bear case",
            "key_risks": ["Risk 1", "Risk 2"],
            "confidence_score": 75,
            "report_markdown": "# Report"
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        ratio_output = {"status": "success", "ratios": {"roe_pct": 15.0}}

        result = run_report_writer("TEST", "Test Company", None, ratio_output, None, None, None)

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "report_markdown" in parsed
        assert "confidence_score" in parsed

    @patch("backend.agents.report_writer.ChatOpenAI")
    def test_no_invented_source_sections(self, mock_chat):
        """Should not include sections for missing data sources."""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "executive_summary": "Based on available data only",
            "bull_case": "Using actual ratio data",
            "bear_case": "Noting data limitations",
            "key_risks": ["Limited data availability"],
            "confidence_score": 40,
            "report_markdown": "# Report with limited data"
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        # Only ratio output provided
        ratio_output = {"status": "success", "ratios": {"roe_pct": 12.0, "debt_to_equity": 60.0}}

        result = run_report_writer("TEST", "Test Company", None, ratio_output, None, None, None)

        # Should only include ratio_analysis in sources
        assert len(result["sources_used"]) == 1
        assert "ratio_analysis" in result["sources_used"]

        # Should note missing sections
        assert len(result["missing_sections"]) == 4

    @patch("backend.agents.report_writer.ChatOpenAI")
    def test_strips_markdown_code_fences(self, mock_chat):
        """Should strip markdown code fences from LLM response."""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """```json
{
  "executive_summary": "Test summary",
  "bull_case": "Bull",
  "bear_case": "Bear",
  "key_risks": ["Risk"],
  "confidence_score": 70,
  "report_markdown": "# Report"
}
```"""
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        ratio_output = {"status": "success", "ratios": {"roe_pct": 15.0}}

        result = run_report_writer("TEST", "Test Company", None, ratio_output, None, None, None)

        assert result["status"] in ["success", "partial_success"]
        assert result["confidence_score"] == 70

    @patch("backend.agents.report_writer.ChatOpenAI")
    def test_handles_partial_success_agent_outputs(self, mock_chat):
        """Should accept partial_success status from agents."""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "executive_summary": "Analysis with partial data",
            "bull_case": "Some positive factors",
            "bear_case": "Some concerns",
            "key_risks": ["Incomplete data"],
            "confidence_score": 55,
            "report_markdown": "# Report"
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        # News and risk with partial_success
        news_output = {"status": "partial_success", "sentiment": "MIXED", "sentiment_score": 0.1, "article_count": 3}
        risk_output = {"status": "partial_success", "risk_level": "MEDIUM", "risk_score": 50}

        result = run_report_writer("TEST", "Test Company", None, None, news_output, None, risk_output)

        # Should accept partial_success outputs
        assert "news_sentiment" in result["sources_used"]
        assert "risk_assessment" in result["sources_used"]

    @patch("backend.agents.report_writer.ChatOpenAI")
    def test_llm_configuration(self, mock_chat):
        """Should configure LLM with correct parameters."""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "executive_summary": "Test",
            "bull_case": "Test",
            "bear_case": "Test",
            "key_risks": [],
            "confidence_score": 50,
            "report_markdown": "# Test"
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        ratio_output = {"status": "success", "ratios": {"roe_pct": 15.0}}

        run_report_writer("TEST", "Test Company", None, ratio_output, None, None, None)

        # Verify LLM was configured correctly
        mock_chat.assert_called_once_with(
            model="gpt-4o",
            temperature=0,
            timeout=120
        )

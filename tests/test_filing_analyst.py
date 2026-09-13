"""
Tests for Filing Analyst agent.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from backend.agents.filing_analyst import run_filing_analyst


class TestFilingAnalyst:
    """Tests for run_filing_analyst function."""

    @patch("backend.agents.filing_analyst.retrieve_for_query")
    @patch("backend.agents.filing_analyst.ChatOpenAI")
    def test_successful_analysis(self, mock_chat, mock_retrieve):
        """Should successfully analyze filings and return structured result."""
        # Mock retrieval responses
        mock_retrieve.return_value = [
            {
                "text": "The company operates in the energy sector with revenue of $50B.",
                "score": 0.95,
                "metadata": {
                    "company": "TESTCO",
                    "file_name": "annual_report_2024.pdf",
                    "page_number": 1
                }
            },
            {
                "text": "Key risk factors include regulatory changes and market volatility.",
                "score": 0.88,
                "metadata": {
                    "company": "TESTCO",
                    "file_name": "annual_report_2024.pdf",
                    "page_number": 15
                }
            }
        ]

        # Mock LLM response
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "filing_summary": "TESTCO is an energy company with $50B revenue.",
            "key_points": [
                "Operates in energy sector",
                "Revenue of $50B",
                "Strong market position"
            ],
            "risks": [
                "Regulatory changes",
                "Market volatility"
            ],
            "outlook": "Planning expansion in renewable energy"
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        result = run_filing_analyst("TEST", "TESTCO")

        assert result["status"] == "success"
        assert result["ticker"] == "TEST"
        assert result["company_name"] == "TESTCO"
        assert "energy company" in result["filing_summary"].lower()
        assert len(result["key_points"]) == 3
        assert len(result["risks"]) == 2
        assert "renewable energy" in result["outlook"].lower()
        assert "annual_report_2024.pdf" in result["sources"]
        assert result["error"] is None

    @patch("backend.agents.filing_analyst.retrieve_for_query")
    def test_no_retrieved_context(self, mock_retrieve):
        """Should handle case when no filing context is retrieved."""
        mock_retrieve.return_value = []

        result = run_filing_analyst("TEST", "TESTCO")

        assert result["status"] == "partial_success"
        assert result["ticker"] == "TEST"
        assert result["company_name"] == "TESTCO"
        assert "no indexed filings" in result["filing_summary"].lower()
        assert result["error"] == "No retrieved context from filings"
        assert result["sources"] == []

    @patch("backend.agents.filing_analyst.retrieve_for_query")
    @patch("backend.agents.filing_analyst.ChatOpenAI")
    def test_llm_json_parsing_error(self, mock_chat, mock_retrieve):
        """Should handle malformed JSON response from LLM gracefully."""
        # Mock retrieval
        mock_retrieve.return_value = [
            {
                "text": "Some filing content",
                "score": 0.9,
                "metadata": {"file_name": "report.pdf"}
            }
        ]

        # Mock LLM with invalid JSON
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON {incomplete"
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        result = run_filing_analyst("TEST", "TESTCO")

        assert result["status"] == "partial_success"
        assert "formatting issue" in result["filing_summary"].lower()
        assert "JSON parse error" in result["error"]
        assert len(result["key_points"]) > 0  # Should have fallback text

    @patch("backend.agents.filing_analyst.retrieve_for_query")
    @patch("backend.agents.filing_analyst.ChatOpenAI")
    def test_llm_exception(self, mock_chat, mock_retrieve):
        """Should handle LLM call exceptions gracefully."""
        # Mock retrieval
        mock_retrieve.return_value = [
            {
                "text": "Filing content",
                "score": 0.9,
                "metadata": {"file_name": "report.pdf"}
            }
        ]

        # Mock LLM to raise exception
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.side_effect = Exception("OpenAI API timeout")
        mock_chat.return_value = mock_llm_instance

        result = run_filing_analyst("TEST", "TESTCO")

        assert result["status"] == "partial_success"
        assert "LLM analysis failed" in result["error"]
        assert len(result["key_points"]) > 0  # Should have fallback info

    @patch("backend.agents.filing_analyst.retrieve_for_query")
    @patch("backend.agents.filing_analyst.ChatOpenAI")
    def test_strips_markdown_code_fences(self, mock_chat, mock_retrieve):
        """Should strip markdown code fences from LLM response."""
        # Mock retrieval
        mock_retrieve.return_value = [
            {
                "text": "Content",
                "score": 0.9,
                "metadata": {"file_name": "report.pdf"}
            }
        ]

        # Mock LLM with markdown fences
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """```json
{
  "filing_summary": "Test summary",
  "key_points": ["Point 1"],
  "risks": ["Risk 1"],
  "outlook": "Positive outlook"
}
```"""
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        result = run_filing_analyst("TEST", "TESTCO")

        assert result["status"] == "success"
        assert result["filing_summary"] == "Test summary"
        assert result["key_points"] == ["Point 1"]

    @patch("backend.agents.filing_analyst.retrieve_for_query")
    @patch("backend.agents.filing_analyst.ChatOpenAI")
    def test_source_metadata_preservation(self, mock_chat, mock_retrieve):
        """Should preserve and deduplicate source file names."""
        # Mock retrieval with multiple chunks from same and different files
        mock_retrieve.return_value = [
            {
                "text": "Content 1",
                "score": 0.9,
                "metadata": {"file_name": "report_2024.pdf"}
            },
            {
                "text": "Content 2",
                "score": 0.85,
                "metadata": {"file_name": "report_2024.pdf"}
            },
            {
                "text": "Content 3",
                "score": 0.8,
                "metadata": {"file_name": "quarterly_q3.pdf"}
            }
        ]

        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "filing_summary": "Summary",
            "key_points": ["Point"],
            "risks": ["Risk"],
            "outlook": "Outlook"
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        result = run_filing_analyst("TEST", "TESTCO")

        assert result["status"] == "success"
        assert len(result["sources"]) == 2
        assert "report_2024.pdf" in result["sources"]
        assert "quarterly_q3.pdf" in result["sources"]
        assert result["sources"] == sorted(result["sources"])  # Should be sorted

    @patch("backend.agents.filing_analyst.retrieve_for_query")
    @patch("backend.agents.filing_analyst.ChatOpenAI")
    def test_json_serializable_result(self, mock_chat, mock_retrieve):
        """Should return JSON-serializable result."""
        # Mock retrieval
        mock_retrieve.return_value = [
            {
                "text": "Content",
                "score": 0.9,
                "metadata": {"file_name": "report.pdf"}
            }
        ]

        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "filing_summary": "Summary",
            "key_points": ["Point 1", "Point 2"],
            "risks": ["Risk 1"],
            "outlook": "Outlook text"
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        result = run_filing_analyst("TEST", "TESTCO")

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        # Verify structure
        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "ticker" in parsed
        assert "company_name" in parsed
        assert "filing_summary" in parsed
        assert "key_points" in parsed
        assert "risks" in parsed
        assert "outlook" in parsed
        assert "sources" in parsed

    @patch("backend.agents.filing_analyst.retrieve_for_query")
    @patch("backend.agents.filing_analyst.ChatOpenAI")
    def test_multiple_retrieval_questions(self, mock_chat, mock_retrieve):
        """Should call retrieve_for_query multiple times for different questions."""
        # Mock retrieval
        mock_retrieve.return_value = [
            {
                "text": "Content",
                "score": 0.9,
                "metadata": {"file_name": "report.pdf"}
            }
        ]

        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "filing_summary": "Summary",
            "key_points": ["Point"],
            "risks": ["Risk"],
            "outlook": "Outlook"
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        run_filing_analyst("TEST", "TESTCO")

        # Should call retrieve_for_query multiple times (6 questions)
        assert mock_retrieve.call_count == 6

        # Verify questions cover required topics
        # Extract query argument (second positional arg or 'query' kwarg)
        call_queries = []
        for call in mock_retrieve.call_args_list:
            args, kwargs = call
            if len(args) > 1:
                call_queries.append(args[1])
            elif 'query' in kwargs:
                call_queries.append(kwargs['query'])

        assert len(call_queries) == 6
        assert any("business" in q.lower() for q in call_queries)
        assert any("revenue" in q.lower() or "profit" in q.lower() for q in call_queries)
        assert any("risk" in q.lower() for q in call_queries)
        assert any("capital" in q.lower() or "outlook" in q.lower() for q in call_queries)

    @patch("backend.agents.filing_analyst.retrieve_for_query")
    def test_retrieval_exception(self, mock_retrieve):
        """Should handle retrieval exceptions gracefully."""
        mock_retrieve.side_effect = Exception("ChromaDB connection failed")

        result = run_filing_analyst("TEST", "TESTCO")

        assert result["status"] == "error"
        assert result["error"] is not None
        assert "ChromaDB connection failed" in result["error"]

    @patch("backend.agents.filing_analyst.retrieve_for_query")
    @patch("backend.agents.filing_analyst.ChatOpenAI")
    def test_llm_configuration(self, mock_chat, mock_retrieve):
        """Should configure LLM with correct parameters."""
        # Mock retrieval
        mock_retrieve.return_value = [
            {
                "text": "Content",
                "score": 0.9,
                "metadata": {"file_name": "report.pdf"}
            }
        ]

        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "filing_summary": "Summary",
            "key_points": ["Point"],
            "risks": ["Risk"],
            "outlook": "Outlook"
        })
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        run_filing_analyst("TEST", "TESTCO")

        # Verify LLM was configured correctly
        mock_chat.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0,
            timeout=60
        )

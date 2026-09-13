"""
FinSight State Definition for LangGraph workflow.

Defines the state structure that flows through the multi-agent pipeline.
"""

from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime


class FinSightState(TypedDict, total=False):
    """
    State for FinSight multi-agent analysis workflow.

    Fields:
        ticker: Stock ticker symbol (e.g., "RELIANCE")
        company_name: Full company name (e.g., "Reliance Industries")
        sector: Industry sector (optional)
        filing_output: Output from filing_analyst agent
        ratio_output: Output from ratio_cruncher agent
        news_output: Output from news_sentinel agent
        technical_output: Output from technical_analyst agent
        risk_output: Output from risk_assessor agent
        final_report: Output from report_writer agent
        errors: List of errors encountered during workflow
        started_at: Workflow start timestamp (ISO 8601 string)
        completed_at: Workflow completion timestamp (ISO 8601 string)
    """
    ticker: str
    company_name: str
    sector: str
    filing_output: Optional[Dict[str, Any]]
    ratio_output: Optional[Dict[str, Any]]
    news_output: Optional[Dict[str, Any]]
    technical_output: Optional[Dict[str, Any]]
    risk_output: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    errors: List[str]
    started_at: str
    completed_at: str

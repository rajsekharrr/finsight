"""
FinSight LangGraph Workflow: Multi-agent orchestration for investment research.

Orchestrates six specialist agents in a structured pipeline:
1. Parallel execution: filing_analyst, ratio_cruncher, news_sentinel, technical_analyst
2. Sequential: risk_assessor (uses ratio + technical outputs)
3. Sequential: report_writer (synthesizes all outputs)
"""

import logging
from datetime import datetime
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from backend.graph.state import FinSightState
from backend.agents.filing_analyst import run_filing_analyst
from backend.agents.ratio_cruncher import run_ratio_cruncher
from backend.agents.news_sentinel import run_news_sentinel
from backend.agents.technical_analyst import run_technical_analyst
from backend.agents.risk_assessor import run_risk_assessor
from backend.agents.report_writer import run_report_writer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_analysis(
    ticker: str,
    company_name: str = "",
    sector: str = ""
) -> Dict[str, Any]:
    """
    Run complete FinSight investment research analysis.

    Orchestrates six specialist agents through LangGraph workflow:
    - Four parallel specialists: filing, ratio, news, technical
    - Sequential risk assessor
    - Sequential report writer

    Args:
        ticker: Stock ticker symbol (e.g., "RELIANCE" or "RELIANCE.NS")
        company_name: Full company name (optional, will fetch if empty)
        sector: Industry sector (optional)

    Returns:
        Final state dict with all agent outputs and final report
    """
    # Normalize ticker
    ticker_normalized = _normalize_ticker(ticker)

    logger.info(f"Starting FinSight analysis for {ticker_normalized} ({company_name})")

    # Initialize state
    initial_state: FinSightState = {
        "ticker": ticker_normalized,
        "company_name": company_name or ticker_normalized,
        "sector": sector,
        "filing_output": None,
        "ratio_output": None,
        "news_output": None,
        "technical_output": None,
        "risk_output": None,
        "final_report": None,
        "errors": [],
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": ""
    }

    # Build workflow graph
    workflow = _build_workflow()

    # Compile and run
    app = workflow.compile()

    try:
        # Execute workflow
        final_state = app.invoke(initial_state)

        # Set completion time
        final_state["completed_at"] = datetime.utcnow().isoformat()

        logger.info("FinSight analysis completed")
        logger.info(f"Errors: {len(final_state.get('errors', []))}")
        logger.info(f"Final report available: {final_state.get('final_report') is not None}")

        return final_state

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        initial_state["errors"].append(f"Workflow execution failed: {str(e)}")
        initial_state["completed_at"] = datetime.utcnow().isoformat()
        return initial_state


def _normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbol.

    Adds .NS suffix for Indian stocks if not present.
    """
    ticker = ticker.strip().upper()

    # If already has exchange suffix, return as-is
    if "." in ticker:
        return ticker

    # For Indian market, add .NS suffix for NSE
    # Common Indian tickers: RELIANCE, TCS, INFY, HDFCBANK, etc.
    return f"{ticker}.NS"


def _build_workflow() -> StateGraph:
    """
    Build LangGraph StateGraph for FinSight analysis.

    Pipeline structure:
    START
      ├─> filing_analyst ─┐
      ├─> ratio_cruncher ─┤
      ├─> news_sentinel ──┼─> risk_assessor ─> report_writer ─> END
      └─> technical_analyst ┘

    Note: We use sequential execution for specialist agents rather than
    LangGraph's Send API because:
    1. The current LangGraph version's Send API requires additional
       boilerplate for error handling in parallel branches
    2. Sequential execution with proper error handling is cleaner
    3. Performance difference is minimal for 4 agents
    4. State management is simpler without complex merging logic

    For future optimization, consider migrating to Send API when:
    - LangGraph improves parallel error handling
    - Agent execution time significantly increases
    - More parallel agents are added
    """
    workflow = StateGraph(FinSightState)

    # Define nodes for each agent
    workflow.add_node("filing_analyst", _filing_analyst_node)
    workflow.add_node("ratio_cruncher", _ratio_cruncher_node)
    workflow.add_node("news_sentinel", _news_sentinel_node)
    workflow.add_node("technical_analyst", _technical_analyst_node)
    workflow.add_node("risk_assessor", _risk_assessor_node)
    workflow.add_node("report_writer", _report_writer_node)

    # Define edges
    # Start with filing analyst
    workflow.set_entry_point("filing_analyst")

    # Sequential execution of specialists
    # (Future optimization: use Send API for parallel execution)
    workflow.add_edge("filing_analyst", "ratio_cruncher")
    workflow.add_edge("ratio_cruncher", "news_sentinel")
    workflow.add_edge("news_sentinel", "technical_analyst")

    # After all specialists, run risk assessor
    workflow.add_edge("technical_analyst", "risk_assessor")

    # Finally, run report writer
    workflow.add_edge("risk_assessor", "report_writer")

    # End workflow
    workflow.add_edge("report_writer", END)

    return workflow


# === Agent Node Functions ===


def _filing_analyst_node(state: FinSightState) -> FinSightState:
    """Run filing analyst agent."""
    logger.info(f"Running filing_analyst for {state['ticker']}")

    try:
        result = run_filing_analyst(state["ticker"], state["company_name"])
        state["filing_output"] = result

        if result.get("status") == "error":
            error_msg = f"filing_analyst error: {result.get('error', 'unknown')}"
            state["errors"].append(error_msg)
            logger.warning(error_msg)
        else:
            logger.info("filing_analyst completed successfully")

    except Exception as e:
        error_msg = f"filing_analyst exception: {str(e)}"
        state["errors"].append(error_msg)
        logger.error(error_msg, exc_info=True)
        state["filing_output"] = {
            "status": "error",
            "error": str(e),
            "ticker": state["ticker"],
            "company_name": state["company_name"]
        }

    return state


def _ratio_cruncher_node(state: FinSightState) -> FinSightState:
    """Run ratio cruncher agent."""
    logger.info(f"Running ratio_cruncher for {state['ticker']}")

    try:
        result = run_ratio_cruncher(state["ticker"], state["company_name"])
        state["ratio_output"] = result

        if result.get("status") == "error":
            error_msg = f"ratio_cruncher error: {result.get('error', 'unknown')}"
            state["errors"].append(error_msg)
            logger.warning(error_msg)
        else:
            logger.info("ratio_cruncher completed successfully")

    except Exception as e:
        error_msg = f"ratio_cruncher exception: {str(e)}"
        state["errors"].append(error_msg)
        logger.error(error_msg, exc_info=True)
        state["ratio_output"] = {
            "status": "error",
            "error": str(e),
            "ticker": state["ticker"],
            "company_name": state["company_name"]
        }

    return state


def _news_sentinel_node(state: FinSightState) -> FinSightState:
    """Run news sentinel agent."""
    logger.info(f"Running news_sentinel for {state['ticker']}")

    try:
        result = run_news_sentinel(state["ticker"], state["company_name"])
        state["news_output"] = result

        if result.get("status") == "error":
            error_msg = f"news_sentinel error: {result.get('error', 'unknown')}"
            state["errors"].append(error_msg)
            logger.warning(error_msg)
        else:
            logger.info("news_sentinel completed successfully")

    except Exception as e:
        error_msg = f"news_sentinel exception: {str(e)}"
        state["errors"].append(error_msg)
        logger.error(error_msg, exc_info=True)
        state["news_output"] = {
            "status": "error",
            "error": str(e),
            "ticker": state["ticker"],
            "company_name": state["company_name"]
        }

    return state


def _technical_analyst_node(state: FinSightState) -> FinSightState:
    """Run technical analyst agent."""
    logger.info(f"Running technical_analyst for {state['ticker']}")

    try:
        result = run_technical_analyst(state["ticker"], state["company_name"])
        state["technical_output"] = result

        if result.get("status") == "error":
            error_msg = f"technical_analyst error: {result.get('error', 'unknown')}"
            state["errors"].append(error_msg)
            logger.warning(error_msg)
        else:
            logger.info("technical_analyst completed successfully")

    except Exception as e:
        error_msg = f"technical_analyst exception: {str(e)}"
        state["errors"].append(error_msg)
        logger.error(error_msg, exc_info=True)
        state["technical_output"] = {
            "status": "error",
            "error": str(e),
            "ticker": state["ticker"],
            "company_name": state["company_name"]
        }

    return state


def _risk_assessor_node(state: FinSightState) -> FinSightState:
    """Run risk assessor agent (uses ratio and technical outputs)."""
    logger.info(f"Running risk_assessor for {state['ticker']}")

    try:
        result = run_risk_assessor(
            ticker=state["ticker"],
            company_name=state["company_name"],
            ratio_output=state.get("ratio_output"),
            technical_output=state.get("technical_output")
        )
        state["risk_output"] = result

        if result.get("status") == "error":
            error_msg = f"risk_assessor error: {result.get('error', 'unknown')}"
            state["errors"].append(error_msg)
            logger.warning(error_msg)
        else:
            logger.info("risk_assessor completed successfully")

    except Exception as e:
        error_msg = f"risk_assessor exception: {str(e)}"
        state["errors"].append(error_msg)
        logger.error(error_msg, exc_info=True)
        state["risk_output"] = {
            "status": "error",
            "error": str(e),
            "ticker": state["ticker"],
            "company_name": state["company_name"]
        }

    return state


def _report_writer_node(state: FinSightState) -> FinSightState:
    """Run report writer agent (synthesizes all outputs)."""
    logger.info(f"Running report_writer for {state['ticker']}")

    try:
        result = run_report_writer(
            ticker=state["ticker"],
            company_name=state["company_name"],
            filing_output=state.get("filing_output"),
            ratio_output=state.get("ratio_output"),
            news_output=state.get("news_output"),
            technical_output=state.get("technical_output"),
            risk_output=state.get("risk_output")
        )
        state["final_report"] = result

        if result.get("status") == "error":
            error_msg = f"report_writer error: {result.get('error', 'unknown')}"
            state["errors"].append(error_msg)
            logger.warning(error_msg)
        else:
            logger.info("report_writer completed successfully")

    except Exception as e:
        error_msg = f"report_writer exception: {str(e)}"
        state["errors"].append(error_msg)
        logger.error(error_msg, exc_info=True)
        state["final_report"] = {
            "status": "error",
            "error": str(e),
            "ticker": state["ticker"],
            "company_name": state["company_name"]
        }

    return state

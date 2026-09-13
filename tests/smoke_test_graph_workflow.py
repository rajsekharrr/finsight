"""
Smoke test for full LangGraph workflow with real pipeline execution.

This requires:
1. Internet connection for yfinance, news APIs
2. OPENAI_API_KEY environment variable
3. Valid Indian stock ticker

Run with: python -m pytest tests/smoke_test_graph_workflow.py -v -s
Or directly: python tests/smoke_test_graph_workflow.py
"""

import sys
from pathlib import Path
import json
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.graph.workflow import run_analysis


def test_smoke_full_pipeline_reliance():
    """Smoke test: Run full pipeline for RELIANCE with real agents."""
    print("\n=== Testing Full LangGraph Pipeline: RELIANCE ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    try:
        result = run_analysis("RELIANCE", "Reliance Industries", "Energy")

        print(f"\n=== Pipeline Execution Result ===")
        print(f"Ticker: {result['ticker']}")
        print(f"Company: {result['company_name']}")
        print(f"Sector: {result['sector']}")
        print(f"Started: {result['started_at']}")
        print(f"Completed: {result['completed_at']}")

        print(f"\n=== Agent Outputs ===")
        agents = [
            ("Filing Analyst", result.get("filing_output")),
            ("Ratio Cruncher", result.get("ratio_output")),
            ("News Sentinel", result.get("news_output")),
            ("Technical Analyst", result.get("technical_output")),
            ("Risk Assessor", result.get("risk_output")),
            ("Report Writer", result.get("final_report")),
        ]

        for agent_name, output in agents:
            if output:
                status = output.get("status", "unknown")
                print(f"  {agent_name}: {status}")
                if status == "error":
                    print(f"    Error: {output.get('error', 'unknown')}")
            else:
                print(f"  {agent_name}: None")

        if result.get("errors"):
            print(f"\n=== Errors ({len(result['errors'])}) ===")
            for error in result["errors"]:
                print(f"  ⚠ {error}")

        if result.get("final_report"):
            print(f"\n=== Final Report Summary ===")
            report = result["final_report"]
            print(f"  Status: {report.get('status')}")
            print(f"  Confidence: {report.get('confidence_score')}/100")
            print(f"  Data Quality: {report.get('data_quality')}")
            print(f"  Sources Used: {', '.join(report.get('sources_used', []))}")
            if report.get('missing_sections'):
                print(f"  Missing: {', '.join(report['missing_sections'])}")

            if report.get('executive_summary'):
                print(f"\n  Executive Summary:")
                print(f"  {report['executive_summary'][:200]}...")

        # Assertions
        assert result["ticker"] == "RELIANCE.NS"
        assert result["company_name"] == "Reliance Industries"
        assert result["started_at"] != ""
        assert result["completed_at"] != ""
        assert result["completed_at"] >= result["started_at"]

        print("\n✓ PASSED - Full pipeline executed successfully")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_smoke_full_pipeline_infy():
    """Smoke test: Run full pipeline for INFY."""
    print("\n=== Testing Full LangGraph Pipeline: INFY ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    try:
        result = run_analysis("INFY", "Infosys", "IT Services")

        print(f"Ticker: {result['ticker']}")
        print(f"Company: {result['company_name']}")
        print(f"Errors: {len(result.get('errors', []))}")

        has_report = result.get("final_report") is not None
        print(f"Report Generated: {has_report}")

        if has_report:
            report = result["final_report"]
            print(f"Report Status: {report.get('status')}")
            print(f"Confidence: {report.get('confidence_score')}/100")

        assert result["ticker"] == "INFY.NS"
        assert result["completed_at"] != ""

        print("✓ PASSED")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_json_serializable():
    """Smoke test: Verify pipeline result is JSON serializable."""
    print("\n=== Testing JSON serialization ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    try:
        result = run_analysis("TCS", "Tata Consultancy Services", "IT Services")

        # Should be JSON serializable
        json_str = json.dumps(result, indent=2)
        assert json_str is not None

        parsed = json.loads(json_str)
        assert "ticker" in parsed
        assert "final_report" in parsed
        assert "errors" in parsed

        print(f"JSON size: {len(json_str)} bytes")
        print("✓ PASSED - Result is JSON serializable")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_error_recovery():
    """Smoke test: Verify pipeline continues with invalid ticker."""
    print("\n=== Testing error recovery ===")

    try:
        result = run_analysis("INVALIDXYZ", "Invalid Company")

        print(f"Ticker: {result['ticker']}")
        print(f"Errors: {len(result.get('errors', []))}")
        print(f"Completed: {result['completed_at'] != ''}")

        # Should complete even with errors
        assert result["completed_at"] != ""
        assert len(result["errors"]) > 0

        print("✓ PASSED - Pipeline handled invalid ticker gracefully")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_state_structure():
    """Smoke test: Verify final state structure."""
    print("\n=== Testing state structure ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    try:
        result = run_analysis("HDFCBANK", "HDFC Bank", "Banking")

        # Verify required fields
        required_fields = [
            "ticker", "company_name", "sector",
            "filing_output", "ratio_output", "news_output",
            "technical_output", "risk_output", "final_report",
            "errors", "started_at", "completed_at"
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        print(f"All {len(required_fields)} required fields present")
        print("✓ PASSED - State structure is correct")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_agent_data_flow():
    """Smoke test: Verify data flows between agents."""
    print("\n=== Testing agent data flow ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    try:
        result = run_analysis("RELIANCE", "Reliance Industries")

        # Check if report writer received inputs
        if result.get("final_report"):
            report = result["final_report"]
            sources = report.get("sources_used", [])
            print(f"Sources used by report writer: {sources}")

            # Should have at least some sources
            assert len(sources) > 0, "Report writer received no agent outputs"

            print(f"✓ Data flowed to report writer: {len(sources)} sources")

        print("✓ PASSED - Agent data flow working")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


if __name__ == "__main__":
    """Run smoke tests directly."""
    print("=" * 60)
    print("LANGGRAPH WORKFLOW SMOKE TESTS")
    print("=" * 60)

    try:
        test_smoke_full_pipeline_reliance()
        test_smoke_full_pipeline_infy()
        test_smoke_json_serializable()
        test_smoke_error_recovery()
        test_smoke_state_structure()
        test_smoke_agent_data_flow()

        print("\n" + "=" * 60)
        print("ALL SMOKE TESTS PASSED ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

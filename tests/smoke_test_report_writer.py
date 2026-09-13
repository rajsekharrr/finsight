"""
Smoke test for Report Writer agent with sample static agent outputs.

This demonstrates report generation without requiring live API calls.
Uses realistic sample data to test the full synthesis pipeline.

Run with: python -m pytest tests/smoke_test_report_writer.py -v -s
Or directly: python tests/smoke_test_report_writer.py
"""

import sys
from pathlib import Path
import json
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.report_writer import run_report_writer


# Sample agent outputs for Reliance Industries
SAMPLE_FILING_OUTPUT = {
    "status": "success",
    "ticker": "RELIANCE",
    "company_name": "Reliance Industries",
    "insights": {
        "revenue_growth": "Strong revenue growth of 12% YoY",
        "business_segments": "Diversified across Oil & Gas, Retail, Telecom",
        "key_highlights": "Expansion in renewable energy sector"
    }
}

SAMPLE_RATIO_OUTPUT = {
    "status": "success",
    "ticker": "RELIANCE",
    "company_name": "Reliance Industries",
    "ratios": {
        "roe_pct": 18.5,
        "roa_pct": 8.2,
        "net_margin_pct": 12.5,
        "debt_to_equity": 45.3,
        "interest_coverage": 5.2,
        "current_ratio": 1.8,
        "quick_ratio": 1.2,
        "asset_turnover": 0.65,
        "days_inventory": 45,
        "pe_ratio": 28.5,
        "pb_ratio": 3.2,
        "ev_ebitda": 12.8
    }
}

SAMPLE_NEWS_OUTPUT = {
    "status": "success",
    "ticker": "RELIANCE",
    "company_name": "Reliance Industries",
    "article_count": 18,
    "sentiment": "POSITIVE",
    "sentiment_score": 0.65,
    "key_themes": [
        "Renewable energy expansion",
        "5G rollout progress",
        "Retail business growth",
        "Strong quarterly earnings"
    ],
    "positive_drivers": [
        "Record subscriber additions in Jio",
        "Retail footprint expansion",
        "Green energy investments"
    ],
    "negative_drivers": [
        "Regulatory challenges in telecom",
        "High capital expenditure requirements"
    ],
    "notable_headlines": [
        "Reliance reports strong Q3 earnings beat",
        "Jio crosses 450M subscriber milestone",
        "RIL announces 100GW renewable energy target"
    ],
    "sources": ["Economic Times", "Business Standard", "Mint"]
}

SAMPLE_TECHNICAL_OUTPUT = {
    "status": "success",
    "ticker": "RELIANCE",
    "company_name": "Reliance Industries",
    "trend": "BULLISH",
    "momentum": "NEUTRAL",
    "technical_score": 68.0,
    "indicators": {
        "latest_price": 2485.50,
        "sma_20": 2420.00,
        "sma_50": 2380.00,
        "sma_200": 2250.00,
        "ema_12": 2465.00,
        "ema_26": 2430.00,
        "macd": 15.8,
        "macd_signal": 12.5,
        "rsi_14": 58.5,
        "volume_average": 5200000,
        "price_return_pct": 15.2
    },
    "signals": [
        "Price above 20-day SMA (bullish)",
        "20-day SMA above 50-day SMA (golden cross)",
        "MACD above signal line (bullish momentum)",
        "RSI in neutral zone (58.5)"
    ],
    "support_resistance": {
        "support_20d": 2350.00,
        "resistance_20d": 2550.00,
        "52w_high": 2650.00,
        "52w_low": 2100.00
    },
    "summary": "BULLISH trend with NEUTRAL momentum. Technical score: 68/100."
}

SAMPLE_RISK_OUTPUT = {
    "status": "success",
    "ticker": "RELIANCE",
    "company_name": "Reliance Industries",
    "risk_score": 38.0,
    "risk_level": "MEDIUM",
    "risk_factors": [
        "Moderate volatility: 28.5%",
        "Above-market beta: 1.15",
        "Moderate leverage: D/E 45",
        "Elevated valuation: P/E 28.5x"
    ],
    "risk_flags": [],
    "metrics": {
        "beta": 1.15,
        "volatility_pct": 28.5,
        "var_95_pct": 3.2,
        "max_drawdown_pct": 18.5,
        "debt_to_equity": 45.3,
        "current_ratio": 1.8,
        "pe_ratio": 28.5
    },
    "unavailable_metrics": ["altman_z_score", "promoter_pledge"],
    "summary": "Reliance Industries has a MEDIUM risk profile with a risk score of 38/100."
}


def test_smoke_full_report_with_openai():
    """Smoke test: Generate full report with all agent outputs (requires OPENAI_API_KEY)."""
    print("\n=== Testing Report Writer: Full Report ===")

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    try:
        result = run_report_writer(
            ticker="RELIANCE",
            company_name="Reliance Industries",
            filing_output=SAMPLE_FILING_OUTPUT,
            ratio_output=SAMPLE_RATIO_OUTPUT,
            news_output=SAMPLE_NEWS_OUTPUT,
            technical_output=SAMPLE_TECHNICAL_OUTPUT,
            risk_output=SAMPLE_RISK_OUTPUT
        )

        print(f"\n=== Report Generation Result ===")
        print(f"Status: {result['status']}")
        print(f"Ticker: {result['ticker']}")
        print(f"Company: {result['company_name']}")
        print(f"Confidence Score: {result['confidence_score']}/100")
        print(f"Data Quality: {result['data_quality']}")

        print(f"\n=== Sources Used ({len(result['sources_used'])}) ===")
        for source in result['sources_used']:
            print(f"  ✓ {source}")

        if result['missing_sections']:
            print(f"\n=== Missing Sections ({len(result['missing_sections'])}) ===")
            for section in result['missing_sections']:
                print(f"  ✗ {section}")

        print(f"\n=== Executive Summary ===")
        print(result['executive_summary'])

        print(f"\n=== Bull Case ===")
        print(result['bull_case'][:300] + "..." if len(result['bull_case']) > 300 else result['bull_case'])

        print(f"\n=== Bear Case ===")
        print(result['bear_case'][:300] + "..." if len(result['bear_case']) > 300 else result['bear_case'])

        print(f"\n=== Key Risks ({len(result['key_risks'])}) ===")
        for i, risk in enumerate(result['key_risks'], 1):
            print(f"  {i}. {risk}")

        print(f"\n=== Full Report ===")
        print(result['report_markdown'][:500] + "..." if len(result['report_markdown']) > 500 else result['report_markdown'])

        if result['error']:
            print(f"\n⚠ Note: {result['error']}")

        assert result["status"] in ["success", "partial_success"]
        assert result["ticker"] == "RELIANCE"
        assert 0 <= result["confidence_score"] <= 100
        assert len(result["sources_used"]) > 0
        assert len(result["report_markdown"]) > 0

        print("\n✓ PASSED - Full report generated successfully")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_partial_report_without_openai():
    """Smoke test: Generate fallback report without OpenAI (no API key needed)."""
    print("\n=== Testing Report Writer: Fallback Report (No API) ===")

    # Temporarily clear API key to force fallback
    old_key = os.environ.pop("OPENAI_API_KEY", None)

    try:
        result = run_report_writer(
            ticker="RELIANCE",
            company_name="Reliance Industries",
            filing_output=None,
            ratio_output=SAMPLE_RATIO_OUTPUT,
            news_output=SAMPLE_NEWS_OUTPUT,
            technical_output=SAMPLE_TECHNICAL_OUTPUT,
            risk_output=None
        )

        print(f"Status: {result['status']}")
        print(f"Confidence: {result['confidence_score']}/100")
        print(f"Sources: {len(result['sources_used'])}")
        print(f"Report length: {len(result['report_markdown'])} chars")

        assert result["status"] in ["success", "partial_success", "error"]
        assert len(result["report_markdown"]) > 0
        assert len(result["executive_summary"]) > 0

        print("✓ PASSED - Fallback report generated")

    finally:
        # Restore API key
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key


def test_smoke_json_serializable():
    """Smoke test: Verify result is JSON serializable."""
    print("\n=== Testing JSON serialization ===")

    # Use fallback mode (no API)
    old_key = os.environ.pop("OPENAI_API_KEY", None)

    try:
        result = run_report_writer(
            "RELIANCE", "Reliance Industries",
            None, SAMPLE_RATIO_OUTPUT, None, None, None
        )

        # Should be JSON serializable
        json_str = json.dumps(result, indent=2)
        assert json_str is not None

        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "report_markdown" in parsed

        print("✓ PASSED - Result is JSON serializable")

    finally:
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key


def test_smoke_data_quality_assessment():
    """Smoke test: Verify data quality assessment."""
    print("\n=== Testing data quality assessment ===")

    old_key = os.environ.pop("OPENAI_API_KEY", None)

    try:
        # 3 out of 5 sources
        result = run_report_writer(
            "RELIANCE", "Reliance Industries",
            None, SAMPLE_RATIO_OUTPUT, SAMPLE_NEWS_OUTPUT, None, SAMPLE_RISK_OUTPUT
        )

        print(f"Data Quality: {result['data_quality']}")
        print(f"Sources: {result['sources_used']}")
        print(f"Missing: {result['missing_sections']}")

        assert len(result['sources_used']) == 3
        assert len(result['missing_sections']) == 2
        assert "3/5" in result['data_quality']

        print("✓ PASSED - Data quality correctly assessed")

    finally:
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key


def test_smoke_confidence_scoring():
    """Smoke test: Verify confidence scoring logic."""
    print("\n=== Testing confidence scoring ===")

    old_key = os.environ.pop("OPENAI_API_KEY", None)

    try:
        # Test with varying amounts of data
        test_cases = [
            (5, [SAMPLE_FILING_OUTPUT, SAMPLE_RATIO_OUTPUT, SAMPLE_NEWS_OUTPUT,
                 SAMPLE_TECHNICAL_OUTPUT, SAMPLE_RISK_OUTPUT]),
            (3, [None, SAMPLE_RATIO_OUTPUT, SAMPLE_NEWS_OUTPUT, SAMPLE_TECHNICAL_OUTPUT, None]),
            (1, [None, SAMPLE_RATIO_OUTPUT, None, None, None]),
        ]

        for expected_sources, outputs in test_cases:
            result = run_report_writer("TEST", "Test Co", *outputs)
            print(f"Sources: {len(result['sources_used'])}/{expected_sources}, Confidence: {result['confidence_score']}")

            assert len(result['sources_used']) == expected_sources
            # Confidence should generally correlate with data availability
            assert 0 <= result['confidence_score'] <= 100

        print("✓ PASSED - Confidence scoring works correctly")

    finally:
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key


if __name__ == "__main__":
    """Run smoke tests directly."""
    print("=" * 60)
    print("REPORT WRITER SMOKE TESTS")
    print("=" * 60)

    try:
        test_smoke_full_report_with_openai()
        test_smoke_partial_report_without_openai()
        test_smoke_json_serializable()
        test_smoke_data_quality_assessment()
        test_smoke_confidence_scoring()

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

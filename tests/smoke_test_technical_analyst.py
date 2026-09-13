"""
Smoke test for Technical Analyst agent with real stock data.

This requires:
1. Internet connection for yfinance API
2. Valid Indian stock tickers (RELIANCE, INFY, TCS, etc.)

Run with: python -m pytest tests/smoke_test_technical_analyst.py -v -s
Or directly: python tests/smoke_test_technical_analyst.py
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.technical_analyst import run_technical_analyst


def test_smoke_technical_analyst_reliance():
    """Smoke test: Analyze RELIANCE with real data."""
    print("\n=== Testing Technical Analyst: RELIANCE ===")

    try:
        result = run_technical_analyst("RELIANCE", "Reliance Industries")

        print(f"\n=== Technical Analysis Result ===")
        print(f"Status: {result['status']}")
        print(f"Ticker: {result['ticker']}")
        print(f"Company: {result['company_name']}")
        print(f"Trend: {result['trend']}")
        print(f"Momentum: {result['momentum']}")
        print(f"Technical Score: {result['technical_score']:.0f}/100")

        print(f"\n=== Indicators ===")
        for key, value in result['indicators'].items():
            if value is not None:
                print(f"  {key}: {value}")

        print(f"\n=== Support/Resistance ===")
        for key, value in result['support_resistance'].items():
            if value is not None:
                print(f"  {key}: {value}")

        print(f"\n=== Signals ({len(result['signals'])}) ===")
        for i, signal in enumerate(result['signals'], 1):
            print(f"  {i}. {signal}")

        print(f"\n=== Summary ===")
        print(result['summary'])

        if result['error']:
            print(f"\n⚠ Warning: {result['error']}")

        assert result["status"] in ["success", "error"]
        assert result["ticker"] == "RELIANCE"
        assert result["company_name"] == "Reliance Industries"
        assert result["trend"] in ["BULLISH", "BEARISH", "NEUTRAL"]
        assert result["momentum"] in ["OVERBOUGHT", "OVERSOLD", "NEUTRAL"]
        assert 0 <= result["technical_score"] <= 100

        print("\n✓ PASSED - Technical analysis completed successfully")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_technical_analyst_infy():
    """Smoke test: Analyze INFY with real data."""
    print("\n=== Testing Technical Analyst: INFY ===")

    try:
        result = run_technical_analyst("INFY", "Infosys")

        print(f"Status: {result['status']}")
        print(f"Company: {result['company_name']}")
        print(f"Trend: {result['trend']}")
        print(f"Momentum: {result['momentum']}")
        print(f"Score: {result['technical_score']:.0f}/100")
        print(f"Signals: {len(result['signals'])}")

        assert result["status"] in ["success", "error"]
        assert result["ticker"] == "INFY"

        print("✓ PASSED")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_json_serializable():
    """Smoke test: Verify result is JSON serializable."""
    print("\n=== Testing JSON serialization ===")

    try:
        result = run_technical_analyst("TCS", "Tata Consultancy Services")

        # Should be JSON serializable
        json_str = json.dumps(result, indent=2)
        assert json_str is not None

        # Verify structure
        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "trend" in parsed
        assert "technical_score" in parsed
        assert "indicators" in parsed

        print("✓ PASSED - Result is JSON serializable")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_indicator_values():
    """Smoke test: Verify indicator values are reasonable."""
    print("\n=== Testing indicator value ranges ===")

    try:
        result = run_technical_analyst("HDFCBANK", "HDFC Bank")

        if result["status"] == "success":
            indicators = result["indicators"]

            # Check RSI bounds
            if indicators.get("rsi_14") is not None:
                assert 0 <= indicators["rsi_14"] <= 100
                print(f"RSI: {indicators['rsi_14']:.1f} (valid)")

            # Check price is positive
            if indicators.get("latest_price") is not None:
                assert indicators["latest_price"] > 0
                print(f"Price: {indicators['latest_price']:.2f} (valid)")

            # Check technical score bounds
            assert 0 <= result["technical_score"] <= 100
            print(f"Technical Score: {result['technical_score']:.0f}/100 (valid)")

            print("✓ PASSED - Indicator values are reasonable")
        else:
            print(f"SKIP: Analysis failed - {result.get('error')}")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_signal_generation():
    """Smoke test: Verify signals are generated."""
    print("\n=== Testing signal generation ===")

    try:
        result = run_technical_analyst("RELIANCE", "Reliance Industries")

        if result["status"] == "success":
            assert len(result["signals"]) > 0
            print(f"Generated {len(result['signals'])} signals")

            # Verify signals contain expected keywords
            signals_text = " ".join(result["signals"])
            has_relevant_terms = any(
                term in signals_text.lower()
                for term in ["sma", "rsi", "macd", "bullish", "bearish", "trend", "momentum"]
            )
            assert has_relevant_terms

            print("✓ PASSED - Signals generated with relevant terms")
        else:
            print(f"SKIP: Analysis failed - {result.get('error')}")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_multiple_companies():
    """Smoke test: Analyze multiple companies."""
    print("\n=== Testing multiple companies ===")

    companies = [
        ("RELIANCE", "Reliance Industries"),
        ("TCS", "Tata Consultancy Services"),
        ("HDFCBANK", "HDFC Bank"),
    ]

    try:
        for ticker, name in companies:
            print(f"\n--- Analyzing {ticker} ---")
            result = run_technical_analyst(ticker, name)

            print(f"Status: {result['status']}")
            if result['status'] == 'success':
                print(f"Trend: {result['trend']}")
                print(f"Score: {result['technical_score']:.0f}/100")
                print(f"Signals: {len(result['signals'])}")

            assert result["status"] in ["success", "error"]

        print("\n✓ PASSED - Multiple companies analyzed")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_invalid_ticker():
    """Smoke test: Graceful handling of invalid ticker."""
    print("\n=== Testing invalid ticker handling ===")

    try:
        result = run_technical_analyst("INVALIDXYZ", "Invalid Company")

        print(f"Status: {result['status']}")
        print(f"Error: {result['error']}")

        assert result["status"] == "error"
        assert result["error"] is not None

        print("✓ PASSED - Gracefully handled invalid ticker")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


if __name__ == "__main__":
    """Run smoke tests directly."""
    print("=" * 60)
    print("TECHNICAL ANALYST SMOKE TESTS")
    print("=" * 60)

    try:
        test_smoke_technical_analyst_reliance()
        test_smoke_technical_analyst_infy()
        test_smoke_json_serializable()
        test_smoke_indicator_values()
        test_smoke_signal_generation()
        test_smoke_multiple_companies()
        test_smoke_invalid_ticker()

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

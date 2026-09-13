"""
Smoke test for Risk Assessor agent with real stock data.

This requires:
1. Internet connection for yfinance API
2. Valid Indian stock tickers (RELIANCE, INFY, TCS, etc.)

Run with: python -m pytest tests/smoke_test_risk_assessor.py -v -s
Or directly: python tests/smoke_test_risk_assessor.py
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.risk_assessor import run_risk_assessor


def test_smoke_risk_assessor_reliance():
    """Smoke test: Assess RELIANCE risk with real data."""
    print("\n=== Testing Risk Assessor: RELIANCE ===")

    try:
        result = run_risk_assessor("RELIANCE", "Reliance Industries")

        print(f"\n=== Risk Assessment Result ===")
        print(f"Status: {result['status']}")
        print(f"Ticker: {result['ticker']}")
        print(f"Company: {result['company_name']}")
        print(f"Risk Score: {result['risk_score']:.0f}/100")
        print(f"Risk Level: {result['risk_level']}")

        print(f"\n=== Risk Metrics ===")
        for key, value in result['metrics'].items():
            print(f"  {key}: {value}")

        if result['unavailable_metrics']:
            print(f"\n=== Unavailable Metrics ({len(result['unavailable_metrics'])}) ===")
            for metric in result['unavailable_metrics']:
                print(f"  - {metric}")

        print(f"\n=== Risk Factors ({len(result['risk_factors'])}) ===")
        for i, factor in enumerate(result['risk_factors'], 1):
            print(f"  {i}. {factor}")

        if result['risk_flags']:
            print(f"\n=== Risk Flags ({len(result['risk_flags'])}) ===")
            for flag in result['risk_flags']:
                print(f"  ⚠ {flag}")

        print(f"\n=== Summary ===")
        print(result['summary'])

        if result['error']:
            print(f"\n⚠ Note: {result['error']}")

        assert result["status"] in ["success", "partial_success", "error"]
        assert result["ticker"] == "RELIANCE"
        assert result["company_name"] == "Reliance Industries"
        assert 0 <= result["risk_score"] <= 100
        assert result["risk_level"] in ["LOW", "MEDIUM", "MEDIUM-HIGH", "HIGH", "VERY HIGH", "UNKNOWN"]

        print("\n✓ PASSED - Risk assessment completed successfully")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_risk_assessor_infy():
    """Smoke test: Assess INFY risk with real data."""
    print("\n=== Testing Risk Assessor: INFY ===")

    try:
        result = run_risk_assessor("INFY", "Infosys")

        print(f"Status: {result['status']}")
        print(f"Company: {result['company_name']}")
        print(f"Risk Score: {result['risk_score']:.0f}/100")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Risk Factors: {len(result['risk_factors'])}")
        print(f"Metrics: {len(result['metrics'])}")

        assert result["status"] in ["success", "partial_success", "error"]
        assert result["ticker"] == "INFY"

        print("✓ PASSED")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_json_serializable():
    """Smoke test: Verify result is JSON serializable."""
    print("\n=== Testing JSON serialization ===")

    try:
        result = run_risk_assessor("TCS", "Tata Consultancy Services")

        # Should be JSON serializable
        json_str = json.dumps(result, indent=2)
        assert json_str is not None

        # Verify structure
        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "risk_score" in parsed
        assert "risk_level" in parsed
        assert "metrics" in parsed

        print("✓ PASSED - Result is JSON serializable")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_risk_metrics_bounds():
    """Smoke test: Verify risk metrics are within expected bounds."""
    print("\n=== Testing risk metrics bounds ===")

    try:
        result = run_risk_assessor("HDFCBANK", "HDFC Bank")

        if result["status"] in ["success", "partial_success"]:
            # Check risk score bounds
            assert 0 <= result["risk_score"] <= 100
            print(f"Risk Score: {result['risk_score']:.0f}/100 (valid)")

            # Check volatility bounds if available
            if "volatility_pct" in result["metrics"]:
                vol = result["metrics"]["volatility_pct"]
                assert 0 < vol < 200
                print(f"Volatility: {vol:.1f}% (valid)")

            # Check beta if available
            if "beta" in result["metrics"]:
                beta = result["metrics"]["beta"]
                assert -5 < beta < 5  # Reasonable beta range
                print(f"Beta: {beta:.2f} (valid)")

            print("✓ PASSED - Metrics within valid bounds")
        else:
            print(f"SKIP: Risk assessment failed - {result.get('error')}")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_risk_level_consistency():
    """Smoke test: Verify risk level matches risk score."""
    print("\n=== Testing risk level consistency ===")

    try:
        result = run_risk_assessor("RELIANCE", "Reliance Industries")

        if result["status"] in ["success", "partial_success"]:
            risk_score = result["risk_score"]
            risk_level = result["risk_level"]

            # Verify consistency
            if risk_score < 20:
                assert risk_level == "LOW"
            elif risk_score < 40:
                assert risk_level == "MEDIUM"
            elif risk_score < 60:
                assert risk_level == "MEDIUM-HIGH"
            elif risk_score < 80:
                assert risk_level == "HIGH"
            else:
                assert risk_level == "VERY HIGH"

            print(f"Risk Score {risk_score:.0f}/100 → {risk_level} (consistent)")
            print("✓ PASSED - Risk level consistent with score")
        else:
            print(f"SKIP: Risk assessment failed")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_multiple_companies():
    """Smoke test: Assess multiple companies."""
    print("\n=== Testing multiple companies ===")

    companies = [
        ("RELIANCE", "Reliance Industries"),
        ("TCS", "Tata Consultancy Services"),
        ("HDFCBANK", "HDFC Bank"),
    ]

    try:
        for ticker, name in companies:
            print(f"\n--- Assessing {ticker} ---")
            result = run_risk_assessor(ticker, name)

            print(f"Status: {result['status']}")
            if result['status'] in ['success', 'partial_success']:
                print(f"Risk Level: {result['risk_level']}")
                print(f"Risk Score: {result['risk_score']:.0f}/100")
                print(f"Risk Factors: {len(result['risk_factors'])}")

            assert result["status"] in ["success", "partial_success", "error"]

        print("\n✓ PASSED - Multiple companies assessed")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_invalid_ticker():
    """Smoke test: Graceful handling of invalid ticker."""
    print("\n=== Testing invalid ticker handling ===")

    try:
        result = run_risk_assessor("INVALIDXYZ", "Invalid Company")

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
    print("RISK ASSESSOR SMOKE TESTS")
    print("=" * 60)

    try:
        test_smoke_risk_assessor_reliance()
        test_smoke_risk_assessor_infy()
        test_smoke_json_serializable()
        test_smoke_risk_metrics_bounds()
        test_smoke_risk_level_consistency()
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

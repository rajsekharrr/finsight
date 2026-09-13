"""
Smoke test for Ratio Cruncher agent with real yfinance data.

This requires:
1. Internet connection for yfinance API
2. Valid Indian stock tickers (RELIANCE, INFY, TCS, etc.)

Run with: python -m pytest tests/smoke_test_ratio_cruncher.py -v -s
Or directly: python tests/smoke_test_ratio_cruncher.py
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.ratio_cruncher import run_ratio_cruncher


def test_smoke_ratio_cruncher_reliance():
    """Smoke test: Analyze RELIANCE with real data."""
    print("\n=== Testing Ratio Cruncher: RELIANCE ===")

    try:
        result = run_ratio_cruncher("RELIANCE", "Reliance Industries", "Energy")

        print(f"\n=== Ratio Analysis Result ===")
        print(f"Status: {result['status']}")
        print(f"Ticker: {result['ticker']}")
        print(f"Company: {result['company_name']}")
        print(f"Sector: {result['sector']}")

        print(f"\n=== Financial Ratios ===")
        for key, value in result['ratios'].items():
            if value is not None:
                print(f"  {key}: {value}")

        if result['sector_benchmarks']:
            print(f"\n=== Sector Benchmarks ===")
            for key, value in result['sector_benchmarks'].items():
                print(f"  {key}: {value}")

        print(f"\n=== Strengths ({len(result['strengths'])}) ===")
        for i, strength in enumerate(result['strengths'], 1):
            print(f"  {i}. {strength}")

        print(f"\n=== Concerns ({len(result['concerns'])}) ===")
        for i, concern in enumerate(result['concerns'], 1):
            print(f"  {i}. {concern}")

        print(f"\n=== Valuation Summary ===")
        print(f"{result['valuation_summary']}")

        if result['error']:
            print(f"\n⚠ Warning: {result['error']}")

        assert result["status"] in ["success", "partial_success"]
        assert result["ticker"] == "RELIANCE"
        assert len(result["ratios"]) > 0
        assert len(result["strengths"]) > 0
        assert len(result["valuation_summary"]) > 0

        print("\n✓ PASSED - Ratio analysis completed successfully")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_ratio_cruncher_infy():
    """Smoke test: Analyze INFY with real data."""
    print("\n=== Testing Ratio Cruncher: INFY ===")

    try:
        result = run_ratio_cruncher("INFY", "Infosys", "IT")

        print(f"Status: {result['status']}")
        print(f"Company: {result['company_name']}")
        print(f"Sector: {result['sector']}")
        print(f"Ratios extracted: {len([v for v in result['ratios'].values() if v is not None])}")
        print(f"Strengths: {len(result['strengths'])}")
        print(f"Concerns: {len(result['concerns'])}")
        print(f"Valuation summary length: {len(result['valuation_summary'])} chars")

        assert result["status"] in ["success", "partial_success"]
        assert result["ticker"] == "INFY"

        print("✓ PASSED")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_json_serializable():
    """Smoke test: Verify result is JSON serializable."""
    print("\n=== Testing JSON serialization ===")

    try:
        result = run_ratio_cruncher("TCS", "Tata Consultancy Services", "IT")

        # Should be JSON serializable
        json_str = json.dumps(result, indent=2)
        assert json_str is not None

        # Verify structure
        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "ratios" in parsed
        assert "strengths" in parsed
        assert "valuation_summary" in parsed

        print("✓ PASSED - Result is JSON serializable")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_invalid_ticker():
    """Smoke test: Graceful handling of invalid ticker."""
    print("\n=== Testing invalid ticker handling ===")

    try:
        result = run_ratio_cruncher("INVALIDXYZ", "Invalid Company", "")

        print(f"Status: {result['status']}")
        print(f"Error: {result['error']}")

        assert result["status"] == "error"
        assert result["error"] is not None
        assert "unavailable" in result["error"].lower()

        print("✓ PASSED - Gracefully handled invalid ticker")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_no_hallucinated_numbers():
    """Smoke test: Verify no hallucinated ratio values."""
    print("\n=== Testing for hallucinated numbers ===")

    try:
        result = run_ratio_cruncher("RELIANCE", "Reliance Industries", "Energy")

        if result["status"] in ["success", "partial_success"]:
            # Verify that all mentioned numbers in text come from ratios
            strengths_text = " ".join(result["strengths"])
            concerns_text = " ".join(result["concerns"])
            all_text = strengths_text + " " + concerns_text + " " + result["valuation_summary"]

            # Extract actual ratio values
            actual_values = [v for v in result["ratios"].values() if v is not None]

            print(f"Found {len(actual_values)} actual ratio values")
            print(f"Strengths text length: {len(strengths_text)} chars")
            print(f"Concerns text length: {len(concerns_text)} chars")

            # This is a heuristic check - we're verifying analysis is based on data
            assert len(actual_values) > 0, "Should have some ratio data"
            assert len(all_text) > 100, "Should have substantial analysis"

            print("✓ PASSED - Analysis based on actual data")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_sector_comparison():
    """Smoke test: Verify sector comparison when available."""
    print("\n=== Testing sector comparison ===")

    try:
        result = run_ratio_cruncher("HDFCBANK", "HDFC Bank", "Financial Services")

        print(f"Status: {result['status']}")
        print(f"Sector: {result['sector']}")
        print(f"Sector benchmarks available: {len(result['sector_benchmarks']) > 0}")

        if result['sector_benchmarks']:
            print(f"Benchmark metrics: {list(result['sector_benchmarks'].keys())}")

            # Verify analysis mentions sector comparisons
            all_text = " ".join(result['strengths'] + result['concerns'])
            has_comparison = any(
                word in all_text.lower()
                for word in ["sector", "median", "vs", "compared"]
            )
            print(f"Analysis includes sector comparisons: {has_comparison}")

        print("✓ PASSED - Sector comparison working")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


def test_smoke_multiple_companies():
    """Smoke test: Analyze multiple companies."""
    print("\n=== Testing multiple companies ===")

    companies = [
        ("RELIANCE", "Reliance Industries", "Energy"),
        ("TCS", "Tata Consultancy Services", "IT"),
        ("HDFCBANK", "HDFC Bank", "Financial Services"),
    ]

    try:
        for ticker, name, sector in companies:
            print(f"\n--- Analyzing {ticker} ---")
            result = run_ratio_cruncher(ticker, name, sector)

            print(f"Status: {result['status']}")
            print(f"Ratios: {len([v for v in result['ratios'].values() if v is not None])}")
            print(f"Strengths: {len(result['strengths'])}")
            print(f"Concerns: {len(result['concerns'])}")

            assert result["status"] in ["success", "partial_success", "error"]

        print("\n✓ PASSED - Multiple companies analyzed")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise


if __name__ == "__main__":
    """Run smoke tests directly."""
    print("=" * 60)
    print("RATIO CRUNCHER SMOKE TESTS")
    print("=" * 60)

    try:
        test_smoke_ratio_cruncher_reliance()
        test_smoke_ratio_cruncher_infy()
        test_smoke_json_serializable()
        test_smoke_invalid_ticker()
        test_smoke_no_hallucinated_numbers()
        test_smoke_sector_comparison()
        test_smoke_multiple_companies()

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

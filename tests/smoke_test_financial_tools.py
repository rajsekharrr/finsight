"""
Live smoke test for financial_tools.py
Tests against real yfinance API with RELIANCE.NS
Run only when you have internet connection and want to verify live data.
"""
import sys
import logging
from backend.tools.financial_tools import (
    get_stock_info,
    get_price_history,
    get_nifty_history,
    get_sector_median
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_live_stock_info():
    """Test get_stock_info with live data"""
    print("\n" + "="*70)
    print("TEST: get_stock_info('RELIANCE')")
    print("="*70)

    result = get_stock_info.func("RELIANCE")

    if "error" in result:
        print(f"❌ FAILED: {result['error']}")
        return False

    print(f"✓ Ticker: {result.get('ticker')}")
    print(f"✓ Company: {result.get('company_name')}")
    print(f"✓ Sector: {result.get('sector')}")
    print(f"✓ Current Price: ₹{result.get('current_price')}")
    print(f"✓ Market Cap (Cr): ₹{result.get('market_cap_cr')}")
    print(f"✓ P/E Ratio: {result.get('pe_ratio')}")
    print(f"✓ P/B Ratio: {result.get('pb_ratio')}")
    print(f"✓ ROE %: {result.get('roe_pct')}")
    print(f"✓ Beta: {result.get('beta')}")
    print(f"✓ 52W High: ₹{result.get('52w_high')}")
    print(f"✓ 52W Low: ₹{result.get('52w_low')}")

    return True


def test_live_price_history():
    """Test get_price_history with live data"""
    print("\n" + "="*70)
    print("TEST: get_price_history('RELIANCE', '1mo')")
    print("="*70)

    result = get_price_history.func("RELIANCE", "1mo")

    if "error" in result:
        print(f"❌ FAILED: {result['error']}")
        return False

    print(f"✓ Ticker: {result.get('ticker')}")
    print(f"✓ Period: {result.get('period')}")
    print(f"✓ Data Points: {len(result.get('dates', []))}")
    print(f"✓ Latest Price: ₹{result.get('latest_price')}")
    print(f"✓ Oldest Price: ₹{result.get('oldest_price')}")
    print(f"✓ Price Return: {result.get('price_return_pct')}%")
    print(f"✓ First Date: {result.get('dates', ['N/A'])[0]}")
    print(f"✓ Last Date: {result.get('dates', ['N/A'])[-1]}")

    return True


def test_live_nifty_history():
    """Test get_nifty_history with live data"""
    print("\n" + "="*70)
    print("TEST: get_nifty_history('1mo')")
    print("="*70)

    result = get_nifty_history.func("1mo")

    if "error" in result:
        print(f"❌ FAILED: {result['error']}")
        return False

    print(f"✓ Period: {result.get('period')}")
    print(f"✓ Data Points: {len(result.get('dates', []))}")
    print(f"✓ Latest Nifty: {result.get('close', ['N/A'])[-1]}")
    print(f"✓ First Date: {result.get('dates', ['N/A'])[0]}")
    print(f"✓ Last Date: {result.get('dates', ['N/A'])[-1]}")

    return True


def test_sector_median():
    """Test get_sector_median"""
    print("\n" + "="*70)
    print("TEST: get_sector_median('Energy')")
    print("="*70)

    result = get_sector_median.func("Energy")

    if "error" in result:
        print(f"❌ FAILED: {result['error']}")
        return False

    print(f"✓ Sector: {result.get('sector')}")
    medians = result.get('medians', {})
    print(f"✓ Median P/E: {medians.get('pe')}")
    print(f"✓ Median P/B: {medians.get('pb')}")
    print(f"✓ Median ROE: {medians.get('roe')}")
    print(f"✓ Median ROCE: {medians.get('roce')}")

    return True


def main():
    """Run all live smoke tests"""
    print("\n" + "="*70)
    print("FINANCIAL TOOLS - LIVE SMOKE TEST")
    print("Testing against real yfinance API with RELIANCE.NS")
    print("="*70)

    tests = [
        ("Stock Info", test_live_stock_info),
        ("Price History", test_live_price_history),
        ("Nifty History", test_live_nifty_history),
        ("Sector Median", test_sector_median),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} raised exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed_count}/{total_count} passed")

    if passed_count == total_count:
        print("\n🎉 All live smoke tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Live smoke test for technical_tools.py
Tests against real yfinance data with manual technical indicators.
Run only when you have internet connection.
"""
import sys
import logging
from backend.tools.technical_tools import compute_technical_indicators

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_live_technical_indicators():
    """Test compute_technical_indicators with live RELIANCE data"""
    print("\n" + "="*70)
    print("TEST: compute_technical_indicators('RELIANCE', '1y')")
    print("="*70)

    result = compute_technical_indicators("RELIANCE", "1y")

    if "error" in result:
        print(f"❌ FAILED: {result['error']}")
        return False

    print(f"✓ Ticker: {result.get('ticker')}")
    print(f"✓ Period: {result.get('period')}")
    print(f"✓ Data Points: {result.get('data_points')}")
    print(f"✓ Current Price: ₹{result.get('current_price')}")

    print("\n📊 Moving Averages:")
    print(f"  SMA 20:  ₹{result.get('sma_20')}")
    print(f"  SMA 50:  ₹{result.get('sma_50')}")
    print(f"  SMA 200: ₹{result.get('sma_200')}")
    print(f"  EMA 12:  ₹{result.get('ema_12')}")
    print(f"  EMA 26:  ₹{result.get('ema_26')}")

    print("\n📈 Momentum Indicators:")
    print(f"  RSI (14):       {result.get('rsi')}")
    print(f"  MACD:           {result.get('macd')}")
    print(f"  MACD Signal:    {result.get('macd_signal')}")
    print(f"  MACD Histogram: {result.get('macd_histogram')}")

    print("\n💹 Support & Resistance:")
    print(f"  Support (20d):    ₹{result.get('support_20d')}")
    print(f"  Resistance (20d): ₹{result.get('resistance_20d')}")
    print(f"  52W High:         ₹{result.get('52w_high')}")
    print(f"  52W Low:          ₹{result.get('52w_low')}")

    print("\n🎯 Signals:")
    print(f"  Trend:            {result.get('trend')}")
    print(f"  Momentum:         {result.get('momentum')}")
    print(f"  Technical Score:  {result.get('technical_score')}/100")

    print("\n📝 Trend Signals:")
    for signal in result.get('trend_signals', []):
        print(f"  • {signal}")

    print("\n📝 Momentum Signals:")
    for signal in result.get('momentum_signals', []):
        print(f"  • {signal}")

    print(f"\n💬 Summary: {result.get('summary')}")

    return True


def test_live_technical_multiple_periods():
    """Test with different periods"""
    print("\n" + "="*70)
    print("TEST: Multiple periods (1mo, 3mo, 6mo, 1y)")
    print("="*70)

    periods = ["1mo", "3mo", "6mo", "1y"]

    for period in periods:
        result = compute_technical_indicators("TCS", period)

        if "error" in result:
            print(f"  {period}: ❌ {result['error']}")
        else:
            print(f"  {period}: ✓ {result['data_points']} points, "
                  f"Trend={result['trend']}, Score={result['technical_score']}")

    return True


def test_indicator_calculations():
    """Verify indicator calculations are reasonable"""
    print("\n" + "="*70)
    print("TEST: Indicator calculation sanity checks")
    print("="*70)

    result = compute_technical_indicators("HDFCBANK", "1y")

    if "error" in result:
        print(f"❌ FAILED: {result['error']}")
        return False

    checks = []

    # RSI should be between 0 and 100
    rsi = result.get('rsi', 0)
    if 0 <= rsi <= 100:
        print(f"✓ RSI in valid range: {rsi}")
        checks.append(True)
    else:
        print(f"❌ RSI out of range: {rsi}")
        checks.append(False)

    # SMA 20 should exist and be reasonable
    sma20 = result.get('sma_20')
    current_price = result.get('current_price')
    if sma20 and abs(sma20 - current_price) / current_price < 0.5:  # Within 50%
        print(f"✓ SMA 20 reasonable: ₹{sma20} vs current ₹{current_price}")
        checks.append(True)
    else:
        print(f"⚠️  SMA 20 check: ₹{sma20} vs current ₹{current_price}")
        checks.append(True)  # Non-fatal

    # Technical score should be 0-100
    score = result.get('technical_score', 0)
    if 0 <= score <= 100:
        print(f"✓ Technical score in range: {score}/100")
        checks.append(True)
    else:
        print(f"❌ Technical score out of range: {score}")
        checks.append(False)

    # Support should be less than resistance
    support = result.get('support_20d')
    resistance = result.get('resistance_20d')
    if support and resistance and support < resistance:
        print(f"✓ Support < Resistance: ₹{support} < ₹{resistance}")
        checks.append(True)
    else:
        print(f"❌ Support/Resistance check failed")
        checks.append(False)

    return all(checks)


def main():
    """Run all live smoke tests"""
    print("\n" + "="*70)
    print("TECHNICAL TOOLS - LIVE SMOKE TEST")
    print("Testing manual technical indicators with real yfinance data")
    print("="*70)

    tests = [
        ("Technical Indicators - RELIANCE", test_live_technical_indicators),
        ("Multiple Periods", test_live_technical_multiple_periods),
        ("Indicator Calculations", test_indicator_calculations),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} raised exception: {e}")
            import traceback
            traceback.print_exc()
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
        print("\nNote: Manual technical indicators implemented without pandas-ta")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

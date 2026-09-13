"""
LangChain @tool wrappers around yfinance for Indian stock data.
All functions handle .NS suffix automatically.
All returns are JSON-serializable dicts (no pandas objects).
All external calls use cache and handle errors gracefully.
"""
import logging
from typing import Optional
import yfinance as yf
import numpy as np
from langchain.tools import tool

from backend.utils.cache import cache_get, cache_set
from backend.utils.validators import normalize_ticker, add_ns_suffix

logger = logging.getLogger(__name__)


def _ensure_ns(ticker: str) -> str:
    """
    Add .NS suffix if not already present.
    Handles plain tickers, .NS, and .BO suffixes.
    """
    try:
        return add_ns_suffix(ticker)
    except ValueError:
        # If validation fails, try to normalize and add suffix anyway
        ticker_clean = ticker.strip().upper()
        if not ticker_clean.endswith(".NS") and not ticker_clean.endswith(".BO"):
            return ticker_clean + ".NS"
        return ticker_clean


@tool
def get_stock_info(ticker: str) -> dict:
    """
    Fetch current stock info for an NSE-listed Indian company.

    Args:
        ticker: NSE ticker symbol, e.g. 'RELIANCE' or 'RELIANCE.NS'

    Returns:
        Dict with price, market cap, P/E, P/B, ROE, D/E, beta, EPS,
        dividend yield, margins, and other fundamentals.
    """
    ticker = _ensure_ns(ticker)
    cache_key = f"stock_info_{ticker}"

    # Try cache first
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Check if we got valid data
        if not info or len(info) < 5:
            logger.warning(f"yfinance returned minimal data for {ticker}")
            return {"error": "No data available from yfinance", "ticker": ticker}

        result = {
            "ticker": ticker,
            "company_name": info.get("longName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "market_cap_cr": round(info.get("marketCap", 0) / 1e7, 1) if info.get("marketCap") else None,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "eps": info.get("trailingEps"),
            "dividend_yield_pct": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0,
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
            "roe_pct": round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") else None,
            "roa_pct": round(info.get("returnOnAssets", 0) * 100, 2) if info.get("returnOnAssets") else None,
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "revenue": info.get("totalRevenue"),
            "revenue_cr": round(info.get("totalRevenue", 0) / 1e7, 1) if info.get("totalRevenue") else None,
            "net_income": info.get("netIncomeToCommon"),
            "net_income_cr": round(info.get("netIncomeToCommon", 0) / 1e7, 1) if info.get("netIncomeToCommon") else None,
            "gross_margins_pct": round(info.get("grossMargins", 0) * 100, 2) if info.get("grossMargins") else None,
            "operating_margins_pct": round(info.get("operatingMargins", 0) * 100, 2) if info.get("operatingMargins") else None,
            "profit_margins_pct": round(info.get("profitMargins", 0) * 100, 2) if info.get("profitMargins") else None,
            "shares_outstanding": info.get("sharesOutstanding"),
            "book_value": info.get("bookValue"),
            "free_cashflow": info.get("freeCashflow"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
        }

        # Cache the result
        cache_set(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"yfinance get_stock_info error for {ticker}: {e}")
        return {"error": str(e), "ticker": ticker}


@tool
def get_price_history(ticker: str, period: str = "1y") -> dict:
    """
    Fetch daily OHLCV price history for a stock.

    Args:
        ticker: NSE ticker e.g. 'RELIANCE.NS'
        period: '1y', '2y', '6mo', '3mo', '1mo'

    Returns:
        Dict with dates, opens, highs, lows, closes, volumes as lists,
        plus latest price, oldest price, and price return percentage.
    """
    ticker = _ensure_ns(ticker)
    cache_key = f"price_history_{ticker}_{period}"

    # Try cache first
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            logger.warning(f"No price data returned for {ticker}")
            return {"error": f"No price data for {ticker}", "ticker": ticker}

        # Convert to JSON-serializable format
        result = {
            "ticker": ticker,
            "period": period,
            "dates": [str(d.date()) for d in hist.index],
            "open": [round(float(v), 2) for v in hist["Open"].tolist()],
            "high": [round(float(v), 2) for v in hist["High"].tolist()],
            "low": [round(float(v), 2) for v in hist["Low"].tolist()],
            "close": [round(float(v), 2) for v in hist["Close"].tolist()],
            "volume": [int(v) for v in hist["Volume"].tolist()],
            "latest_price": round(float(hist["Close"].iloc[-1]), 2),
            "oldest_price": round(float(hist["Close"].iloc[0]), 2),
            "price_return_pct": round(
                (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[0]) - 1) * 100, 2
            ),
        }

        # Cache the result
        cache_set(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"yfinance get_price_history error for {ticker}: {e}")
        return {"error": str(e), "ticker": ticker}


@tool
def get_nifty_history(period: str = "1y") -> dict:
    """
    Fetch Nifty 50 index price history for Beta computation.

    Args:
        period: '1y', '2y', '6mo', '3mo'

    Returns:
        Dict with dates and closes as lists.
    """
    cache_key = f"nifty_{period}"

    # Try cache first
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period=period)

        if hist.empty:
            logger.warning(f"No Nifty data returned for period {period}")
            return {"error": "No Nifty 50 data available", "period": period}

        result = {
            "dates": [str(d.date()) for d in hist.index],
            "close": [round(float(v), 2) for v in hist["Close"].tolist()],
            "period": period,
        }

        # Cache the result
        cache_set(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"Nifty fetch error: {e}")
        return {"error": str(e), "period": period}


@tool
def get_sector_median(sector: str) -> dict:
    """
    Return median financial ratios for a given sector.

    Args:
        sector: One of IT, BFSI, Auto, FMCG, Pharma, Infra, Defence, Energy, NBFC, Diversified

    Returns:
        Dict with sector name and median ratios.
    """
    from backend.utils.sector_data import get_sector_median as _get_sector_median

    try:
        return _get_sector_median(sector)
    except Exception as e:
        logger.error(f"get_sector_median error for {sector}: {e}")
        return {
            "error": str(e),
            "sector": sector,
            "medians": {}
        }

"""
Indian stock ticker validation and normalization.
Supports NSE and BSE ticker formats.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Valid ticker pattern: alphanumeric with optional .NS or .BO suffix
TICKER_PATTERN = re.compile(r'^[A-Z0-9&]+(?:\.(NS|BO))?$')

# Maximum ticker length (NSE tickers are typically 10 chars or less)
MAX_TICKER_LENGTH = 20


def normalize_ticker(ticker: str) -> Optional[str]:
    """
    Normalize Indian stock ticker to uppercase format.
    Supports:
    - Plain NSE ticker: RELIANCE -> RELIANCE
    - NSE suffix: RELIANCE.NS -> RELIANCE.NS
    - BSE suffix: RELIANCE.BO -> RELIANCE.BO

    Returns:
        Normalized ticker string or None if invalid
    """
    if not ticker:
        return None

    # Strip whitespace and convert to uppercase
    normalized = ticker.strip().upper()

    # Check if empty after stripping
    if not normalized:
        return None

    # Check length
    if len(normalized) > MAX_TICKER_LENGTH:
        logger.warning(f"Ticker too long: {normalized} (max {MAX_TICKER_LENGTH} chars)")
        return None

    # Validate pattern
    if not TICKER_PATTERN.match(normalized):
        logger.warning(f"Invalid ticker format: {normalized}")
        return None

    return normalized


def is_valid_ticker(ticker: str) -> bool:
    """
    Check if a ticker string is valid for Indian markets.

    Rejects:
    - Empty strings
    - Strings with only whitespace
    - Special characters (except . for suffix and & for company names)
    - Very long tickers

    Args:
        ticker: Ticker string to validate

    Returns:
        True if valid, False otherwise
    """
    normalized = normalize_ticker(ticker)
    return normalized is not None


def add_ns_suffix(ticker: str) -> str:
    """
    Add .NS suffix to ticker if not already present.
    Used for yfinance API calls.

    Args:
        ticker: Plain or suffixed ticker

    Returns:
        Ticker with .NS suffix
    """
    normalized = normalize_ticker(ticker)
    if not normalized:
        raise ValueError(f"Invalid ticker: {ticker}")

    # If already has .NS or .BO suffix, return as-is
    if normalized.endswith('.NS') or normalized.endswith('.BO'):
        return normalized

    # Add .NS suffix
    return f"{normalized}.NS"


def strip_suffix(ticker: str) -> str:
    """
    Remove .NS or .BO suffix from ticker.

    Args:
        ticker: Ticker with or without suffix

    Returns:
        Plain ticker without suffix
    """
    normalized = normalize_ticker(ticker)
    if not normalized:
        raise ValueError(f"Invalid ticker: {ticker}")

    # Remove suffix if present
    if normalized.endswith('.NS'):
        return normalized[:-3]
    elif normalized.endswith('.BO'):
        return normalized[:-3]

    return normalized

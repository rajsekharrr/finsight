"""
Load sector medians from static JSON file.
Provides sector benchmarking data for ratio analysis.
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SECTOR_FILE = Path("static/sector_medians.json")
_cached_data: Optional[dict] = None


def get_sector_medians() -> dict:
    """
    Load and return all sector median data.
    Caches result in memory after first load.
    Handles missing or invalid JSON gracefully.
    """
    global _cached_data

    if _cached_data is not None:
        return _cached_data

    try:
        if not _SECTOR_FILE.exists():
            logger.error(f"Sector medians file not found at {_SECTOR_FILE}")
            # Return minimal fallback data
            return {
                "Diversified": {
                    "pe": 22.0,
                    "pb": 3.5,
                    "roe": 15.0,
                    "description": "Default fallback sector"
                }
            }

        with open(_SECTOR_FILE, 'r', encoding='utf-8') as f:
            _cached_data = json.load(f)

        logger.debug(f"Loaded {len(_cached_data)} sector definitions")
        return _cached_data

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in sector medians file: {e}")
        return {"Diversified": {"pe": 22.0, "pb": 3.5, "roe": 15.0}}
    except Exception as e:
        logger.error(f"Error loading sector medians: {e}")
        return {"Diversified": {"pe": 22.0, "pb": 3.5, "roe": 15.0}}


def get_sector_median(sector: str) -> dict:
    """
    Return median financial ratios for a given sector.
    Falls back to 'Diversified' for unknown sectors.

    Args:
        sector: Sector name (e.g., 'IT', 'BFSI', 'Auto')

    Returns:
        Dict with sector medians and metadata
    """
    medians = get_sector_medians()

    # Normalize sector name
    sector_normalized = sector.strip()

    # Try exact match first
    if sector_normalized in medians:
        return {
            "sector": sector_normalized,
            "medians": medians[sector_normalized]
        }

    # Try case-insensitive match
    sector_upper = sector_normalized.upper()
    for key in medians:
        if key.upper() == sector_upper:
            return {
                "sector": key,
                "medians": medians[key]
            }

    # Try partial match (e.g., "Technology" -> "IT")
    for key in medians:
        if sector_upper in key.upper() or key.upper() in sector_upper:
            return {
                "sector": key,
                "medians": medians[key]
            }

    # Fallback to Diversified
    logger.debug(f"Unknown sector '{sector}', using Diversified fallback")
    return {
        "sector": "Diversified",
        "medians": medians.get("Diversified", {})
    }


def detect_sector_from_yfinance(sector_str: str) -> str:
    """
    Map yfinance sector string to our sector key.
    Used to normalize sector names from yfinance API.
    """
    mapping = {
        "Technology": "IT",
        "Financial Services": "BFSI",
        "Consumer Defensive": "FMCG",
        "Healthcare": "Pharma",
        "Industrials": "Infra",
        "Energy": "Energy",
        "Basic Materials": "Energy",
        "Consumer Cyclical": "Auto",
        "Utilities": "Infra",
        "Communication Services": "IT",
        "Real Estate": "Infra",
    }

    return mapping.get(sector_str, "Diversified")

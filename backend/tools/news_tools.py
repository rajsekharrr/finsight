"""
Google News RSS fetcher for Indian equity news.
No API key required. Uses feedparser.
Never scrapes NSE or BSE websites (Terms of Use restriction).
"""
import logging
import feedparser
from datetime import datetime
from typing import List, Dict, Optional

from backend.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# Economic Times RSS feeds (optional fallback sources)
ECONOMIC_TIMES_MARKETS_RSS = "https://economictimes.indiatimes.com/markets/rss.cms"
ECONOMIC_TIMES_RESULTS_RSS = "https://economictimes.indiatimes.com/markets/earnings/rss.cms"


def fetch_google_news(company_name: str, ticker: str = "", limit: int = 20) -> List[Dict]:
    """
    Fetch recent news for a company from Google News RSS.

    Args:
        company_name: Company name (e.g., "Reliance Industries")
        ticker: Optional ticker symbol (e.g., "RELIANCE" or "RELIANCE.NS")
        limit: Maximum number of articles to return

    Returns:
        List of article dicts with title, published, source, summary, link.
        Returns empty list on error (non-fatal).
    """
    cache_key = f"news_{company_name.replace(' ', '_')}"

    # Try cache first
    cached = cache_get(cache_key)
    if cached:
        logger.debug(f"News cache hit for {company_name}")
        return cached

    # Build query - company name + NSE/stock context for India filter
    query = company_name.replace(" ", "+")
    if ticker:
        # Remove .NS/.BO suffix for cleaner query
        ticker_clean = ticker.replace(".NS", "").replace(".BO", "")
        query += f"+{ticker_clean}"
    query += "+NSE+stock"

    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        logger.info(f"Fetching news for {company_name} from Google News RSS")
        feed = feedparser.parse(url)

        articles = []
        for entry in feed.entries[:limit]:
            articles.append({
                "title": entry.get("title", ""),
                "published": entry.get("published", ""),
                "source": entry.get("source", {}).get("title", "Google News"),
                "summary": entry.get("summary", ""),
                "link": entry.get("link", ""),
            })

        # If no results, try broader query without ticker
        if not articles and ticker:
            logger.warning(f"No news found for {company_name}, trying broader query")
            url2 = f"https://news.google.com/rss/search?q={company_name.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
            feed2 = feedparser.parse(url2)

            for entry in feed2.entries[:limit]:
                articles.append({
                    "title": entry.get("title", ""),
                    "published": entry.get("published", ""),
                    "source": entry.get("source", {}).get("title", "Google News"),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                })

        logger.info(f"Fetched {len(articles)} news articles for {company_name}")

        # Cache the result
        cache_set(cache_key, articles)
        return articles

    except Exception as e:
        logger.error(f"News fetch error for {company_name}: {e}")
        # Non-fatal: return empty list
        return []


def fetch_economic_times_markets(limit: int = 20) -> List[Dict]:
    """
    Fetch general Indian markets news from Economic Times RSS.

    Args:
        limit: Maximum number of articles

    Returns:
        List of article dicts
    """
    cache_key = "news_et_markets"

    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        feed = feedparser.parse(ECONOMIC_TIMES_MARKETS_RSS)

        articles = []
        for entry in feed.entries[:limit]:
            articles.append({
                "title": entry.get("title", ""),
                "published": entry.get("published", ""),
                "source": "Economic Times",
                "summary": entry.get("summary", ""),
                "link": entry.get("link", ""),
            })

        cache_set(cache_key, articles)
        return articles

    except Exception as e:
        logger.error(f"Economic Times fetch error: {e}")
        return []

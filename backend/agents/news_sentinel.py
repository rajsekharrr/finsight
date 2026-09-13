"""
News Sentinel Agent: Analyzes recent news sentiment and themes for a company.

Fetches recent news articles and uses LLM to analyze sentiment, identify key themes,
and extract positive/negative drivers.
"""

import logging
import json
import re
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from backend.tools.news_tools import fetch_google_news

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_news_sentinel(ticker: str, company_name: str) -> Dict[str, Any]:
    """
    Analyze recent news sentiment and themes for a company.

    Args:
        ticker: Stock ticker symbol (e.g., "RELIANCE")
        company_name: Full company name (e.g., "Reliance Industries")

    Returns:
        Dict with keys:
        - status: "success", "partial_success", or "error"
        - ticker: Stock ticker
        - company_name: Company name
        - article_count: Number of articles analyzed
        - sentiment: Overall sentiment (POSITIVE, NEGATIVE, NEUTRAL, MIXED, UNKNOWN)
        - sentiment_score: Numeric score from -1 (negative) to 1 (positive)
        - key_themes: List of identified themes
        - positive_drivers: List of positive news drivers
        - negative_drivers: List of negative news drivers
        - notable_headlines: List of important headlines
        - sources: List of source links
        - error: Error message if applicable
    """
    result = {
        "status": "error",
        "ticker": ticker,
        "company_name": company_name,
        "article_count": 0,
        "sentiment": "UNKNOWN",
        "sentiment_score": 0.0,
        "key_themes": [],
        "positive_drivers": [],
        "negative_drivers": [],
        "notable_headlines": [],
        "sources": [],
        "error": None
    }

    try:
        logger.info(f"Starting news sentiment analysis for {ticker} ({company_name})")

        # Fetch news articles
        articles = fetch_google_news(company_name, ticker, limit=20)

        if not articles:
            logger.warning(f"No news articles found for {company_name}")
            result["status"] = "partial_success"
            result["error"] = "No recent news articles found"
            return result

        logger.info(f"Fetched {len(articles)} news articles")
        result["article_count"] = len(articles)

        # Extract source links and headlines
        sources = [article.get("link", "") for article in articles if article.get("link")]
        result["sources"] = sources

        headlines = [article.get("title", "") for article in articles if article.get("title")]

        # Build context for LLM
        articles_text = "\n\n".join([
            f"Headline: {article.get('title', '')}\n"
            f"Source: {article.get('source', '')}\n"
            f"Published: {article.get('published', '')}\n"
            f"Summary: {article.get('summary', '')}"
            for article in articles
        ])

        # Create prompt for LLM sentiment analysis
        prompt = f"""You are a financial news analyst. Analyze the following {len(articles)} recent news articles about {company_name} ({ticker}).

News Articles:
{articles_text}

Provide a comprehensive sentiment analysis in JSON format with this exact structure:

{{
  "sentiment": "<POSITIVE|NEGATIVE|NEUTRAL|MIXED>",
  "sentiment_score": <float between -1.0 and 1.0>,
  "key_themes": [
    "Theme 1",
    "Theme 2",
    "... (3-7 main themes from the news)"
  ],
  "positive_drivers": [
    "Positive driver 1",
    "Positive driver 2",
    "... (specific positive news points, or empty list if none)"
  ],
  "negative_drivers": [
    "Negative driver 1",
    "Negative driver 2",
    "... (specific negative news points, or empty list if none)"
  ],
  "notable_headlines": [
    "Most important headline 1",
    "Most important headline 2",
    "... (3-5 most significant headlines)"
  ]
}}

Guidelines:
- sentiment: Use POSITIVE if mostly good news, NEGATIVE if mostly bad news, NEUTRAL if balanced/routine, MIXED if conflicting signals
- sentiment_score: -1.0 (very negative) to 1.0 (very positive), 0 is neutral
- Identify concrete themes and drivers from the actual articles
- Select the most impactful headlines for notable_headlines

Return ONLY the JSON object, no markdown code fences or additional text."""

        # Call LLM for sentiment analysis
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                timeout=60
            )

            logger.info("Calling LLM for sentiment analysis")
            response = llm.invoke(prompt)
            response_text = response.content.strip()

            logger.info(f"LLM response length: {len(response_text)} characters")

            # Strip markdown code fences if present
            response_text = re.sub(r'^```(?:json)?\s*\n?', '', response_text)
            response_text = re.sub(r'\n?```\s*$', '', response_text)
            response_text = response_text.strip()

            # Parse JSON response
            try:
                analysis = json.loads(response_text)

                # Validate and extract sentiment
                sentiment = analysis.get("sentiment", "UNKNOWN").upper()
                valid_sentiments = ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED", "UNKNOWN"]
                if sentiment not in valid_sentiments:
                    logger.warning(f"Invalid sentiment '{sentiment}', defaulting to UNKNOWN")
                    sentiment = "UNKNOWN"

                result["sentiment"] = sentiment

                # Validate and extract sentiment score
                sentiment_score = analysis.get("sentiment_score", 0.0)
                try:
                    sentiment_score = float(sentiment_score)
                    # Clamp to [-1, 1] range
                    sentiment_score = max(-1.0, min(1.0, sentiment_score))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid sentiment_score, defaulting to 0.0")
                    sentiment_score = 0.0

                result["sentiment_score"] = sentiment_score

                # Extract other fields
                result["key_themes"] = analysis.get("key_themes", [])
                result["positive_drivers"] = analysis.get("positive_drivers", [])
                result["negative_drivers"] = analysis.get("negative_drivers", [])
                result["notable_headlines"] = analysis.get("notable_headlines", [])

                result["status"] = "success"
                logger.info(f"News sentiment analysis completed: {sentiment} ({sentiment_score:.2f})")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
                logger.error(f"Response text: {response_text[:500]}")

                result["status"] = "partial_success"
                result["sentiment"] = "UNKNOWN"
                result["error"] = f"JSON parse error: {str(e)}"

                # Provide basic fallback analysis
                result["key_themes"] = [f"Analysis available but parsing failed"]
                result["notable_headlines"] = headlines[:5] if len(headlines) >= 5 else headlines

        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            result["status"] = "partial_success"
            result["sentiment"] = "UNKNOWN"
            result["error"] = f"LLM analysis failed: {str(e)}"

            # Provide basic fallback data
            result["key_themes"] = [f"Fetched {len(articles)} articles but analysis failed"]
            result["notable_headlines"] = headlines[:5] if len(headlines) >= 5 else headlines

    except Exception as e:
        logger.error(f"News sentiment analysis failed for {ticker}: {e}", exc_info=True)
        result["status"] = "error"
        result["error"] = str(e)

    return result

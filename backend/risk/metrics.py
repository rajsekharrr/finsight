"""
Risk Metrics: Pure calculation functions for financial risk assessment.

All functions use code-based calculations only - no API calls, no LLM.
Handles missing, empty, or invalid inputs gracefully.
"""

import logging
import numpy as np
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


def calculate_beta(
    stock_returns: List[float],
    market_returns: List[float]
) -> Optional[float]:
    """
    Calculate beta coefficient (systematic risk vs market).

    Beta measures how much a stock moves relative to the market:
    - Beta > 1: More volatile than market
    - Beta = 1: Moves with market
    - Beta < 1: Less volatile than market
    - Beta < 0: Moves opposite to market

    Args:
        stock_returns: List of stock returns (decimals, e.g., 0.05 = 5%)
        market_returns: List of market returns (same length as stock_returns)

    Returns:
        Beta coefficient, or None if calculation fails
    """
    try:
        if not stock_returns or not market_returns:
            logger.warning("Empty returns provided for beta calculation")
            return None

        if len(stock_returns) != len(market_returns):
            logger.warning(f"Mismatched lengths: stock {len(stock_returns)}, market {len(market_returns)}")
            return None

        if len(stock_returns) < 2:
            logger.warning("Insufficient data points for beta calculation")
            return None

        # Convert to numpy arrays
        stock_arr = np.array(stock_returns, dtype=float)
        market_arr = np.array(market_returns, dtype=float)

        # Remove NaN values
        mask = ~(np.isnan(stock_arr) | np.isnan(market_arr))
        stock_arr = stock_arr[mask]
        market_arr = market_arr[mask]

        if len(stock_arr) < 2:
            logger.warning("Insufficient valid data after removing NaN")
            return None

        # Calculate covariance and variance
        covariance = np.cov(stock_arr, market_arr)[0, 1]
        market_variance = np.var(market_arr, ddof=1)

        if market_variance == 0:
            logger.warning("Market variance is zero")
            return None

        beta = covariance / market_variance
        return float(beta)

    except Exception as e:
        logger.error(f"Beta calculation error: {e}")
        return None


def calculate_var(
    returns: List[float],
    confidence: float = 0.95
) -> Optional[float]:
    """
    Calculate Value at Risk (VaR) using historical simulation method.

    VaR estimates the maximum expected loss over a given time period
    at a specified confidence level.

    Args:
        returns: List of historical returns (decimals)
        confidence: Confidence level (0.95 = 95%)

    Returns:
        VaR as a positive percentage (e.g., 2.5 means 2.5% max loss),
        or None if calculation fails
    """
    try:
        if not returns:
            logger.warning("Empty returns provided for VaR calculation")
            return None

        if len(returns) < 10:
            logger.warning("Insufficient data points for VaR calculation")
            return None

        if not (0 < confidence < 1):
            logger.warning(f"Invalid confidence level: {confidence}")
            return None

        # Convert to numpy array
        returns_arr = np.array(returns, dtype=float)

        # Remove NaN values
        returns_arr = returns_arr[~np.isnan(returns_arr)]

        if len(returns_arr) < 10:
            logger.warning("Insufficient valid data after removing NaN")
            return None

        # Calculate VaR at the specified confidence level
        # For 95% confidence, we look at the 5th percentile (worst 5% of returns)
        percentile = (1 - confidence) * 100
        var = np.percentile(returns_arr, percentile)

        # Return as positive percentage (magnitude of loss)
        return float(abs(var) * 100)

    except Exception as e:
        logger.error(f"VaR calculation error: {e}")
        return None


def calculate_volatility(
    returns: List[float],
    annualize: bool = True
) -> Optional[float]:
    """
    Calculate volatility (standard deviation of returns).

    Args:
        returns: List of historical returns (decimals)
        annualize: If True, annualize the volatility (assumes daily returns)

    Returns:
        Volatility as percentage (e.g., 25.0 means 25% volatility),
        or None if calculation fails
    """
    try:
        if not returns:
            logger.warning("Empty returns provided for volatility calculation")
            return None

        if len(returns) < 2:
            logger.warning("Insufficient data points for volatility calculation")
            return None

        # Convert to numpy array
        returns_arr = np.array(returns, dtype=float)

        # Remove NaN values
        returns_arr = returns_arr[~np.isnan(returns_arr)]

        if len(returns_arr) < 2:
            logger.warning("Insufficient valid data after removing NaN")
            return None

        # Calculate standard deviation
        volatility = np.std(returns_arr, ddof=1)

        # Annualize if requested (assumes ~252 trading days per year)
        if annualize:
            volatility = volatility * np.sqrt(252)

        # Return as percentage
        return float(volatility * 100)

    except Exception as e:
        logger.error(f"Volatility calculation error: {e}")
        return None


def calculate_max_drawdown(prices: List[float]) -> Optional[float]:
    """
    Calculate maximum drawdown (peak-to-trough decline).

    Maximum drawdown measures the largest peak-to-trough decline
    in the price series.

    Args:
        prices: List of historical prices

    Returns:
        Maximum drawdown as positive percentage (e.g., 15.0 means 15% drawdown),
        or None if calculation fails
    """
    try:
        if not prices:
            logger.warning("Empty prices provided for drawdown calculation")
            return None

        if len(prices) < 2:
            logger.warning("Insufficient data points for drawdown calculation")
            return None

        # Convert to numpy array
        prices_arr = np.array(prices, dtype=float)

        # Remove NaN values
        prices_arr = prices_arr[~np.isnan(prices_arr)]

        if len(prices_arr) < 2:
            logger.warning("Insufficient valid data after removing NaN")
            return None

        # Calculate running maximum
        running_max = np.maximum.accumulate(prices_arr)

        # Calculate drawdown at each point
        drawdowns = (prices_arr - running_max) / running_max

        # Get maximum drawdown (most negative value)
        max_drawdown = np.min(drawdowns)

        # Return as positive percentage
        return float(abs(max_drawdown) * 100)

    except Exception as e:
        logger.error(f"Max drawdown calculation error: {e}")
        return None


def calculate_altman_z_score(
    working_capital: Optional[float],
    retained_earnings: Optional[float],
    ebit: Optional[float],
    market_value_equity: Optional[float],
    sales: Optional[float],
    total_assets: Optional[float],
    total_liabilities: Optional[float]
) -> Optional[float]:
    """
    Calculate Altman Z-Score for bankruptcy prediction.

    Z-Score formula for public companies:
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    Where:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Value of Equity / Total Liabilities
    X5 = Sales / Total Assets

    Interpretation:
    - Z > 2.99: Safe zone
    - 1.81 < Z < 2.99: Grey zone
    - Z < 1.81: Distress zone

    Args:
        working_capital: Current assets - current liabilities
        retained_earnings: Cumulative retained earnings
        ebit: Earnings before interest and tax
        market_value_equity: Market cap
        sales: Total revenue/sales
        total_assets: Total assets
        total_liabilities: Total liabilities

    Returns:
        Altman Z-Score, or None if insufficient data
    """
    try:
        # Check if we have minimum required data
        required_fields = [total_assets, total_liabilities]
        if any(x is None or x <= 0 for x in required_fields):
            logger.warning("Insufficient data for Z-Score calculation")
            return None

        # X1: Working Capital / Total Assets
        x1 = 0.0
        if working_capital is not None and total_assets > 0:
            x1 = working_capital / total_assets

        # X2: Retained Earnings / Total Assets
        x2 = 0.0
        if retained_earnings is not None and total_assets > 0:
            x2 = retained_earnings / total_assets

        # X3: EBIT / Total Assets
        x3 = 0.0
        if ebit is not None and total_assets > 0:
            x3 = ebit / total_assets

        # X4: Market Value of Equity / Total Liabilities
        x4 = 0.0
        if market_value_equity is not None and total_liabilities > 0:
            x4 = market_value_equity / total_liabilities

        # X5: Sales / Total Assets
        x5 = 0.0
        if sales is not None and total_assets > 0:
            x5 = sales / total_assets

        # Calculate Z-Score
        z_score = (
            1.2 * x1 +
            1.4 * x2 +
            3.3 * x3 +
            0.6 * x4 +
            1.0 * x5
        )

        return float(z_score)

    except Exception as e:
        logger.error(f"Altman Z-Score calculation error: {e}")
        return None


def classify_risk_level(score: float) -> str:
    """
    Classify risk level based on risk score.

    Args:
        score: Risk score (0-100, where higher = higher risk)

    Returns:
        Risk level string: LOW, MEDIUM, MEDIUM-HIGH, HIGH, VERY HIGH
    """
    if score < 20:
        return "LOW"
    elif score < 40:
        return "MEDIUM"
    elif score < 60:
        return "MEDIUM-HIGH"
    elif score < 80:
        return "HIGH"
    else:
        return "VERY HIGH"

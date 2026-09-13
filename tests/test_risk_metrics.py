"""
Tests for risk metrics calculations.
"""

import pytest
from backend.risk.metrics import (
    calculate_beta,
    calculate_var,
    calculate_volatility,
    calculate_max_drawdown,
    calculate_altman_z_score,
    classify_risk_level
)


class TestRiskMetrics:
    """Tests for risk metric calculation functions."""

    def test_calculate_beta_valid(self):
        """Should calculate beta correctly with valid data."""
        stock_returns = [0.01, 0.02, -0.01, 0.03, -0.02]
        market_returns = [0.015, 0.01, -0.005, 0.02, -0.01]

        beta = calculate_beta(stock_returns, market_returns)

        assert beta is not None
        assert isinstance(beta, float)
        # Beta should be reasonable (typically between -2 and 3)
        assert -3 < beta < 4

    def test_calculate_beta_empty_data(self):
        """Should return None for empty data."""
        assert calculate_beta([], []) is None
        assert calculate_beta([0.01], []) is None

    def test_calculate_beta_mismatched_lengths(self):
        """Should return None for mismatched lengths."""
        stock_returns = [0.01, 0.02, 0.03]
        market_returns = [0.01, 0.02]

        assert calculate_beta(stock_returns, market_returns) is None

    def test_calculate_beta_insufficient_data(self):
        """Should return None for insufficient data points."""
        stock_returns = [0.01]
        market_returns = [0.01]

        assert calculate_beta(stock_returns, market_returns) is None

    def test_calculate_var_valid(self):
        """Should calculate VaR correctly with valid data."""
        returns = [0.01, 0.02, -0.03, 0.015, -0.025, 0.005, -0.01, 0.02, -0.015, 0.01] * 2

        var = calculate_var(returns, confidence=0.95)

        assert var is not None
        assert isinstance(var, float)
        assert var > 0  # Should be positive (magnitude of loss)
        assert var < 100  # Should be reasonable percentage

    def test_calculate_var_empty_data(self):
        """Should return None for empty data."""
        assert calculate_var([]) is None

    def test_calculate_var_insufficient_data(self):
        """Should return None for insufficient data points."""
        returns = [0.01, 0.02, -0.01]  # Less than 10 points

        assert calculate_var(returns) is None

    def test_calculate_var_invalid_confidence(self):
        """Should return None for invalid confidence level."""
        returns = [0.01] * 20

        assert calculate_var(returns, confidence=1.5) is None
        assert calculate_var(returns, confidence=0) is None

    def test_calculate_volatility_valid(self):
        """Should calculate volatility correctly."""
        returns = [0.01, 0.02, -0.01, 0.015, -0.02, 0.005] * 5

        vol = calculate_volatility(returns, annualize=True)

        assert vol is not None
        assert isinstance(vol, float)
        assert vol > 0
        assert vol < 200  # Should be reasonable percentage

    def test_calculate_volatility_not_annualized(self):
        """Should calculate non-annualized volatility."""
        returns = [0.01, 0.02, -0.01, 0.015, -0.02]

        vol_daily = calculate_volatility(returns, annualize=False)
        vol_annual = calculate_volatility(returns, annualize=True)

        assert vol_daily is not None
        assert vol_annual is not None
        # Annual should be larger (sqrt(252) factor)
        assert vol_annual > vol_daily

    def test_calculate_volatility_empty_data(self):
        """Should return None for empty data."""
        assert calculate_volatility([]) is None

    def test_calculate_volatility_insufficient_data(self):
        """Should return None for insufficient data."""
        assert calculate_volatility([0.01]) is None

    def test_calculate_max_drawdown_valid(self):
        """Should calculate max drawdown correctly."""
        prices = [100, 105, 110, 95, 90, 100, 105]

        max_dd = calculate_max_drawdown(prices)

        assert max_dd is not None
        assert isinstance(max_dd, float)
        # Max drawdown is from 110 to 90 = 18.18%
        assert 15 < max_dd < 20

    def test_calculate_max_drawdown_no_drawdown(self):
        """Should handle case with no drawdown (only increases)."""
        prices = [100, 105, 110, 115, 120]

        max_dd = calculate_max_drawdown(prices)

        assert max_dd is not None
        assert max_dd == 0

    def test_calculate_max_drawdown_empty_data(self):
        """Should return None for empty data."""
        assert calculate_max_drawdown([]) is None

    def test_calculate_max_drawdown_insufficient_data(self):
        """Should return None for insufficient data."""
        assert calculate_max_drawdown([100]) is None

    def test_calculate_altman_z_score_valid(self):
        """Should calculate Altman Z-Score with valid data."""
        z_score = calculate_altman_z_score(
            working_capital=1000000,
            retained_earnings=2000000,
            ebit=500000,
            market_value_equity=5000000,
            sales=3000000,
            total_assets=8000000,
            total_liabilities=3000000
        )

        assert z_score is not None
        assert isinstance(z_score, float)
        # Z-score typically ranges from -5 to 20
        assert -10 < z_score < 25

    def test_calculate_altman_z_score_missing_total_assets(self):
        """Should return None if total_assets is missing."""
        z_score = calculate_altman_z_score(
            working_capital=1000000,
            retained_earnings=2000000,
            ebit=500000,
            market_value_equity=5000000,
            sales=3000000,
            total_assets=None,
            total_liabilities=3000000
        )

        assert z_score is None

    def test_calculate_altman_z_score_partial_data(self):
        """Should handle partial data (some None values)."""
        z_score = calculate_altman_z_score(
            working_capital=None,  # Missing
            retained_earnings=2000000,
            ebit=None,  # Missing
            market_value_equity=5000000,
            sales=3000000,
            total_assets=8000000,
            total_liabilities=3000000
        )

        # Should still calculate with available data
        assert z_score is not None
        assert isinstance(z_score, float)

    def test_calculate_altman_z_score_zero_total_assets(self):
        """Should return None if total_assets is zero."""
        z_score = calculate_altman_z_score(
            working_capital=1000000,
            retained_earnings=2000000,
            ebit=500000,
            market_value_equity=5000000,
            sales=3000000,
            total_assets=0,
            total_liabilities=3000000
        )

        assert z_score is None

    def test_classify_risk_level_low(self):
        """Should classify low risk correctly."""
        assert classify_risk_level(10) == "LOW"
        assert classify_risk_level(0) == "LOW"
        assert classify_risk_level(19) == "LOW"

    def test_classify_risk_level_medium(self):
        """Should classify medium risk correctly."""
        assert classify_risk_level(20) == "MEDIUM"
        assert classify_risk_level(30) == "MEDIUM"
        assert classify_risk_level(39) == "MEDIUM"

    def test_classify_risk_level_medium_high(self):
        """Should classify medium-high risk correctly."""
        assert classify_risk_level(40) == "MEDIUM-HIGH"
        assert classify_risk_level(50) == "MEDIUM-HIGH"
        assert classify_risk_level(59) == "MEDIUM-HIGH"

    def test_classify_risk_level_high(self):
        """Should classify high risk correctly."""
        assert classify_risk_level(60) == "HIGH"
        assert classify_risk_level(70) == "HIGH"
        assert classify_risk_level(79) == "HIGH"

    def test_classify_risk_level_very_high(self):
        """Should classify very high risk correctly."""
        assert classify_risk_level(80) == "VERY HIGH"
        assert classify_risk_level(90) == "VERY HIGH"
        assert classify_risk_level(100) == "VERY HIGH"

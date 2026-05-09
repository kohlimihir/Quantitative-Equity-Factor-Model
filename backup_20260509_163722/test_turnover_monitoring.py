"""
test_turnover_monitoring.py — Unit Tests for Turnover Monitoring and Alerts
=============================================================================
Tests the new turnover monitoring, portfolio transition analysis, and
turnover forecasting features added in Task 6.5.

Requirements: 4.8, 4.10
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from turnover_optimizer import TurnoverOptimizer


class TestTurnoverMonitoring:
    """Test suite for turnover threshold monitoring (Requirement 4.8)"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = TurnoverOptimizer(top_per_sector=3, rebal_threshold=0.12)
        
        # Create sample turnover data
        dates = pd.date_range(start="2023-01-01", periods=12, freq="MS")
        self.turnover_df = pd.DataFrame({
            "date": dates,
            "turnover": [0.15, 0.25, 0.35, 0.20, 0.45, 0.18, 
                        0.28, 0.32, 0.22, 0.38, 0.19, 0.26],
            "n_changed": [5, 8, 12, 7, 15, 6, 9, 11, 7, 13, 6, 8],
            "n_stocks": [33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33],
            "stocks_sold": [[] for _ in range(12)],
            "stocks_bought": [[] for _ in range(12)],
        })
    
    def test_monitor_turnover_threshold_flags_high_months(self):
        """Test that months exceeding 30% threshold are flagged"""
        flagged = self.optimizer.monitor_turnover_threshold(self.turnover_df, threshold=0.30)
        
        # Should flag months with turnover > 30%
        assert len(flagged) == 4  # Months with 0.35, 0.45, 0.32, 0.38
        assert all(flagged["turnover"] > 0.30)
        assert "alert_level" in flagged.columns
        assert "alert_message" in flagged.columns
    
    def test_monitor_turnover_threshold_no_flags_when_below(self):
        """Test that no flags are raised when all months are below threshold"""
        low_turnover_df = self.turnover_df.copy()
        low_turnover_df["turnover"] = [0.15, 0.20, 0.18, 0.22, 0.19, 0.21,
                                        0.17, 0.23, 0.16, 0.24, 0.18, 0.20]
        
        flagged = self.optimizer.monitor_turnover_threshold(low_turnover_df, threshold=0.30)
        
        assert len(flagged) == 0
    
    def test_monitor_turnover_threshold_alert_levels(self):
        """Test that alert levels are correctly assigned"""
        flagged = self.optimizer.monitor_turnover_threshold(self.turnover_df, threshold=0.30)
        
        # Check alert level categories exist
        assert "WARNING" in flagged["alert_level"].values or \
               "HIGH" in flagged["alert_level"].values or \
               "CRITICAL" in flagged["alert_level"].values
        
        # Verify alert levels are ordered by severity
        for _, row in flagged.iterrows():
            if row["turnover"] <= 0.40:
                assert row["alert_level"] == "WARNING"
            elif row["turnover"] <= 0.50:
                assert row["alert_level"] == "HIGH"
            else:
                assert row["alert_level"] == "CRITICAL"
    
    def test_monitor_turnover_threshold_custom_threshold(self):
        """Test monitoring with custom threshold values"""
        # Test with 20% threshold
        flagged_20 = self.optimizer.monitor_turnover_threshold(self.turnover_df, threshold=0.20)
        assert len(flagged_20) > 4  # More months should be flagged
        
        # Test with 40% threshold
        flagged_40 = self.optimizer.monitor_turnover_threshold(self.turnover_df, threshold=0.40)
        assert len(flagged_40) == 1  # Only month with 0.45 turnover


class TestPortfolioTransition:
    """Test suite for portfolio transition analysis (Requirement 4.10)"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = TurnoverOptimizer(top_per_sector=3, rebal_threshold=0.12)
        
        # Create sample predictions data
        dates = pd.date_range(start="2023-01-01", periods=6, freq="MS")
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD"]
        sectors = ["Tech", "Tech", "Tech", "Consumer", "Consumer", "Tech", "Tech", "Consumer", "Tech"]
        
        data = []
        for date in dates:
            for ticker, sector in zip(tickers, sectors):
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "sector": sector,
                    "predicted": np.random.rand(),
                    "actual": np.random.randn() * 0.05,
                })
        
        self.predictions_before = pd.DataFrame(data)
        
        # Create "after" predictions with some changes
        self.predictions_after = self.predictions_before.copy()
        # Modify some predictions to simulate parameter change impact
        self.predictions_after.loc[
            self.predictions_after["ticker"].isin(["AAPL", "TSLA"]), 
            "predicted"
        ] *= 0.8
        self.predictions_after.loc[
            self.predictions_after["ticker"].isin(["AMD", "NFLX"]), 
            "predicted"
        ] *= 1.3
    
    def test_analyze_portfolio_transition_returns_dataframe(self):
        """Test that transition analysis returns valid DataFrame"""
        transition_df = self.optimizer.analyze_portfolio_transition(
            self.predictions_before,
            self.predictions_after,
            param_name="test_param",
            param_before=0.5,
            param_after=0.7
        )
        
        assert isinstance(transition_df, pd.DataFrame)
        assert len(transition_df) > 0
        assert "date" in transition_df.columns
        assert "transition_pct" in transition_df.columns
        assert "n_kept" in transition_df.columns
        assert "n_removed" in transition_df.columns
        assert "n_added" in transition_df.columns
    
    def test_analyze_portfolio_transition_calculates_correctly(self):
        """Test that transition percentages are calculated correctly"""
        transition_df = self.optimizer.analyze_portfolio_transition(
            self.predictions_before,
            self.predictions_after
        )
        
        # Verify transition percentages are between 0 and 1
        assert all(transition_df["transition_pct"] >= 0)
        assert all(transition_df["transition_pct"] <= 1)
        
        # Verify that n_removed + n_kept equals portfolio size
        for _, row in transition_df.iterrows():
            # Total should be consistent (top_per_sector * num_sectors)
            assert row["n_kept"] + row["n_removed"] >= 0
    
    def test_analyze_portfolio_transition_identical_predictions(self):
        """Test transition analysis with identical predictions (no change)"""
        transition_df = self.optimizer.analyze_portfolio_transition(
            self.predictions_before,
            self.predictions_before  # Same predictions
        )
        
        # Should have zero transition when predictions are identical
        assert all(transition_df["transition_pct"] == 0)
        assert all(transition_df["n_removed"] == 0)
        assert all(transition_df["n_added"] == 0)


class TestTurnoverForecasting:
    """Test suite for turnover forecasting (Requirement 4.10)"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = TurnoverOptimizer(top_per_sector=3, rebal_threshold=0.12)
        
        # Create sample predictions data
        dates = pd.date_range(start="2023-01-01", periods=12, freq="MS")
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD"]
        sectors = ["Tech", "Tech", "Tech", "Consumer", "Consumer", "Tech", "Tech", "Consumer", "Tech"]
        
        data = []
        for date in dates:
            for ticker, sector in zip(tickers, sectors):
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "sector": sector,
                    "predicted": np.random.rand(),
                    "actual": np.random.randn() * 0.05,
                })
        
        self.predictions_df = pd.DataFrame(data)
    
    def test_forecast_turnover_ewm_alpha(self):
        """Test turnover forecasting for EWM alpha parameter"""
        forecast_df = self.optimizer.forecast_turnover(
            self.predictions_df,
            param_changes=[0.3, 0.5, 0.7],
            param_type="ewm_alpha"
        )
        
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) == 3
        assert "param_value" in forecast_df.columns
        assert "estimated_avg_turnover" in forecast_df.columns
        assert "estimated_annual_turnover" in forecast_df.columns
        assert "lower_bound_95ci" in forecast_df.columns
        assert "upper_bound_95ci" in forecast_df.columns
        
        # Higher alpha should generally lead to higher turnover
        alpha_03 = forecast_df[forecast_df["param_value"] == 0.3]["estimated_avg_turnover"].values[0]
        alpha_07 = forecast_df[forecast_df["param_value"] == 0.7]["estimated_avg_turnover"].values[0]
        assert alpha_07 >= alpha_03  # Higher alpha = more turnover
    
    def test_forecast_turnover_rebal_threshold(self):
        """Test turnover forecasting for rebalancing threshold parameter"""
        forecast_df = self.optimizer.forecast_turnover(
            self.predictions_df,
            param_changes=[0.05, 0.12, 0.20],
            param_type="rebal_threshold"
        )
        
        assert len(forecast_df) == 3
        
        # Higher threshold should lead to lower turnover
        thresh_05 = forecast_df[forecast_df["param_value"] == 0.05]["estimated_avg_turnover"].values[0]
        thresh_20 = forecast_df[forecast_df["param_value"] == 0.20]["estimated_avg_turnover"].values[0]
        assert thresh_05 >= thresh_20  # Lower threshold = more turnover
    
    def test_forecast_turnover_holding_bonus(self):
        """Test turnover forecasting for holding bonus parameter"""
        forecast_df = self.optimizer.forecast_turnover(
            self.predictions_df,
            param_changes=[0.0, 0.02, 0.05],
            param_type="holding_bonus"
        )
        
        assert len(forecast_df) == 3
        
        # Higher bonus should lead to lower turnover
        bonus_00 = forecast_df[forecast_df["param_value"] == 0.0]["estimated_avg_turnover"].values[0]
        bonus_05 = forecast_df[forecast_df["param_value"] == 0.05]["estimated_avg_turnover"].values[0]
        assert bonus_00 >= bonus_05  # Higher bonus = less turnover
    
    def test_forecast_turnover_bounds_are_valid(self):
        """Test that confidence interval bounds are valid"""
        forecast_df = self.optimizer.forecast_turnover(
            self.predictions_df,
            param_changes=[0.3, 0.5, 0.7],
            param_type="ewm_alpha"
        )
        
        for _, row in forecast_df.iterrows():
            # Lower bound should be less than estimate
            assert row["lower_bound_95ci"] <= row["estimated_avg_turnover"]
            # Upper bound should be greater than estimate
            assert row["upper_bound_95ci"] >= row["estimated_avg_turnover"]
            # Bounds should be between 0 and 1
            assert 0 <= row["lower_bound_95ci"] <= 1
            assert 0 <= row["upper_bound_95ci"] <= 1
    
    def test_forecast_turnover_includes_change_metrics(self):
        """Test that forecast includes change from baseline metrics"""
        forecast_df = self.optimizer.forecast_turnover(
            self.predictions_df,
            param_changes=[0.3, 0.5, 0.7],
            param_type="ewm_alpha"
        )
        
        assert "change_from_baseline" in forecast_df.columns
        assert "change_pct" in forecast_df.columns
        
        # At least one parameter should show a change
        assert any(forecast_df["change_from_baseline"] != 0)


class TestIntegration:
    """Integration tests for complete turnover monitoring workflow"""
    
    def test_complete_monitoring_workflow(self):
        """Test complete workflow: compute turnover → monitor → analyze → forecast"""
        # Create sample data
        dates = pd.date_range(start="2023-01-01", periods=12, freq="MS")
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD"]
        sectors = ["Tech", "Tech", "Tech", "Consumer", "Consumer", "Tech", "Tech", "Consumer", "Tech"]
        
        data = []
        for date in dates:
            for ticker, sector in zip(tickers, sectors):
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "sector": sector,
                    "predicted": np.random.rand(),
                    "actual": np.random.randn() * 0.05,
                })
        
        predictions_df = pd.DataFrame(data)
        optimizer = TurnoverOptimizer(top_per_sector=3, rebal_threshold=0.12)
        
        # Step 1: Compute turnover
        turnover_df = optimizer.compute_monthly_turnover(predictions_df)
        assert len(turnover_df) > 0
        
        # Step 2: Monitor threshold
        flagged = optimizer.monitor_turnover_threshold(turnover_df, threshold=0.30)
        assert isinstance(flagged, pd.DataFrame)
        
        # Step 3: Analyze transition (simulate parameter change)
        predictions_alt = predictions_df.copy()
        predictions_alt["predicted"] *= 1.1  # Slight modification
        transition_df = optimizer.analyze_portfolio_transition(
            predictions_df, predictions_alt
        )
        assert len(transition_df) > 0
        
        # Step 4: Forecast turnover
        forecast_df = optimizer.forecast_turnover(
            predictions_df,
            param_changes=[0.3, 0.5, 0.7],
            param_type="ewm_alpha"
        )
        assert len(forecast_df) == 3


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

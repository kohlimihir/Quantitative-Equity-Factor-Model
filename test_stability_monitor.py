"""
test_stability_monitor.py — Unit tests for stability monitoring system
========================================================================
Tests the monthly IC tracking, distribution analysis, and negative IC
month identification functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from stability_monitor import StabilityMonitor, run_stability_monitoring


class TestStabilityMonitor(unittest.TestCase):
    """Test suite for StabilityMonitor class."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic prediction data
        np.random.seed(42)
        
        dates = pd.date_range(start="2022-01-31", periods=12, freq="ME")
        tickers = [f"TICK{i}" for i in range(50)]
        sectors = ["Tech", "Finance", "Healthcare", "Consumer", "Energy"]
        
        data = []
        for date in dates:
            for ticker in tickers:
                # Create predictions with some correlation to actuals
                actual = np.random.normal(0.01, 0.05)
                # Add noise to create imperfect correlation
                predicted = actual + np.random.normal(0, 0.03)
                
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "sector": np.random.choice(sectors),
                    "actual": actual,
                    "predicted": predicted
                })
        
        self.predictions_df = pd.DataFrame(data)
        self.monitor = StabilityMonitor()
    
    def test_compute_monthly_ic(self):
        """Test monthly IC computation."""
        monthly_ic = self.monitor.compute_monthly_ic(self.predictions_df)
        
        # Check that we have IC for all months
        self.assertEqual(len(monthly_ic), 12)
        
        # Check that IC values are in valid range [-1, 1]
        valid_ic = monthly_ic.dropna()
        self.assertTrue(all(valid_ic >= -1))
        self.assertTrue(all(valid_ic <= 1))
        
        # Check that index is dates
        self.assertTrue(all(isinstance(idx, pd.Timestamp) for idx in monthly_ic.index))
        
        print(f"✓ Monthly IC computation: {len(valid_ic)} months computed")
    
    def test_analyze_ic_distribution(self):
        """Test IC distribution analysis."""
        monthly_ic = self.monitor.compute_monthly_ic(self.predictions_df)
        distribution = self.monitor.analyze_ic_distribution(monthly_ic)
        
        # Check that all expected metrics are present
        expected_keys = [
            "mean", "median", "std", "min", "max",
            "p5", "p25", "p75", "p95",
            "win_rate", "ic_ir", "n_months"
        ]
        for key in expected_keys:
            self.assertIn(key, distribution)
        
        # Check that metrics are in valid ranges
        self.assertTrue(-1 <= distribution["mean"] <= 1)
        self.assertTrue(0 <= distribution["std"] <= 2)
        self.assertTrue(0 <= distribution["win_rate"] <= 1)
        self.assertEqual(distribution["n_months"], 12)
        
        print(f"✓ IC distribution: mean={distribution['mean']:.4f}, "
              f"win_rate={distribution['win_rate']:.2%}")
    
    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        monthly_ic = self.monitor.compute_monthly_ic(self.predictions_df)
        win_rate = self.monitor.calculate_win_rate(monthly_ic)
        
        # Check that win rate is in valid range
        self.assertTrue(0 <= win_rate <= 1)
        
        # Check with different threshold
        win_rate_05 = self.monitor.calculate_win_rate(monthly_ic, threshold=0.05)
        self.assertTrue(0 <= win_rate_05 <= 1)
        
        # Win rate with higher threshold should be lower or equal
        self.assertTrue(win_rate_05 <= win_rate)
        
        print(f"✓ Win rate: {win_rate:.2%} (threshold=0.0), "
              f"{win_rate_05:.2%} (threshold=0.05)")
    
    def test_identify_negative_ic_months(self):
        """Test negative IC month identification."""
        monthly_ic = self.monitor.compute_monthly_ic(self.predictions_df)
        negative_months_df = self.monitor.identify_negative_ic_months(
            self.predictions_df, monthly_ic
        )
        
        # Check DataFrame structure
        if len(negative_months_df) > 0:
            expected_cols = [
                "date", "ic", "n_stocks",
                "pred_mean", "pred_std", "actual_mean", "actual_std",
                "worst_sector", "worst_sector_ic", "n_sectors_negative"
            ]
            for col in expected_cols:
                self.assertIn(col, negative_months_df.columns)
            
            # Check that all IC values are negative
            self.assertTrue(all(negative_months_df["ic"] < 0))
            
            # Check that n_stocks is positive
            self.assertTrue(all(negative_months_df["n_stocks"] > 0))
            
            print(f"✓ Negative IC months: {len(negative_months_df)} identified")
        else:
            print("✓ No negative IC months (all months positive)")
    
    def test_generate_stability_report(self):
        """Test comprehensive stability report generation."""
        report = self.monitor.generate_stability_report(
            self.predictions_df,
            save_path=None  # Don't save during test
        )
        
        # Check that all report components are present
        self.assertIn("monthly_ic", report)
        self.assertIn("distribution", report)
        self.assertIn("negative_months", report)
        self.assertIn("monthly_ic_history", report)
        
        # Check monthly_ic is a Series
        self.assertIsInstance(report["monthly_ic"], pd.Series)
        
        # Check distribution is a dict
        self.assertIsInstance(report["distribution"], dict)
        
        # Check negative_months is a DataFrame
        self.assertIsInstance(report["negative_months"], pd.DataFrame)
        
        print(f"✓ Stability report generated successfully")
    
    def test_with_perfect_predictions(self):
        """Test with perfect predictions (IC should be 1.0)."""
        # Create data with perfect predictions
        dates = pd.date_range(start="2022-01-31", periods=6, freq="ME")
        tickers = [f"TICK{i}" for i in range(30)]
        
        data = []
        for date in dates:
            for ticker in tickers:
                actual = np.random.normal(0.01, 0.05)
                predicted = actual  # Perfect prediction
                
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "sector": "Tech",
                    "actual": actual,
                    "predicted": predicted
                })
        
        perfect_df = pd.DataFrame(data)
        monitor = StabilityMonitor()
        monthly_ic = monitor.compute_monthly_ic(perfect_df)
        
        # IC should be very close to 1.0 (allowing for numerical precision)
        valid_ic = monthly_ic.dropna()
        self.assertTrue(all(valid_ic > 0.99))
        
        print(f"✓ Perfect predictions: mean IC={valid_ic.mean():.4f}")
    
    def test_with_random_predictions(self):
        """Test with random predictions (IC should be near 0)."""
        # Create data with random predictions
        dates = pd.date_range(start="2022-01-31", periods=6, freq="ME")
        tickers = [f"TICK{i}" for i in range(30)]
        
        np.random.seed(123)
        data = []
        for date in dates:
            for ticker in tickers:
                actual = np.random.normal(0.01, 0.05)
                predicted = np.random.normal(0.01, 0.05)  # Independent random
                
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "sector": "Tech",
                    "actual": actual,
                    "predicted": predicted
                })
        
        random_df = pd.DataFrame(data)
        monitor = StabilityMonitor()
        monthly_ic = monitor.compute_monthly_ic(random_df)
        
        # IC should be close to 0 (allowing for random variation)
        valid_ic = monthly_ic.dropna()
        mean_ic = valid_ic.mean()
        self.assertTrue(abs(mean_ic) < 0.3)  # Should be near 0
        
        print(f"✓ Random predictions: mean IC={mean_ic:.4f} (expected ~0)")
    
    def test_run_stability_monitoring(self):
        """Test convenience function."""
        report = run_stability_monitoring(
            self.predictions_df,
            save_reports=False
        )
        
        # Check that report is returned
        self.assertIsInstance(report, dict)
        self.assertIn("monthly_ic", report)
        self.assertIn("distribution", report)
        
        print(f"✓ Convenience function works correctly")
    
    def test_regime_detection_disabled(self):
        """Test that regime detection can be disabled."""
        report = self.monitor.generate_stability_report(
            self.predictions_df,
            enable_regime_detection=False,
            save_path=None
        )
        
        # Check that regime data is empty when disabled
        self.assertEqual(len(report.get("regime_data", [])), 0)
        self.assertEqual(len(report.get("regime_performance", {})), 0)
        
        print("✓ Regime detection can be disabled")
    
    def test_detect_market_regimes(self):
        """Test market regime detection."""
        # Create synthetic price data
        dates = pd.date_range(start="2022-01-01", periods=500, freq="D")
        
        # Create price series with different regimes
        prices = []
        base_price = 100
        for i, date in enumerate(dates):
            if i < 100:
                # High volatility period
                change = np.random.normal(0, 0.03)
            elif i < 300:
                # Trending period
                change = 0.001 + np.random.normal(0, 0.005)
            else:
                # Mean reverting period
                change = -0.0005 * (prices[-1] - base_price) / base_price if prices else 0
                change += np.random.normal(0, 0.005)
            
            new_price = (prices[-1] if prices else base_price) * (1 + change)
            prices.append(new_price)
        
        prices_df = pd.DataFrame({"SPY": prices}, index=dates)
        
        # Detect regimes
        regime_df = self.monitor.detect_market_regimes(
            prices_df=prices_df,
            predictions_df=self.predictions_df
        )
        
        # Check that regimes were detected
        self.assertGreater(len(regime_df), 0)
        self.assertIn("regime", regime_df.columns)
        self.assertIn("volatility", regime_df.columns)
        self.assertIn("trend_strength", regime_df.columns)
        self.assertIn("mean_reversion_score", regime_df.columns)
        
        # Check that regime types are valid
        valid_regimes = {"HIGH_VOLATILITY", "TRENDING", "MEAN_REVERTING"}
        detected_regimes = set(regime_df["regime"].unique())
        self.assertTrue(detected_regimes.issubset(valid_regimes))
        
        print(f"✓ Regime detection: {len(regime_df)} months, "
              f"{len(detected_regimes)} regime types")
    
    def test_analyze_performance_by_regime(self):
        """Test regime-specific performance analysis."""
        # Create synthetic regime data
        dates = self.predictions_df["date"].unique()
        regimes = np.random.choice(
            ["HIGH_VOLATILITY", "TRENDING", "MEAN_REVERTING"],
            size=len(dates)
        )
        
        regime_df = pd.DataFrame({
            "date": dates,
            "regime": regimes,
            "volatility": np.random.uniform(0.1, 0.3, len(dates)),
            "trend_strength": np.random.uniform(0, 0.1, len(dates)),
            "mean_reversion_score": np.random.uniform(-0.5, 0.5, len(dates))
        })
        
        # Analyze performance by regime
        regime_performance = self.monitor.analyze_performance_by_regime(
            self.predictions_df,
            regime_df=regime_df
        )
        
        # Check that performance metrics are computed for each regime
        self.assertGreater(len(regime_performance), 0)
        
        for regime, perf in regime_performance.items():
            self.assertIn("mean", perf)
            self.assertIn("std", perf)
            self.assertIn("win_rate", perf)
            self.assertIn("n_months", perf)
            self.assertIn("n_predictions", perf)
            
            # Check that metrics are in valid ranges
            self.assertTrue(-1 <= perf["mean"] <= 1)
            self.assertTrue(0 <= perf["std"] <= 2)
            self.assertTrue(0 <= perf["win_rate"] <= 1)
            self.assertGreater(perf["n_months"], 0)
            self.assertGreater(perf["n_predictions"], 0)
        
        print(f"✓ Regime performance analysis: {len(regime_performance)} regimes")
    
    def test_analyze_regime_transitions(self):
        """Test regime transition impact analysis."""
        # Create synthetic regime data with transitions
        dates = sorted(self.predictions_df["date"].unique())
        regimes = []
        
        # Create regime sequence with transitions
        for i in range(len(dates)):
            if i < 4:
                regimes.append("MEAN_REVERTING")
            elif i < 8:
                regimes.append("HIGH_VOLATILITY")
            else:
                regimes.append("TRENDING")
        
        regime_df = pd.DataFrame({
            "date": dates,
            "regime": regimes,
            "volatility": np.random.uniform(0.1, 0.3, len(dates)),
            "trend_strength": np.random.uniform(0, 0.1, len(dates)),
            "mean_reversion_score": np.random.uniform(-0.5, 0.5, len(dates))
        })
        
        # Analyze transitions
        transitions_df = self.monitor.analyze_regime_transitions(
            self.predictions_df,
            regime_df=regime_df
        )
        
        # Check that transitions were identified
        if len(transitions_df) > 0:
            self.assertIn("transition_type", transitions_df.columns)
            self.assertIn("mean_ic", transitions_df.columns)
            self.assertIn("win_rate", transitions_df.columns)
            
            # Check that transition types are formatted correctly
            for trans_type in transitions_df["transition_type"]:
                self.assertIn("→", trans_type)
            
            print(f"✓ Regime transitions: {len(transitions_df)} transition types")
        else:
            print("✓ Regime transitions: insufficient data for analysis")
    
    def test_compute_rolling_ic(self):
        """Test rolling IC computation."""
        monthly_ic = self.monitor.compute_monthly_ic(self.predictions_df)
        rolling_ic_df = self.monitor.compute_rolling_ic(monthly_ic, windows=[3, 6])
        
        # Check that rolling IC DataFrame is returned
        self.assertIsInstance(rolling_ic_df, pd.DataFrame)
        
        # Check that original IC is included
        self.assertIn("ic", rolling_ic_df.columns)
        
        # Check that rolling metrics are computed for each window
        for window in [3, 6]:
            self.assertIn(f"rolling_{window}m_mean", rolling_ic_df.columns)
            self.assertIn(f"rolling_{window}m_std", rolling_ic_df.columns)
            self.assertIn(f"rolling_{window}m_ir", rolling_ic_df.columns)
            self.assertIn(f"rolling_{window}m_win_rate", rolling_ic_df.columns)
        
        # Check that rolling values are in valid ranges
        for window in [3, 6]:
            rolling_mean = rolling_ic_df[f"rolling_{window}m_mean"].dropna()
            if len(rolling_mean) > 0:
                self.assertTrue(all(rolling_mean >= -1))
                self.assertTrue(all(rolling_mean <= 1))
            
            rolling_win_rate = rolling_ic_df[f"rolling_{window}m_win_rate"].dropna()
            if len(rolling_win_rate) > 0:
                self.assertTrue(all(rolling_win_rate >= 0))
                self.assertTrue(all(rolling_win_rate <= 1))
        
        print(f"✓ Rolling IC: computed for windows [3, 6] months")
    
    def test_compute_ic_autocorrelation(self):
        """Test IC autocorrelation computation."""
        monthly_ic = self.monitor.compute_monthly_ic(self.predictions_df)
        ic_autocorr = self.monitor.compute_ic_autocorrelation(monthly_ic, max_lag=6)
        
        # Check that autocorrelation is computed for all lags
        self.assertIsInstance(ic_autocorr, dict)
        self.assertEqual(len(ic_autocorr), 6)
        
        # Check that all lags are present
        for lag in range(1, 7):
            self.assertIn(lag, ic_autocorr)
        
        # Check that autocorrelation values are in valid range [-1, 1]
        for lag, autocorr in ic_autocorr.items():
            if not np.isnan(autocorr):
                self.assertTrue(-1 <= autocorr <= 1)
        
        print(f"✓ IC autocorrelation: computed for lags 1-6")
    
    def test_check_stability_thresholds(self):
        """Test stability threshold monitoring."""
        monthly_ic = self.monitor.compute_monthly_ic(self.predictions_df)
        stability_check = self.monitor.check_stability_thresholds(
            monthly_ic,
            std_threshold=0.15,
            mean_threshold=0.01,
            win_rate_threshold=0.60
        )
        
        # Check that stability check returns expected structure
        self.assertIsInstance(stability_check, dict)
        self.assertIn("passed", stability_check)
        self.assertIn("flags", stability_check)
        self.assertIn("metrics", stability_check)
        self.assertIn("thresholds", stability_check)
        
        # Check that passed is a boolean
        self.assertIsInstance(stability_check["passed"], bool)
        
        # Check that flags is a list
        self.assertIsInstance(stability_check["flags"], list)
        
        # Check that metrics are present
        self.assertIn("mean_ic", stability_check["metrics"])
        self.assertIn("std_ic", stability_check["metrics"])
        self.assertIn("win_rate", stability_check["metrics"])
        
        # Check that thresholds are present
        self.assertIn("std_threshold", stability_check["thresholds"])
        self.assertIn("mean_threshold", stability_check["thresholds"])
        self.assertIn("win_rate_threshold", stability_check["thresholds"])
        
        # Check that metrics are in valid ranges
        self.assertTrue(-1 <= stability_check["metrics"]["mean_ic"] <= 1)
        self.assertTrue(0 <= stability_check["metrics"]["std_ic"] <= 2)
        self.assertTrue(0 <= stability_check["metrics"]["win_rate"] <= 1)
        
        print(f"✓ Stability thresholds: passed={stability_check['passed']}, "
              f"flags={stability_check['flags']}")
    
    def test_stability_report_includes_rolling_metrics(self):
        """Test that stability report includes rolling metrics."""
        report = self.monitor.generate_stability_report(
            self.predictions_df,
            save_path=None,
            enable_regime_detection=False
        )
        
        # Check that rolling metrics are included
        self.assertIn("rolling_ic", report)
        self.assertIn("ic_autocorrelation", report)
        self.assertIn("stability_check", report)
        
        # Check that rolling_ic is a DataFrame
        self.assertIsInstance(report["rolling_ic"], pd.DataFrame)
        
        # Check that ic_autocorrelation is a dict
        self.assertIsInstance(report["ic_autocorrelation"], dict)
        
        # Check that stability_check is a dict
        self.assertIsInstance(report["stability_check"], dict)
        
        print(f"✓ Stability report includes rolling metrics")
    
    def test_rolling_ic_with_insufficient_data(self):
        """Test rolling IC with insufficient data."""
        # Create data with only 2 months (less than 3-month window)
        dates = pd.date_range(start="2022-01-31", periods=2, freq="ME")
        tickers = [f"TICK{i}" for i in range(30)]
        
        data = []
        for date in dates:
            for ticker in tickers:
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "actual": np.random.normal(0.01, 0.05),
                    "predicted": np.random.normal(0.01, 0.05)
                })
        
        short_df = pd.DataFrame(data)
        monitor = StabilityMonitor()
        monthly_ic = monitor.compute_monthly_ic(short_df)
        rolling_ic_df = monitor.compute_rolling_ic(monthly_ic, windows=[3, 6])
        
        # Should return DataFrame but with NaN for rolling metrics
        self.assertIsInstance(rolling_ic_df, pd.DataFrame)
        
        print("✓ Rolling IC with insufficient data handled correctly")
    
    def test_ic_autocorrelation_with_insufficient_data(self):
        """Test IC autocorrelation with insufficient data."""
        # Create data with only 3 months (less than max_lag + 2)
        dates = pd.date_range(start="2022-01-31", periods=3, freq="ME")
        tickers = [f"TICK{i}" for i in range(30)]
        
        data = []
        for date in dates:
            for ticker in tickers:
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "actual": np.random.normal(0.01, 0.05),
                    "predicted": np.random.normal(0.01, 0.05)
                })
        
        short_df = pd.DataFrame(data)
        monitor = StabilityMonitor()
        monthly_ic = monitor.compute_monthly_ic(short_df)
        ic_autocorr = monitor.compute_ic_autocorrelation(monthly_ic, max_lag=6)
        
        # Should return empty dict
        self.assertEqual(len(ic_autocorr), 0)
        
        print("✓ IC autocorrelation with insufficient data handled correctly")


class TestStabilityMonitorEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["date", "ticker", "actual", "predicted"])
        monitor = StabilityMonitor()
        
        monthly_ic = monitor.compute_monthly_ic(empty_df)
        self.assertEqual(len(monthly_ic), 0)
        
        print("✓ Empty DataFrame handled correctly")
    
    def test_single_month(self):
        """Test with single month of data."""
        data = []
        date = pd.Timestamp("2022-01-31")
        for i in range(50):
            data.append({
                "date": date,
                "ticker": f"TICK{i}",
                "actual": np.random.normal(0.01, 0.05),
                "predicted": np.random.normal(0.01, 0.05)
            })
        
        single_month_df = pd.DataFrame(data)
        monitor = StabilityMonitor()
        monthly_ic = monitor.compute_monthly_ic(single_month_df)
        
        self.assertEqual(len(monthly_ic), 1)
        
        print("✓ Single month handled correctly")
    
    def test_missing_values(self):
        """Test with missing values in predictions/actuals."""
        dates = pd.date_range(start="2022-01-31", periods=6, freq="ME")
        tickers = [f"TICK{i}" for i in range(30)]
        
        data = []
        for date in dates:
            for i, ticker in enumerate(tickers):
                # Introduce some NaN values
                actual = np.random.normal(0.01, 0.05) if i % 5 != 0 else np.nan
                predicted = np.random.normal(0.01, 0.05) if i % 7 != 0 else np.nan
                
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "actual": actual,
                    "predicted": predicted
                })
        
        missing_df = pd.DataFrame(data)
        monitor = StabilityMonitor()
        monthly_ic = monitor.compute_monthly_ic(missing_df)
        
        # Should still compute IC for months with enough valid data
        valid_ic = monthly_ic.dropna()
        self.assertGreater(len(valid_ic), 0)
        
        print(f"✓ Missing values handled: {len(valid_ic)}/{len(monthly_ic)} months valid")
    
    def test_constant_predictions(self):
        """Test with constant predictions (no variance)."""
        dates = pd.date_range(start="2022-01-31", periods=3, freq="ME")
        tickers = [f"TICK{i}" for i in range(30)]
        
        data = []
        for date in dates:
            for ticker in tickers:
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "actual": np.random.normal(0.01, 0.05),
                    "predicted": 0.5  # Constant prediction
                })
        
        constant_df = pd.DataFrame(data)
        monitor = StabilityMonitor()
        monthly_ic = monitor.compute_monthly_ic(constant_df)
        
        # IC should be 0 for constant predictions
        valid_ic = monthly_ic.dropna()
        self.assertTrue(all(abs(valid_ic) < 0.01))
        
        print("✓ Constant predictions handled (IC=0)")


if __name__ == "__main__":
    print("="*70)
    print("   STABILITY MONITOR TEST SUITE")
    print("="*70)
    
    # Run tests
    unittest.main(verbosity=2)

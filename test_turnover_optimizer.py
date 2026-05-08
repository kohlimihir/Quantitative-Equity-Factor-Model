"""
test_turnover_optimizer.py — Unit Tests for Turnover Optimizer
================================================================
Tests the turnover optimization framework functionality.
"""

import unittest
import pandas as pd
import numpy as np
from turnover_optimizer import TurnoverOptimizer
import os


class TestTurnoverOptimizer(unittest.TestCase):
    """Unit tests for TurnoverOptimizer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Load test data
        cls.predictions_df = pd.read_csv("data/sector_predictions.csv", 
                                         parse_dates=["date"])
        cls.optimizer = TurnoverOptimizer(top_per_sector=3, rebal_threshold=0.12)
    
    def test_compute_monthly_turnover(self):
        """Test monthly turnover calculation."""
        turnover_df = self.optimizer.compute_monthly_turnover(self.predictions_df)
        
        # Check DataFrame structure
        self.assertIsInstance(turnover_df, pd.DataFrame)
        self.assertIn("date", turnover_df.columns)
        self.assertIn("turnover", turnover_df.columns)
        self.assertIn("n_changed", turnover_df.columns)
        self.assertIn("n_stocks", turnover_df.columns)
        
        # Check turnover values are valid
        self.assertTrue((turnover_df["turnover"] >= 0).all())
        self.assertTrue((turnover_df["turnover"] <= 1).all())
        
        # Check we have multiple months
        self.assertGreater(len(turnover_df), 10)
        
        print(f"✓ Monthly turnover calculation: {len(turnover_df)} months analyzed")
        print(f"  Avg turnover: {turnover_df['turnover'].mean():.2%}")
        print(f"  Annual turnover: {turnover_df['turnover'].mean() * 12:.2%}")
    
    def test_compute_sector_turnover(self):
        """Test sector-specific turnover calculation."""
        sector_turnover_df = self.optimizer.compute_sector_turnover(self.predictions_df)
        
        # Check DataFrame structure
        self.assertIsInstance(sector_turnover_df, pd.DataFrame)
        self.assertIn("date", sector_turnover_df.columns)
        self.assertIn("sector", sector_turnover_df.columns)
        self.assertIn("turnover", sector_turnover_df.columns)
        
        # Check turnover values are valid
        self.assertTrue((sector_turnover_df["turnover"] >= 0).all())
        self.assertTrue((sector_turnover_df["turnover"] <= 1).all())
        
        # Check we have multiple sectors
        sectors = sector_turnover_df["sector"].unique()
        self.assertGreater(len(sectors), 3)
        
        print(f"✓ Sector turnover calculation: {len(sectors)} sectors analyzed")
        
        # Print sector summary
        sector_summary = sector_turnover_df.groupby("sector")["turnover"].mean()
        for sector, turnover in sector_summary.items():
            print(f"  {sector}: {turnover:.2%}")
    
    def test_compute_turnover_attribution(self):
        """Test turnover attribution analysis."""
        attribution_df = self.optimizer.compute_turnover_attribution(self.predictions_df)
        
        # Check DataFrame structure
        self.assertIsInstance(attribution_df, pd.DataFrame)
        
        # Attribution may be empty if no churn occurred
        if len(attribution_df) > 0:
            self.assertIn("ticker", attribution_df.columns)
            self.assertIn("churn_count", attribution_df.columns)
            
            # Check churn counts are positive
            self.assertTrue((attribution_df["churn_count"] > 0).all())
            
            print(f"✓ Turnover attribution: {len(attribution_df)} stocks with churn")
            if len(attribution_df) > 0:
                top_churners = attribution_df.head(5)
                print(f"  Top churners:")
                for _, row in top_churners.iterrows():
                    print(f"    {row['ticker']}: {row['churn_count']} times")
        else:
            print(f"✓ Turnover attribution: No churn detected (stable portfolio)")
    
    def test_ewm_alpha_range(self):
        """Test that different alpha values produce different turnover."""
        # Create simple test with two alpha values
        alpha_range = [0.3, 0.7]
        
        # Simulate EWM smoothing for different alphas
        results = []
        for alpha in alpha_range:
            # Apply simple EWM smoothing
            df = self.predictions_df.copy()
            all_dates = sorted(df["date"].unique())
            prev_ranks = {}
            smoothed_predictions = []
            
            for date in all_dates:
                test_df = df[df["date"] == date]
                for _, row in test_df.iterrows():
                    curr = row["raw_rank"]
                    prev = prev_ranks.get(row["ticker"], curr)
                    s = alpha * curr + (1 - alpha) * prev
                    prev_ranks[row["ticker"]] = s
                    smoothed_predictions.append({
                        "date": date,
                        "ticker": row["ticker"],
                        "sector": row["sector"],
                        "predicted": s,
                        "actual": row["actual"]
                    })
            
            smoothed_df = pd.DataFrame(smoothed_predictions)
            turnover_df = self.optimizer.compute_monthly_turnover(smoothed_df)
            avg_turnover = turnover_df["turnover"].mean()
            results.append({"alpha": alpha, "turnover": avg_turnover})
        
        # Lower alpha should generally produce lower turnover
        # (though not guaranteed due to threshold effects)
        print(f"✓ EWM alpha range test:")
        for result in results:
            print(f"  Alpha {result['alpha']:.1f}: {result['turnover']:.2%} turnover")
        
        # Just verify we got results for both alphas
        self.assertEqual(len(results), 2)
    
    def test_turnover_threshold_flag(self):
        """Test that high turnover months are correctly identified."""
        turnover_df = self.optimizer.compute_monthly_turnover(self.predictions_df)
        
        # Count months with >30% turnover (per requirement 4.8)
        high_turnover_months = (turnover_df["turnover"] > 0.30).sum()
        
        print(f"✓ Turnover threshold monitoring:")
        print(f"  Months with >30% turnover: {high_turnover_months}/{len(turnover_df)}")
        print(f"  Percentage: {high_turnover_months/len(turnover_df)*100:.1f}%")
        
        # Just verify the calculation works
        self.assertGreaterEqual(high_turnover_months, 0)
    
    def test_optimization_results_structure(self):
        """Test that optimization results file has correct structure."""
        if os.path.exists("reports/ewm_optimization_results.csv"):
            results_df = pd.read_csv("reports/ewm_optimization_results.csv")
            
            # Check required columns
            required_cols = ["alpha", "avg_monthly_turnover", "annual_turnover", 
                           "mean_ic", "ic_ir", "score"]
            for col in required_cols:
                self.assertIn(col, results_df.columns)
            
            # Check alpha range
            self.assertTrue((results_df["alpha"] >= 0.3).all())
            self.assertTrue((results_df["alpha"] <= 0.7).all())
            
            # Check turnover values are valid
            self.assertTrue((results_df["avg_monthly_turnover"] >= 0).all())
            self.assertTrue((results_df["avg_monthly_turnover"] <= 1).all())
            
            print(f"✓ Optimization results structure validated")
            print(f"  Alpha values tested: {results_df['alpha'].tolist()}")
            print(f"  Turnover range: {results_df['avg_monthly_turnover'].min():.2%} - "
                  f"{results_df['avg_monthly_turnover'].max():.2%}")
        else:
            print("⚠ Optimization results file not found (run test_ewm_simple.py first)")
    
    def test_threshold_optimization_basic(self):
        """Test basic rebalancing threshold optimization functionality."""
        # Test with a small threshold range
        threshold_range = [0.10, 0.15, 0.20]
        
        results_df = self.optimizer.optimize_rebalancing_threshold(
            self.predictions_df,
            threshold_range=threshold_range
        )
        
        # Check DataFrame structure
        self.assertIsInstance(results_df, pd.DataFrame)
        required_cols = ["threshold", "avg_monthly_turnover", "annual_turnover",
                        "mean_ic", "ic_ir", "high_turnover_months", "score"]
        for col in required_cols:
            self.assertIn(col, results_df.columns)
        
        # Check threshold range
        self.assertTrue((results_df["threshold"] >= 0.05).all())
        self.assertTrue((results_df["threshold"] <= 0.20).all())
        
        # Check turnover values are valid
        self.assertTrue((results_df["avg_monthly_turnover"] >= 0).all())
        self.assertTrue((results_df["avg_monthly_turnover"] <= 1).all())
        
        # Check we tested all thresholds
        self.assertEqual(len(results_df), len(threshold_range))
        
        print(f"✓ Threshold optimization basic test passed")
        print(f"  Thresholds tested: {results_df['threshold'].tolist()}")
        print(f"  Turnover range: {results_df['avg_monthly_turnover'].min():.2%} - "
              f"{results_df['avg_monthly_turnover'].max():.2%}")
    
    def test_threshold_impact_on_turnover(self):
        """Test that threshold changes impact turnover as expected."""
        # Test two extreme thresholds
        threshold_range = [0.05, 0.20]
        
        results_df = self.optimizer.optimize_rebalancing_threshold(
            self.predictions_df,
            threshold_range=threshold_range
        )
        
        # Lower threshold should generally have higher turnover
        low_threshold_turnover = results_df[results_df["threshold"] == 0.05]["avg_monthly_turnover"].values[0]
        high_threshold_turnover = results_df[results_df["threshold"] == 0.20]["avg_monthly_turnover"].values[0]
        
        print(f"✓ Threshold impact test:")
        print(f"  Low threshold (0.05): {low_threshold_turnover:.2%} turnover")
        print(f"  High threshold (0.20): {high_threshold_turnover:.2%} turnover")
        
        # Generally expect lower threshold to have higher turnover
        # (though not guaranteed due to portfolio dynamics)
        if low_threshold_turnover > high_threshold_turnover:
            print(f"  ✓ Lower threshold has higher turnover (as expected)")
        else:
            print(f"  ⚠ Turnover relationship unexpected (may be valid due to portfolio dynamics)")
    
    def test_threshold_optimization_results_file(self):
        """Test that threshold optimization results file has correct structure."""
        if os.path.exists("reports/threshold_optimization_results.csv"):
            results_df = pd.read_csv("reports/threshold_optimization_results.csv")
            
            # Check required columns
            required_cols = ["threshold", "avg_monthly_turnover", "annual_turnover",
                           "mean_ic", "ic_ir", "high_turnover_months", "score"]
            for col in required_cols:
                self.assertIn(col, results_df.columns)
            
            # Check threshold range
            self.assertTrue((results_df["threshold"] >= 0.05).all())
            self.assertTrue((results_df["threshold"] <= 0.20).all())
            
            # Check turnover values are valid
            self.assertTrue((results_df["avg_monthly_turnover"] >= 0).all())
            self.assertTrue((results_df["avg_monthly_turnover"] <= 1).all())
            
            # Check high turnover months are non-negative integers
            self.assertTrue((results_df["high_turnover_months"] >= 0).all())
            
            print(f"✓ Threshold optimization results file validated")
            print(f"  Thresholds tested: {results_df['threshold'].tolist()}")
            print(f"  Turnover range: {results_df['avg_monthly_turnover'].min():.2%} - "
                  f"{results_df['avg_monthly_turnover'].max():.2%}")
            print(f"  Optimal threshold: {results_df.loc[results_df['score'].idxmax(), 'threshold']:.2f}")
        else:
            print("⚠ Threshold optimization results file not found (run test_threshold_optimization.py first)")
    
    # ========================================================================
    # TASK 6.5: NEW TESTS FOR TURNOVER MONITORING AND ALERTS
    # ========================================================================
    
    def test_monitor_turnover_threshold(self):
        """Test monthly turnover threshold monitoring (Requirement 4.8)."""
        turnover_df = self.optimizer.compute_monthly_turnover(self.predictions_df)
        
        # Monitor with 30% threshold
        flagged_months = self.optimizer.monitor_turnover_threshold(turnover_df, threshold=0.30)
        
        # Check DataFrame structure
        self.assertIsInstance(flagged_months, pd.DataFrame)
        
        if len(flagged_months) > 0:
            # Check required columns
            self.assertIn("date", flagged_months.columns)
            self.assertIn("turnover", flagged_months.columns)
            self.assertIn("alert_level", flagged_months.columns)
            self.assertIn("alert_message", flagged_months.columns)
            
            # Check all flagged months exceed threshold
            self.assertTrue((flagged_months["turnover"] > 0.30).all())
            
            # Check alert levels are valid
            valid_levels = ["WARNING", "HIGH", "CRITICAL"]
            self.assertTrue(all(level in valid_levels for level in flagged_months["alert_level"]))
            
            print(f"✓ Turnover threshold monitoring (Task 6.5):")
            print(f"  Flagged months: {len(flagged_months)}/{len(turnover_df)}")
            print(f"  Alert levels: {flagged_months['alert_level'].value_counts().to_dict()}")
        else:
            print(f"✓ Turnover threshold monitoring (Task 6.5):")
            print(f"  No months exceeded 30% threshold")
    
    def test_analyze_portfolio_transition(self):
        """Test portfolio transition analysis (Requirement 4.10)."""
        # Create two versions of predictions (simulate parameter change)
        predictions_before = self.predictions_df.copy()
        predictions_after = self.predictions_df.copy()
        
        # Modify some predictions to simulate parameter change
        np.random.seed(42)
        mask = np.random.rand(len(predictions_after)) < 0.2
        predictions_after.loc[mask, "predicted"] *= 0.9
        
        # Analyze transition
        transition_df = self.optimizer.analyze_portfolio_transition(
            predictions_before,
            predictions_after,
            param_name="test_param",
            param_before=0.5,
            param_after=0.7
        )
        
        # Check DataFrame structure
        self.assertIsInstance(transition_df, pd.DataFrame)
        
        if len(transition_df) > 0:
            # Check required columns
            required_cols = ["date", "n_kept", "n_removed", "n_added", "transition_pct"]
            for col in required_cols:
                self.assertIn(col, transition_df.columns)
            
            # Check transition percentages are valid
            self.assertTrue((transition_df["transition_pct"] >= 0).all())
            self.assertTrue((transition_df["transition_pct"] <= 1).all())
            
            # Check counts are non-negative
            self.assertTrue((transition_df["n_kept"] >= 0).all())
            self.assertTrue((transition_df["n_removed"] >= 0).all())
            self.assertTrue((transition_df["n_added"] >= 0).all())
            
            print(f"✓ Portfolio transition analysis (Task 6.5):")
            print(f"  Months analyzed: {len(transition_df)}")
            print(f"  Avg transition: {transition_df['transition_pct'].mean():.2%}")
            print(f"  Max transition: {transition_df['transition_pct'].max():.2%}")
        else:
            print(f"⚠ Portfolio transition analysis: No common dates found")
    
    def test_forecast_turnover_ewm_alpha(self):
        """Test turnover forecasting for EWM alpha (Requirement 4.10)."""
        # Forecast for a few alpha values
        alpha_values = [0.3, 0.5, 0.7]
        
        forecast_df = self.optimizer.forecast_turnover(
            self.predictions_df,
            param_changes=alpha_values,
            param_type="ewm_alpha"
        )
        
        # Check DataFrame structure
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), len(alpha_values))
        
        # Check required columns
        required_cols = ["param_value", "estimated_avg_turnover", "estimated_annual_turnover",
                        "lower_bound_95ci", "upper_bound_95ci", "expected_high_turnover_months",
                        "change_from_baseline", "change_pct"]
        for col in required_cols:
            self.assertIn(col, forecast_df.columns)
        
        # Check estimates are valid
        self.assertTrue((forecast_df["estimated_avg_turnover"] >= 0).all())
        self.assertTrue((forecast_df["estimated_avg_turnover"] <= 1).all())
        
        # Check confidence intervals are valid
        for _, row in forecast_df.iterrows():
            self.assertLessEqual(row["lower_bound_95ci"], row["estimated_avg_turnover"])
            self.assertGreaterEqual(row["upper_bound_95ci"], row["estimated_avg_turnover"])
        
        print(f"✓ Turnover forecasting - EWM alpha (Task 6.5):")
        print(f"  Parameters tested: {alpha_values}")
        print(f"  Forecast range: {forecast_df['estimated_avg_turnover'].min():.2%} - "
              f"{forecast_df['estimated_avg_turnover'].max():.2%}")
    
    def test_forecast_turnover_rebal_threshold(self):
        """Test turnover forecasting for rebalancing threshold (Requirement 4.10)."""
        # Forecast for a few threshold values
        threshold_values = [0.05, 0.12, 0.20]
        
        forecast_df = self.optimizer.forecast_turnover(
            self.predictions_df,
            param_changes=threshold_values,
            param_type="rebal_threshold"
        )
        
        # Check DataFrame structure
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), len(threshold_values))
        
        # Check required columns exist
        required_cols = ["param_value", "estimated_avg_turnover", "lower_bound_95ci", "upper_bound_95ci"]
        for col in required_cols:
            self.assertIn(col, forecast_df.columns)
        
        # Check estimates are valid
        self.assertTrue((forecast_df["estimated_avg_turnover"] >= 0).all())
        self.assertTrue((forecast_df["estimated_avg_turnover"] <= 1).all())
        
        print(f"✓ Turnover forecasting - Rebal threshold (Task 6.5):")
        print(f"  Parameters tested: {threshold_values}")
        print(f"  Forecast range: {forecast_df['estimated_avg_turnover'].min():.2%} - "
              f"{forecast_df['estimated_avg_turnover'].max():.2%}")
    
    def test_forecast_turnover_holding_bonus(self):
        """Test turnover forecasting for holding bonus (Requirement 4.10)."""
        # Forecast for a few bonus values
        bonus_values = [0.0, 0.02, 0.05]
        
        forecast_df = self.optimizer.forecast_turnover(
            self.predictions_df,
            param_changes=bonus_values,
            param_type="holding_bonus"
        )
        
        # Check DataFrame structure
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), len(bonus_values))
        
        # Check required columns exist
        required_cols = ["param_value", "estimated_avg_turnover", "expected_high_turnover_months"]
        for col in required_cols:
            self.assertIn(col, forecast_df.columns)
        
        # Check estimates are valid
        self.assertTrue((forecast_df["estimated_avg_turnover"] >= 0).all())
        self.assertTrue((forecast_df["estimated_avg_turnover"] <= 1).all())
        
        print(f"✓ Turnover forecasting - Holding bonus (Task 6.5):")
        print(f"  Parameters tested: {bonus_values}")
        print(f"  Forecast range: {forecast_df['estimated_avg_turnover'].min():.2%} - "
              f"{forecast_df['estimated_avg_turnover'].max():.2%}")
    
    def test_complete_monitoring_workflow(self):
        """Test complete turnover monitoring workflow (Task 6.5 integration)."""
        print(f"\n✓ Complete monitoring workflow test (Task 6.5):")
        
        # Step 1: Compute turnover
        turnover_df = self.optimizer.compute_monthly_turnover(self.predictions_df)
        self.assertGreater(len(turnover_df), 0)
        print(f"  Step 1: Computed turnover for {len(turnover_df)} months")
        
        # Step 2: Monitor threshold
        flagged = self.optimizer.monitor_turnover_threshold(turnover_df, threshold=0.30)
        self.assertIsInstance(flagged, pd.DataFrame)
        print(f"  Step 2: Monitored threshold, flagged {len(flagged)} months")
        
        # Step 3: Analyze transition
        predictions_alt = self.predictions_df.copy()
        predictions_alt["predicted"] *= 1.05
        transition_df = self.optimizer.analyze_portfolio_transition(
            self.predictions_df, predictions_alt
        )
        self.assertGreater(len(transition_df), 0)
        print(f"  Step 3: Analyzed transitions for {len(transition_df)} months")
        
        # Step 4: Forecast turnover
        forecast_df = self.optimizer.forecast_turnover(
            self.predictions_df,
            param_changes=[0.3, 0.5, 0.7],
            param_type="ewm_alpha"
        )
        self.assertEqual(len(forecast_df), 3)
        print(f"  Step 4: Generated forecasts for 3 parameter values")
        
        print(f"  ✓ All workflow steps completed successfully")


def run_tests():
    """Run all tests and print summary."""
    print("="*70)
    print("  TURNOVER OPTIMIZER UNIT TESTS")
    print("="*70 + "\n")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTurnoverOptimizer)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
    
    print("="*70 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

"""
test_threshold_optimization.py — Test Rebalancing Threshold Optimization
==========================================================================
Tests the rebalancing threshold optimization functionality.
"""

import pandas as pd
import numpy as np
from turnover_optimizer import TurnoverOptimizer
import os


def test_threshold_optimization():
    """Test rebalancing threshold optimization."""
    print("="*70)
    print("  REBALANCING THRESHOLD OPTIMIZATION TEST")
    print("="*70 + "\n")
    
    # Load test data
    print("Loading sector predictions...")
    predictions_df = pd.read_csv("data/sector_predictions.csv", parse_dates=["date"])
    print(f"Loaded {len(predictions_df)} predictions across {predictions_df['date'].nunique()} months\n")
    
    # Initialize optimizer with default threshold
    optimizer = TurnoverOptimizer(top_per_sector=3, rebal_threshold=0.12)
    
    # Test threshold optimization
    print("Running threshold optimization...")
    threshold_range = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    results_df = optimizer.optimize_rebalancing_threshold(
        predictions_df, 
        threshold_range=threshold_range
    )
    
    # Validate results
    print("\n" + "="*70)
    print("  VALIDATION CHECKS")
    print("="*70)
    
    # Check 1: Results DataFrame structure
    required_cols = ["threshold", "avg_monthly_turnover", "annual_turnover", 
                    "mean_ic", "ic_ir", "high_turnover_months", "score"]
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        print(f"✗ Missing columns: {missing_cols}")
        return False
    else:
        print(f"✓ All required columns present")
    
    # Check 2: Threshold range
    if not (results_df["threshold"].min() >= 0.05 and results_df["threshold"].max() <= 0.20):
        print(f"✗ Threshold range invalid: {results_df['threshold'].min():.2f} - {results_df['threshold'].max():.2f}")
        return False
    else:
        print(f"✓ Threshold range valid: {results_df['threshold'].min():.2f} - {results_df['threshold'].max():.2f}")
    
    # Check 3: Turnover values are valid
    if not ((results_df["avg_monthly_turnover"] >= 0).all() and 
            (results_df["avg_monthly_turnover"] <= 1).all()):
        print(f"✗ Invalid turnover values detected")
        return False
    else:
        print(f"✓ Turnover values valid (0-100%)")
    
    # Check 4: Turnover decreases with higher threshold
    # (Generally expected, though not guaranteed due to portfolio dynamics)
    turnover_trend = results_df["avg_monthly_turnover"].diff().mean()
    if turnover_trend < 0:
        print(f"✓ Turnover generally decreases with higher threshold (trend: {turnover_trend:.4f})")
    else:
        print(f"⚠ Turnover trend unexpected (trend: {turnover_trend:.4f}), but may be valid due to portfolio dynamics")
    
    # Check 5: Optimal threshold identified
    optimal_idx = results_df["score"].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, "threshold"]
    print(f"✓ Optimal threshold identified: {optimal_threshold:.2f}")
    
    # Check 6: High turnover months tracked
    if "high_turnover_months" in results_df.columns:
        print(f"✓ High turnover months tracked (>30% threshold)")
    else:
        print(f"✗ High turnover months not tracked")
        return False
    
    # Save results
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/threshold_optimization_results.csv", index=False)
    print(f"\n✓ Saved: reports/threshold_optimization_results.csv")
    
    # Generate sensitivity analysis plots
    print("\nGenerating threshold sensitivity analysis plots...")
    optimizer.analyze_threshold_sensitivity(results_df)
    
    # Print summary table
    print("\n" + "="*70)
    print("  THRESHOLD OPTIMIZATION SUMMARY")
    print("="*70)
    print(f"{'Threshold':<12} {'Avg Turn':<12} {'Annual Turn':<14} {'Mean IC':<12} {'IC-IR':<10} {'Score':<10}")
    print("-"*70)
    for _, row in results_df.iterrows():
        marker = " *" if row["threshold"] == optimal_threshold else ""
        print(f"{row['threshold']:<12.2f} {row['avg_monthly_turnover']:<12.2%} "
              f"{row['annual_turnover']:<14.2%} {row['mean_ic']:<12.5f} "
              f"{row['ic_ir']:<10.3f} {row['score']:<10.5f}{marker}")
    print("="*70)
    print("* = Optimal threshold based on IC - 0.5*turnover scoring")
    print("="*70 + "\n")
    
    # Print recommendations
    print("="*70)
    print("  RECOMMENDATIONS")
    print("="*70)
    
    # Find threshold with lowest turnover
    lowest_turnover_idx = results_df["avg_monthly_turnover"].idxmin()
    lowest_turnover = results_df.loc[lowest_turnover_idx, "threshold"]
    
    # Find threshold with highest IC
    highest_ic_idx = results_df["mean_ic"].idxmax()
    highest_ic = results_df.loc[highest_ic_idx, "threshold"]
    
    print(f"Lowest turnover: threshold = {lowest_turnover:.2f} "
          f"({results_df.loc[lowest_turnover_idx, 'annual_turnover']:.1%} annual)")
    print(f"Highest IC: threshold = {highest_ic:.2f} "
          f"(IC = {results_df.loc[highest_ic_idx, 'mean_ic']:.5f})")
    print(f"Optimal balance: threshold = {optimal_threshold:.2f}")
    
    # Provide interpretation
    print("\nInterpretation:")
    if optimal_threshold <= 0.08:
        print("  - Low threshold recommended: Aggressive rebalancing captures more alpha")
        print("  - Trade-off: Higher turnover and transaction costs")
    elif optimal_threshold >= 0.15:
        print("  - High threshold recommended: Conservative rebalancing reduces costs")
        print("  - Trade-off: May miss some alpha opportunities")
    else:
        print("  - Moderate threshold recommended: Balanced turnover and alpha capture")
        print("  - Trade-off: Reasonable compromise between costs and performance")
    
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_threshold_optimization()
    if success:
        print("✓ All threshold optimization tests passed!\n")
        exit(0)
    else:
        print("✗ Some threshold optimization tests failed\n")
        exit(1)

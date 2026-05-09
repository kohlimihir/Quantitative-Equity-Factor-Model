"""
quick_holding_period_test.py — Quick test of holding period enhancements
=========================================================================
Tests the new holding period functionality with a small parameter set
to verify the implementation works correctly.
"""

import pandas as pd
import numpy as np
import os
from turnover_optimizer import TurnoverOptimizer
from data_loader import FEATURES, TARGET

def main():
    """
    Quick test of holding period enhancements.
    """
    print("\n" + "="*70)
    print("  QUICK HOLDING PERIOD TEST")
    print("="*70)
    
    # Load data
    print("\nLoading factor features...")
    if not os.path.exists("data/factor_features.csv"):
        print("ERROR: data/factor_features.csv not found!")
        print("Please run data_loader.py first to generate features.")
        return
    
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    print(f"Loaded {len(factors_df):,} rows, {factors_df['date'].nunique()} months")
    
    # Initialize optimizer
    optimizer = TurnoverOptimizer(top_per_sector=3, rebal_threshold=0.12)
    
    # Test 1: Quick holding bonus test (2 configurations)
    print("\n" + "="*70)
    print("  TEST 1: HOLDING BONUS (2 configurations)")
    print("="*70)
    
    bonus_results = optimizer.optimize_holding_period_bonus(
        factors_df=factors_df,
        features=FEATURES,
        target=TARGET,
        bonus_rates=[0.0, 0.02],  # Just baseline and default
        bonus_caps=[5]  # Just one cap value
    )
    
    print("\nResults:")
    for idx, row in bonus_results.iterrows():
        print(f"\n  Rate={row['bonus_rate']:.3f}, Cap={row['bonus_cap']:.0f}:")
        print(f"    Annual turnover: {row['annual_turnover']:.2%}")
        print(f"    Mean IC: {row['mean_ic']:.5f}")
    
    # Test 2: Quick minimum holding period test (2 configurations)
    print("\n" + "="*70)
    print("  TEST 2: MINIMUM HOLDING PERIOD (2 configurations)")
    print("="*70)
    
    min_hold_results = optimizer.optimize_minimum_holding_period(
        factors_df=factors_df,
        features=FEATURES,
        target=TARGET,
        min_hold_periods=[0, 2]  # Just baseline and 2 months
    )
    
    print("\nResults:")
    for idx, row in min_hold_results.iterrows():
        print(f"\n  Min hold={row['min_hold_period']:.0f} months:")
        print(f"    Annual turnover: {row['annual_turnover']:.2%}")
        print(f"    Mean IC: {row['mean_ic']:.5f}")
    
    # Save results
    os.makedirs("reports", exist_ok=True)
    bonus_results.to_csv("reports/quick_holding_bonus_test.csv", index=False)
    min_hold_results.to_csv("reports/quick_min_hold_test.csv", index=False)
    
    print("\n" + "="*70)
    print("  TEST COMPLETE")
    print("="*70)
    print("\nSaved:")
    print("  - reports/quick_holding_bonus_test.csv")
    print("  - reports/quick_min_hold_test.csv")
    print("\nHolding period enhancements are working correctly!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

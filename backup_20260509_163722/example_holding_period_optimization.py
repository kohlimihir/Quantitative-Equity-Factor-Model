"""
example_holding_period_optimization.py — Holding Period Enhancement Demo
==========================================================================
Demonstrates the new holding period optimization capabilities:

1. Holding-period bonus optimization (configurable rates and caps)
2. Minimum holding period constraint optimization
3. Impact analysis on turnover and IC

This script tests different holding period parameters to find the optimal
balance between turnover reduction and IC preservation.

Requirements: 4.4, 4.6
"""

import pandas as pd
import numpy as np
import os
from turnover_optimizer import TurnoverOptimizer
from data_loader import FEATURES, TARGET

def main():
    """
    Run holding period optimization analysis.
    """
    print("\n" + "="*70)
    print("  HOLDING PERIOD ENHANCEMENT DEMONSTRATION")
    print("="*70)
    print("\nThis demo tests holding period enhancements:")
    print("  1. Configurable holding-period bonuses (rates and caps)")
    print("  2. Minimum holding period constraints")
    print("  3. Impact analysis on turnover and IC")
    print("="*70 + "\n")
    
    # Load data
    print("Loading factor features...")
    if not os.path.exists("data/factor_features.csv"):
        print("ERROR: data/factor_features.csv not found!")
        print("Please run data_loader.py first to generate features.")
        return
    
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    print(f"Loaded {len(factors_df):,} rows, {factors_df['date'].nunique()} months")
    
    # Initialize optimizer
    optimizer = TurnoverOptimizer(top_per_sector=3, rebal_threshold=0.12)
    
    # Test 1: Holding-period bonus optimization
    print("\n" + "="*70)
    print("  TEST 1: HOLDING-PERIOD BONUS OPTIMIZATION")
    print("="*70)
    print("\nTesting different bonus rates and caps to find optimal balance...")
    print("This will take several minutes as it runs walk-forward validation")
    print("for each parameter combination.\n")
    
    bonus_results = optimizer.optimize_holding_period_bonus(
        factors_df=factors_df,
        features=FEATURES,
        target=TARGET,
        bonus_rates=[0.0, 0.01, 0.02, 0.03, 0.05],
        bonus_caps=[3, 5, 8]
    )
    
    # Save results
    os.makedirs("reports", exist_ok=True)
    bonus_results.to_csv("reports/holding_bonus_optimization.csv", index=False)
    print(f"\nSaved: reports/holding_bonus_optimization.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("  HOLDING BONUS OPTIMIZATION SUMMARY")
    print("="*70)
    print("\nTop 5 configurations by score (IC - 0.5*turnover):")
    top5 = bonus_results.nlargest(5, "score")
    for idx, row in top5.iterrows():
        print(f"\n  Rate={row['bonus_rate']:.3f}, Cap={row['bonus_cap']:.0f} months:")
        print(f"    Max bonus: {row['max_bonus']:.3f}")
        print(f"    Annual turnover: {row['annual_turnover']:.2%}")
        print(f"    Mean IC: {row['mean_ic']:.5f}")
        print(f"    IC-IR: {row['ic_ir']:.5f}")
        print(f"    Score: {row['score']:.5f}")
    
    # Test 2: Minimum holding period optimization
    print("\n" + "="*70)
    print("  TEST 2: MINIMUM HOLDING PERIOD OPTIMIZATION")
    print("="*70)
    print("\nTesting different minimum holding periods (hard constraint)...")
    print("This will take several minutes as it runs walk-forward validation")
    print("for each minimum holding period.\n")
    
    min_hold_results = optimizer.optimize_minimum_holding_period(
        factors_df=factors_df,
        features=FEATURES,
        target=TARGET,
        min_hold_periods=[0, 1, 2, 3]
    )
    
    # Save results
    min_hold_results.to_csv("reports/min_holding_period_optimization.csv", index=False)
    print(f"\nSaved: reports/min_holding_period_optimization.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("  MINIMUM HOLDING PERIOD SUMMARY")
    print("="*70)
    print("\nAll configurations:")
    for idx, row in min_hold_results.iterrows():
        print(f"\n  Min hold period: {row['min_hold_period']:.0f} months:")
        print(f"    Annual turnover: {row['annual_turnover']:.2%}")
        print(f"    Mean IC: {row['mean_ic']:.5f}")
        print(f"    IC-IR: {row['ic_ir']:.5f}")
        print(f"    Score: {row['score']:.5f}")
    
    # Test 3: Combined impact analysis
    print("\n" + "="*70)
    print("  TEST 3: COMBINED IMPACT ANALYSIS")
    print("="*70)
    
    # Compare baseline (no bonus, no min hold) vs optimal configurations
    baseline = bonus_results[
        (bonus_results["bonus_rate"] == 0.0) & 
        (bonus_results["bonus_cap"] == 3)
    ].iloc[0]
    
    optimal_bonus = bonus_results.loc[bonus_results["score"].idxmax()]
    optimal_min_hold = min_hold_results.loc[min_hold_results["score"].idxmax()]
    
    print("\nBaseline (no holding enhancements):")
    print(f"  Annual turnover: {baseline['annual_turnover']:.2%}")
    print(f"  Mean IC: {baseline['mean_ic']:.5f}")
    print(f"  IC-IR: {baseline['ic_ir']:.5f}")
    
    print("\nOptimal holding bonus:")
    print(f"  Rate={optimal_bonus['bonus_rate']:.3f}, Cap={optimal_bonus['bonus_cap']:.0f}")
    print(f"  Annual turnover: {optimal_bonus['annual_turnover']:.2%} "
          f"({(optimal_bonus['annual_turnover']-baseline['annual_turnover']):.2%} vs baseline)")
    print(f"  Mean IC: {optimal_bonus['mean_ic']:.5f} "
          f"({optimal_bonus['mean_ic']-baseline['mean_ic']:+.5f} vs baseline)")
    print(f"  IC-IR: {optimal_bonus['ic_ir']:.5f} "
          f"({optimal_bonus['ic_ir']-baseline['ic_ir']:+.5f} vs baseline)")
    
    print("\nOptimal minimum holding period:")
    print(f"  Min hold: {optimal_min_hold['min_hold_period']:.0f} months")
    print(f"  Annual turnover: {optimal_min_hold['annual_turnover']:.2%} "
          f"({(optimal_min_hold['annual_turnover']-baseline['annual_turnover']):.2%} vs baseline)")
    print(f"  Mean IC: {optimal_min_hold['mean_ic']:.5f} "
          f"({optimal_min_hold['mean_ic']-baseline['mean_ic']:+.5f} vs baseline)")
    print(f"  IC-IR: {optimal_min_hold['ic_ir']:.5f} "
          f"({optimal_min_hold['ic_ir']-baseline['ic_ir']:+.5f} vs baseline)")
    
    print("\n" + "="*70)
    print("  RECOMMENDATIONS")
    print("="*70)
    
    # Determine which approach is better
    if optimal_bonus["score"] > optimal_min_hold["score"]:
        print("\nRECOMMENDATION: Use holding-period bonus")
        print(f"  - Set bonus_rate={optimal_bonus['bonus_rate']:.3f}")
        print(f"  - Set bonus_cap={optimal_bonus['bonus_cap']:.0f} months")
        print(f"  - Expected annual turnover: {optimal_bonus['annual_turnover']:.2%}")
        print(f"  - Expected IC: {optimal_bonus['mean_ic']:.5f}")
    else:
        print("\nRECOMMENDATION: Use minimum holding period")
        print(f"  - Set min_hold_period={optimal_min_hold['min_hold_period']:.0f} months")
        print(f"  - Expected annual turnover: {optimal_min_hold['annual_turnover']:.2%}")
        print(f"  - Expected IC: {optimal_min_hold['mean_ic']:.5f}")
    
    print("\nNOTE: You can also combine both approaches for maximum turnover reduction.")
    print("="*70 + "\n")
    
    print("Holding period optimization complete!")
    print("\nGenerated reports:")
    print("  - reports/holding_bonus_optimization.csv")
    print("  - reports/min_holding_period_optimization.csv")


if __name__ == "__main__":
    main()

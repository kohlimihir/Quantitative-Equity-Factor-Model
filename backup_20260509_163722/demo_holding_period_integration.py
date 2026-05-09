"""
demo_holding_period_integration.py — Integration Demo
======================================================
Demonstrates how holding period enhancements integrate with the
existing portfolio construction and turnover analysis system.

This script shows:
1. How to use configurable holding period parameters
2. How to analyze the impact on turnover and IC
3. How to compare different configurations
"""

import pandas as pd
import numpy as np
import os
from sector_neutralisation import walk_forward_sector_neutral, add_sector_features, evaluate_sector_neutral
from turnover_optimizer import TurnoverOptimizer

def main():
    """
    Integration demonstration of holding period enhancements.
    """
    print("\n" + "="*70)
    print("  HOLDING PERIOD INTEGRATION DEMO")
    print("="*70)
    print("\nThis demo shows how holding period enhancements integrate with")
    print("the existing portfolio construction and turnover analysis system.")
    print("="*70 + "\n")
    
    # Load data
    print("Loading factor features...")
    if not os.path.exists("data/factor_features.csv"):
        print("ERROR: data/factor_features.csv not found!")
        print("Please run data_loader.py first.")
        return
    
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    print(f"Loaded {len(factors_df):,} rows, {factors_df['date'].nunique()} months")
    
    # Prepare data
    factors_df, features = add_sector_features(factors_df)
    
    # Configuration 1: Baseline (no holding enhancements)
    print("\n" + "="*70)
    print("  CONFIGURATION 1: BASELINE (No Holding Enhancements)")
    print("="*70)
    
    results_baseline = walk_forward_sector_neutral(
        factors_df, 
        features,
        ewm_alpha=0.5,
        hold_bonus_per_month=0.0,  # No bonus
        hold_bonus_cap=0,
        min_hold_period=0
    )
    
    metrics_baseline = evaluate_sector_neutral(results_baseline)
    
    # Analyze turnover
    optimizer = TurnoverOptimizer(top_per_sector=3, rebal_threshold=0.12)
    turnover_baseline = optimizer.compute_monthly_turnover(results_baseline)
    
    print("\nTurnover Analysis:")
    print(f"  Avg monthly turnover: {turnover_baseline['turnover'].mean():.2%}")
    print(f"  Annual turnover: {turnover_baseline['turnover'].mean() * 12:.2%}")
    
    # Configuration 2: With holding bonus
    print("\n" + "="*70)
    print("  CONFIGURATION 2: WITH HOLDING BONUS")
    print("="*70)
    
    results_bonus = walk_forward_sector_neutral(
        factors_df, 
        features,
        ewm_alpha=0.5,
        hold_bonus_per_month=0.02,  # 2% bonus per month
        hold_bonus_cap=5,            # Cap at 5 months
        min_hold_period=0
    )
    
    metrics_bonus = evaluate_sector_neutral(results_bonus)
    
    # Analyze turnover
    turnover_bonus = optimizer.compute_monthly_turnover(results_bonus)
    
    print("\nTurnover Analysis:")
    print(f"  Avg monthly turnover: {turnover_bonus['turnover'].mean():.2%}")
    print(f"  Annual turnover: {turnover_bonus['turnover'].mean() * 12:.2%}")
    
    # Configuration 3: With minimum holding period
    print("\n" + "="*70)
    print("  CONFIGURATION 3: WITH MINIMUM HOLDING PERIOD")
    print("="*70)
    
    results_min_hold = walk_forward_sector_neutral(
        factors_df, 
        features,
        ewm_alpha=0.5,
        hold_bonus_per_month=0.0,
        hold_bonus_cap=0,
        min_hold_period=2  # Must hold for 2 months
    )
    
    metrics_min_hold = evaluate_sector_neutral(results_min_hold)
    
    # Analyze turnover
    turnover_min_hold = optimizer.compute_monthly_turnover(results_min_hold)
    
    print("\nTurnover Analysis:")
    print(f"  Avg monthly turnover: {turnover_min_hold['turnover'].mean():.2%}")
    print(f"  Annual turnover: {turnover_min_hold['turnover'].mean() * 12:.2%}")
    
    # Configuration 4: Combined (bonus + minimum hold)
    print("\n" + "="*70)
    print("  CONFIGURATION 4: COMBINED (Bonus + Min Hold)")
    print("="*70)
    
    results_combined = walk_forward_sector_neutral(
        factors_df, 
        features,
        ewm_alpha=0.5,
        hold_bonus_per_month=0.02,
        hold_bonus_cap=5,
        min_hold_period=1  # 1 month minimum
    )
    
    metrics_combined = evaluate_sector_neutral(results_combined)
    
    # Analyze turnover
    turnover_combined = optimizer.compute_monthly_turnover(results_combined)
    
    print("\nTurnover Analysis:")
    print(f"  Avg monthly turnover: {turnover_combined['turnover'].mean():.2%}")
    print(f"  Annual turnover: {turnover_combined['turnover'].mean() * 12:.2%}")
    
    # Comparison Summary
    print("\n" + "="*70)
    print("  COMPARISON SUMMARY")
    print("="*70)
    
    configs = [
        ("Baseline", metrics_baseline, turnover_baseline),
        ("Holding Bonus", metrics_bonus, turnover_bonus),
        ("Min Hold Period", metrics_min_hold, turnover_min_hold),
        ("Combined", metrics_combined, turnover_combined)
    ]
    
    print("\n{:<20} {:>12} {:>12} {:>12}".format(
        "Configuration", "Annual Turn", "Mean IC", "IC-IR"
    ))
    print("-" * 70)
    
    for name, metrics, turnover_df in configs:
        annual_turn = turnover_df['turnover'].mean() * 12
        print("{:<20} {:>11.2%} {:>12.5f} {:>12.5f}".format(
            name, annual_turn, metrics['Mean_IC'], metrics['IC_IR']
        ))
    
    # Calculate improvements
    print("\n" + "="*70)
    print("  IMPROVEMENTS VS BASELINE")
    print("="*70)
    
    baseline_turn = turnover_baseline['turnover'].mean() * 12
    baseline_ic = metrics_baseline['Mean_IC']
    
    for name, metrics, turnover_df in configs[1:]:  # Skip baseline
        annual_turn = turnover_df['turnover'].mean() * 12
        turn_change = annual_turn - baseline_turn
        ic_change = metrics['Mean_IC'] - baseline_ic
        
        print(f"\n{name}:")
        print(f"  Turnover change: {turn_change:+.2%} ({turn_change/baseline_turn*100:+.1f}%)")
        print(f"  IC change: {ic_change:+.5f} ({ic_change/baseline_ic*100:+.1f}%)")
    
    # Save results
    os.makedirs("reports", exist_ok=True)
    
    results_baseline.to_csv("reports/holding_demo_baseline.csv", index=False)
    results_bonus.to_csv("reports/holding_demo_bonus.csv", index=False)
    results_min_hold.to_csv("reports/holding_demo_min_hold.csv", index=False)
    results_combined.to_csv("reports/holding_demo_combined.csv", index=False)
    
    turnover_baseline.to_csv("reports/holding_demo_turnover_baseline.csv", index=False)
    turnover_bonus.to_csv("reports/holding_demo_turnover_bonus.csv", index=False)
    turnover_min_hold.to_csv("reports/holding_demo_turnover_min_hold.csv", index=False)
    turnover_combined.to_csv("reports/holding_demo_turnover_combined.csv", index=False)
    
    print("\n" + "="*70)
    print("  DEMO COMPLETE")
    print("="*70)
    print("\nGenerated reports:")
    print("  - reports/holding_demo_*.csv (predictions)")
    print("  - reports/holding_demo_turnover_*.csv (turnover analysis)")
    print("\nKey Takeaways:")
    print("  1. Holding bonuses reduce turnover while maintaining/improving IC")
    print("  2. Minimum holding periods may increase turnover (forced holds)")
    print("  3. Combined approach provides balanced turnover reduction")
    print("  4. All configurations are easily configurable via parameters")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

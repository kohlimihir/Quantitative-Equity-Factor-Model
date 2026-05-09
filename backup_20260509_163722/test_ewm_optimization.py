"""
test_ewm_optimization.py — Test EWM Parameter Optimization
===========================================================
Demonstrates the EWM alpha parameter optimization functionality
of the turnover_optimizer module.
"""

import pandas as pd
import numpy as np
from turnover_optimizer import TurnoverOptimizer

def main():
    print("="*70)
    print("  EWM PARAMETER OPTIMIZATION TEST")
    print("="*70)
    
    # Load factor features
    print("\nLoading factor features...")
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    
    # Load features list
    from data_loader import FEATURES
    
    # Initialize optimizer
    optimizer = TurnoverOptimizer(top_per_sector=3, rebal_threshold=0.12)
    
    # Run EWM alpha optimization
    print("\nRunning EWM alpha optimization...")
    print("This will test alpha values: [0.3, 0.4, 0.5, 0.6, 0.7]")
    print("Lower alpha = more smoothing = less turnover")
    print("Higher alpha = less smoothing = more turnover\n")
    
    optimization_results = optimizer.optimize_ewm_alpha(
        factors_df=factors_df,
        features=FEATURES,
        target="Next_Month_Return",
        alpha_range=[0.3, 0.4, 0.5, 0.6, 0.7]
    )
    
    # Save results
    optimization_results.to_csv("reports/ewm_optimization_results.csv", index=False)
    print(f"Saved: reports/ewm_optimization_results.csv")
    
    # Generate sensitivity analysis plots
    print("\nGenerating parameter sensitivity analysis plots...")
    optimizer.analyze_parameter_sensitivity(optimization_results)
    
    # Print detailed results table
    print("\n" + "="*70)
    print("  DETAILED OPTIMIZATION RESULTS")
    print("="*70)
    print(f"{'Alpha':<8} {'Avg Turn':<12} {'Annual Turn':<14} {'Mean IC':<12} {'IC-IR':<10} {'Score':<10}")
    print("-"*70)
    for _, row in optimization_results.iterrows():
        print(f"{row['alpha']:<8.2f} {row['avg_monthly_turnover']:<12.2%} "
              f"{row['annual_turnover']:<14.2%} {row['mean_ic']:<12.5f} "
              f"{row['ic_ir']:<10.3f} {row['score']:<10.5f}")
    print("="*70)
    
    # Recommendations
    optimal_idx = optimization_results["score"].idxmax()
    optimal_row = optimization_results.loc[optimal_idx]
    
    print("\n" + "="*70)
    print("  RECOMMENDATIONS")
    print("="*70)
    print(f"\nOptimal Alpha: {optimal_row['alpha']:.2f}")
    print(f"  - Balances turnover reduction with IC preservation")
    print(f"  - Expected annual turnover: {optimal_row['annual_turnover']:.2%}")
    print(f"  - Expected mean IC: {optimal_row['mean_ic']:.5f}")
    print(f"  - Expected IC-IR: {optimal_row['ic_ir']:.3f}")
    
    # Compare with baseline (alpha=0.5)
    baseline_idx = optimization_results[optimization_results["alpha"] == 0.5].index[0]
    baseline_row = optimization_results.loc[baseline_idx]
    
    print(f"\nComparison with baseline (alpha=0.5):")
    print(f"  Turnover change: {optimal_row['annual_turnover'] - baseline_row['annual_turnover']:+.2%}")
    print(f"  IC change: {optimal_row['mean_ic'] - baseline_row['mean_ic']:+.5f}")
    print(f"  IC-IR change: {optimal_row['ic_ir'] - baseline_row['ic_ir']:+.3f}")
    
    # Identify tradeoffs
    min_turnover_idx = optimization_results["avg_monthly_turnover"].idxmin()
    max_ic_idx = optimization_results["mean_ic"].idxmax()
    
    print(f"\nTradeoff Analysis:")
    print(f"  Minimum turnover: alpha={optimization_results.loc[min_turnover_idx, 'alpha']:.2f} "
          f"({optimization_results.loc[min_turnover_idx, 'annual_turnover']:.2%} annual, "
          f"IC={optimization_results.loc[min_turnover_idx, 'mean_ic']:.5f})")
    print(f"  Maximum IC: alpha={optimization_results.loc[max_ic_idx, 'alpha']:.2f} "
          f"(IC={optimization_results.loc[max_ic_idx, 'mean_ic']:.5f}, "
          f"{optimization_results.loc[max_ic_idx, 'annual_turnover']:.2%} annual turnover)")
    
    print("\n" + "="*70)
    print("  EWM OPTIMIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - reports/ewm_optimization_results.csv")
    print("  - reports/ewm_parameter_sensitivity.png")
    print("\nReview the sensitivity plots to visualize the turnover vs IC tradeoff.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

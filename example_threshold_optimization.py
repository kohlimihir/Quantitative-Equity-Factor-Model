"""
example_threshold_optimization.py — Rebalancing Threshold Optimization Example
===============================================================================
Demonstrates how to use the rebalancing threshold optimization functionality
to find the optimal threshold that balances turnover reduction with IC preservation.

This example shows:
1. Loading predictions data
2. Running threshold optimization across a range of values
3. Analyzing the turnover vs IC tradeoff
4. Generating sensitivity analysis visualizations
5. Providing actionable recommendations
"""

import pandas as pd
import os
from turnover_optimizer import TurnoverOptimizer


def main():
    """Run threshold optimization example."""
    print("\n" + "="*70)
    print("  REBALANCING THRESHOLD OPTIMIZATION EXAMPLE")
    print("="*70 + "\n")
    
    # Step 1: Load predictions data
    print("Step 1: Loading predictions data...")
    predictions_df = pd.read_csv("data/sector_predictions.csv", parse_dates=["date"])
    print(f"  Loaded {len(predictions_df)} predictions")
    print(f"  Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
    print(f"  Months: {predictions_df['date'].nunique()}")
    print(f"  Stocks: {predictions_df['ticker'].nunique()}")
    print(f"  Sectors: {predictions_df['sector'].nunique()}\n")
    
    # Step 2: Initialize optimizer
    print("Step 2: Initializing turnover optimizer...")
    optimizer = TurnoverOptimizer(
        top_per_sector=3,      # 3 stocks per sector
        rebal_threshold=0.12   # Default 12% rebalancing threshold
    )
    print("  Optimizer configured:")
    print(f"    - Top per sector: {optimizer.top_per_sector}")
    print(f"    - Default rebalancing threshold: {optimizer.rebal_threshold:.2%}\n")
    
    # Step 3: Analyze baseline (current threshold=0.12)
    print("Step 3: Analyzing baseline performance (threshold=0.12)...")
    baseline_turnover = optimizer.compute_monthly_turnover(predictions_df)
    print(f"  Baseline avg monthly turnover: {baseline_turnover['turnover'].mean():.2%}")
    print(f"  Baseline annual turnover: {baseline_turnover['turnover'].mean() * 12:.2%}")
    print(f"  Months with >30% turnover: {(baseline_turnover['turnover'] > 0.30).sum()}\n")
    
    # Step 4: Run threshold optimization
    print("Step 4: Running threshold optimization...")
    print("  Testing thresholds: 0.05, 0.08, 0.10, 0.12, 0.15, 0.20")
    print("  This will test different rebalancing thresholds to find the optimal balance")
    print("  between turnover reduction and IC preservation.\n")
    
    threshold_range = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    results_df = optimizer.optimize_rebalancing_threshold(
        predictions_df,
        threshold_range=threshold_range
    )
    
    # Step 5: Save results
    print("Step 5: Saving optimization results...")
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/threshold_optimization_results.csv", index=False)
    print("  Saved: reports/threshold_optimization_results.csv\n")
    
    # Step 6: Generate sensitivity analysis
    print("Step 6: Generating sensitivity analysis visualizations...")
    optimizer.analyze_threshold_sensitivity(results_df)
    print("  Generated 4 plots showing:")
    print("    1. Turnover vs Threshold")
    print("    2. IC vs Threshold")
    print("    3. Turnover vs IC Tradeoff")
    print("    4. High Turnover Months vs Threshold\n")
    
    # Step 7: Provide recommendations
    print("="*70)
    print("  OPTIMIZATION RESULTS & RECOMMENDATIONS")
    print("="*70 + "\n")
    
    # Find optimal threshold
    optimal_idx = results_df["score"].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, "threshold"]
    
    # Find threshold with lowest turnover
    lowest_turnover_idx = results_df["avg_monthly_turnover"].idxmin()
    lowest_turnover_threshold = results_df.loc[lowest_turnover_idx, "threshold"]
    
    # Find threshold with highest IC
    highest_ic_idx = results_df["mean_ic"].idxmax()
    highest_ic_threshold = results_df.loc[highest_ic_idx, "threshold"]
    
    print("Key Findings:")
    print(f"  1. Optimal threshold (best balance): {optimal_threshold:.2f}")
    print(f"     - Avg monthly turnover: {results_df.loc[optimal_idx, 'avg_monthly_turnover']:.2%}")
    print(f"     - Annual turnover: {results_df.loc[optimal_idx, 'annual_turnover']:.2%}")
    print(f"     - Mean IC: {results_df.loc[optimal_idx, 'mean_ic']:.5f}")
    print(f"     - Months with >30% turnover: {results_df.loc[optimal_idx, 'high_turnover_months']}")
    print()
    
    print(f"  2. Lowest turnover threshold: {lowest_turnover_threshold:.2f}")
    print(f"     - Avg monthly turnover: {results_df.loc[lowest_turnover_idx, 'avg_monthly_turnover']:.2%}")
    print(f"     - Annual turnover: {results_df.loc[lowest_turnover_idx, 'annual_turnover']:.2%}")
    print(f"     - Mean IC: {results_df.loc[lowest_turnover_idx, 'mean_ic']:.5f}")
    print()
    
    print(f"  3. Highest IC threshold: {highest_ic_threshold:.2f}")
    print(f"     - Avg monthly turnover: {results_df.loc[highest_ic_idx, 'avg_monthly_turnover']:.2%}")
    print(f"     - Annual turnover: {results_df.loc[highest_ic_idx, 'annual_turnover']:.2%}")
    print(f"     - Mean IC: {results_df.loc[highest_ic_idx, 'mean_ic']:.5f}")
    print()
    
    # Calculate improvement vs baseline
    baseline_idx = results_df[results_df["threshold"] == 0.12].index[0]
    baseline_turnover_val = results_df.loc[baseline_idx, "avg_monthly_turnover"]
    optimal_turnover_val = results_df.loc[optimal_idx, "avg_monthly_turnover"]
    turnover_reduction = (baseline_turnover_val - optimal_turnover_val) / baseline_turnover_val
    
    print("Improvement vs Baseline (threshold=0.12):")
    if optimal_threshold != 0.12:
        print(f"  - Turnover reduction: {turnover_reduction:.1%}")
        print(f"  - Annual turnover change: {results_df.loc[baseline_idx, 'annual_turnover']:.1%} → "
              f"{results_df.loc[optimal_idx, 'annual_turnover']:.1%}")
        print(f"  - IC change: {results_df.loc[baseline_idx, 'mean_ic']:.5f} → "
              f"{results_df.loc[optimal_idx, 'mean_ic']:.5f}")
    else:
        print(f"  - Current threshold (0.12) is already optimal!")
    print()
    
    # Provide interpretation
    print("Interpretation:")
    if optimal_threshold <= 0.08:
        print("  ✓ Low threshold recommended (aggressive rebalancing)")
        print("    - Benefit: Captures more alpha by quickly replacing underperformers")
        print("    - Cost: Higher turnover and transaction costs")
        print("    - Best for: Low transaction cost environments, high alpha signals")
    elif optimal_threshold >= 0.15:
        print("  ✓ High threshold recommended (conservative rebalancing)")
        print("    - Benefit: Significantly reduces turnover and transaction costs")
        print("    - Cost: May miss some alpha opportunities")
        print("    - Best for: High transaction cost environments, stable portfolios")
    else:
        print("  ✓ Moderate threshold recommended (balanced approach)")
        print("    - Benefit: Reasonable compromise between costs and performance")
        print("    - Cost: Neither maximizes alpha nor minimizes costs")
        print("    - Best for: Most practical scenarios")
    print()
    
    # Provide actionable next steps
    print("Next Steps:")
    print("  1. Review the sensitivity analysis plots in reports/threshold_parameter_sensitivity.png")
    print("  2. Consider your transaction cost assumptions (currently using 0.5 penalty)")
    print("  3. Test the optimal threshold in your production configuration")
    print("  4. Monitor actual turnover and IC after implementing the change")
    print("  5. Consider combining with EWM alpha optimization for further improvements")
    print()
    
    print("="*70 + "\n")
    
    print("✓ Threshold optimization complete!")
    print(f"  Results saved to: reports/threshold_optimization_results.csv")
    print(f"  Plots saved to: reports/threshold_parameter_sensitivity.png\n")


if __name__ == "__main__":
    main()

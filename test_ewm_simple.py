"""
test_ewm_simple.py — Simple EWM Parameter Test
===============================================
Tests EWM parameter optimization using pre-computed predictions
to demonstrate the turnover vs IC tradeoff analysis.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from turnover_optimizer import TurnoverOptimizer

def simulate_ewm_smoothing(predictions_df, alpha):
    """
    Apply EWM smoothing with different alpha values to existing predictions.
    
    This simulates what would happen if we used different alpha values
    in the original model training.
    """
    df = predictions_df.copy()
    all_dates = sorted(df["date"].unique())
    results = []
    prev_ranks = {}
    
    for date in all_dates:
        test_df = df[df["date"] == date]
        
        # Apply EWM smoothing with custom alpha
        smoothed = []
        for _, row in test_df.iterrows():
            curr = row["raw_rank"]  # Use raw_rank as the current signal
            prev = prev_ranks.get(row["ticker"], curr)
            s = alpha * curr + (1 - alpha) * prev
            smoothed.append(s)
            prev_ranks[row["ticker"]] = s
        
        test_df = test_df.copy()
        test_df["smoothed_rank"] = smoothed
        results.append(test_df)
    
    return pd.concat(results, ignore_index=True)

def main():
    print("="*70)
    print("  SIMPLE EWM PARAMETER OPTIMIZATION TEST")
    print("="*70)
    
    # Load existing sector predictions
    print("\nLoading sector predictions...")
    predictions_df = pd.read_csv("data/sector_predictions.csv", parse_dates=["date"])
    
    # Initialize optimizer
    optimizer = TurnoverOptimizer(top_per_sector=3, rebal_threshold=0.12)
    
    # Test different alpha values
    alpha_range = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []
    
    print("\n" + "="*70)
    print("  EWM ALPHA PARAMETER OPTIMIZATION")
    print("="*70)
    print(f"Testing alpha values: {alpha_range}")
    print(f"Alpha interpretation:")
    print(f"  - Lower alpha (0.3): More smoothing, less turnover, potentially lower IC")
    print(f"  - Higher alpha (0.7): Less smoothing, more turnover, potentially higher IC")
    print("="*70 + "\n")
    
    for alpha in alpha_range:
        print(f"Testing alpha = {alpha:.2f}...")
        
        # Apply EWM smoothing with this alpha
        smoothed_df = simulate_ewm_smoothing(predictions_df, alpha)
        smoothed_df["predicted"] = smoothed_df["smoothed_rank"]
        
        # Compute turnover
        turnover_df = optimizer.compute_monthly_turnover(smoothed_df, rank_col="predicted")
        avg_turnover = turnover_df["turnover"].mean()
        max_turnover = turnover_df["turnover"].max()
        annual_turnover = avg_turnover * 12
        
        # Compute IC
        monthly_ic = smoothed_df.groupby("date").apply(
            lambda g: spearmanr(g["actual"], g["predicted"])[0]
            if len(g) > 1 else 0
        )
        mean_ic = monthly_ic.mean()
        ic_std = monthly_ic.std()
        ic_ir = mean_ic / ic_std if ic_std > 0 else 0
        
        results.append({
            "alpha": alpha,
            "avg_monthly_turnover": avg_turnover,
            "max_monthly_turnover": max_turnover,
            "annual_turnover": annual_turnover,
            "mean_ic": mean_ic,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "turnover_ic_ratio": avg_turnover / mean_ic if mean_ic > 0 else np.inf,
        })
        
        print(f"  Avg monthly turnover: {avg_turnover:.2%}")
        print(f"  Annual turnover: {annual_turnover:.2%}")
        print(f"  Mean IC: {mean_ic:.5f}")
        print(f"  IC-IR: {ic_ir:.5f}")
        print(f"  Turnover/IC ratio: {avg_turnover/mean_ic if mean_ic > 0 else np.inf:.2f}\n")
    
    results_df = pd.DataFrame(results)
    
    # Calculate score (maximize IC - turnover_penalty * turnover)
    turnover_penalty = 0.5
    results_df["score"] = results_df["mean_ic"] - turnover_penalty * results_df["avg_monthly_turnover"]
    
    # Save results
    results_df.to_csv("reports/ewm_optimization_results.csv", index=False)
    print(f"Saved: reports/ewm_optimization_results.csv")
    
    # Generate sensitivity analysis plots
    print("\nGenerating parameter sensitivity analysis plots...")
    optimizer.analyze_parameter_sensitivity(results_df)
    
    # Print detailed results table
    print("\n" + "="*70)
    print("  DETAILED OPTIMIZATION RESULTS")
    print("="*70)
    print(f"{'Alpha':<8} {'Avg Turn':<12} {'Annual Turn':<14} {'Mean IC':<12} {'IC-IR':<10} {'Score':<10}")
    print("-"*70)
    for _, row in results_df.iterrows():
        print(f"{row['alpha']:<8.2f} {row['avg_monthly_turnover']:<12.2%} "
              f"{row['annual_turnover']:<14.2%} {row['mean_ic']:<12.5f} "
              f"{row['ic_ir']:<10.3f} {row['score']:<10.5f}")
    print("="*70)
    
    # Recommendations
    optimal_idx = results_df["score"].idxmax()
    optimal_row = results_df.loc[optimal_idx]
    
    print("\n" + "="*70)
    print("  RECOMMENDATIONS")
    print("="*70)
    print(f"\nOptimal Alpha: {optimal_row['alpha']:.2f}")
    print(f"  - Balances turnover reduction with IC preservation")
    print(f"  - Expected annual turnover: {optimal_row['annual_turnover']:.2%}")
    print(f"  - Expected mean IC: {optimal_row['mean_ic']:.5f}")
    print(f"  - Expected IC-IR: {optimal_row['ic_ir']:.3f}")
    
    # Compare with baseline (alpha=0.5)
    baseline_idx = results_df[results_df["alpha"] == 0.5].index[0]
    baseline_row = results_df.loc[baseline_idx]
    
    print(f"\nComparison with baseline (alpha=0.5):")
    print(f"  Turnover change: {optimal_row['annual_turnover'] - baseline_row['annual_turnover']:+.2%}")
    print(f"  IC change: {optimal_row['mean_ic'] - baseline_row['mean_ic']:+.5f}")
    print(f"  IC-IR change: {optimal_row['ic_ir'] - baseline_row['ic_ir']:+.3f}")
    
    # Identify tradeoffs
    min_turnover_idx = results_df["avg_monthly_turnover"].idxmin()
    max_ic_idx = results_df["mean_ic"].idxmax()
    
    print(f"\nTradeoff Analysis:")
    print(f"  Minimum turnover: alpha={results_df.loc[min_turnover_idx, 'alpha']:.2f} "
          f"({results_df.loc[min_turnover_idx, 'annual_turnover']:.2%} annual, "
          f"IC={results_df.loc[min_turnover_idx, 'mean_ic']:.5f})")
    print(f"  Maximum IC: alpha={results_df.loc[max_ic_idx, 'alpha']:.2f} "
          f"(IC={results_df.loc[max_ic_idx, 'mean_ic']:.5f}, "
          f"{results_df.loc[max_ic_idx, 'annual_turnover']:.2%} annual turnover)")
    
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

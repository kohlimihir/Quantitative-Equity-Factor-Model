"""
example_ewm_integration.py — Integration Example for EWM Optimization
======================================================================
Demonstrates how to integrate EWM parameter optimization into the
main model pipeline.
"""

import pandas as pd
from turnover_optimizer import TurnoverOptimizer
from data_loader import FEATURES

def run_ewm_optimization_pipeline():
    """
    Complete pipeline for EWM parameter optimization.
    
    This example shows how to:
    1. Load factor features
    2. Run EWM optimization
    3. Analyze results
    4. Select optimal parameters
    5. Generate reports
    """
    
    print("="*70)
    print("  EWM PARAMETER OPTIMIZATION PIPELINE")
    print("="*70)
    
    # Step 1: Load data
    print("\n[1/5] Loading factor features...")
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    print(f"  Loaded {len(factors_df):,} rows, {factors_df['date'].nunique()} months")
    
    # Step 2: Initialize optimizer
    print("\n[2/5] Initializing turnover optimizer...")
    optimizer = TurnoverOptimizer(
        top_per_sector=3,      # 3 stocks per sector
        rebal_threshold=0.12   # 12% rebalancing threshold
    )
    print("  Optimizer configured:")
    print(f"    - Top per sector: {optimizer.top_per_sector}")
    print(f"    - Rebalancing threshold: {optimizer.rebal_threshold:.2%}")
    
    # Step 3: Analyze baseline (current alpha=0.5)
    print("\n[3/5] Analyzing baseline performance (alpha=0.5)...")
    baseline_predictions = pd.read_csv("data/sector_predictions.csv", 
                                       parse_dates=["date"])
    baseline_turnover = optimizer.compute_monthly_turnover(baseline_predictions)
    
    print(f"  Baseline metrics:")
    print(f"    - Avg monthly turnover: {baseline_turnover['turnover'].mean():.2%}")
    print(f"    - Annual turnover: {baseline_turnover['turnover'].mean() * 12:.2%}")
    print(f"    - Max monthly turnover: {baseline_turnover['turnover'].max():.2%}")
    
    # Step 4: Run optimization (using simplified approach for speed)
    print("\n[4/5] Running EWM parameter optimization...")
    print("  Testing alpha values: [0.3, 0.4, 0.5, 0.6, 0.7]")
    print("  (Using simplified approach with existing predictions)")
    
    # Use simplified optimization with existing predictions
    from scipy.stats import spearmanr
    import numpy as np
    
    alpha_range = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []
    
    for alpha in alpha_range:
        # Simulate EWM smoothing with this alpha
        df = baseline_predictions.copy()
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
        
        # Compute metrics
        turnover_df = optimizer.compute_monthly_turnover(smoothed_df)
        avg_turnover = turnover_df["turnover"].mean()
        annual_turnover = avg_turnover * 12
        
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
            "annual_turnover": annual_turnover,
            "mean_ic": mean_ic,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate score
    turnover_penalty = 0.5
    results_df["score"] = results_df["mean_ic"] - turnover_penalty * results_df["avg_monthly_turnover"]
    
    # Step 5: Analyze results and make recommendations
    print("\n[5/5] Analyzing results and generating recommendations...")
    
    optimal_idx = results_df["score"].idxmax()
    optimal_row = results_df.loc[optimal_idx]
    
    print("\n" + "="*70)
    print("  OPTIMIZATION RESULTS")
    print("="*70)
    
    print("\nDetailed Results:")
    print(f"{'Alpha':<8} {'Monthly Turn':<14} {'Annual Turn':<14} {'IC':<10} {'IC-IR':<10}")
    print("-"*70)
    for _, row in results_df.iterrows():
        marker = " ← OPTIMAL" if row["alpha"] == optimal_row["alpha"] else ""
        print(f"{row['alpha']:<8.2f} {row['avg_monthly_turnover']:<14.2%} "
              f"{row['annual_turnover']:<14.2%} {row['mean_ic']:<10.5f} "
              f"{row['ic_ir']:<10.3f}{marker}")
    
    print("\n" + "="*70)
    print("  RECOMMENDATION")
    print("="*70)
    print(f"\nOptimal Alpha: {optimal_row['alpha']:.2f}")
    print(f"\nExpected Performance:")
    print(f"  - Annual turnover: {optimal_row['annual_turnover']:.2%}")
    print(f"  - Mean IC: {optimal_row['mean_ic']:.5f}")
    print(f"  - IC-IR: {optimal_row['ic_ir']:.3f}")
    
    # Compare with baseline
    baseline_idx = results_df[results_df["alpha"] == 0.5].index[0]
    baseline_row = results_df.loc[baseline_idx]
    
    print(f"\nImprovement vs Baseline (alpha=0.5):")
    turnover_change = optimal_row["annual_turnover"] - baseline_row["annual_turnover"]
    ic_change = optimal_row["mean_ic"] - baseline_row["mean_ic"]
    
    print(f"  - Turnover: {turnover_change:+.2%} "
          f"({'reduction' if turnover_change < 0 else 'increase'})")
    print(f"  - IC: {ic_change:+.5f} "
          f"({'improvement' if ic_change > 0 else 'degradation'})")
    
    # Implementation guidance
    print("\n" + "="*70)
    print("  IMPLEMENTATION GUIDANCE")
    print("="*70)
    print(f"\nTo implement the optimal alpha ({optimal_row['alpha']:.2f}):")
    print(f"\n1. Update sector_neutralisation.py:")
    print(f"   EWM_ALPHA = {optimal_row['alpha']:.2f}  # Changed from 0.50")
    print(f"\n2. Update configs/baseline.json:")
    print(f'   "turnover": {{')
    print(f'     "ewm_smoothing": {{')
    print(f'       "alpha": {optimal_row["alpha"]:.2f}')
    print(f'     }}')
    print(f'   }}')
    print(f"\n3. Re-run the full pipeline:")
    print(f"   python run_all.py")
    print(f"\n4. Validate results:")
    print(f"   - Check that turnover is reduced")
    print(f"   - Verify IC is maintained or improved")
    print(f"   - Monitor transaction costs")
    
    # Save results
    results_df.to_csv("reports/ewm_optimization_results.csv", index=False)
    print(f"\n✓ Saved: reports/ewm_optimization_results.csv")
    
    # Generate plots
    optimizer.analyze_parameter_sensitivity(results_df)
    
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the sensitivity plots: reports/ewm_parameter_sensitivity.png")
    print("  2. Validate the recommendation with out-of-sample data")
    print("  3. Implement the optimal alpha in production")
    print("  4. Monitor performance and adjust if needed")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_ewm_optimization_pipeline()

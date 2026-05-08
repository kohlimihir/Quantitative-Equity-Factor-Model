"""
demo_stability_monitoring.py — Demonstration of Stability Monitoring System
============================================================================
Shows how to use the stability monitor to track model performance consistency
and identify periods of poor performance.

This demo:
1. Loads existing model predictions
2. Computes monthly IC and distribution metrics
3. Identifies negative IC months
4. Generates comprehensive stability report
"""

import pandas as pd
import os

from stability_monitor import StabilityMonitor, run_stability_monitoring


def demo_basic_usage():
    """Demonstrate basic stability monitoring usage."""
    print("\n" + "="*70)
    print("   DEMO: Basic Stability Monitoring")
    print("="*70)
    
    # Load predictions
    predictions_file = "data/sector_predictions.csv"
    if not os.path.exists(predictions_file):
        print(f"\n⚠ Predictions file not found: {predictions_file}")
        print("  Please run the model pipeline first to generate predictions.")
        return
    
    print(f"\nLoading predictions from {predictions_file}...")
    predictions_df = pd.read_csv(predictions_file, parse_dates=["date"])
    print(f"  Loaded {len(predictions_df):,} predictions")
    print(f"  Date range: {predictions_df['date'].min().date()} to {predictions_df['date'].max().date()}")
    print(f"  Months: {predictions_df['date'].nunique()}")
    print(f"  Stocks: {predictions_df['ticker'].nunique()}")
    
    # Run stability monitoring
    print("\n" + "-"*70)
    report = run_stability_monitoring(
        predictions_df,
        pred_col="predicted",
        actual_col="actual",
        save_reports=True
    )
    
    # Access report components
    print("\n" + "-"*70)
    print("Report Components:")
    print(f"  - Monthly IC: {len(report['monthly_ic'])} months")
    print(f"  - Distribution metrics: {len(report['distribution'])} metrics")
    print(f"  - Negative IC months: {len(report['negative_months'])} months")
    
    return report


def demo_custom_analysis():
    """Demonstrate custom stability analysis."""
    print("\n" + "="*70)
    print("   DEMO: Custom Stability Analysis")
    print("="*70)
    
    # Load predictions
    predictions_file = "data/sector_predictions.csv"
    if not os.path.exists(predictions_file):
        print(f"\n⚠ Predictions file not found: {predictions_file}")
        return
    
    predictions_df = pd.read_csv(predictions_file, parse_dates=["date"])
    
    # Create monitor instance
    monitor = StabilityMonitor()
    
    # Step 1: Compute monthly IC
    print("\nStep 1: Computing monthly IC...")
    monthly_ic = monitor.compute_monthly_ic(predictions_df)
    
    # Step 2: Analyze distribution
    print("\nStep 2: Analyzing IC distribution...")
    distribution = monitor.analyze_ic_distribution(monthly_ic)
    
    print(f"\n  Key Metrics:")
    print(f"    Mean IC:     {distribution['mean']:+.5f}")
    print(f"    IC-IR:       {distribution['ic_ir']:.5f}")
    print(f"    Win Rate:    {distribution['win_rate']:.2%}")
    print(f"    Volatility:  {distribution['std']:.5f}")
    
    # Step 3: Calculate win rate with different thresholds
    print("\nStep 3: Win rate at different thresholds...")
    thresholds = [0.0, 0.02, 0.05, 0.10]
    for threshold in thresholds:
        win_rate = monitor.calculate_win_rate(monthly_ic, threshold=threshold)
        print(f"    IC > {threshold:.2f}: {win_rate:.2%}")
    
    # Step 4: Identify negative IC months
    print("\nStep 4: Analyzing negative IC months...")
    negative_months = monitor.identify_negative_ic_months(
        predictions_df, monthly_ic
    )
    
    if len(negative_months) > 0:
        print(f"\n  Worst 5 months:")
        worst_5 = negative_months.nsmallest(5, "ic")
        for _, row in worst_5.iterrows():
            print(f"    {row['date'].strftime('%Y-%m')}: IC={row['ic']:+.4f}, "
                  f"worst sector={row['worst_sector']}")
    
    return monitor


def demo_sector_analysis():
    """Demonstrate sector-specific stability analysis."""
    print("\n" + "="*70)
    print("   DEMO: Sector-Specific Stability")
    print("="*70)
    
    # Load predictions
    predictions_file = "data/sector_predictions.csv"
    if not os.path.exists(predictions_file):
        print(f"\n⚠ Predictions file not found: {predictions_file}")
        return
    
    predictions_df = pd.read_csv(predictions_file, parse_dates=["date"])
    
    # Analyze each sector separately
    sectors = predictions_df["sector"].unique()
    print(f"\nAnalyzing {len(sectors)} sectors...")
    
    sector_results = []
    
    for sector in sorted(sectors):
        sector_data = predictions_df[predictions_df["sector"] == sector].copy()
        
        monitor = StabilityMonitor()
        monthly_ic = monitor.compute_monthly_ic(sector_data)
        distribution = monitor.analyze_ic_distribution(monthly_ic)
        
        sector_results.append({
            "sector": sector,
            "mean_ic": distribution["mean"],
            "ic_ir": distribution["ic_ir"],
            "win_rate": distribution["win_rate"],
            "std_ic": distribution["std"],
            "n_months": distribution["n_months"]
        })
    
    # Display results
    results_df = pd.DataFrame(sector_results)
    results_df = results_df.sort_values("mean_ic", ascending=False)
    
    print("\n  Sector Performance Ranking:")
    print(f"  {'Sector':<15} {'Mean IC':>10} {'IC-IR':>8} {'Win Rate':>10} {'Volatility':>12}")
    print("  " + "-"*60)
    for _, row in results_df.iterrows():
        print(f"  {row['sector']:<15} {row['mean_ic']:>+10.5f} {row['ic_ir']:>8.3f} "
              f"{row['win_rate']:>9.1%} {row['std_ic']:>12.5f}")
    
    # Identify problematic sectors
    print("\n  Sectors needing attention:")
    problematic = results_df[
        (results_df["mean_ic"] < 0.01) | 
        (results_df["win_rate"] < 0.5) |
        (results_df["std_ic"] > 0.15)
    ]
    
    if len(problematic) > 0:
        for _, row in problematic.iterrows():
            issues = []
            if row["mean_ic"] < 0.01:
                issues.append("low IC")
            if row["win_rate"] < 0.5:
                issues.append("low win rate")
            if row["std_ic"] > 0.15:
                issues.append("high volatility")
            print(f"    {row['sector']}: {', '.join(issues)}")
    else:
        print("    None - all sectors performing well!")
    
    return results_df


def demo_time_series_analysis():
    """Demonstrate time series analysis of IC."""
    print("\n" + "="*70)
    print("   DEMO: Time Series IC Analysis")
    print("="*70)
    
    # Load predictions
    predictions_file = "data/sector_predictions.csv"
    if not os.path.exists(predictions_file):
        print(f"\n⚠ Predictions file not found: {predictions_file}")
        return
    
    predictions_df = pd.read_csv(predictions_file, parse_dates=["date"])
    
    monitor = StabilityMonitor()
    monthly_ic = monitor.compute_monthly_ic(predictions_df)
    
    # Compute rolling IC using new method
    print("\n" + "-"*70)
    print("Rolling IC Analysis (using new compute_rolling_ic method):")
    rolling_ic_df = monitor.compute_rolling_ic(monthly_ic, windows=[3, 6])
    
    # Display latest rolling metrics
    if len(rolling_ic_df) > 0:
        latest = rolling_ic_df.iloc[-1]
        print(f"\n  Latest Rolling Metrics:")
        print(f"    Current IC:           {latest['ic']:+.5f}")
        
        if 'rolling_3m_mean' in rolling_ic_df.columns:
            print(f"    3-month rolling mean: {latest['rolling_3m_mean']:+.5f}")
            print(f"    3-month rolling std:  {latest['rolling_3m_std']:.5f}")
            print(f"    3-month IC-IR:        {latest['rolling_3m_ir']:.5f}")
            print(f"    3-month win rate:     {latest['rolling_3m_win_rate']:.2%}")
        
        if 'rolling_6m_mean' in rolling_ic_df.columns:
            print(f"\n    6-month rolling mean: {latest['rolling_6m_mean']:+.5f}")
            print(f"    6-month rolling std:  {latest['rolling_6m_std']:.5f}")
            print(f"    6-month IC-IR:        {latest['rolling_6m_ir']:.5f}")
            print(f"    6-month win rate:     {latest['rolling_6m_win_rate']:.2%}")
    
    # Compute IC autocorrelation using new method
    print("\n" + "-"*70)
    print("IC Autocorrelation Analysis (using new compute_ic_autocorrelation method):")
    ic_autocorr = monitor.compute_ic_autocorrelation(monthly_ic, max_lag=6)
    
    # Check stability thresholds using new method
    print("\n" + "-"*70)
    print("Stability Threshold Monitoring (using new check_stability_thresholds method):")
    stability_check = monitor.check_stability_thresholds(
        monthly_ic,
        std_threshold=0.15,
        mean_threshold=0.01,
        win_rate_threshold=0.60
    )
    
    # Identify trends
    print("\n" + "-"*70)
    print("IC Trend Analysis:")
    recent_6m = monthly_ic.tail(6).mean() if len(monthly_ic) >= 6 else None
    earlier_6m = monthly_ic.iloc[-12:-6].mean() if len(monthly_ic) >= 12 else None
    
    if recent_6m is not None and earlier_6m is not None:
        trend = recent_6m - earlier_6m
        print(f"  Recent 6 months:  {recent_6m:+.5f}")
        print(f"  Previous 6 months: {earlier_6m:+.5f}")
        print(f"  Trend:            {trend:+.5f} ({'improving' if trend > 0 else 'declining'})")
    
    # Identify streaks
    print("\nIC Streaks:")
    positive_streak = 0
    negative_streak = 0
    current_streak = 0
    
    for ic in reversed(monthly_ic.dropna().values):
        if ic > 0:
            if current_streak >= 0:
                current_streak += 1
            else:
                break
        else:
            if current_streak <= 0:
                current_streak -= 1
            else:
                break
    
    if current_streak > 0:
        print(f"  Current positive streak: {current_streak} months")
    elif current_streak < 0:
        print(f"  Current negative streak: {abs(current_streak)} months")
    else:
        print(f"  No current streak")
    
    return rolling_ic_df, ic_autocorr, stability_check


if __name__ == "__main__":
    print("\n" + "="*70)
    print("   STABILITY MONITORING DEMONSTRATION")
    print("="*70)
    
    # Run all demos
    try:
        # Demo 1: Basic usage
        report = demo_basic_usage()
        
        # Demo 2: Custom analysis
        monitor = demo_custom_analysis()
        
        # Demo 3: Sector analysis
        sector_results = demo_sector_analysis()
        
        # Demo 4: Time series analysis
        rolling_results = demo_time_series_analysis()
        
        print("\n" + "="*70)
        print("   DEMONSTRATION COMPLETE")
        print("="*70)
        print("\nAll stability monitoring features demonstrated successfully!")
        print("\nNew Rolling Metrics Features:")
        print("  ✓ Rolling 3-month and 6-month IC calculations")
        print("  ✓ IC autocorrelation measurement for signal persistence")
        print("  ✓ Stability threshold monitoring (flags IC std dev >0.15)")
        print("\nGenerated reports:")
        print("  - reports/stability_report_monthly_ic.csv")
        print("  - reports/stability_report_distribution.csv")
        print("  - reports/stability_report_negative_months.csv")
        print("  - reports/stability_report_rolling_ic.csv (NEW)")
        print("  - reports/stability_report_ic_autocorrelation.csv (NEW)")
        print("  - reports/stability_report_stability_check.csv (NEW)")
        
    except Exception as e:
        print(f"\n⚠ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

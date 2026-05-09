"""
test_optimized_turnover.py - Quick test of optimized turnover parameters
===========================================================================
Tests the combination of:
- EWM alpha = 0.7 (more smoothing)
- Rebalancing threshold = 0.25 (higher barrier)
- Holding bonus = 0.03/month (stronger incumbent advantage)
- Min hold period = 3 months (forced holding)
- Top 4 per sector (wider portfolio)
"""

import pandas as pd
import numpy as np
from sector_neutralisation import (
    add_sector_features, walk_forward_sector_neutral,
    evaluate_sector_neutral, build_sector_aware_portfolio
)

print("="*70)
print("  OPTIMIZED TURNOVER PARAMETER TEST")
print("="*70)
print("\nOptimized Parameters:")
print("  - EWM Alpha: 0.70 (vs 0.50 baseline)")
print("  - Rebalancing Threshold: 0.25 (vs 0.12 baseline)")
print("  - Holding Bonus: 0.03/month (vs 0.02 baseline)")
print("  - Min Hold Period: 3 months (vs 0 baseline)")
print("  - Top per Sector: 4 (vs 3 baseline)")
print("="*70)

# Load data
print("\n[1/5] Loading factor features...")
factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
print(f"  Loaded {len(factors_df):,} rows, {factors_df['date'].nunique()} months")

# Add sector features
print("\n[2/5] Preparing sector features...")
factors_df_sector, z_features = add_sector_features(factors_df)

# Run walk-forward with optimized parameters
print("\n[3/5] Running walk-forward with optimized parameters...")
print("  This will take a few minutes...")
sector_results = walk_forward_sector_neutral(
    factors_df_sector,
    z_features,
    min_train_months=24,
    ewm_alpha=0.70,              # Optimized
    hold_bonus_per_month=0.03,   # Optimized
    hold_bonus_cap=5,
    min_hold_period=3            # Optimized
)

# Save results
sector_results.to_csv("data/sector_predictions_optimized.csv", index=False)
print(f"\n  ✓ Saved predictions: data/sector_predictions_optimized.csv")

# Evaluate
print("\n[4/5] Evaluating model performance...")
metrics = evaluate_sector_neutral(
    sector_results,
    old_path="data/lgbm_predictions.csv"
)

# Build portfolio with optimized parameters
print("\n[5/5] Building portfolio with optimized parameters...")
sector_port = build_sector_aware_portfolio(
    sector_results,
    top_per_sector=4,            # Optimized
    rebal_threshold=0.25         # Optimized
)

# Calculate turnover
print("\n" + "="*70)
print("  TURNOVER ANALYSIS")
print("="*70)

turnover_list = []
prev_held = set()

for date, g in sector_results.groupby("date"):
    # Get current holdings (top 4 per sector)
    current_held = set()
    for sector, sg in g.groupby("sector"):
        top = sg.nlargest(4, "predicted")["ticker"].values
        current_held.update(top)
    
    if prev_held:
        sold = prev_held - current_held
        bought = current_held - prev_held
        turnover = len(sold) / len(prev_held) if prev_held else 0
        turnover_list.append({
            "date": date,
            "stocks_sold": len(sold),
            "stocks_bought": len(bought),
            "n_changed": len(sold),
            "n_stocks": len(current_held),
            "turnover": turnover
        })
    
    prev_held = current_held

turnover_df = pd.DataFrame(turnover_list)
turnover_df.to_csv("reports/turnover_optimized.csv", index=False)

avg_monthly_turnover = turnover_df["turnover"].mean()
annual_turnover = avg_monthly_turnover * 12
max_monthly_turnover = turnover_df["turnover"].max()
months_above_30 = (turnover_df["turnover"] > 0.30).sum()

print(f"\nTurnover Metrics:")
print(f"  Avg Monthly Turnover: {avg_monthly_turnover:.1%}")
print(f"  Annual Turnover: {annual_turnover:.0%}")
print(f"  Max Monthly Turnover: {max_monthly_turnover:.1%}")
print(f"  Months >30% turnover: {months_above_30}/{len(turnover_df)}")

# Performance metrics
rf = 0.045 / 12
sr = sector_port["portfolio_return"]
sharpe = ((sr-rf).mean()*12) / (sr.std()*np.sqrt(12))
cum = (1+sr).prod() - 1
mdd = ((1+sr).cumprod()/(1+sr).cumprod().cummax()-1).min()

print(f"\nPortfolio Performance:")
print(f"  Sharpe Ratio: {sharpe:.3f}")
print(f"  Cumulative Return: {cum:.2%}")
print(f"  Maximum Drawdown: {mdd:.2%}")

# Comparison with baseline
print("\n" + "="*70)
print("  COMPARISON WITH BASELINE")
print("="*70)

baseline_turnover_df = pd.read_csv("reports/monthly_turnover.csv")
baseline_avg = baseline_turnover_df["turnover"].mean()
baseline_annual = baseline_avg * 12

print(f"\nTurnover:")
print(f"  Baseline: {baseline_annual:.0%} annually ({baseline_avg:.1%} monthly)")
print(f"  Optimized: {annual_turnover:.0%} annually ({avg_monthly_turnover:.1%} monthly)")
print(f"  Reduction: {baseline_annual - annual_turnover:.0%} ({(baseline_annual - annual_turnover)/baseline_annual*100:.1f}%)")

# Target assessment
target_annual = 200  # 200% target
print(f"\nTarget Assessment:")
print(f"  Target: {target_annual}% annually")
print(f"  Current: {annual_turnover:.0%} annually")
if annual_turnover <= target_annual:
    print(f"  Status: ✓ TARGET MET!")
else:
    print(f"  Status: ⚠ Above target by {annual_turnover - target_annual:.0%}")

print("\n" + "="*70)
print("  TEST COMPLETE")
print("="*70)
print("\nSaved files:")
print("  - data/sector_predictions_optimized.csv")
print("  - reports/turnover_optimized.csv")
print("="*70)

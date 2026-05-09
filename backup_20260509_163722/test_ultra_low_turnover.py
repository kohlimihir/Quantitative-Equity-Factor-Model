"""
test_ultra_low_turnover.py - Ultra-aggressive turnover reduction test
=======================================================================
Tests extreme parameters to achieve <200% annual turnover:
- EWM alpha = 0.85 (very high smoothing)
- Rebalancing threshold = 0.35 (very high barrier)
- Holding bonus = 0.05/month (very strong incumbent advantage)
- Min hold period = 6 months (long forced holding)
- Top 3 per sector (narrower portfolio for stability)
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import FEATURES, TARGET, SECTOR_MAP
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# Ultra-aggressive parameters
ULTRA_EWM_ALPHA = 0.85
ULTRA_REBAL_THRESHOLD = 0.35
ULTRA_HOLD_BONUS = 0.05
ULTRA_MIN_HOLD = 6
ULTRA_TOP_PER_SECTOR = 3

print("="*70)
print("  ULTRA-LOW TURNOVER TEST (<200% TARGET)")
print("="*70)
print("\nUltra-Aggressive Parameters:")
print(f"  - EWM Alpha: {ULTRA_EWM_ALPHA} (vs 0.70 optimized, 0.50 baseline)")
print(f"  - Rebalancing Threshold: {ULTRA_REBAL_THRESHOLD} (vs 0.25 optimized, 0.12 baseline)")
print(f"  - Holding Bonus: {ULTRA_HOLD_BONUS}/month (vs 0.03 optimized, 0.02 baseline)")
print(f"  - Min Hold Period: {ULTRA_MIN_HOLD} months (vs 3 optimized, 0 baseline)")
print(f"  - Top per Sector: {ULTRA_TOP_PER_SECTOR} (vs 4 optimized, 3 baseline)")
print("="*70)

# Load data
print("\n[1/4] Loading data...")
factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
print(f"  Loaded {len(factors_df):,} rows, {factors_df['date'].nunique()} months")

# Add sector column
if "sector" not in factors_df.columns:
    factors_df["sector"] = factors_df["ticker"].map(SECTOR_MAP)
factors_df = factors_df.dropna(subset=["sector"])

# Walk-forward with ultra-aggressive parameters
print("\n[2/4] Running walk-forward with ultra-aggressive parameters...")
print("  This will take a few minutes...")

all_dates = sorted(factors_df["date"].unique())
results = []
prev_ranks = {}
hold_tenure = {}
hold_start_date = {}
prev_held_all = set()

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 80,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.2,
    "reg_lambda": 0.2,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for i, test_date in enumerate(all_dates):
    if i < 24:  # min_train_months
        continue
    
    train_df = factors_df[factors_df["date"].isin(all_dates[:i])]
    test_df = factors_df[factors_df["date"] == test_date]
    
    if len(train_df) < 200 or len(test_df) == 0:
        continue
    
    # Prepare features
    X_train_df = train_df[FEATURES].copy()
    for col in FEATURES:
        med = X_train_df[col].median()
        X_train_df[col] = X_train_df[col].fillna(med)
    X_train_df = X_train_df.fillna(0)
    
    X_test_df = test_df[FEATURES].copy()
    for col in FEATURES:
        med = X_test_df[col].median()
        X_test_df[col] = X_test_df[col].fillna(med)
    X_test_df = X_test_df.fillna(0)
    
    X_train = X_train_df.values
    y_train = train_df[TARGET].values
    X_test = X_test_df.values
    y_test = test_df[TARGET].values
    
    # Train model
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train)
    raw_scores = model.predict(X_test)
    
    temp = test_df.copy()
    temp["raw_score"] = raw_scores
    
    # Within-sector percentile rank
    temp["raw_rank"] = temp.groupby("sector")["raw_score"].rank(pct=True)
    
    # Ultra-aggressive EWM smoothing
    smoothed = []
    for _, row in temp.iterrows():
        curr = row["raw_rank"]
        prev = prev_ranks.get(row["ticker"], curr)
        s = ULTRA_EWM_ALPHA * prev + (1 - ULTRA_EWM_ALPHA) * curr  # Note: reversed for more smoothing
        
        # Ultra-strong holding bonus
        ticker = row["ticker"]
        if ticker in prev_held_all:
            tenure = min(hold_tenure.get(ticker, 0) + 1, 10)  # Higher cap
            s += tenure * ULTRA_HOLD_BONUS
            hold_tenure[ticker] = tenure
        else:
            hold_tenure[ticker] = 0
        
        smoothed.append(s)
        prev_ranks[row["ticker"]] = s
    
    temp["smoothed_rank"] = smoothed
    
    # Ultra-long minimum holding period
    if ULTRA_MIN_HOLD > 0:
        for ticker in prev_held_all:
            if ticker in hold_start_date:
                start_idx = all_dates.index(hold_start_date[ticker])
                current_idx = all_dates.index(test_date)
                months_held = current_idx - start_idx
                
                if months_held < ULTRA_MIN_HOLD:
                    ticker_rows = temp[temp["ticker"] == ticker]
                    if not ticker_rows.empty:
                        current_rank = temp.loc[temp["ticker"] == ticker, "smoothed_rank"].values[0]
                        temp.loc[temp["ticker"] == ticker, "smoothed_rank"] = current_rank + 20.0  # Very high boost
    
    # Track selected stocks
    month_selected = set()
    for sector, sg in temp.groupby("sector"):
        top = sg.nlargest(ULTRA_TOP_PER_SECTOR, "smoothed_rank")["ticker"].values
        month_selected.update(top)
    
    # Update hold start dates
    for ticker in month_selected:
        if ticker not in prev_held_all:
            hold_start_date[ticker] = test_date
    
    for ticker in list(hold_start_date.keys()):
        if ticker not in month_selected:
            del hold_start_date[ticker]
    
    prev_held_all = month_selected
    
    for _, row in temp.iterrows():
        results.append({
            "date": test_date,
            "ticker": row["ticker"],
            "sector": row["sector"],
            "actual": row[TARGET],
            "predicted": row["smoothed_rank"],
            "raw_rank": row["raw_rank"],
            "raw_score": row["raw_score"],
        })

results_df = pd.DataFrame(results)
print(f"  Walk-forward complete: {len(results_df):,} predictions, {results_df['date'].nunique()} months")

# Evaluate IC
print("\n[3/4] Evaluating performance...")
monthly_ic = results_df.groupby("date").apply(
    lambda g: g["actual"].corr(g["predicted"], method="spearman")
)
mean_ic = monthly_ic.mean()
ic_ir = mean_ic / monthly_ic.std() if monthly_ic.std() > 0 else 0

print(f"  Mean IC: {mean_ic:.5f}")
print(f"  IC-IR: {ic_ir:.5f}")
print(f"  Positive IC months: {(monthly_ic > 0).sum()}/{len(monthly_ic)}")

# Build portfolio with ultra-aggressive threshold
print("\n[4/4] Building portfolio and calculating turnover...")
portfolio_rets = []
prev_held = {}
turnover_list = []

for date, g in results_df.groupby("date"):
    month_portfolio = []
    month_selected_all = set()
    
    for sector, sg in g.groupby("sector"):
        sg_sorted = sg.sort_values("predicted", ascending=False)
        naive_top = set(sg_sorted.head(ULTRA_TOP_PER_SECTOR)["ticker"].values)
        held = prev_held.get(sector, naive_top)
        final = set()
        
        # Ultra-high rebalancing threshold
        for h in held:
            h_row = sg_sorted[sg_sorted["ticker"] == h]
            if h_row.empty:
                continue
            h_rank = h_row["predicted"].values[0]
            chall = sg_sorted[~sg_sorted["ticker"].isin(held)]
            if chall.empty or chall["predicted"].max() - h_rank <= ULTRA_REBAL_THRESHOLD:
                final.add(h)
        
        remaining = ULTRA_TOP_PER_SECTOR - len(final)
        if remaining > 0:
            others = sg_sorted[~sg_sorted["ticker"].isin(final)]
            final.update(others.head(remaining)["ticker"].values)
        if len(final) < ULTRA_TOP_PER_SECTOR:
            final = naive_top
        
        sel = g[g["ticker"].isin(final)]
        month_portfolio.extend(sel["actual"].values)
        prev_held[sector] = final
        month_selected_all.update(final)
    
    # Calculate turnover
    prev_all = set()
    for s in prev_held.values():
        prev_all.update(s)
    
    if prev_all:  # Only calculate if we have previous holdings
        sold = prev_all - month_selected_all
        bought = month_selected_all - prev_all
        turnover = len(sold) / len(prev_all) if prev_all else 0
        
        turnover_list.append({
            "date": date,
            "stocks_sold": len(sold),
            "stocks_bought": len(bought),
            "n_changed": len(sold),
            "n_stocks": len(month_selected_all),
            "turnover": turnover
        })
    
    portfolio_rets.append({
        "date": date,
        "portfolio_return": np.mean(month_portfolio) if month_portfolio else 0,
        "n_stocks": len(month_portfolio),
    })

# Results
turnover_df = pd.DataFrame(turnover_list)
port_df = pd.DataFrame(portfolio_rets).set_index("date")

avg_monthly_turnover = turnover_df["turnover"].mean()
annual_turnover = avg_monthly_turnover * 12
max_monthly_turnover = turnover_df["turnover"].max()
months_above_30 = (turnover_df["turnover"] > 0.30).sum()

rf = 0.045 / 12
sr = port_df["portfolio_return"]
sharpe = ((sr-rf).mean()*12) / (sr.std()*np.sqrt(12))
cum = (1+sr).prod() - 1
mdd = ((1+sr).cumprod()/(1+sr).cumprod().cummax()-1).min()

print("\n" + "="*70)
print("  RESULTS")
print("="*70)

print(f"\nTurnover Metrics:")
print(f"  Avg Monthly Turnover: {avg_monthly_turnover:.1%}")
print(f"  Annual Turnover: {annual_turnover:.0%}")
print(f"  Max Monthly Turnover: {max_monthly_turnover:.1%}")
print(f"  Months >30% turnover: {months_above_30}/{len(turnover_df)}")

print(f"\nPortfolio Performance:")
print(f"  Mean IC: {mean_ic:.5f}")
print(f"  IC-IR: {ic_ir:.5f}")
print(f"  Sharpe Ratio: {sharpe:.3f}")
print(f"  Cumulative Return: {cum:.2%}")
print(f"  Maximum Drawdown: {mdd:.2%}")

# Comparison
print("\n" + "="*70)
print("  COMPARISON")
print("="*70)

baseline_turnover_df = pd.read_csv("reports/monthly_turnover.csv")
baseline_avg = baseline_turnover_df["turnover"].mean()
baseline_annual = baseline_avg * 12

optimized_turnover_df = pd.read_csv("reports/turnover_optimized.csv")
optimized_avg = optimized_turnover_df["turnover"].mean()
optimized_annual = optimized_avg * 12

print(f"\nTurnover Evolution:")
print(f"  Baseline:    {baseline_annual:.0%} annually ({baseline_avg:.1%} monthly)")
print(f"  Optimized:   {optimized_annual:.0%} annually ({optimized_avg:.1%} monthly)")
print(f"  Ultra-Low:   {annual_turnover:.0%} annually ({avg_monthly_turnover:.1%} monthly)")
print(f"  Total Reduction: {baseline_annual - annual_turnover:.0%} ({(baseline_annual - annual_turnover)/baseline_annual*100:.1f}%)")

target_annual = 200
print(f"\nTarget Assessment:")
print(f"  Target: {target_annual}% annually")
print(f"  Current: {annual_turnover:.0%} annually")
if annual_turnover <= target_annual:
    print(f"  Status: ✓✓✓ TARGET MET! ✓✓✓")
    print(f"  Below target by: {target_annual - annual_turnover:.0%}")
else:
    print(f"  Status: ⚠ Still above target by {annual_turnover - target_annual:.0%}")
    print(f"  Progress: {(baseline_annual - annual_turnover)/(baseline_annual - target_annual)*100:.1f}% of the way to target")

# Save results
turnover_df.to_csv("reports/turnover_ultra_low.csv", index=False)
results_df.to_csv("data/sector_predictions_ultra_low.csv", index=False)

print("\n" + "="*70)
print("  TEST COMPLETE")
print("="*70)
print("\nSaved files:")
print("  - data/sector_predictions_ultra_low.csv")
print("  - reports/turnover_ultra_low.csv")
print("="*70)

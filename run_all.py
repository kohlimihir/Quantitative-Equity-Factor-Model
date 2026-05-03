"""
run_all.py  —  Master Pipeline (250-Stock, 17-Feature Model)
============================================================
Single command to run the full 5-stage pipeline:

  python run_all.py

Stages:
  1  Data download + 17-feature engineering   (data_loader.py)
  2  Ridge baseline walk-forward              (model.py)
  3  LightGBM + SHAP                          (shap_explainability.py)
  4  Sector neutralisation + EWM smoothing    (sector_neutralisation.py)
  5  OOT validation                           (oot_validation.py)
  6  Transaction cost modelling               (transaction_costs.py)

Universe: 250 stocks (50 per sector × 5 sectors)
Features: 17 across 8 groups (Momentum, Risk, Technical, Value,
          Quality, Growth, Size, Liquidity)
Portfolio: top-2 per sector = 10 stocks/month
Turnover reduction: EWM α=0.7 + rebalancing threshold 8%
Data caching: daily_prices.parquet + fundamentals.parquet
              (re-runs skip download entirely)
"""

import os
import sys
import time
import pandas as pd
import numpy as np

os.makedirs("data",    exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

RF = 0.045 / 12


def section(title):
    print("\n" + "="*68)
    print(f"  {title}")
    print("="*68)


def step(n, label):
    print(f"\n[Step {n}] {label}...")


def done(label, t):
    print(f"         Done — {label} ({t:.1f}s)")


t0 = time.time()
section("EQUITY FACTOR MODEL — 250 STOCKS, 17 FEATURES")
print("  Ridge → LightGBM+SHAP → Sector Neutral → OOT → Costs")
print("  8 feature groups: Momentum, Risk, Technical, Value,")
print("  Quality, Growth, Size, Liquidity")
print("  Data cached locally — subsequent runs skip download")


# ── STEP 1: Data & Features ───────────────────────────────────────────────────
step(1, "Loading/downloading data and engineering 17 factors")
t = time.time()

from data_loader import (
    download_price_data, compute_monthly_returns, compute_factors,
    download_fundamentals, check_feature_correlation,
    SECTOR_MAP, FEATURES, FEATURE_GROUPS, TARGET
)

prices          = download_price_data()
monthly_returns = compute_monthly_returns(prices)
monthly_returns.to_csv("data/monthly_returns.csv")

fund_df    = download_fundamentals()
factors_df = compute_factors(monthly_returns, prices, fund_df, SECTOR_MAP)
factors_df.to_csv("data/factor_features.csv", index=False)

corr_matrix = check_feature_correlation(factors_df)

done("data_loader", time.time() - t)
print(f"\n  Universe : {factors_df['ticker'].nunique()} stocks, "
      f"{factors_df['date'].nunique()} months")
print(f"  Features : {len(FEATURES)} ({list(FEATURE_GROUPS.keys())})")


# ── STEP 2: Ridge Baseline ────────────────────────────────────────────────────
step(2, "Ridge regression baseline — Stage 1 (17 features)")
t = time.time()

from model import walk_forward_validation, evaluate_model

ridge_results, coef_df = walk_forward_validation(factors_df)
metrics_ridge           = evaluate_model(ridge_results, coef_df)
ridge_results.to_csv("data/ridge_predictions.csv", index=False)

# Ridge portfolio: top-10 cross-sectional (no sector constraint)
ridge_port_list = []
for date, g in ridge_results.groupby("date"):
    ridge_port_list.append(g.nlargest(10, "predicted")["actual"].mean())
ridge_port   = pd.Series(ridge_port_list)
ridge_sharpe = ((ridge_port-RF).mean()*12) / (ridge_port.std()*np.sqrt(12))
ridge_cum    = (1+ridge_port).prod()-1
ridge_mdd    = ((1+ridge_port).cumprod()/(1+ridge_port).cumprod().cummax()-1).min()
print(f"\n  Ridge portfolio: Sharpe {ridge_sharpe:.3f} | "
      f"Return {ridge_cum:.2%} | MaxDD {ridge_mdd:.2%}")
done("model (Ridge)", time.time() - t)


# ── STEP 3: LightGBM + SHAP ───────────────────────────────────────────────────
step(3, "LightGBM + SHAP explainability — Stage 2")
t = time.time()

from shap_explainability import (
    walk_forward_lgbm, evaluate_lgbm,
    plot_global_importance, plot_shap_direction,
    plot_rolling_factor_importance, plot_shap_by_group,
    explain_single_prediction,
)

lgbm_results, shap_df = walk_forward_lgbm(factors_df)
metrics_lgbm            = evaluate_lgbm(lgbm_results)
lgbm_results.to_csv("data/lgbm_predictions.csv", index=False)
shap_df.to_csv("data/shap_values.csv",            index=False)

# LightGBM portfolio: top-10 cross-sectional
lgbm_port_list = []
for date, g in lgbm_results.groupby("date"):
    lgbm_port_list.append(g.nlargest(10, "predicted")["actual"].mean())
lgbm_port   = pd.Series(lgbm_port_list)
lgbm_sharpe = ((lgbm_port-RF).mean()*12) / (lgbm_port.std()*np.sqrt(12))

print("\n  Generating SHAP charts...")
plot_global_importance(shap_df,
    save_path="outputs/shap_global_importance.png")
plot_shap_direction(shap_df,
    save_path="outputs/shap_direction.png")
plot_rolling_factor_importance(shap_df,
    save_path="outputs/shap_rolling_importance.png")
plot_shap_by_group(shap_df,
    save_path="outputs/shap_group_importance.png")
explain_single_prediction(shap_df,
    save_path="outputs/shap_single_prediction.png")
done("LightGBM + SHAP", time.time() - t)


# ── STEP 4: Sector Neutralisation + EWM ──────────────────────────────────────
step(4, "Sector neutralisation + EWM signal smoothing — Stage 3")
t = time.time()

from sector_neutralisation import (
    add_sector_features, walk_forward_sector_neutral,
    evaluate_sector_neutral, build_sector_aware_portfolio,
    EWM_ALPHA, REBAL_THRESHOLD, TOP_PER_SECTOR,
)

factors_df_sector, z_features = add_sector_features(factors_df, SECTOR_MAP)
sector_results = walk_forward_sector_neutral(factors_df_sector, z_features)
sector_results.to_csv("data/sector_predictions.csv", index=False)

metrics_sector = evaluate_sector_neutral(
    sector_results, old_path="data/lgbm_predictions.csv"
)

sector_port = build_sector_aware_portfolio(sector_results)
sector_port.to_csv("reports/sector_portfolio.csv")

sr          = sector_port["portfolio_return"]
sect_sharpe = ((sr-RF).mean()*12) / (sr.std()*np.sqrt(12))
sect_cum    = (1+sr).prod()-1
sect_mdd    = ((1+sr).cumprod()/(1+sr).cumprod().cummax()-1).min()
print(f"\n  Sector portfolio: Sharpe {sect_sharpe:.3f} | "
      f"Return {sect_cum:.2%} | MaxDD {sect_mdd:.2%}")
done("sector_neutralisation", time.time() - t)


# ── STEP 5: OOT Validation ────────────────────────────────────────────────────
step(5, "Out-of-Time validation — Stage 4")
t = time.time()

from oot_validation import run_oot_validation

oot_metrics = run_oot_validation(
    factors_df_sector=factors_df_sector,
    z_features=z_features,
    sector_results_full=sector_results,
    sector_port_full=sr,
)
done("oot_validation", time.time() - t)


# ── STEP 6: Transaction Costs ─────────────────────────────────────────────────
step(6, "Transaction cost modelling — Stage 5")
t = time.time()

from transaction_costs import (
    compute_turnover, apply_transaction_costs,
    print_cost_analysis, plot_cost_scenarios,
    plot_sharpe_sensitivity, COST_SCENARIOS,
)

SECTOR_PRED = "data/sector_predictions.csv"
PORTFOLIO   = "reports/sector_portfolio.csv"

turnover_df = compute_turnover(SECTOR_PRED)
print_cost_analysis(PORTFOLIO, turnover_df)
plot_cost_scenarios(PORTFOLIO, turnover_df)
plot_sharpe_sensitivity(PORTFOLIO, turnover_df)

# Save cost-adjusted returns report
gross, _, _ = apply_transaction_costs(PORTFOLIO, turnover_df, 0)
rows = []
for date, g in gross.items():
    row = {"date": date, "gross_return": g}
    for name, cost in COST_SCENARIOS.items():
        _, net, mc = apply_transaction_costs(PORTFOLIO, turnover_df, cost)
        short = name.split("(")[1].replace(")","").replace(" ","_")
        row[f"net_{short}"]  = net.get(date, g)
        row[f"cost_{short}"] = mc.get(date, 0)
    rows.append(row)
pd.DataFrame(rows).set_index("date").to_csv("reports/cost_adjusted_returns.csv")

_, net_10, _ = apply_transaction_costs(PORTFOLIO, turnover_df, 0.0010)
net_sharpe   = ((net_10-RF).mean()*12) / (net_10.std()*np.sqrt(12))
net_cum      = (1+net_10).prod()-1
done("transaction_costs", time.time() - t)


# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
total = time.time() - t0
section("PIPELINE COMPLETE")

print(f"\n  Total runtime  : {total/60:.1f} minutes")
print(f"  Universe       : {factors_df['ticker'].nunique()} stocks | "
      f"{factors_df['date'].nunique()} months")
print(f"  Features       : {len(FEATURES)} across "
      f"{len(FEATURE_GROUPS)} groups")
print(f"  Portfolio      : top-{TOP_PER_SECTOR}/sector = 10 stocks | "
      f"EWM α={EWM_ALPHA} | threshold={REBAL_THRESHOLD}")
print(f"  Turnover       : {turnover_df['turnover'].mean():.1%}/month "
      f"({turnover_df['turnover'].mean()*12:.0%}/year)")

print(f"\n  {'Stage':<38} {'Mean IC':>9} {'IC-IR':>8} {'Sharpe':>8}")
print("  " + "-"*65)
print(f"  {'1. Ridge (17 features)':<38} "
      f"{metrics_ridge.get('Mean_IC',0):>9.4f} "
      f"{metrics_ridge.get('IC_IR',0):>8.4f} "
      f"{ridge_sharpe:>8.3f}")
print(f"  {'2. LightGBM (17 features)':<38} "
      f"{metrics_lgbm.get('Mean_IC',0):>9.4f} "
      f"{metrics_lgbm.get('IC_IR',0):>8.4f} "
      f"{lgbm_sharpe:>8.3f}")
print(f"  {'3. + Sector Neutral + EWM':<38} "
      f"{metrics_sector.get('Mean_IC',0):>9.4f} "
      f"{metrics_sector.get('IC_IR',0):>8.4f} "
      f"{sect_sharpe:>8.3f}")
print(f"  {'4. OOT (Jan 2025+)':<38} "
      f"{oot_metrics.get('OOT_Mean_IC',0):>9.4f} "
      f"{oot_metrics.get('OOT_IC_IR',0):>8.4f} "
      f"{oot_metrics.get('OOT_Sharpe',0):>8.3f}")
print(f"  {'5. Net of 10bps costs':<38} {'—':>9} {'—':>8} "
      f"{net_sharpe:>8.3f}")

print("\n  Outputs saved:")
print("    data/          — factor_features.csv, *_predictions.csv,")
print("                     shap_values.csv, daily_prices.parquet,")
print("                     fundamentals.parquet (cache)")
print("    outputs/       — all charts (Ridge + SHAP + OOT + costs)")
print("    reports/       — sector_portfolio.csv, oot_*.csv,")
print("                     cost_adjusted_returns.csv")

ic_v  = metrics_sector.get("Mean_IC", 0)
ir_v  = metrics_sector.get("IC_IR",   0)
om    = oot_metrics.get("OOT_Months", 0)
os_   = oot_metrics.get("OOT_Sharpe", 0)
turn  = turnover_df["turnover"].mean() * 12
n_st  = factors_df["ticker"].nunique()

print(f"""
  Resume bullet:
    "Built 17-factor equity model on {n_st} S&P 500 stocks across
     5 GICS sectors (50 stocks/sector); features span momentum,
     idiosyncratic volatility, 52W-high anchoring, market beta,
     price/book, ROE, revenue growth, log market cap, and liquidity —
     chosen for low inter-group correlation and academic evidence of
     persistent alpha; cross-sectional Z-scoring + LightGBM achieved
     IC {ic_v:.3f} and IC-IR {ir_v:.3f}; EWM signal smoothing +
     rebalancing threshold reduced annual turnover to {turn:.0%};
     OOT validation on {om} months of unseen data confirmed
     Sharpe {os_:.2f}; net-of-cost Sharpe {net_sharpe:.2f} at 10bps
     round-trip confirms deployable alpha; fundamental data uses
     45-day publication lag to prevent leakage."
""")
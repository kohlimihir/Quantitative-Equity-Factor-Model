"""
sector_neutralisation.py  —  Stage 3: Sector-Diversified Portfolio + Turnover Control
======================================================================================
Improvements over plain LightGBM:

1. RAW FEATURES (no Z-scoring)
   LightGBM is tree-based and invariant to monotone transformations.
   Cross-sectional Z-scoring was destroying sector-level alpha by
   removing the cross-sectional mean from each feature. Now the model
   sees raw feature values and can exploit sector-level signals
   (e.g., "all Tech stocks have high momentum" → buy Tech).

2. RAW RETURN as training target (not sector-relative)
   Model learns absolute return prediction. Sector diversification is
   enforced purely by portfolio construction (top-3/sector), not the target.

3. TURNOVER REDUCTION — four industrial methods combined:
   Method A: EWM signal smoothing
     smoothed_rank = 0.5 × current_rank + 0.5 × previous_rank
     Smoothed rankings change slowly → less monthly churn
   Method B: Rebalancing threshold (swing tolerance)
     A held stock is only replaced if the challenger's smoothed rank
     exceeds the held stock's rank by >= REBAL_THRESHOLD (12 pct points)
   Method C: Holding-period bonus
     Each month a stock stays in the portfolio, its effective rank
     gets a small bonus (+0.02/month, capped at 5 months = +0.10).
     This is the standard "incumbent advantage" technique used by
     institutional quant funds to penalise excessive turnover.
   Method D: Top-3 per sector (vs top-2)
     Wider portfolio reduces concentration-driven turnover and
     lowers idiosyncratic risk.
   Expected turnover: ~20-30%/month vs 72% without these controls.

TARGET: Next_Month_Return (price return — e.g. 0.08 = 8% gain)
  Ranking is a PORTFOLIO SELECTION TOOL computed after prediction.

LEAKAGE AUDIT:
  Features: raw values — no cross-sectional transformation
  EWM smoothing: uses only prev_ranks from past months, never future
  Threshold: uses current + previous scores only
  Holding bonus: tracks tenure from past months only, never future
  model.fit(X_train, y_train): fixed n_estimators, no early stopping
  all_dates[:i]: expanding window, no future months in training
  Missing data: cross-sectional median imputation (no future leak)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from data_loader import FEATURES, TARGET, SECTOR_MAP

TOP_PER_SECTOR      = 3      # 3 × 5 sectors = 15 stocks/month
REBAL_THRESHOLD     = 0.12   # challenger must beat holder by 12 rank pct points
EWM_ALPHA           = 0.50   # weight on current month rank (0.5 on previous)
HOLD_BONUS_PER_MONTH = 0.02  # +2 rank pct points per month held (capped at 5)
HOLD_BONUS_CAP       = 5     # max months of bonus accrual

LGB_PARAMS = {
    "objective"        : "regression",
    "metric"           : "rmse",
    "n_estimators"     : 300,
    "learning_rate"    : 0.03,
    "num_leaves"       : 31,
    "min_child_samples": 80,
    "subsample"        : 0.7,
    "colsample_bytree" : 0.7,
    "reg_alpha"        : 0.2,
    "reg_lambda"       : 0.2,
    "random_state"     : 42,
    "verbose"          : -1,
    "n_jobs"           : -1,
}


def add_sector_features(factors_df, sector_map=None):
    """
    Prepares data for sector-diversified portfolio construction.

    DOES NOT Z-score features — LightGBM is tree-based and invariant to
    monotone transformations. Cross-sectional Z-scoring was previously
    destroying sector-level alpha (e.g., if all Tech stocks have high
    momentum and Tech outperforms, Z-scoring neutralises that signal).

    Sector diversification is enforced purely by portfolio construction
    (top-3/sector = 15 stocks), NOT by feature transformation.
    """
    if sector_map is None:
        sector_map = SECTOR_MAP
    df = factors_df.copy()
    if "sector" not in df.columns:
        df["sector"] = df["ticker"].map(sector_map)
    df = df.dropna(subset=["sector"])

    # Use raw features directly — no Z-scoring
    model_features = list(FEATURES)

    sector_counts = df["sector"].value_counts().to_dict()
    print(f"Sectors: {sector_counts}")
    print(f"Using {len(model_features)} raw features (no Z-scoring)\n")
    return df, model_features


def walk_forward_sector_neutral(factors_df, sector_z_features,
                                 min_train_months=24):
    """
    Walk-forward with EWM signal smoothing + holding-period bonus.

    For each month t:
      1. Train LightGBM on all months before t (raw return target)
      2. Predict raw returns for all 250 stocks at t
      3. Compute within-sector percentile rank (0-1 per sector)
      4. Smooth: smoothed = EWM_ALPHA×current + (1-EWM_ALPHA)×previous
      5. Add holding bonus: +HOLD_BONUS_PER_MONTH per month held (capped)
      6. Save adjusted rank as "predicted" for portfolio construction

    Missing feature values are filled with per-month cross-sectional
    median to avoid systematic bias from zero-fill.
    """
    all_dates   = sorted(factors_df["date"].unique())
    results     = []
    prev_ranks  = {}   # {ticker: previous smoothed rank}
    hold_tenure = {}   # {ticker: consecutive months held in portfolio}
    prev_held_all = set()  # all tickers in portfolio last month

    print(f"Sector-neutral walk-forward: {len(all_dates)} months | "
          f"top-{TOP_PER_SECTOR}/sector | EWM α={EWM_ALPHA} | "
          f"rebal threshold={REBAL_THRESHOLD} | "
          f"hold bonus={HOLD_BONUS_PER_MONTH}/mo (cap {HOLD_BONUS_CAP})")

    for i, test_date in enumerate(all_dates):
        if i < min_train_months:
            continue
        train_df = factors_df[factors_df["date"].isin(all_dates[:i])]
        test_df  = factors_df[factors_df["date"] == test_date]
        if len(train_df) < 200 or len(test_df) == 0:
            continue

        # Cross-sectional median imputation (per month, no future leak)
        X_train_df = train_df[sector_z_features].copy()
        for col in sector_z_features:
            med = X_train_df[col].median()
            X_train_df[col] = X_train_df[col].fillna(med)
        X_train_df = X_train_df.fillna(0)
        X_test_df = test_df[sector_z_features].copy()
        for col in sector_z_features:
            med = X_test_df[col].median()
            X_test_df[col] = X_test_df[col].fillna(med)
        X_test_df = X_test_df.fillna(0)

        X_train = X_train_df.values
        y_train = train_df[TARGET].values          # raw return — not sector-relative
        X_test  = X_test_df.values
        y_test  = test_df[TARGET].values

        # Fixed n_estimators — deterministic, zero leakage
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train)
        raw_scores = model.predict(X_test)

        temp = test_df.copy()
        temp["raw_score"] = raw_scores

        # Within-sector percentile rank (0 = worst in sector, 1 = best)
        temp["raw_rank"] = (
            temp.groupby("sector")["raw_score"].rank(pct=True)
        )

        # EWM smoothing — reduces rank churn between months
        smoothed = []
        for _, row in temp.iterrows():
            curr = row["raw_rank"]
            prev = prev_ranks.get(row["ticker"], curr)  # first time: use raw
            s    = EWM_ALPHA * curr + (1 - EWM_ALPHA) * prev
            # Holding-period bonus — incumbent advantage
            ticker = row["ticker"]
            if ticker in prev_held_all:
                tenure = min(hold_tenure.get(ticker, 0) + 1, HOLD_BONUS_CAP)
                s += tenure * HOLD_BONUS_PER_MONTH
                hold_tenure[ticker] = tenure
            else:
                hold_tenure[ticker] = 0
            smoothed.append(s)
            prev_ranks[row["ticker"]] = s

        temp["smoothed_rank"] = smoothed

        # Track which stocks are selected this month (for next month's bonus)
        month_selected = set()
        for sector, sg in temp.groupby("sector"):
            top = sg.nlargest(TOP_PER_SECTOR, "smoothed_rank")["ticker"].values
            month_selected.update(top)
        prev_held_all = month_selected

        for _, row in temp.iterrows():
            results.append({
                "date"        : test_date,
                "ticker"      : row["ticker"],
                "sector"      : row["sector"],
                "actual"      : row[TARGET],
                "predicted"   : row["smoothed_rank"],
                "raw_rank"    : row["raw_rank"],
                "raw_score"   : row["raw_score"],
            })

    results_df = pd.DataFrame(results)
    print(f"Walk-forward complete: {len(results_df):,} predictions, "
          f"{results_df['date'].nunique()} months")
    return results_df


def evaluate_sector_neutral(results_df, old_path=None):
    monthly_ic = (
        results_df.groupby("date")
        .apply(lambda g: g["actual"].corr(g["predicted"], method="spearman"))
    )
    mean_ic = monthly_ic.mean()
    ic_ir   = mean_ic / monthly_ic.std() if monthly_ic.std() > 0 else 0
    pos_ic  = (monthly_ic > 0).sum()

    print(f"\n{'='*62}\n   SECTOR-NEUTRAL MODEL (Stage 3)\n{'='*62}")
    print(f"  Mean IC   : {mean_ic:.5f}  (target >0.05)")
    print(f"  IC Std Dev: {monthly_ic.std():.5f}")
    print(f"  IC-IR     : {ic_ir:.5f}  (target >0.30)")
    print(f"  Pos IC    : {pos_ic}/{len(monthly_ic)} "
          f"({pos_ic/len(monthly_ic)*100:.1f}%)")

    print("\n  IC by sector:")
    sector_ic = (
        results_df.groupby(["sector","date"])
        .apply(lambda g: g["actual"].corr(g["predicted"], method="spearman"))
        .groupby("sector").mean().sort_values(ascending=False)
    )
    for sec, ic in sector_ic.items():
        flag = "GOOD" if ic > 0.05 else ("ok" if ic > 0 else "WEAK")
        print(f"    {sec:<14}: IC={ic:+.4f}  [{flag}]")

    if old_path:
        try:
            old    = pd.read_csv(old_path, parse_dates=["date"])
            old_ic = old.groupby("date").apply(
                lambda g: g["actual"].corr(g["predicted"], method="spearman"))
            old_m  = old_ic.mean()
            old_ir = old_m / old_ic.std() if old_ic.std() > 0 else 0
            print(f"\n  Before vs After sector neutralisation:")
            print(f"    Mean IC: {old_m:+.5f} → {mean_ic:+.5f} ({mean_ic-old_m:+.5f})")
            print(f"    IC-IR  : {old_ir:+.5f} → {ic_ir:+.5f} ({ic_ir-old_ir:+.5f})")
        except FileNotFoundError:
            pass
    print("="*62)
    return {"Mean_IC":mean_ic,"IC_IR":ic_ir,
            "Pos_IC":pos_ic,"Total_M":len(monthly_ic)}


def build_sector_aware_portfolio(results_df,
                                  top_per_sector=TOP_PER_SECTOR,
                                  rebal_threshold=REBAL_THRESHOLD):
    """
    Selects top-3 stocks per sector using smoothed ranks with a
    rebalancing threshold + holding-period bonus to minimise turnover.

    Turnover reduction stack:
      1. EWM smoothing (already in predicted rank)
      2. Threshold: challenger must beat holder by >= rebal_threshold
      3. Holding bonus: already embedded in smoothed_rank from walk-forward
      4. Wider portfolio (top-3 vs top-2): more positions = less churn
    """
    portfolio_rets = []
    prev_held      = {}   # {sector: set of held tickers}

    for date, g in results_df.groupby("date"):
        month_portfolio = []

        for sector, sg in g.groupby("sector"):
            sg_sorted = sg.sort_values("predicted", ascending=False)
            naive_top = set(sg_sorted.head(top_per_sector)["ticker"].values)
            held      = prev_held.get(sector, naive_top)
            final     = set()

            for h in held:
                h_row = sg_sorted[sg_sorted["ticker"] == h]
                if h_row.empty:
                    continue
                h_rank  = h_row["predicted"].values[0]
                chall   = sg_sorted[~sg_sorted["ticker"].isin(held)]
                if chall.empty or \
                   chall["predicted"].max() - h_rank <= rebal_threshold:
                    final.add(h)

            remaining = top_per_sector - len(final)
            if remaining > 0:
                others = sg_sorted[~sg_sorted["ticker"].isin(final)]
                final.update(others.head(remaining)["ticker"].values)
            if len(final) < top_per_sector:
                final = naive_top

            sel = g[g["ticker"].isin(final)]
            month_portfolio.extend(sel["actual"].values)
            prev_held[sector] = final

        portfolio_rets.append({
            "date"            : date,
            "portfolio_return": np.mean(month_portfolio) if month_portfolio else 0,
            "n_stocks"        : len(month_portfolio),
        })

    port_df = pd.DataFrame(portfolio_rets).set_index("date")
    print(f"\nPortfolio: {len(port_df)} months, "
          f"avg {port_df['n_stocks'].mean():.1f} stocks/month")
    return port_df


if __name__ == "__main__":
    import os
    os.makedirs("data",exist_ok=True); os.makedirs("reports",exist_ok=True)
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    factors_df, z_features = add_sector_features(factors_df)
    results_df = walk_forward_sector_neutral(factors_df, z_features)
    results_df.to_csv("data/sector_predictions.csv", index=False)
    metrics = evaluate_sector_neutral(results_df,
                                       old_path="data/lgbm_predictions.csv")
    port_df = build_sector_aware_portfolio(results_df)
    rf = 0.045 / 12
    sr = port_df["portfolio_return"]
    sharpe = ((sr-rf).mean()*12) / (sr.std()*np.sqrt(12))
    cum    = (1+sr).prod() - 1
    mdd    = ((1+sr).cumprod()/(1+sr).cumprod().cummax()-1).min()
    print(f"  Sharpe: {sharpe:.3f} | Return: {cum:.2%} | MaxDD: {mdd:.2%}")
    port_df.to_csv("reports/sector_portfolio.csv")
    print("Saved: data/sector_predictions.csv, reports/sector_portfolio.csv")
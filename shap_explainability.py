"""
shap_explainability.py  —  Stage 2: LightGBM + SHAP (17 features)
==================================================================
With 250 stocks and 17 features spanning price, fundamental, and risk
signals, LightGBM captures non-linear interactions Ridge cannot:
  - High P/E × high momentum → different signal than low P/E × high momentum
  - Low Beta × near 52W high → strong quality+anchoring combo
  - High IdioVol × high MaxRet → double lottery premium → underperformance

TARGET: Next_Month_Return (price return forecasting, not rank).
LEAKAGE: fixed n_estimators, no early stopping, no test eval_set.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

from data_loader import FEATURES, TARGET, FEATURE_GROUPS

FEATURE_LABELS = {
    "Mom_12_1"      : "12M Momentum",
    "Mom_6_1"       : "6M Momentum",
    "Mom_1"         : "1M Reversal",
    "Vol_12"        : "12M Volatility",
    "IdioVol"       : "Idiosyncratic Vol",
    "Beta_12"       : "Market Beta",
    "High52W"       : "52W High Ratio",
    "Trend_MA"      : "Price/MA Trend",
    "MaxRet_1M"     : "Max Daily Return",
    "PB_ratio"      : "Price/Book",
    "PE_TTM"        : "P/E (TTM)",
    "ROE"           : "Return on Equity",
    "GrossMargin"   : "Gross Margin",
    "RevGrowth_YoY" : "Revenue Growth",
    "EarnGrowth_YoY": "Earnings Growth",
    "LogMktCap"     : "Log Mkt Cap",
    "VolRatio"      : "Volume Ratio",
}

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

PALETTE = ["#1f4e79","#c55a11","#1e6b3a","#8b1a1a","#9b59b6",
           "#1abc9c","#f39c12","#2ecc71","#e74c3c","#3498db",
           "#e67e22","#16a085","#8e44ad","#d35400","#27ae60",
           "#c0392b","#2980b9"]
GREY = "#a6a6a6"

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"#f8f9fa",
    "axes.spines.top":False,"axes.spines.right":False,
    "font.family":"DejaVu Sans",
})


def walk_forward_lgbm(factors_df, min_train_months=24):
    """
    Expanding-window walk-forward with LightGBM.
    250 stocks × growing window = thousands of training rows.
    Fixed n_estimators — deterministic, zero leakage.
    """
    all_dates    = sorted(factors_df["date"].unique())
    results      = []
    shap_records = []

    print(f"LightGBM walk-forward: {len(all_dates)} months, "
          f"~{factors_df['ticker'].nunique()} stocks/month, "
          f"{len(FEATURES)} features")

    for i, test_date in enumerate(all_dates):
        if i < min_train_months:
            continue
        train_df = factors_df[factors_df["date"].isin(all_dates[:i])]
        test_df  = factors_df[factors_df["date"] == test_date]
        if len(train_df) < 200 or len(test_df) == 0:
            continue

        X_train = train_df[FEATURES].fillna(0).values
        y_train = train_df[TARGET].values
        X_test  = test_df[FEATURES].fillna(0).values
        y_test  = test_df[TARGET].values

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # SHAP computed post-fit — no leakage
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        for j, ticker in enumerate(test_df["ticker"].values):
            results.append({
                "date"     : test_date,
                "ticker"   : ticker,
                "sector"   : test_df["sector"].values[j],
                "actual"   : y_test[j],
                "predicted": y_pred[j],
            })
            rec = {"date":test_date,"ticker":ticker,
                   "predicted":y_pred[j],"actual":y_test[j]}
            for k, feat in enumerate(FEATURES):
                rec[f"shap_{feat}"] = shap_values[j, k]
                rec[f"feat_{feat}"] = X_test[j, k]
            shap_records.append(rec)

    results_df = pd.DataFrame(results)
    shap_df    = pd.DataFrame(shap_records)
    print(f"Complete: {len(results_df):,} predictions, "
          f"{results_df['date'].nunique()} months")
    return results_df, shap_df


def evaluate_lgbm(results_df):
    actual, predicted = results_df["actual"], results_df["predicted"]
    rmse = np.sqrt(((actual - predicted)**2).mean())
    r2   = 1 - ((actual-predicted)**2).sum() / ((actual-actual.mean())**2).sum()
    monthly_ic = results_df.groupby("date").apply(
        lambda g: g["actual"].corr(g["predicted"], method="spearman"))
    mean_ic = monthly_ic.mean()
    ic_ir   = mean_ic / monthly_ic.std() if monthly_ic.std() > 0 else 0
    dir_acc = ((actual > 0) == (predicted > 0)).mean()

    print(f"\n{'='*60}\n   LIGHTGBM (Stage 2, 17 features)\n{'='*60}")
    print(f"  RMSE: {rmse:.5f} | R²: {r2:.5f}")
    print(f"  Mean IC: {mean_ic:.5f} | IC-IR: {ic_ir:.5f} | DirAcc: {dir_acc:.2%}")
    print("="*60)
    return {"RMSE":rmse,"R2":r2,"Mean_IC":mean_ic,"IC_IR":ic_ir,"DirAcc":dir_acc}


def plot_global_importance(shap_df, save_path="outputs/shap_global_importance.png"):
    shap_cols = [f"shap_{f}" for f in FEATURES if f"shap_{f}" in shap_df.columns]
    mean_abs  = shap_df[shap_cols].abs().mean().sort_values(ascending=True)
    labels    = [FEATURE_LABELS.get(c.replace("shap_",""), c.replace("shap_",""))
                 for c in mean_abs.index]
    colours   = [PALETTE[0] if v == mean_abs.max() else GREY for v in mean_abs.values]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, mean_abs.values, color=colours, height=0.6)
    for bar, val in zip(bars, mean_abs.values):
        ax.text(val + 0.00002, bar.get_y() + bar.get_height()/2,
                f"{val:.5f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Factor Importance — LightGBM SHAP (17 features)",
                 fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved: {save_path}")


def plot_shap_direction(shap_df, save_path="outputs/shap_direction.png"):
    rows = []
    for feat in FEATURES:
        sc, fc = f"shap_{feat}", f"feat_{feat}"
        if sc not in shap_df or fc not in shap_df:
            continue
        med  = shap_df[fc].median()
        high = shap_df.loc[shap_df[fc] >= med, sc].mean()
        low  = shap_df.loc[shap_df[fc] <  med, sc].mean()
        rows.append({"factor": FEATURE_LABELS.get(feat, feat),
                     "high": high, "low": low})
    df  = pd.DataFrame(rows)
    x   = np.arange(len(df)); w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x-w/2, df["high"], w, color=PALETTE[0], label="High value", alpha=0.85)
    ax.bar(x+w/2, df["low"],  w, color=PALETTE[1], label="Low value",  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["factor"], fontsize=8, rotation=25, ha="right")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel("Average SHAP value")
    ax.set_title("Factor Direction: High vs Low Value", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved: {save_path}")


def plot_rolling_factor_importance(shap_df,
                                    save_path="outputs/shap_rolling_importance.png"):
    shap_cols = [f"shap_{f}" for f in FEATURES if f"shap_{f}" in shap_df.columns]
    monthly   = shap_df.groupby("date")[shap_cols].apply(lambda g: g.abs().mean())
    if not hasattr(monthly, "columns"):
        return
    monthly.columns = [FEATURE_LABELS.get(c.replace("shap_",""), c.replace("shap_",""))
                       for c in monthly.columns]
    rolling = monthly.rolling(12).mean().dropna()
    fig, ax = plt.subplots(figsize=(14, 6))
    for col, colour in zip(rolling.columns, PALETTE):
        ax.plot(rolling.index, rolling[col], lw=1.6, label=col, color=colour)
    ax.set_title("Rolling 12M Factor Importance — 17 Features (SHAP)",
                 fontweight="bold", pad=12)
    ax.set_ylabel("Mean |SHAP| (rolling 12M)")
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved: {save_path}")


def plot_shap_by_group(shap_df, save_path="outputs/shap_group_importance.png"):
    """Shows importance aggregated by feature group — easier to read for 17 features."""
    group_shap = {}
    for grp, feats in FEATURE_GROUPS.items():
        cols = [f"shap_{f}" for f in feats if f"shap_{f}" in shap_df.columns]
        if cols:
            group_shap[grp] = shap_df[cols].abs().mean().mean()

    grps   = list(group_shap.keys())
    vals   = list(group_shap.values())
    sorted_pairs = sorted(zip(vals, grps), reverse=False)
    vals, grps   = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(grps))]
    bars = ax.barh(grps, vals, color=colours, height=0.55)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.000005, bar.get_y() + bar.get_height()/2,
                f"{val:.5f}", va="center", fontsize=9)
    ax.set_xlabel("Mean |SHAP| across features in group")
    ax.set_title("SHAP Importance by Feature Group", fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved: {save_path}")


def explain_single_prediction(shap_df, ticker=None, date=None,
                               save_path="outputs/shap_single_prediction.png"):
    if ticker is None:
        latest = shap_df["date"].max()
        row = shap_df[shap_df["date"] == latest].nlargest(1, "predicted").iloc[0]
    else:
        date = date or shap_df["date"].max()
        mask = (shap_df["ticker"] == ticker) & (shap_df["date"] == date)
        row  = shap_df[mask].iloc[0] if mask.sum() > 0 else \
               shap_df[shap_df["date"] == shap_df["date"].max()].nlargest(1, "predicted").iloc[0]

    shap_vals   = {}
    for f in FEATURES:
        if f"shap_{f}" in row.index:
            shap_vals[FEATURE_LABELS.get(f, f)] = row[f"shap_{f}"]

    baseline    = shap_df["predicted"].mean()
    final_pred  = row["predicted"]
    shap_sorted = dict(sorted(shap_vals.items(), key=lambda x: abs(x[1])))

    running = baseline
    positions, values, starts = [], [], []
    for feat, val in shap_sorted.items():
        positions.append(feat); values.append(val); starts.append(running)
        running += val

    colours = [PALETTE[0] if v >= 0 else PALETTE[1] for v in values]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(positions, values, left=starts, color=colours, height=0.55, alpha=0.85)
    ax.axvline(baseline,   color=GREY,      lw=1.5, linestyle="--",
               label=f"Baseline ({baseline:.4f})")
    ax.axvline(final_pred, color=PALETTE[0], lw=2,  linestyle="-",
               label=f"Prediction ({final_pred:.4f})")
    for i, (val, start) in enumerate(zip(values, starts)):
        if abs(val) > 0.0005:
            ax.text(start+val/2, i, f"{val:+.4f}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    ax.set_title(
        f"SHAP Waterfall — {row['ticker']} "
        f"({pd.Timestamp(row['date']).strftime('%b %Y')})",
        fontweight="bold", pad=12)
    ax.set_xlabel("Predicted return contribution")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Saved: {save_path}")

    print(f"\nExplaining: {row['ticker']} "
          f"({pd.Timestamp(row['date']).strftime('%b %Y')})")
    print(f"  Baseline  : {baseline:+.4f}")
    for feat, val in shap_sorted.items():
        print(f"  {feat:<22}: {val:+.5f}  {'↑' if val>0 else '↓'}")
    print(f"  Predicted : {final_pred:+.4f}")
    if "actual" in row.index:
        print(f"  Actual    : {row['actual']:+.4f}")


if __name__ == "__main__":
    import os
    os.makedirs("data",exist_ok=True); os.makedirs("outputs",exist_ok=True)
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    results_df, shap_df = walk_forward_lgbm(factors_df)
    results_df.to_csv("data/lgbm_predictions.csv",  index=False)
    shap_df.to_csv("data/shap_values.csv",           index=False)
    evaluate_lgbm(results_df)
    plot_global_importance(shap_df)
    plot_shap_direction(shap_df)
    plot_rolling_factor_importance(shap_df)
    plot_shap_by_group(shap_df)
    explain_single_prediction(shap_df)
    print("\nDone.")
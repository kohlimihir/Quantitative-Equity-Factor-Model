"""
model.py  —  Stage 1: Ridge Regression Baseline (17 features)
=============================================================
Intentionally simple. Establishes an honest baseline before adding
complexity. With 250 stocks × 24+ months = ~6,000 training rows,
Ridge coefficients are statistically reliable.

TARGET: Next_Month_Return (price return, not rank).
LEAKAGE: scaler fit on train only; all_dates[:i] expanding window.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

from data_loader import FEATURES, TARGET, FEATURE_GROUPS


def walk_forward_validation(factors_df, min_train_months=24):
    """
    Expanding-window walk-forward. Train on [0..i-1], predict i.
    250 stocks × 24 months = ~6,000 rows at first prediction.
    """
    all_dates = sorted(factors_df["date"].unique())
    results, coef_list = [], []
    print(f"Ridge walk-forward: {len(all_dates)} months, "
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

        # Scaler fit ONLY on train — no test statistics leak in
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        coef_list.append(dict(zip(FEATURES, model.coef_)))

        for j, ticker in enumerate(test_df["ticker"].values):
            results.append({
                "date"     : test_date,
                "ticker"   : ticker,
                "sector"   : test_df["sector"].values[j],
                "actual"   : y_test[j],
                "predicted": y_pred[j],
            })

    results_df = pd.DataFrame(results)
    coef_df    = pd.DataFrame(coef_list)
    print(f"Complete: {len(results_df):,} predictions, "
          f"{results_df['date'].nunique()} months")
    return results_df, coef_df


def evaluate_model(results_df, coef_df):
    actual, predicted = results_df["actual"], results_df["predicted"]
    rmse    = np.sqrt(mean_squared_error(actual, predicted))
    r2      = r2_score(actual, predicted)
    monthly_ic = results_df.groupby("date").apply(
        lambda g: g["actual"].corr(g["predicted"], method="spearman"))
    mean_ic = monthly_ic.mean()
    ic_ir   = mean_ic / monthly_ic.std() if monthly_ic.std() > 0 else 0
    dir_acc = ((actual > 0) == (predicted > 0)).mean()

    print(f"\n{'='*60}\n   RIDGE BASELINE (17 features)\n{'='*60}")
    print(f"  RMSE: {rmse:.5f} | R²: {r2:.5f}")
    print(f"  Mean IC: {mean_ic:.5f} | IC-IR: {ic_ir:.5f} | DirAcc: {dir_acc:.2%}")
    print(f"\n  Factor coefficients by group:")
    avg = coef_df.mean()
    for grp, feats in FEATURE_GROUPS.items():
        print(f"    {grp}:")
        for f in feats:
            if f in avg.index:
                print(f"      {f:<18}: {avg[f]:+.5f}  "
                      f"({'↑' if avg[f]>0 else '↓'})")
    print("="*60)
    return {"RMSE":rmse,"R2":r2,"Mean_IC":mean_ic,"IC_IR":ic_ir,"DirAcc":dir_acc}


if __name__ == "__main__":
    import os; os.makedirs("data",exist_ok=True)
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    results_df, coef_df = walk_forward_validation(factors_df)
    evaluate_model(results_df, coef_df)
    results_df.to_csv("data/ridge_predictions.csv", index=False)
    print("Saved: data/ridge_predictions.csv")
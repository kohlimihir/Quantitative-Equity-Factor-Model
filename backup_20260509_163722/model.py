"""
model.py  —  Stage 1: Ridge Regression Baseline (19 features)
=============================================================
Intentionally simple. Establishes an honest baseline before adding
complexity. With 250 stocks × 24+ months = ~6,000 training rows,
Ridge coefficients are statistically reliable.

TARGET: Next_Month_Return (price return, not rank).
LEAKAGE: scaler fit on train only; all_dates[:i] expanding window.
         Missing data: cross-sectional median imputation (no future leak).
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

        X_train_df = train_df[FEATURES].copy()
        for col in FEATURES:
            med = X_train_df[col].median()
            X_train_df[col] = X_train_df[col].fillna(med)
        X_train_df = X_train_df.fillna(0)  # fallback if median is NaN
        X_test_df = test_df[FEATURES].copy()
        for col in FEATURES:
            med = X_test_df[col].median()
            X_test_df[col] = X_test_df[col].fillna(med)
        X_test_df = X_test_df.fillna(0)    # fallback if median is NaN

        X_train = X_train_df.values
        y_train = train_df[TARGET].values
        X_test  = X_test_df.values
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


def compute_rolling_ic(monthly_ic, windows=[3, 6, 12]):
    """
    Compute rolling IC statistics with configurable windows.
    
    Args:
        monthly_ic: Series of monthly IC values indexed by date
        windows: List of window sizes in months (default: [3, 6, 12])
    
    Returns:
        DataFrame with rolling IC statistics for each window
        
    Validates: Requirements 3.5
    """
    rolling_stats = {}
    
    for window in windows:
        if len(monthly_ic) < window:
            continue
        
        rolling_mean = monthly_ic.rolling(window=window).mean()
        rolling_std = monthly_ic.rolling(window=window).std()
        rolling_ir = rolling_mean / rolling_std
        
        rolling_stats[f"rolling_{window}m_ic"] = rolling_mean
        rolling_stats[f"rolling_{window}m_std"] = rolling_std
        rolling_stats[f"rolling_{window}m_ir"] = rolling_ir
    
    if rolling_stats:
        return pd.DataFrame(rolling_stats, index=monthly_ic.index)
    else:
        return pd.DataFrame()


def compute_feature_importance(coef_df, feature_names=None):
    """
    Compute feature importance from model coefficients.
    
    Analyzes coefficient stability and magnitude across time to identify
    the most important and consistent features.
    
    Args:
        coef_df: DataFrame with model coefficients over time
        feature_names: List of feature names (default: all columns in coef_df)
    
    Returns:
        DataFrame with feature importance metrics
        
    Validates: Requirements 3.8
    """
    if feature_names is None:
        feature_names = coef_df.columns.tolist()
    
    importance_records = []
    
    for feature in feature_names:
        if feature not in coef_df.columns:
            continue
        
        coefs = coef_df[feature].values
        
        # Remove NaN values
        valid_coefs = coefs[~np.isnan(coefs)]
        
        if len(valid_coefs) == 0:
            continue
        
        # Compute importance metrics
        mean_coef = np.mean(valid_coefs)
        abs_mean_coef = np.mean(np.abs(valid_coefs))
        std_coef = np.std(valid_coefs)
        stability = 1.0 / (1.0 + std_coef) if std_coef > 0 else 1.0
        
        # Importance score: combines magnitude and stability
        importance_score = abs_mean_coef * stability
        
        importance_records.append({
            "feature": feature,
            "mean_coef": mean_coef,
            "abs_mean_coef": abs_mean_coef,
            "std_coef": std_coef,
            "stability": stability,
            "importance_score": importance_score
        })
    
    importance_df = pd.DataFrame(importance_records)
    importance_df = importance_df.sort_values("importance_score", ascending=False)
    
    return importance_df


def plot_insample_vs_validation_performance(
    results_df,
    monthly_ic,
    rolling_ic_df=None,
    output_path="outputs/insample_vs_validation.png"
):
    """
    Create in-sample vs validation performance comparison plots.
    
    Generates visualizations comparing:
    - Monthly IC over time
    - Rolling IC statistics
    - Cumulative performance
    
    Args:
        results_df: DataFrame with predictions and actuals
        monthly_ic: Series of monthly IC values
        rolling_ic_df: DataFrame with rolling IC statistics (optional)
        output_path: Path to save the plot
        
    Validates: Requirements 3.9, 3.10
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
    except ImportError:
        print("  ⚠️  Warning: matplotlib not installed. Skipping plot generation.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.patch.set_facecolor("white")
    
    NAVY = "#1f4e79"
    TEAL = "#0f6e56"
    ORANGE = "#c55a11"
    GREY = "#a6a6a6"
    
    # Plot 1: Monthly IC with rolling averages
    ax1 = axes[0]
    ax1.set_facecolor("#f8f9fa")
    
    # Bar plot of monthly IC
    ax1.bar(monthly_ic.index, monthly_ic.values, color=GREY, alpha=0.5, 
            width=20, label="Monthly IC")
    
    # Add rolling IC lines if available
    if rolling_ic_df is not None and not rolling_ic_df.empty:
        if "rolling_3m_ic" in rolling_ic_df.columns:
            ax1.plot(rolling_ic_df.index, rolling_ic_df["rolling_3m_ic"], 
                    color=TEAL, linewidth=2, label="3-month rolling IC")
        if "rolling_6m_ic" in rolling_ic_df.columns:
            ax1.plot(rolling_ic_df.index, rolling_ic_df["rolling_6m_ic"], 
                    color=NAVY, linewidth=2, label="6-month rolling IC")
    
    ax1.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax1.axhline(0.05, color=ORANGE, linewidth=1.2, linestyle="--", 
                label="IC=0.05 threshold")
    ax1.set_title("Monthly IC with Rolling Averages", fontweight="bold", fontsize=12)
    ax1.set_ylabel("IC (Spearman)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, alpha=0.3, linestyle=":")
    
    # Plot 2: Rolling IC Information Ratio
    ax2 = axes[1]
    ax2.set_facecolor("#f8f9fa")
    
    if rolling_ic_df is not None and not rolling_ic_df.empty:
        if "rolling_3m_ir" in rolling_ic_df.columns:
            ax2.plot(rolling_ic_df.index, rolling_ic_df["rolling_3m_ir"], 
                    color=TEAL, linewidth=2, marker="o", markersize=3,
                    label="3-month IC-IR")
        if "rolling_6m_ir" in rolling_ic_df.columns:
            ax2.plot(rolling_ic_df.index, rolling_ic_df["rolling_6m_ir"], 
                    color=NAVY, linewidth=2, marker="s", markersize=3,
                    label="6-month IC-IR")
        
        ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax2.axhline(1.0, color=ORANGE, linewidth=1.2, linestyle="--", 
                    label="IC-IR=1.0 threshold")
        ax2.set_title("Rolling IC Information Ratio", fontweight="bold", fontsize=12)
        ax2.set_ylabel("IC-IR")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(True, alpha=0.3, linestyle=":")
    else:
        ax2.text(0.5, 0.5, "Rolling IC-IR data not available", 
                ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Rolling IC Information Ratio", fontweight="bold", fontsize=12)
    
    # Plot 3: Cumulative IC (sum of monthly ICs as proxy for cumulative performance)
    ax3 = axes[2]
    ax3.set_facecolor("#f8f9fa")
    
    cumulative_ic = monthly_ic.cumsum()
    ax3.plot(cumulative_ic.index, cumulative_ic.values, color=NAVY, 
            linewidth=2.5, label="Cumulative IC")
    ax3.fill_between(cumulative_ic.index, 0, cumulative_ic.values, 
                     alpha=0.2, color=NAVY)
    ax3.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax3.set_title("Cumulative IC Over Time", fontweight="bold", fontsize=12)
    ax3.set_ylabel("Cumulative IC")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(True, alpha=0.3, linestyle=":")
    
    plt.tight_layout()
    
    # Save figure
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ In-sample vs validation plot saved to {output_path}")


def evaluate_model(results_df, coef_df):
    actual, predicted = results_df["actual"], results_df["predicted"]
    rmse    = np.sqrt(mean_squared_error(actual, predicted))
    r2      = r2_score(actual, predicted)
    monthly_ic = results_df.groupby("date").apply(
        lambda g: g["actual"].corr(g["predicted"], method="spearman"))
    mean_ic = monthly_ic.mean()
    ic_ir   = mean_ic / monthly_ic.std() if monthly_ic.std() > 0 else 0
    dir_acc = ((actual > 0) == (predicted > 0)).mean()

    print(f"\n{'='*60}\n   RIDGE BASELINE (19 features)\n{'='*60}")
    print(f"  RMSE: {rmse:.5f} | R²: {r2:.5f}")
    print(f"  Mean IC: {mean_ic:.5f} | IC-IR: {ic_ir:.5f} | DirAcc: {dir_acc:.2%}")
    
    # Compute rolling IC statistics
    print(f"\n  Rolling IC Statistics:")
    rolling_ic_df = compute_rolling_ic(monthly_ic, windows=[3, 6, 12])
    if not rolling_ic_df.empty:
        for window in [3, 6, 12]:
            col_ic = f"rolling_{window}m_ic"
            col_ir = f"rolling_{window}m_ir"
            if col_ic in rolling_ic_df.columns:
                latest_ic = rolling_ic_df[col_ic].dropna().iloc[-1] if len(rolling_ic_df[col_ic].dropna()) > 0 else np.nan
                latest_ir = rolling_ic_df[col_ir].dropna().iloc[-1] if len(rolling_ic_df[col_ir].dropna()) > 0 else np.nan
                print(f"    {window}-month: IC={latest_ic:.5f}, IC-IR={latest_ir:.5f}")
    
    # Compute feature importance
    print(f"\n  Feature Importance (Top 10):")
    importance_df = compute_feature_importance(coef_df, FEATURES)
    for idx, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:<18}: {row['importance_score']:.5f} "
              f"(coef={row['mean_coef']:+.5f}, stability={row['stability']:.3f})")
    
    print(f"\n  Factor coefficients by group:")
    avg = coef_df.mean()
    for grp, feats in FEATURE_GROUPS.items():
        print(f"    {grp}:")
        for f in feats:
            if f in avg.index:
                print(f"      {f:<18}: {avg[f]:+.5f}  "
                      f"({'↑' if avg[f]>0 else '↓'})")
    
    # Generate in-sample vs validation performance plots
    plot_insample_vs_validation_performance(results_df, monthly_ic, rolling_ic_df)
    
    print("="*60)
    return {
        "RMSE": rmse,
        "R2": r2,
        "Mean_IC": mean_ic,
        "IC_IR": ic_ir,
        "DirAcc": dir_acc,
        "rolling_ic": rolling_ic_df,
        "feature_importance": importance_df
    }


if __name__ == "__main__":
    import os; os.makedirs("data",exist_ok=True)
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    results_df, coef_df = walk_forward_validation(factors_df)
    evaluate_model(results_df, coef_df)
    results_df.to_csv("data/ridge_predictions.csv", index=False)
    print("Saved: data/ridge_predictions.csv")
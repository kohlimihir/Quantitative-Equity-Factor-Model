"""
feature_analyzer.py — Feature Quality Enhancement System
=========================================================
Analyzes feature predictive power, correlation, and stability to support
feature selection and engineering decisions.

CAPABILITIES:
  1. Monthly IC computation for each feature (Spearman rank correlation)
  2. Pairwise feature correlation analysis with configurable thresholds
  3. Feature stability measurement (rank correlation across consecutive months)
  4. Feature coverage analysis (missing data rate detection)
  5. Incremental IC contribution measurement (feature value-add analysis)

LEAKAGE PREVENTION:
  All analyses use only historical data. IC is computed between features
  at time t and returns at time t+1 (already in the dataset as Next_Month_Return).
  No future information is used.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.7, 2.8, 2.10
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from data_loader import FEATURES, TARGET, FEATURE_GROUPS

# Optional visualization imports (gracefully handled if not available)
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def compute_monthly_ic(
    factors_df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute monthly Information Coefficient (IC) for each feature.
    
    IC = Spearman rank correlation between feature values and subsequent returns.
    
    Args:
        factors_df: DataFrame with columns [date, ticker, features..., Next_Month_Return]
        features: List of feature names to analyze (default: all FEATURES)
    
    Returns:
        DataFrame with columns [date, feature, ic] containing monthly IC values
        
    Validates: Requirements 2.1
    """
    if features is None:
        features = [f for f in FEATURES if f in factors_df.columns]
    
    all_dates = sorted(factors_df["date"].unique())
    ic_records = []
    
    print(f"\nComputing monthly IC for {len(features)} features across {len(all_dates)} months...")
    
    for date in all_dates:
        month_data = factors_df[factors_df["date"] == date].copy()
        
        if len(month_data) < 10:  # Need minimum sample size
            continue
            
        target_values = month_data[TARGET].values
        
        # Skip if target has too many NaNs or no variance
        if np.isnan(target_values).sum() > len(target_values) * 0.5:
            continue
        if np.nanstd(target_values) < 1e-10:
            continue
        
        for feature in features:
            feature_values = month_data[feature].values
            
            # Skip if feature has too many NaNs
            if np.isnan(feature_values).sum() > len(feature_values) * 0.5:
                ic_records.append({
                    "date": date,
                    "feature": feature,
                    "ic": np.nan
                })
                continue
            
            # Remove rows where either feature or target is NaN
            valid_mask = ~(np.isnan(feature_values) | np.isnan(target_values))
            if valid_mask.sum() < 10:  # Need minimum valid samples
                ic_records.append({
                    "date": date,
                    "feature": feature,
                    "ic": np.nan
                })
                continue
            
            valid_features = feature_values[valid_mask]
            valid_targets = target_values[valid_mask]
            
            # Check for variance
            if np.std(valid_features) < 1e-10:
                ic_records.append({
                    "date": date,
                    "feature": feature,
                    "ic": np.nan
                })
                continue
            
            # Compute Spearman correlation
            try:
                ic, _ = spearmanr(valid_features, valid_targets)
                ic_records.append({
                    "date": date,
                    "feature": feature,
                    "ic": ic if not np.isnan(ic) else 0.0
                })
            except Exception:
                ic_records.append({
                    "date": date,
                    "feature": feature,
                    "ic": np.nan
                })
    
    ic_df = pd.DataFrame(ic_records)
    
    # Summary statistics
    ic_summary = ic_df.groupby("feature")["ic"].agg([
        ("mean_ic", "mean"),
        ("std_ic", "std"),
        ("median_ic", "median"),
        ("min_ic", "min"),
        ("max_ic", "max"),
        ("count", "count")
    ]).reset_index()
    
    print(f"  Computed {len(ic_df)} monthly IC values")
    print(f"\n  Top 5 features by mean IC:")
    top_features = ic_summary.nlargest(5, "mean_ic")
    for _, row in top_features.iterrows():
        print(f"    {row['feature']:<18}: {row['mean_ic']:+.5f} "
              f"(std: {row['std_ic']:.5f}, median: {row['median_ic']:+.5f})")
    
    print(f"\n  Bottom 5 features by mean IC:")
    bottom_features = ic_summary.nsmallest(5, "mean_ic")
    for _, row in bottom_features.iterrows():
        print(f"    {row['feature']:<18}: {row['mean_ic']:+.5f} "
              f"(std: {row['std_ic']:.5f}, median: {row['median_ic']:+.5f})")
    
    return ic_df


def compute_pairwise_correlations(
    factors_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    threshold: float = 0.75,
    method: str = "spearman"
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    Compute pairwise feature correlations and identify highly correlated pairs.
    
    Computes mean correlation across all months to get stable estimates.
    
    Args:
        factors_df: DataFrame with feature columns
        features: List of feature names to analyze (default: all FEATURES)
        threshold: Correlation threshold for flagging pairs (default: 0.75)
        method: Correlation method - "spearman" or "pearson" (default: "spearman")
    
    Returns:
        Tuple of (mean_correlation_matrix, flagged_pairs)
        - mean_correlation_matrix: DataFrame with mean correlations
        - flagged_pairs: List of (feature1, feature2, correlation) tuples
        
    Validates: Requirements 2.3
    """
    if features is None:
        features = [f for f in FEATURES if f in factors_df.columns]
    
    print(f"\nComputing pairwise correlations for {len(features)} features...")
    print(f"  Method: {method}, Threshold: {threshold}")
    
    all_dates = sorted(factors_df["date"].unique())
    monthly_corrs = []
    
    for date in all_dates:
        month_data = factors_df[factors_df["date"] == date][features].copy()
        
        if len(month_data) < 10:
            continue
        
        # Drop features with too many NaNs for this month
        valid_features = []
        for feat in features:
            if month_data[feat].notna().sum() >= 10:
                valid_features.append(feat)
        
        if len(valid_features) < 2:
            continue
        
        try:
            corr_matrix = month_data[valid_features].corr(method=method)
            monthly_corrs.append(corr_matrix)
        except Exception:
            continue
    
    if not monthly_corrs:
        print("  Warning: Could not compute correlations")
        return pd.DataFrame(), []
    
    # Compute mean correlation across all months
    mean_corr = pd.concat(monthly_corrs).groupby(level=0).mean()
    
    # Identify highly correlated pairs
    flagged_pairs = []
    features_list = list(mean_corr.index)
    for i, feat1 in enumerate(features_list):
        for j, feat2 in enumerate(features_list):
            if j <= i:  # Skip diagonal and lower triangle
                continue
            
            if feat1 not in mean_corr.index or feat2 not in mean_corr.columns:
                continue
            
            corr_value = mean_corr.loc[feat1, feat2]
            
            if abs(corr_value) > threshold:
                flagged_pairs.append((feat1, feat2, corr_value))
    
    # Sort by absolute correlation (descending)
    flagged_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f"  Computed mean correlation matrix from {len(monthly_corrs)} months")
    
    if flagged_pairs:
        print(f"\n  ⚠  {len(flagged_pairs)} highly correlated pairs (|corr| > {threshold}):")
        for feat1, feat2, corr in flagged_pairs[:10]:  # Show top 10
            print(f"    {feat1:<18} × {feat2:<18}: {corr:+.3f}")
        if len(flagged_pairs) > 10:
            print(f"    ... and {len(flagged_pairs) - 10} more pairs")
    else:
        print(f"  ✓ No pairs exceed |corr| > {threshold} — feature set is clean")
    
    # Overall correlation statistics
    corr_values = mean_corr.values
    upper_tri = corr_values[np.triu_indices_from(corr_values, k=1)]
    print(f"\n  Correlation statistics:")
    print(f"    Mean |corr|: {np.abs(upper_tri).mean():.3f}")
    print(f"    Max |corr|:  {np.abs(upper_tri).max():.3f}")
    print(f"    Median |corr|: {np.median(np.abs(upper_tri)):.3f}")
    
    return mean_corr, flagged_pairs


def compute_feature_stability(
    factors_df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Measure feature stability by computing rank correlation across consecutive months.
    
    Stable features maintain similar cross-sectional rankings over time.
    High stability indicates reliable, persistent signals.
    
    Args:
        factors_df: DataFrame with columns [date, ticker, features...]
        features: List of feature names to analyze (default: all FEATURES)
    
    Returns:
        DataFrame with columns [feature, mean_stability, std_stability, median_stability]
        where stability is the Spearman correlation of feature ranks between
        consecutive months for the same stocks.
        
    Validates: Requirements 2.7
    """
    if features is None:
        features = [f for f in FEATURES if f in factors_df.columns]
    
    print(f"\nComputing feature stability for {len(features)} features...")
    
    all_dates = sorted(factors_df["date"].unique())
    stability_records = []
    
    for i in range(len(all_dates) - 1):
        date_t0 = all_dates[i]
        date_t1 = all_dates[i + 1]
        
        # Get data for both months
        data_t0 = factors_df[factors_df["date"] == date_t0].set_index("ticker")
        data_t1 = factors_df[factors_df["date"] == date_t1].set_index("ticker")
        
        # Find common tickers
        common_tickers = data_t0.index.intersection(data_t1.index)
        
        if len(common_tickers) < 10:
            continue
        
        for feature in features:
            if feature not in data_t0.columns or feature not in data_t1.columns:
                continue
            
            # Get feature values for common tickers
            values_t0 = data_t0.loc[common_tickers, feature].values
            values_t1 = data_t1.loc[common_tickers, feature].values
            
            # Remove pairs where either value is NaN
            valid_mask = ~(np.isnan(values_t0) | np.isnan(values_t1))
            
            if valid_mask.sum() < 10:
                stability_records.append({
                    "date_t0": date_t0,
                    "date_t1": date_t1,
                    "feature": feature,
                    "stability": np.nan
                })
                continue
            
            valid_t0 = values_t0[valid_mask]
            valid_t1 = values_t1[valid_mask]
            
            # Check for variance
            if np.std(valid_t0) < 1e-10 or np.std(valid_t1) < 1e-10:
                stability_records.append({
                    "date_t0": date_t0,
                    "date_t1": date_t1,
                    "feature": feature,
                    "stability": np.nan
                })
                continue
            
            # Compute rank correlation between consecutive months
            try:
                stability, _ = spearmanr(valid_t0, valid_t1)
                stability_records.append({
                    "date_t0": date_t0,
                    "date_t1": date_t1,
                    "feature": feature,
                    "stability": stability if not np.isnan(stability) else 0.0
                })
            except Exception:
                stability_records.append({
                    "date_t0": date_t0,
                    "date_t1": date_t1,
                    "feature": feature,
                    "stability": np.nan
                })
    
    stability_df = pd.DataFrame(stability_records)
    
    # Aggregate by feature
    stability_summary = stability_df.groupby("feature")["stability"].agg([
        ("mean_stability", "mean"),
        ("std_stability", "std"),
        ("median_stability", "median"),
        ("min_stability", "min"),
        ("max_stability", "max"),
        ("count", "count")
    ]).reset_index()
    
    print(f"  Computed {len(stability_df)} month-to-month stability measurements")
    
    print(f"\n  Most stable features (high rank correlation across months):")
    top_stable = stability_summary.nlargest(5, "mean_stability")
    for _, row in top_stable.iterrows():
        print(f"    {row['feature']:<18}: {row['mean_stability']:.5f} "
              f"(std: {row['std_stability']:.5f}, median: {row['median_stability']:.5f})")
    
    print(f"\n  Least stable features (low rank correlation across months):")
    bottom_stable = stability_summary.nsmallest(5, "mean_stability")
    for _, row in bottom_stable.iterrows():
        print(f"    {row['feature']:<18}: {row['mean_stability']:.5f} "
              f"(std: {row['std_stability']:.5f}, median: {row['median_stability']:.5f})")
    
    return stability_summary


def analyze_feature_coverage(
    factors_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    threshold: float = 0.30
) -> pd.DataFrame:
    """
    Analyze feature coverage (missing data rates).
    
    Args:
        factors_df: DataFrame with feature columns
        features: List of feature names to analyze (default: all FEATURES)
        threshold: Missing data threshold for flagging (default: 0.30 = 30%)
    
    Returns:
        DataFrame with columns [feature, missing_rate, total_rows, missing_rows]
    """
    if features is None:
        features = [f for f in FEATURES if f in factors_df.columns]
    
    print(f"\nAnalyzing feature coverage for {len(features)} features...")
    
    coverage_records = []
    total_rows = len(factors_df)
    
    for feature in features:
        missing_rows = factors_df[feature].isna().sum()
        missing_rate = missing_rows / total_rows
        
        coverage_records.append({
            "feature": feature,
            "missing_rate": missing_rate,
            "coverage_rate": 1 - missing_rate,
            "total_rows": total_rows,
            "missing_rows": missing_rows,
            "valid_rows": total_rows - missing_rows
        })
    
    coverage_df = pd.DataFrame(coverage_records)
    coverage_df = coverage_df.sort_values("missing_rate", ascending=False)
    
    # Flag features with high missing rates
    flagged = coverage_df[coverage_df["missing_rate"] > threshold]
    
    if len(flagged) > 0:
        print(f"\n  ⚠  {len(flagged)} features with missing rate > {threshold:.0%}:")
        for _, row in flagged.iterrows():
            print(f"    {row['feature']:<18}: {row['missing_rate']:.1%} missing "
                  f"({row['missing_rows']:,} / {row['total_rows']:,} rows)")
    else:
        print(f"  ✓ All features have missing rate ≤ {threshold:.0%}")
    
    print(f"\n  Coverage summary:")
    print(f"    Mean missing rate: {coverage_df['missing_rate'].mean():.1%}")
    print(f"    Median missing rate: {coverage_df['missing_rate'].median():.1%}")
    print(f"    Max missing rate: {coverage_df['missing_rate'].max():.1%}")
    
    return coverage_df


def compute_incremental_ic(
    factors_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    baseline_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Measure incremental IC contribution of each feature.
    
    For each feature, computes:
    - Baseline IC: IC using all features EXCEPT the target feature
    - Full IC: IC using all features INCLUDING the target feature
    - Incremental IC: Full IC - Baseline IC
    
    This measures how much predictive power each feature adds when included
    in the model versus excluded.
    
    Args:
        factors_df: DataFrame with all features and target
        features: List of feature names to analyze (default: all FEATURES)
        baseline_features: Optional list of features to always include in baseline
                          (default: None, meaning each feature is tested individually)
    
    Returns:
        DataFrame with columns [feature, baseline_ic, full_ic, incremental_ic]
        
    Validates: Requirements 2.4
    """
    if features is None:
        features = [f for f in FEATURES if f in factors_df.columns]
    
    print(f"\nComputing incremental IC contribution for {len(features)} features...")
    
    all_dates = sorted(factors_df["date"].unique())
    incremental_records = []
    
    for feature_to_test in features:
        # Define baseline features (all except the one being tested)
        if baseline_features is not None:
            baseline_feats = baseline_features.copy()
        else:
            baseline_feats = [f for f in features if f != feature_to_test]
        
        # Compute IC for each month with and without the feature
        baseline_ics = []
        full_ics = []
        
        for date in all_dates:
            month_data = factors_df[factors_df["date"] == date].copy()
            
            if len(month_data) < 10:
                continue
            
            target_values = month_data[TARGET].values
            
            # Skip if target has too many NaNs or no variance
            if np.isnan(target_values).sum() > len(target_values) * 0.5:
                continue
            if np.nanstd(target_values) < 1e-10:
                continue
            
            # Compute baseline IC (without feature_to_test)
            if len(baseline_feats) > 0:
                baseline_data = month_data[baseline_feats].copy()
                
                # Remove rows with too many NaNs
                valid_mask = baseline_data.notna().sum(axis=1) >= len(baseline_feats) * 0.5
                valid_mask = valid_mask & ~np.isnan(target_values)
                
                if valid_mask.sum() >= 10:
                    # Create composite signal from baseline features (simple average of ranks)
                    baseline_signal = baseline_data.loc[valid_mask].rank(pct=True).mean(axis=1).values
                    valid_targets = target_values[valid_mask]
                    
                    if np.std(baseline_signal) > 1e-10:
                        try:
                            baseline_ic, _ = spearmanr(baseline_signal, valid_targets)
                            if not np.isnan(baseline_ic):
                                baseline_ics.append(baseline_ic)
                        except Exception:
                            pass
            
            # Compute full IC (with feature_to_test)
            full_feats = baseline_feats + [feature_to_test]
            full_data = month_data[full_feats].copy()
            
            # Remove rows with too many NaNs
            valid_mask = full_data.notna().sum(axis=1) >= len(full_feats) * 0.5
            valid_mask = valid_mask & ~np.isnan(target_values)
            
            if valid_mask.sum() >= 10:
                # Create composite signal from all features (simple average of ranks)
                full_signal = full_data.loc[valid_mask].rank(pct=True).mean(axis=1).values
                valid_targets = target_values[valid_mask]
                
                if np.std(full_signal) > 1e-10:
                    try:
                        full_ic, _ = spearmanr(full_signal, valid_targets)
                        if not np.isnan(full_ic):
                            full_ics.append(full_ic)
                    except Exception:
                        pass
        
        # Compute mean ICs
        baseline_ic_mean = np.mean(baseline_ics) if len(baseline_ics) > 0 else 0.0
        full_ic_mean = np.mean(full_ics) if len(full_ics) > 0 else 0.0
        incremental_ic = full_ic_mean - baseline_ic_mean
        
        incremental_records.append({
            "feature": feature_to_test,
            "baseline_ic": baseline_ic_mean,
            "full_ic": full_ic_mean,
            "incremental_ic": incremental_ic,
            "n_months_baseline": len(baseline_ics),
            "n_months_full": len(full_ics)
        })
    
    incremental_df = pd.DataFrame(incremental_records)
    incremental_df = incremental_df.sort_values("incremental_ic", ascending=False)
    
    print(f"  Computed incremental IC for {len(incremental_df)} features")
    
    print(f"\n  Top 5 features by incremental IC contribution:")
    for _, row in incremental_df.head(5).iterrows():
        print(f"    {row['feature']:<18}: +{row['incremental_ic']:+.5f} "
              f"(baseline: {row['baseline_ic']:.5f}, full: {row['full_ic']:.5f})")
    
    print(f"\n  Bottom 5 features by incremental IC contribution:")
    for _, row in incremental_df.tail(5).iterrows():
        print(f"    {row['feature']:<18}: {row['incremental_ic']:+.5f} "
              f"(baseline: {row['baseline_ic']:.5f}, full: {row['full_ic']:.5f})")
    
    # Flag features with negative incremental IC
    negative_contrib = incremental_df[incremental_df["incremental_ic"] < 0]
    if len(negative_contrib) > 0:
        print(f"\n  ⚠  {len(negative_contrib)} features have NEGATIVE incremental IC:")
        for _, row in negative_contrib.iterrows():
            print(f"    {row['feature']:<18}: {row['incremental_ic']:+.5f} "
                  f"(removing this feature would IMPROVE IC)")
    
    return incremental_df


def generate_feature_quality_report(
    factors_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    correlation_threshold: float = 0.75,
    coverage_threshold: float = 0.30,
    include_incremental_ic: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Generate comprehensive feature quality report.
    
    Combines IC analysis, correlation analysis, stability measurement,
    coverage analysis, and incremental IC contribution into a single
    comprehensive report.
    
    Args:
        factors_df: DataFrame with all features and target
        features: List of feature names to analyze (default: all FEATURES)
        correlation_threshold: Threshold for flagging correlated pairs
        coverage_threshold: Threshold for flagging high missing rates
        include_incremental_ic: Whether to compute incremental IC (default: True)
    
    Returns:
        Dictionary containing:
        - "ic_monthly": Monthly IC values
        - "ic_summary": IC summary statistics by feature
        - "correlation_matrix": Mean correlation matrix
        - "correlated_pairs": List of highly correlated pairs
        - "stability": Feature stability measurements
        - "coverage": Feature coverage analysis
        - "incremental_ic": Incremental IC contribution (if include_incremental_ic=True)
        - "overall_ranking": Combined feature ranking
    """
    if features is None:
        features = [f for f in FEATURES if f in factors_df.columns]
    
    print("\n" + "="*70)
    print("  FEATURE QUALITY ANALYSIS REPORT")
    print("="*70)
    
    # 1. Monthly IC computation
    ic_monthly = compute_monthly_ic(factors_df, features)
    ic_summary = ic_monthly.groupby("feature")["ic"].agg([
        ("mean_ic", "mean"),
        ("std_ic", "std"),
        ("median_ic", "median"),
        ("ic_ir", lambda x: x.mean() / x.std() if x.std() > 0 else 0)
    ]).reset_index()
    
    # 2. Pairwise correlations
    corr_matrix, corr_pairs = compute_pairwise_correlations(
        factors_df, features, correlation_threshold
    )
    
    # 3. Feature stability
    stability = compute_feature_stability(factors_df, features)
    
    # 4. Feature coverage
    coverage = analyze_feature_coverage(factors_df, features, coverage_threshold)
    
    # 5. Incremental IC contribution (optional)
    incremental_ic = None
    if include_incremental_ic:
        incremental_ic = compute_incremental_ic(factors_df, features)
    
    # 6. Create overall ranking
    print("\n" + "="*70)
    print("  OVERALL FEATURE RANKING")
    print("="*70)
    
    # Merge all metrics
    ranking = ic_summary.copy()
    ranking = ranking.merge(
        stability[["feature", "mean_stability", "std_stability"]],
        on="feature",
        how="left"
    )
    ranking = ranking.merge(
        coverage[["feature", "missing_rate", "coverage_rate"]],
        on="feature",
        how="left"
    )
    
    # Add incremental IC if computed
    if incremental_ic is not None:
        ranking = ranking.merge(
            incremental_ic[["feature", "incremental_ic", "baseline_ic", "full_ic"]],
            on="feature",
            how="left"
        )
    
    # Compute composite score (higher is better)
    # Normalize each metric to [0, 1] range
    ranking["ic_score"] = (ranking["mean_ic"] - ranking["mean_ic"].min()) / \
                          (ranking["mean_ic"].max() - ranking["mean_ic"].min() + 1e-10)
    ranking["stability_score"] = (ranking["mean_stability"] - ranking["mean_stability"].min()) / \
                                 (ranking["mean_stability"].max() - ranking["mean_stability"].min() + 1e-10)
    ranking["coverage_score"] = ranking["coverage_rate"]
    
    # Composite score: weighted average (40% IC, 30% stability, 30% coverage)
    # Updated from 50/30/20 to 40/30/30 to balance coverage importance
    ranking["composite_score"] = (
        0.4 * ranking["ic_score"] +
        0.3 * ranking["stability_score"] +
        0.3 * ranking["coverage_score"]
    )
    
    ranking = ranking.sort_values("composite_score", ascending=False)
    
    print("\n  Top 10 features by composite score:")
    print(f"  {'Rank':<6}{'Feature':<18}{'Mean IC':<12}{'Stability':<12}"
          f"{'Coverage':<12}{'Score':<10}")
    print("  " + "-"*68)
    
    for idx, (_, row) in enumerate(ranking.head(10).iterrows(), 1):
        print(f"  {idx:<6}{row['feature']:<18}{row['mean_ic']:>+10.5f}  "
              f"{row['mean_stability']:>10.5f}  {row['coverage_rate']:>10.1%}  "
              f"{row['composite_score']:>8.3f}")
    
    print("\n  Bottom 5 features by composite score:")
    for idx, (_, row) in enumerate(ranking.tail(5).iterrows(), len(ranking)-4):
        print(f"  {idx:<6}{row['feature']:<18}{row['mean_ic']:>+10.5f}  "
              f"{row['mean_stability']:>10.5f}  {row['coverage_rate']:>10.1%}  "
              f"{row['composite_score']:>8.3f}")
    
    print("\n" + "="*70)
    
    result = {
        "ic_monthly": ic_monthly,
        "ic_summary": ic_summary,
        "correlation_matrix": corr_matrix,
        "correlated_pairs": corr_pairs,
        "stability": stability,
        "coverage": coverage,
        "overall_ranking": ranking
    }
    
    if incremental_ic is not None:
        result["incremental_ic"] = incremental_ic
    
    return result


def generate_feature_recommendations(
    ranking_df: pd.DataFrame,
    corr_pairs: List[Tuple[str, str, float]],
    coverage_df: pd.DataFrame,
    incremental_ic_df: Optional[pd.DataFrame] = None,
    low_ic_threshold: float = 0.01,
    high_missing_threshold: float = 0.30,
    high_corr_threshold: float = 0.75,
    negative_incremental_threshold: float = -0.001
) -> Dict[str, List[Dict[str, str]]]:
    """
    Generate actionable feature recommendations based on quality metrics.
    
    Analyzes features and provides recommendations for:
    - Features to REMOVE (low IC, negative incremental IC, high missing rate)
    - Features to KEEP (high quality, strong predictive power)
    - Features to ENGINEER (opportunities for improvement)
    - Correlated pairs to ADDRESS (redundancy issues)
    
    Args:
        ranking_df: Overall feature ranking with composite scores
        corr_pairs: List of highly correlated feature pairs
        coverage_df: Feature coverage analysis
        incremental_ic_df: Incremental IC contribution (optional)
        low_ic_threshold: Threshold for flagging low IC features
        high_missing_threshold: Threshold for flagging high missing rates
        high_corr_threshold: Threshold for flagging high correlations
        negative_incremental_threshold: Threshold for flagging negative incremental IC
    
    Returns:
        Dictionary with recommendation categories:
        - "remove": Features recommended for removal
        - "keep": Features recommended to keep
        - "engineer": Features with engineering opportunities
        - "correlated_pairs": Correlated pairs to address
        
    Validates: Requirements 2.10
    """
    recommendations = {
        "remove": [],
        "keep": [],
        "engineer": [],
        "correlated_pairs": []
    }
    
    print("\n" + "="*70)
    print("  FEATURE RECOMMENDATIONS")
    print("="*70)
    
    # 1. Identify features to REMOVE
    print("\n📌 FEATURES TO REMOVE:")
    remove_count = 0
    
    # Low IC features
    low_ic_features = ranking_df[ranking_df["mean_ic"].abs() < low_ic_threshold]
    for _, row in low_ic_features.iterrows():
        recommendations["remove"].append({
            "feature": row["feature"],
            "reason": f"Low IC (|{row['mean_ic']:.5f}| < {low_ic_threshold})",
            "priority": "HIGH"
        })
        print(f"  ❌ {row['feature']:<18}: Low IC (|{row['mean_ic']:.5f}| < {low_ic_threshold})")
        remove_count += 1
    
    # Negative incremental IC features
    if incremental_ic_df is not None:
        negative_ic = incremental_ic_df[
            incremental_ic_df["incremental_ic"] < negative_incremental_threshold
        ]
        for _, row in negative_ic.iterrows():
            if row["feature"] not in [r["feature"] for r in recommendations["remove"]]:
                recommendations["remove"].append({
                    "feature": row["feature"],
                    "reason": f"Negative incremental IC ({row['incremental_ic']:+.5f})",
                    "priority": "HIGH"
                })
                print(f"  ❌ {row['feature']:<18}: Negative incremental IC "
                      f"({row['incremental_ic']:+.5f}) — removing improves model")
                remove_count += 1
    
    # High missing rate features
    high_missing = coverage_df[coverage_df["missing_rate"] > high_missing_threshold]
    for _, row in high_missing.iterrows():
        if row["feature"] not in [r["feature"] for r in recommendations["remove"]]:
            recommendations["remove"].append({
                "feature": row["feature"],
                "reason": f"High missing rate ({row['missing_rate']:.1%} > {high_missing_threshold:.0%})",
                "priority": "MEDIUM"
            })
            print(f"  ⚠️  {row['feature']:<18}: High missing rate "
                  f"({row['missing_rate']:.1%} > {high_missing_threshold:.0%})")
            remove_count += 1
    
    if remove_count == 0:
        print("  ✓ No features recommended for removal")
    
    # 2. Identify features to KEEP
    print("\n📌 FEATURES TO KEEP (Top performers):")
    top_features = ranking_df.nlargest(10, "composite_score")
    for idx, (_, row) in enumerate(top_features.iterrows(), 1):
        recommendations["keep"].append({
            "feature": row["feature"],
            "reason": f"High composite score ({row['composite_score']:.3f}), "
                     f"IC={row['mean_ic']:+.5f}, stability={row['mean_stability']:.3f}",
            "rank": idx
        })
        print(f"  ✅ #{idx:<2} {row['feature']:<18}: Score={row['composite_score']:.3f}, "
              f"IC={row['mean_ic']:+.5f}, Stability={row['mean_stability']:.3f}")
    
    # 3. Identify engineering opportunities
    print("\n📌 ENGINEERING OPPORTUNITIES:")
    engineer_count = 0
    
    # Features with moderate IC but low stability (could benefit from smoothing)
    unstable_features = ranking_df[
        (ranking_df["mean_ic"].abs() >= low_ic_threshold) &
        (ranking_df["mean_stability"] < 0.5)
    ]
    for _, row in unstable_features.head(5).iterrows():
        recommendations["engineer"].append({
            "feature": row["feature"],
            "opportunity": f"Low stability ({row['mean_stability']:.3f}) — consider EWM smoothing",
            "current_ic": f"{row['mean_ic']:+.5f}"
        })
        print(f"  🔧 {row['feature']:<18}: Low stability ({row['mean_stability']:.3f}) "
              f"— consider EWM smoothing or rolling averages")
        engineer_count += 1
    
    # Features with good IC but moderate missing rates (could benefit from better imputation)
    moderate_missing = coverage_df[
        (coverage_df["missing_rate"] > 0.10) &
        (coverage_df["missing_rate"] <= high_missing_threshold)
    ]
    for _, row in moderate_missing.head(5).iterrows():
        feature_ic = ranking_df[ranking_df["feature"] == row["feature"]]["mean_ic"].values
        if len(feature_ic) > 0 and abs(feature_ic[0]) >= low_ic_threshold:
            if row["feature"] not in [r["feature"] for r in recommendations["engineer"]]:
                recommendations["engineer"].append({
                    "feature": row["feature"],
                    "opportunity": f"Moderate missing rate ({row['missing_rate']:.1%}) "
                                  f"— consider improved imputation",
                    "current_ic": f"{feature_ic[0]:+.5f}"
                })
                print(f"  🔧 {row['feature']:<18}: Moderate missing rate "
                      f"({row['missing_rate']:.1%}) — consider improved imputation")
                engineer_count += 1
    
    if engineer_count == 0:
        print("  ✓ No immediate engineering opportunities identified")
    
    # 4. Address correlated pairs
    print("\n📌 CORRELATED PAIRS TO ADDRESS:")
    if corr_pairs:
        for feat1, feat2, corr in corr_pairs[:10]:  # Top 10 pairs
            # Determine which feature to keep based on composite score
            score1 = ranking_df[ranking_df["feature"] == feat1]["composite_score"].values
            score2 = ranking_df[ranking_df["feature"] == feat2]["composite_score"].values
            
            if len(score1) > 0 and len(score2) > 0:
                keep_feat = feat1 if score1[0] > score2[0] else feat2
                remove_feat = feat2 if keep_feat == feat1 else feat1
                
                recommendations["correlated_pairs"].append({
                    "feature1": feat1,
                    "feature2": feat2,
                    "correlation": f"{corr:+.3f}",
                    "recommendation": f"Keep {keep_feat}, consider removing {remove_feat}"
                })
                print(f"  🔗 {feat1:<18} × {feat2:<18}: {corr:+.3f} "
                      f"— keep {keep_feat}, consider removing {remove_feat}")
    else:
        print("  ✓ No highly correlated pairs found")
    
    print("\n" + "="*70)
    
    return recommendations


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: str = "reports/feature_correlation_heatmap.png",
    figsize: Tuple[int, int] = (14, 12),
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0
) -> None:
    """
    Generate and save a correlation heatmap visualization.
    
    Creates a color-coded heatmap showing pairwise feature correlations
    with annotations for high correlation values.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        output_path: Path to save the heatmap image
        figsize: Figure size (width, height) in inches
        cmap: Colormap name (default: RdBu_r for red-blue diverging)
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        
    Validates: Requirements 2.10
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  ⚠️  Warning: matplotlib/seaborn not installed. Skipping heatmap generation.")
        print("     Install with: pip install matplotlib seaborn")
        return
    
    if corr_matrix.empty:
        print("  ⚠️  Warning: Empty correlation matrix. Skipping heatmap generation.")
        return
    
    print(f"\nGenerating correlation heatmap...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate heatmap
    sns.heatmap(
        corr_matrix,
        annot=False,  # Don't annotate all cells (too cluttered)
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        ax=ax
    )
    
    # Customize plot
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Features", fontsize=12, fontweight="bold")
    ax.set_ylabel("Features", fontsize=12, fontweight="bold")
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Correlation heatmap saved to {output_path}")


def save_feature_quality_report(
    report: Dict[str, pd.DataFrame],
    recommendations: Dict[str, List[Dict[str, str]]],
    output_dir: str = "reports"
) -> None:
    """
    Save comprehensive feature quality report to disk.
    
    Saves all report components including:
    - CSV files for all metrics
    - Correlation heatmap visualization
    - Text report with recommendations
    
    Args:
        report: Dictionary containing all report components
        recommendations: Feature recommendations dictionary
        output_dir: Directory to save reports
        
    Validates: Requirements 2.10
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("  SAVING FEATURE QUALITY REPORTS")
    print("="*70)
    
    # Save CSV reports
    report["ic_monthly"].to_csv(
        os.path.join(output_dir, "feature_ic_monthly.csv"), index=False
    )
    report["ic_summary"].to_csv(
        os.path.join(output_dir, "feature_ic_summary.csv"), index=False
    )
    report["stability"].to_csv(
        os.path.join(output_dir, "feature_stability.csv"), index=False
    )
    report["coverage"].to_csv(
        os.path.join(output_dir, "feature_coverage.csv"), index=False
    )
    report["overall_ranking"].to_csv(
        os.path.join(output_dir, "feature_ranking.csv"), index=False
    )
    
    if "incremental_ic" in report:
        report["incremental_ic"].to_csv(
            os.path.join(output_dir, "feature_incremental_ic.csv"), index=False
        )
    
    if not report["correlation_matrix"].empty:
        report["correlation_matrix"].to_csv(
            os.path.join(output_dir, "feature_correlation_matrix.csv")
        )
    
    if report["correlated_pairs"]:
        corr_pairs_df = pd.DataFrame(
            report["correlated_pairs"],
            columns=["feature1", "feature2", "correlation"]
        )
        corr_pairs_df.to_csv(
            os.path.join(output_dir, "feature_correlated_pairs.csv"), index=False
        )
    
    # Generate correlation heatmap
    if not report["correlation_matrix"].empty:
        plot_correlation_heatmap(
            report["correlation_matrix"],
            output_path=os.path.join(output_dir, "feature_correlation_heatmap.png")
        )
    
    # Save recommendations as text report
    recommendations_path = os.path.join(output_dir, "feature_recommendations.txt")
    with open(recommendations_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("  FEATURE QUALITY RECOMMENDATIONS\n")
        f.write("="*70 + "\n\n")
        
        # Features to remove
        f.write("📌 FEATURES TO REMOVE:\n")
        f.write("-" * 70 + "\n")
        if recommendations["remove"]:
            for rec in recommendations["remove"]:
                f.write(f"  ❌ {rec['feature']:<18} [{rec['priority']}]\n")
                f.write(f"     Reason: {rec['reason']}\n\n")
        else:
            f.write("  ✓ No features recommended for removal\n\n")
        
        # Features to keep
        f.write("\n📌 FEATURES TO KEEP (Top Performers):\n")
        f.write("-" * 70 + "\n")
        for rec in recommendations["keep"]:
            f.write(f"  ✅ #{rec['rank']:<2} {rec['feature']:<18}\n")
            f.write(f"     {rec['reason']}\n\n")
        
        # Engineering opportunities
        f.write("\n📌 ENGINEERING OPPORTUNITIES:\n")
        f.write("-" * 70 + "\n")
        if recommendations["engineer"]:
            for rec in recommendations["engineer"]:
                f.write(f"  🔧 {rec['feature']:<18} (Current IC: {rec['current_ic']})\n")
                f.write(f"     {rec['opportunity']}\n\n")
        else:
            f.write("  ✓ No immediate engineering opportunities identified\n\n")
        
        # Correlated pairs
        f.write("\n📌 CORRELATED PAIRS TO ADDRESS:\n")
        f.write("-" * 70 + "\n")
        if recommendations["correlated_pairs"]:
            for rec in recommendations["correlated_pairs"]:
                f.write(f"  🔗 {rec['feature1']:<18} × {rec['feature2']:<18}\n")
                f.write(f"     Correlation: {rec['correlation']}\n")
                f.write(f"     Recommendation: {rec['recommendation']}\n\n")
        else:
            f.write("  ✓ No highly correlated pairs found\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"\n✓ Feature quality reports saved to {output_dir}/")
    print("  CSV Reports:")
    print("    - feature_ic_monthly.csv")
    print("    - feature_ic_summary.csv")
    print("    - feature_stability.csv")
    print("    - feature_coverage.csv")
    print("    - feature_ranking.csv")
    if "incremental_ic" in report:
        print("    - feature_incremental_ic.csv")
    print("    - feature_correlation_matrix.csv")
    if report["correlated_pairs"]:
        print("    - feature_correlated_pairs.csv")
    print("  Visualizations:")
    if not report["correlation_matrix"].empty:
        print("    - feature_correlation_heatmap.png")
    print("  Text Reports:")
    print("    - feature_recommendations.txt")
    print("\n" + "="*70)


if __name__ == "__main__":
    import os
    
    # Load factor data
    factors_path = os.path.join("data", "factor_features.csv")
    if not os.path.exists(factors_path):
        print(f"Error: {factors_path} not found. Run data_loader.py first.")
        exit(1)
    
    print(f"Loading factor data from {factors_path}...")
    factors_df = pd.read_csv(factors_path, parse_dates=["date"])
    print(f"  Loaded {len(factors_df):,} rows, {factors_df['ticker'].nunique()} stocks, "
          f"{factors_df['date'].nunique()} months")
    
    # Generate comprehensive report
    report = generate_feature_quality_report(
        factors_df,
        correlation_threshold=0.75,
        coverage_threshold=0.30,
        include_incremental_ic=True
    )
    
    # Generate recommendations
    recommendations = generate_feature_recommendations(
        ranking_df=report["overall_ranking"],
        corr_pairs=report["correlated_pairs"],
        coverage_df=report["coverage"],
        incremental_ic_df=report.get("incremental_ic"),
        low_ic_threshold=0.01,
        high_missing_threshold=0.30,
        high_corr_threshold=0.75,
        negative_incremental_threshold=-0.001
    )
    
    # Save all reports
    save_feature_quality_report(report, recommendations, output_dir="reports")

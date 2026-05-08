"""
test_feature_analyzer.py — Unit tests for feature_analyzer.py
==============================================================
Tests the feature analysis framework to ensure correct computation
of IC, correlations, and stability metrics.
"""

import pandas as pd
import numpy as np
from feature_analyzer import (
    compute_monthly_ic,
    compute_pairwise_correlations,
    compute_feature_stability,
    analyze_feature_coverage,
    compute_incremental_ic,
    generate_feature_quality_report
)


def test_compute_monthly_ic():
    """Test monthly IC computation."""
    print("\n=== Testing compute_monthly_ic ===")
    
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=12, freq="ME")
    tickers = [f"STOCK{i}" for i in range(50)]
    
    records = []
    for date in dates:
        for ticker in tickers:
            # Create feature with some predictive power
            feature_val = np.random.randn()
            target_val = 0.5 * feature_val + 0.5 * np.random.randn()
            
            records.append({
                "date": date,
                "ticker": ticker,
                "TestFeature": feature_val,
                "Next_Month_Return": target_val
            })
    
    df = pd.DataFrame(records)
    
    # Compute IC
    ic_df = compute_monthly_ic(df, features=["TestFeature"])
    
    # Verify structure
    assert "date" in ic_df.columns
    assert "feature" in ic_df.columns
    assert "ic" in ic_df.columns
    assert len(ic_df) == 12  # One per month
    
    # Verify IC values are reasonable (should be positive due to correlation)
    mean_ic = ic_df["ic"].mean()
    print(f"  Mean IC: {mean_ic:.4f}")
    assert mean_ic > 0, "Expected positive IC for correlated data"
    
    print("  ✓ compute_monthly_ic test passed")


def test_compute_pairwise_correlations():
    """Test pairwise correlation computation."""
    print("\n=== Testing compute_pairwise_correlations ===")
    
    # Create synthetic data with correlated features
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=12, freq="ME")
    tickers = [f"STOCK{i}" for i in range(50)]
    
    records = []
    for date in dates:
        for ticker in tickers:
            base = np.random.randn()
            records.append({
                "date": date,
                "ticker": ticker,
                "Feature1": base,
                "Feature2": 0.9 * base + 0.1 * np.random.randn(),  # Highly correlated
                "Feature3": np.random.randn(),  # Independent
                "Next_Month_Return": np.random.randn()
            })
    
    df = pd.DataFrame(records)
    
    # Compute correlations
    corr_matrix, flagged_pairs = compute_pairwise_correlations(
        df, features=["Feature1", "Feature2", "Feature3"], threshold=0.75
    )
    
    # Verify structure
    assert not corr_matrix.empty
    assert "Feature1" in corr_matrix.index
    assert "Feature2" in corr_matrix.columns
    
    # Verify Feature1 and Feature2 are flagged as correlated
    assert len(flagged_pairs) > 0, "Expected to find correlated pairs"
    pair_features = [(p[0], p[1]) for p in flagged_pairs]
    assert ("Feature1", "Feature2") in pair_features or ("Feature2", "Feature1") in pair_features
    
    print(f"  Found {len(flagged_pairs)} correlated pairs")
    print("  ✓ compute_pairwise_correlations test passed")


def test_compute_feature_stability():
    """Test feature stability computation."""
    print("\n=== Testing compute_feature_stability ===")
    
    # Create synthetic data with stable and unstable features
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=12, freq="ME")
    tickers = [f"STOCK{i}" for i in range(50)]
    
    # Create stable feature (persistent rankings)
    stable_values = {ticker: np.random.randn() for ticker in tickers}
    
    records = []
    for date in dates:
        for ticker in tickers:
            records.append({
                "date": date,
                "ticker": ticker,
                "StableFeature": stable_values[ticker] + 0.1 * np.random.randn(),
                "UnstableFeature": np.random.randn(),  # Random each month
                "Next_Month_Return": np.random.randn()
            })
    
    df = pd.DataFrame(records)
    
    # Compute stability
    stability_df = compute_feature_stability(
        df, features=["StableFeature", "UnstableFeature"]
    )
    
    # Verify structure
    assert "feature" in stability_df.columns
    assert "mean_stability" in stability_df.columns
    assert len(stability_df) == 2
    
    # Verify stable feature has higher stability
    stable_row = stability_df[stability_df["feature"] == "StableFeature"].iloc[0]
    unstable_row = stability_df[stability_df["feature"] == "UnstableFeature"].iloc[0]
    
    print(f"  StableFeature stability: {stable_row['mean_stability']:.4f}")
    print(f"  UnstableFeature stability: {unstable_row['mean_stability']:.4f}")
    
    assert stable_row["mean_stability"] > unstable_row["mean_stability"], \
        "Expected stable feature to have higher stability"
    
    print("  ✓ compute_feature_stability test passed")


def test_analyze_feature_coverage():
    """Test feature coverage analysis."""
    print("\n=== Testing analyze_feature_coverage ===")
    
    # Create synthetic data with missing values
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=12, freq="ME")
    tickers = [f"STOCK{i}" for i in range(100)]
    
    records = []
    for date in dates:
        for ticker in tickers:
            records.append({
                "date": date,
                "ticker": ticker,
                "FullCoverage": np.random.randn(),
                "PartialCoverage": np.random.randn() if np.random.rand() > 0.5 else np.nan,
                "Next_Month_Return": np.random.randn()
            })
    
    df = pd.DataFrame(records)
    
    # Compute coverage
    coverage_df = analyze_feature_coverage(
        df, features=["FullCoverage", "PartialCoverage"], threshold=0.30
    )
    
    # Verify structure
    assert "feature" in coverage_df.columns
    assert "missing_rate" in coverage_df.columns
    assert "coverage_rate" in coverage_df.columns
    assert len(coverage_df) == 2
    
    # Verify FullCoverage has 0% missing
    full_row = coverage_df[coverage_df["feature"] == "FullCoverage"].iloc[0]
    assert full_row["missing_rate"] == 0.0
    
    # Verify PartialCoverage has ~50% missing
    partial_row = coverage_df[coverage_df["feature"] == "PartialCoverage"].iloc[0]
    print(f"  PartialCoverage missing rate: {partial_row['missing_rate']:.1%}")
    assert 0.4 < partial_row["missing_rate"] < 0.6, "Expected ~50% missing rate"
    
    print("  ✓ analyze_feature_coverage test passed")


def test_compute_incremental_ic():
    """Test incremental IC contribution measurement."""
    print("\n=== Testing compute_incremental_ic ===")
    
    # Create synthetic data where Feature1 adds value, Feature2 doesn't
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=12, freq="ME")
    tickers = [f"STOCK{i}" for i in range(50)]
    
    records = []
    for date in dates:
        for ticker in tickers:
            base_signal = np.random.randn()
            # Feature1 has predictive power
            feature1 = base_signal + 0.2 * np.random.randn()
            # Feature2 is pure noise
            feature2 = np.random.randn()
            # Target is correlated with base_signal (and thus Feature1)
            target = 0.6 * base_signal + 0.4 * np.random.randn()
            
            records.append({
                "date": date,
                "ticker": ticker,
                "Feature1": feature1,
                "Feature2": feature2,
                "Next_Month_Return": target
            })
    
    df = pd.DataFrame(records)
    
    # Compute incremental IC
    incremental_df = compute_incremental_ic(
        df, features=["Feature1", "Feature2"]
    )
    
    # Verify structure
    assert "feature" in incremental_df.columns
    assert "baseline_ic" in incremental_df.columns
    assert "full_ic" in incremental_df.columns
    assert "incremental_ic" in incremental_df.columns
    assert len(incremental_df) == 2
    
    # Verify Feature1 has positive incremental IC
    feat1_row = incremental_df[incremental_df["feature"] == "Feature1"].iloc[0]
    feat2_row = incremental_df[incremental_df["feature"] == "Feature2"].iloc[0]
    
    print(f"  Feature1 incremental IC: {feat1_row['incremental_ic']:+.4f}")
    print(f"  Feature2 incremental IC: {feat2_row['incremental_ic']:+.4f}")
    
    # Feature1 should have higher incremental IC than Feature2
    assert feat1_row["incremental_ic"] > feat2_row["incremental_ic"], \
        "Expected Feature1 to have higher incremental IC than Feature2"
    
    print("  ✓ compute_incremental_ic test passed")


def test_generate_feature_quality_report():
    """Test comprehensive report generation."""
    print("\n=== Testing generate_feature_quality_report ===")
    
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=12, freq="ME")
    tickers = [f"STOCK{i}" for i in range(50)]
    
    records = []
    for date in dates:
        for ticker in tickers:
            base = np.random.randn()
            records.append({
                "date": date,
                "ticker": ticker,
                "Feature1": base,
                "Feature2": 0.8 * base + 0.2 * np.random.randn(),
                "Feature3": np.random.randn(),
                "Next_Month_Return": 0.3 * base + 0.7 * np.random.randn()
            })
    
    df = pd.DataFrame(records)
    
    # Generate report with incremental IC
    report = generate_feature_quality_report(
        df, features=["Feature1", "Feature2", "Feature3"],
        include_incremental_ic=True
    )
    
    # Verify all components are present
    assert "ic_monthly" in report
    assert "ic_summary" in report
    assert "correlation_matrix" in report
    assert "correlated_pairs" in report
    assert "stability" in report
    assert "coverage" in report
    assert "incremental_ic" in report
    assert "overall_ranking" in report
    
    # Verify data structures
    assert not report["ic_monthly"].empty
    assert not report["ic_summary"].empty
    assert not report["stability"].empty
    assert not report["coverage"].empty
    assert not report["incremental_ic"].empty
    assert not report["overall_ranking"].empty
    
    # Verify ranking has composite score
    assert "composite_score" in report["overall_ranking"].columns
    
    # Verify incremental IC is in ranking
    assert "incremental_ic" in report["overall_ranking"].columns
    
    print("  ✓ generate_feature_quality_report test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("  FEATURE ANALYZER UNIT TESTS")
    print("="*70)
    
    test_compute_monthly_ic()
    test_compute_pairwise_correlations()
    test_compute_feature_stability()
    test_analyze_feature_coverage()
    test_compute_incremental_ic()
    test_generate_feature_quality_report()
    
    print("\n" + "="*70)
    print("  ✓ ALL TESTS PASSED")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()

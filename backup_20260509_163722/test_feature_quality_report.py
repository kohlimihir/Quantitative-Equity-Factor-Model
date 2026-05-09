"""
test_feature_quality_report.py — Unit Tests for Task 3.4
=========================================================
Tests for feature quality report generation, correlation heatmap,
and feature recommendation system.

Requirements: 2.10
"""

import pandas as pd
import numpy as np
import os
from feature_analyzer import (
    generate_feature_quality_report,
    generate_feature_recommendations,
    plot_correlation_heatmap,
    save_feature_quality_report
)


def create_test_data():
    """Create synthetic test data with known characteristics."""
    np.random.seed(42)
    
    dates = pd.date_range("2020-01-01", periods=24, freq="ME")
    tickers = [f"STOCK{i}" for i in range(100)]
    
    records = []
    for date in dates:
        for ticker in tickers:
            # Create features with different characteristics
            base_signal = np.random.randn()
            
            # GoodFeature1: High IC, stable, good coverage
            good_feat1 = base_signal + 0.1 * np.random.randn()
            
            # GoodFeature2: High IC, stable, good coverage
            good_feat2 = base_signal + 0.15 * np.random.randn()
            
            # CorrelatedFeature: Highly correlated with GoodFeature1
            corr_feat = 0.9 * good_feat1 + 0.1 * np.random.randn()
            
            # LowICFeature: Low predictive power
            low_ic_feat = np.random.randn()
            
            # UnstableFeature: High IC but unstable
            unstable_feat = base_signal + np.random.randn()
            
            # HighMissingFeature: Good IC but high missing rate
            high_missing_feat = base_signal if np.random.rand() > 0.4 else np.nan
            
            # NegativeICFeature: Negative incremental IC
            negative_ic_feat = -0.3 * base_signal + 0.7 * np.random.randn()
            
            # Target correlated with base_signal
            target = 0.5 * base_signal + 0.5 * np.random.randn()
            
            records.append({
                "date": date,
                "ticker": ticker,
                "GoodFeature1": good_feat1,
                "GoodFeature2": good_feat2,
                "CorrelatedFeature": corr_feat,
                "LowICFeature": low_ic_feat,
                "UnstableFeature": unstable_feat,
                "HighMissingFeature": high_missing_feat,
                "NegativeICFeature": negative_ic_feat,
                "Next_Month_Return": target
            })
    
    return pd.DataFrame(records)


def test_generate_feature_quality_report():
    """Test comprehensive feature quality report generation."""
    print("\n=== Testing generate_feature_quality_report ===")
    
    df = create_test_data()
    
    features = [
        "GoodFeature1", "GoodFeature2", "CorrelatedFeature",
        "LowICFeature", "UnstableFeature", "HighMissingFeature",
        "NegativeICFeature"
    ]
    
    # Generate report
    report = generate_feature_quality_report(
        df,
        features=features,
        correlation_threshold=0.75,
        coverage_threshold=0.30,
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
    
    # Verify ranking has all required columns
    required_cols = [
        "feature", "mean_ic", "mean_stability", "coverage_rate",
        "composite_score", "incremental_ic"
    ]
    for col in required_cols:
        assert col in report["overall_ranking"].columns, f"Missing column: {col}"
    
    # Verify correlated pairs detected
    assert len(report["correlated_pairs"]) > 0, "Should detect correlated pairs"
    
    # Verify GoodFeature1 and CorrelatedFeature are flagged
    pair_features = [(p[0], p[1]) for p in report["correlated_pairs"]]
    assert any(
        ("GoodFeature1" in pair and "CorrelatedFeature" in pair)
        for pair in pair_features
    ), "Should detect GoodFeature1 and CorrelatedFeature correlation"
    
    print("  ✓ Report structure validated")
    print(f"  ✓ Found {len(report['correlated_pairs'])} correlated pairs")
    print("  ✓ generate_feature_quality_report test passed")
    
    return report


def test_generate_feature_recommendations():
    """Test feature recommendation system."""
    print("\n=== Testing generate_feature_recommendations ===")
    
    df = create_test_data()
    
    features = [
        "GoodFeature1", "GoodFeature2", "CorrelatedFeature",
        "LowICFeature", "UnstableFeature", "HighMissingFeature",
        "NegativeICFeature"
    ]
    
    # Generate report first
    report = generate_feature_quality_report(
        df,
        features=features,
        correlation_threshold=0.75,
        coverage_threshold=0.30,
        include_incremental_ic=True
    )
    
    # Generate recommendations
    recommendations = generate_feature_recommendations(
        ranking_df=report["overall_ranking"],
        corr_pairs=report["correlated_pairs"],
        coverage_df=report["coverage"],
        incremental_ic_df=report["incremental_ic"],
        low_ic_threshold=0.01,
        high_missing_threshold=0.30,
        high_corr_threshold=0.75,
        negative_incremental_threshold=-0.001
    )
    
    # Verify structure
    assert "remove" in recommendations
    assert "keep" in recommendations
    assert "engineer" in recommendations
    assert "correlated_pairs" in recommendations
    
    # Verify recommendations are lists
    assert isinstance(recommendations["remove"], list)
    assert isinstance(recommendations["keep"], list)
    assert isinstance(recommendations["engineer"], list)
    assert isinstance(recommendations["correlated_pairs"], list)
    
    # Verify we have some recommendations
    assert len(recommendations["keep"]) > 0, "Should have features to keep"
    
    # Verify LowICFeature is recommended for removal
    remove_features = [r["feature"] for r in recommendations["remove"]]
    assert "LowICFeature" in remove_features, "LowICFeature should be recommended for removal"
    
    # Verify HighMissingFeature is flagged
    assert any(
        "HighMissingFeature" in r["feature"]
        for r in recommendations["remove"] + recommendations["engineer"]
    ), "HighMissingFeature should be flagged"
    
    # Verify correlated pairs are addressed
    assert len(recommendations["correlated_pairs"]) > 0, "Should have correlated pairs"
    
    # Verify recommendation structure
    if recommendations["remove"]:
        first_remove = recommendations["remove"][0]
        assert "feature" in first_remove
        assert "reason" in first_remove
        assert "priority" in first_remove
    
    if recommendations["keep"]:
        first_keep = recommendations["keep"][0]
        assert "feature" in first_keep
        assert "reason" in first_keep
        assert "rank" in first_keep
    
    if recommendations["engineer"]:
        first_engineer = recommendations["engineer"][0]
        assert "feature" in first_engineer
        assert "opportunity" in first_engineer
        assert "current_ic" in first_engineer
    
    if recommendations["correlated_pairs"]:
        first_pair = recommendations["correlated_pairs"][0]
        assert "feature1" in first_pair
        assert "feature2" in first_pair
        assert "correlation" in first_pair
        assert "recommendation" in first_pair
    
    print("  ✓ Recommendation structure validated")
    print(f"  ✓ Remove: {len(recommendations['remove'])} features")
    print(f"  ✓ Keep: {len(recommendations['keep'])} features")
    print(f"  ✓ Engineer: {len(recommendations['engineer'])} opportunities")
    print(f"  ✓ Correlated pairs: {len(recommendations['correlated_pairs'])} pairs")
    print("  ✓ generate_feature_recommendations test passed")
    
    return recommendations


def test_plot_correlation_heatmap():
    """Test correlation heatmap generation."""
    print("\n=== Testing plot_correlation_heatmap ===")
    
    # Create a small correlation matrix
    features = ["Feature1", "Feature2", "Feature3", "Feature4"]
    corr_data = np.array([
        [1.0, 0.8, 0.3, -0.2],
        [0.8, 1.0, 0.4, -0.1],
        [0.3, 0.4, 1.0, 0.6],
        [-0.2, -0.1, 0.6, 1.0]
    ])
    corr_matrix = pd.DataFrame(corr_data, index=features, columns=features)
    
    # Test heatmap generation
    output_path = "reports/test_correlation_heatmap.png"
    os.makedirs("reports", exist_ok=True)
    
    plot_correlation_heatmap(
        corr_matrix,
        output_path=output_path,
        figsize=(8, 6)
    )
    
    # Check if file was created (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt
        assert os.path.exists(output_path), "Heatmap file should be created"
        print(f"  ✓ Heatmap saved to {output_path}")
        
        # Clean up test file
        if os.path.exists(output_path):
            os.remove(output_path)
            print("  ✓ Test file cleaned up")
    except ImportError:
        print("  ⚠️  Matplotlib not available, skipping heatmap file check")
    
    print("  ✓ plot_correlation_heatmap test passed")


def test_save_feature_quality_report():
    """Test comprehensive report saving."""
    print("\n=== Testing save_feature_quality_report ===")
    
    df = create_test_data()
    
    features = [
        "GoodFeature1", "GoodFeature2", "CorrelatedFeature",
        "LowICFeature", "UnstableFeature", "HighMissingFeature"
    ]
    
    # Generate report
    report = generate_feature_quality_report(
        df,
        features=features,
        correlation_threshold=0.75,
        coverage_threshold=0.30,
        include_incremental_ic=True
    )
    
    # Generate recommendations
    recommendations = generate_feature_recommendations(
        ranking_df=report["overall_ranking"],
        corr_pairs=report["correlated_pairs"],
        coverage_df=report["coverage"],
        incremental_ic_df=report["incremental_ic"],
        low_ic_threshold=0.01,
        high_missing_threshold=0.30,
        high_corr_threshold=0.75
    )
    
    # Save reports
    output_dir = "reports"
    save_feature_quality_report(report, recommendations, output_dir=output_dir)
    
    # Verify CSV files were created
    expected_files = [
        "feature_ic_monthly.csv",
        "feature_ic_summary.csv",
        "feature_stability.csv",
        "feature_coverage.csv",
        "feature_ranking.csv",
        "feature_incremental_ic.csv",
        "feature_correlation_matrix.csv",
        "feature_recommendations.txt"
    ]
    
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        assert os.path.exists(filepath), f"Missing file: {filename}"
        print(f"  ✓ {filename} created")
    
    # Verify recommendations text file content
    recommendations_path = os.path.join(output_dir, "feature_recommendations.txt")
    with open(recommendations_path, "r") as f:
        content = f.read()
        assert "FEATURES TO REMOVE" in content
        assert "FEATURES TO KEEP" in content
        assert "ENGINEERING OPPORTUNITIES" in content
        assert "CORRELATED PAIRS" in content
        print("  ✓ Recommendations text file has correct structure")
    
    # Check if heatmap was created (if matplotlib available)
    heatmap_path = os.path.join(output_dir, "feature_correlation_heatmap.png")
    try:
        import matplotlib.pyplot as plt
        if os.path.exists(heatmap_path):
            print(f"  ✓ Correlation heatmap created")
    except ImportError:
        print("  ⚠️  Matplotlib not available, heatmap not created")
    
    print("  ✓ save_feature_quality_report test passed")


def test_recommendations_identify_low_ic_features():
    """Test that recommendations correctly identify low IC features."""
    print("\n=== Testing low IC feature identification ===")
    
    df = create_test_data()
    
    features = ["GoodFeature1", "LowICFeature"]
    
    report = generate_feature_quality_report(
        df, features=features, include_incremental_ic=True
    )
    
    recommendations = generate_feature_recommendations(
        ranking_df=report["overall_ranking"],
        corr_pairs=report["correlated_pairs"],
        coverage_df=report["coverage"],
        incremental_ic_df=report["incremental_ic"],
        low_ic_threshold=0.01
    )
    
    # LowICFeature should be recommended for removal
    remove_features = [r["feature"] for r in recommendations["remove"]]
    assert "LowICFeature" in remove_features
    
    # GoodFeature1 should be in keep list
    keep_features = [r["feature"] for r in recommendations["keep"]]
    assert "GoodFeature1" in keep_features
    
    print("  ✓ Low IC features correctly identified")


def test_recommendations_identify_high_missing_features():
    """Test that recommendations correctly identify high missing rate features."""
    print("\n=== Testing high missing rate feature identification ===")
    
    df = create_test_data()
    
    features = ["GoodFeature1", "HighMissingFeature"]
    
    report = generate_feature_quality_report(
        df, features=features, include_incremental_ic=True
    )
    
    recommendations = generate_feature_recommendations(
        ranking_df=report["overall_ranking"],
        corr_pairs=report["correlated_pairs"],
        coverage_df=report["coverage"],
        incremental_ic_df=report["incremental_ic"],
        high_missing_threshold=0.30
    )
    
    # HighMissingFeature should be flagged (either remove or engineer)
    all_flagged = (
        [r["feature"] for r in recommendations["remove"]] +
        [r["feature"] for r in recommendations["engineer"]]
    )
    assert "HighMissingFeature" in all_flagged
    
    print("  ✓ High missing rate features correctly identified")


def test_recommendations_identify_correlated_pairs():
    """Test that recommendations correctly identify correlated pairs."""
    print("\n=== Testing correlated pair identification ===")
    
    df = create_test_data()
    
    features = ["GoodFeature1", "CorrelatedFeature"]
    
    report = generate_feature_quality_report(
        df, features=features, correlation_threshold=0.75, include_incremental_ic=True
    )
    
    recommendations = generate_feature_recommendations(
        ranking_df=report["overall_ranking"],
        corr_pairs=report["correlated_pairs"],
        coverage_df=report["coverage"],
        incremental_ic_df=report["incremental_ic"],
        high_corr_threshold=0.75
    )
    
    # Should have correlated pairs
    assert len(recommendations["correlated_pairs"]) > 0
    
    # Check that GoodFeature1 and CorrelatedFeature are in the pairs
    pair_features = set()
    for pair in recommendations["correlated_pairs"]:
        pair_features.add(pair["feature1"])
        pair_features.add(pair["feature2"])
    
    assert "GoodFeature1" in pair_features or "CorrelatedFeature" in pair_features
    
    print("  ✓ Correlated pairs correctly identified")


def test_empty_correlation_matrix_handling():
    """Test that empty correlation matrix is handled gracefully."""
    print("\n=== Testing empty correlation matrix handling ===")
    
    # Create empty correlation matrix
    empty_corr = pd.DataFrame()
    
    # Should not raise an error
    plot_correlation_heatmap(empty_corr, output_path="reports/test_empty.png")
    
    print("  ✓ Empty correlation matrix handled gracefully")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("  FEATURE QUALITY REPORT TESTS (Task 3.4)")
    print("="*70)
    
    test_generate_feature_quality_report()
    test_generate_feature_recommendations()
    test_plot_correlation_heatmap()
    test_save_feature_quality_report()
    test_recommendations_identify_low_ic_features()
    test_recommendations_identify_high_missing_features()
    test_recommendations_identify_correlated_pairs()
    test_empty_correlation_matrix_handling()
    
    print("\n" + "="*70)
    print("  ✓ ALL TESTS PASSED")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()

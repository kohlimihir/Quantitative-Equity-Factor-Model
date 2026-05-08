"""
test_feature_engineering.py — Unit Tests for Feature Engineering Enhancements
==============================================================================
Tests for Task 3.3: interaction features, non-linear transformations,
and sector-relative features.

Requirements: 2.5, 2.6, 2.9
"""

import pandas as pd
import numpy as np
from data_loader import (
    add_interaction_features,
    add_nonlinear_transformations,
    add_sector_relative_features,
    apply_feature_engineering,
    FEATURE_GROUPS,
    TARGET
)


def create_test_data():
    """Create synthetic test data for feature engineering tests."""
    np.random.seed(42)
    
    dates = pd.date_range("2020-01-01", periods=6, freq="ME")
    tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "BAC", "GS"]
    sectors = ["Technology", "Technology", "Technology", "Financials", "Financials", "Financials"]
    
    data = []
    for date in dates:
        for ticker, sector in zip(tickers, sectors):
            data.append({
                "date": date,
                "ticker": ticker,
                "sector": sector,
                "Mom_12_1": np.random.uniform(-0.2, 0.3),
                "Mom_6_1": np.random.uniform(-0.15, 0.25),
                "Mom_1": np.random.uniform(-0.1, 0.15),
                "Vol_12": np.random.uniform(0.1, 0.4),
                "IdioVol": np.random.uniform(0.05, 0.3),
                "Beta_12": np.random.uniform(0.5, 1.5),
                "High52W": np.random.uniform(0.7, 1.0),
                "Trend_MA": np.random.uniform(0.9, 1.1),
                "MaxRet_1M": np.random.uniform(0.01, 0.05),
                "PB_ratio": np.random.uniform(1.0, 5.0),
                "PE_TTM": np.random.uniform(10.0, 30.0),
                "EV_EBITDA": np.random.uniform(8.0, 20.0),
                "ROE": np.random.uniform(0.05, 0.25),
                "GrossMargin": np.random.uniform(0.3, 0.7),
                "CashFlowYield": np.random.uniform(0.02, 0.08),
                "RevGrowth_YoY": np.random.uniform(-0.1, 0.2),
                "EarnGrowth_YoY": np.random.uniform(-0.15, 0.25),
                "LogMktCap": np.random.uniform(20.0, 25.0),
                "VolRatio": np.random.uniform(0.8, 1.2),
                TARGET: np.random.uniform(-0.05, 0.05),
            })
    
    return pd.DataFrame(data)


def test_add_interaction_features_disabled():
    """Test that interaction features are not added when disabled."""
    print("\n=== Testing add_interaction_features (disabled) ===")
    
    df = create_test_data()
    original_columns = set(df.columns)
    
    config = {"enable_interactions": False}
    result = add_interaction_features(df.copy(), config)
    
    # Should not add any new columns
    assert set(result.columns) == original_columns
    print("  ✓ No interaction features added when disabled")


def test_add_interaction_features_enabled():
    """Test that interaction features are correctly created."""
    print("\n=== Testing add_interaction_features (enabled) ===")
    
    df = create_test_data()
    original_columns = set(df.columns)
    
    config = {
        "enable_interactions": True,
        "interaction_pairs": [
            ("G1 Momentum", "G4 Value"),
            ("G2 Risk", "G7 Size"),
        ]
    }
    
    result = add_interaction_features(df.copy(), config)
    
    # Should add new interaction columns
    new_columns = set(result.columns) - original_columns
    assert len(new_columns) > 0
    print(f"  ✓ Added {len(new_columns)} interaction features")
    
    # Check specific interactions exist
    assert "Mom_12_1_x_PB_ratio" in result.columns
    assert "Mom_12_1_x_PE_TTM" in result.columns
    assert "Vol_12_x_LogMktCap" in result.columns
    print("  ✓ Expected interaction features created")
    
    # Verify interaction values are correct (product of two features)
    for idx in result.index[:5]:
        mom_val = result.loc[idx, "Mom_12_1"]
        pb_val = result.loc[idx, "PB_ratio"]
        interaction_val = result.loc[idx, "Mom_12_1_x_PB_ratio"]
        
        if not np.isnan(mom_val) and not np.isnan(pb_val):
            expected = mom_val * pb_val
            assert np.isclose(interaction_val, expected, rtol=1e-5) or np.isnan(interaction_val)
    
    print("  ✓ Interaction values computed correctly")


def test_add_interaction_features_handles_nan():
    """Test that interaction features handle NaN values gracefully."""
    print("\n=== Testing add_interaction_features (NaN handling) ===")
    
    df = create_test_data()
    
    # Introduce some NaN values
    df.loc[0, "Mom_12_1"] = np.nan
    df.loc[1, "PB_ratio"] = np.nan
    
    config = {
        "enable_interactions": True,
        "interaction_pairs": [("G1 Momentum", "G4 Value")]
    }
    
    result = add_interaction_features(df.copy(), config)
    
    # Check that NaN propagates correctly
    assert np.isnan(result.loc[0, "Mom_12_1_x_PB_ratio"])
    assert np.isnan(result.loc[1, "Mom_12_1_x_PB_ratio"])
    print("  ✓ NaN values handled correctly in interactions")


def test_add_nonlinear_transformations_disabled():
    """Test that non-linear transformations are not added when disabled."""
    print("\n=== Testing add_nonlinear_transformations (disabled) ===")
    
    df = create_test_data()
    original_columns = set(df.columns)
    
    config = {"enable_nonlinear": False}
    result = add_nonlinear_transformations(df.copy(), config)
    
    # Should not add any new columns
    assert set(result.columns) == original_columns
    print("  ✓ No non-linear features added when disabled")


def test_add_nonlinear_transformations_enabled():
    """Test that non-linear transformations are correctly created."""
    print("\n=== Testing add_nonlinear_transformations (enabled) ===")
    
    df = create_test_data()
    original_columns = set(df.columns)
    
    config = {
        "enable_nonlinear": True,
        "nonlinear_features": ["Mom_12_1", "Vol_12", "LogMktCap"]
    }
    
    result = add_nonlinear_transformations(df.copy(), config)
    
    # Should add 3 transformations per feature (log, sqrt, rank)
    new_columns = set(result.columns) - original_columns
    expected_count = 3 * 3  # 3 features × 3 transformations
    assert len(new_columns) == expected_count
    print(f"  ✓ Added {len(new_columns)} non-linear features")
    
    # Check specific transformations exist
    assert "Mom_12_1_log" in result.columns
    assert "Mom_12_1_sqrt" in result.columns
    assert "Mom_12_1_rank" in result.columns
    assert "Vol_12_log" in result.columns
    assert "LogMktCap_sqrt" in result.columns
    print("  ✓ Expected transformation features created")


def test_nonlinear_log_transformation():
    """Test log transformation handles positive and negative values."""
    print("\n=== Testing log transformation ===")
    
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="ME"),
        "ticker": ["AAPL"] * 5,
        "sector": ["Technology"] * 5,
        "Mom_12_1": [0.1, -0.1, 0.0, 0.5, -0.3],
        TARGET: [0.01] * 5,
    })
    
    config = {
        "enable_nonlinear": True,
        "nonlinear_features": ["Mom_12_1"]
    }
    
    result = add_nonlinear_transformations(df.copy(), config)
    
    # Check log transformation values
    assert result.loc[0, "Mom_12_1_log"] > 0  # log(1 + 0.1) > 0
    assert result.loc[1, "Mom_12_1_log"] < 0  # -log(1 + 0.1) < 0
    assert result.loc[2, "Mom_12_1_log"] == 0  # log(1 + 0) = 0
    print("  ✓ Log transformation handles positive/negative/zero values")


def test_nonlinear_sqrt_transformation():
    """Test sqrt transformation handles positive and negative values."""
    print("\n=== Testing sqrt transformation ===")
    
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="ME"),
        "ticker": ["AAPL"] * 5,
        "sector": ["Technology"] * 5,
        "Mom_12_1": [0.25, -0.25, 0.0, 1.0, -1.0],
        TARGET: [0.01] * 5,
    })
    
    config = {
        "enable_nonlinear": True,
        "nonlinear_features": ["Mom_12_1"]
    }
    
    result = add_nonlinear_transformations(df.copy(), config)
    
    # Check sqrt transformation values
    assert np.isclose(result.loc[0, "Mom_12_1_sqrt"], 0.5)  # sqrt(0.25) = 0.5
    assert np.isclose(result.loc[1, "Mom_12_1_sqrt"], -0.5)  # -sqrt(0.25) = -0.5
    assert result.loc[2, "Mom_12_1_sqrt"] == 0.0  # sqrt(0) = 0
    assert np.isclose(result.loc[3, "Mom_12_1_sqrt"], 1.0)  # sqrt(1) = 1
    assert np.isclose(result.loc[4, "Mom_12_1_sqrt"], -1.0)  # -sqrt(1) = -1
    print("  ✓ Sqrt transformation handles positive/negative/zero values")


def test_nonlinear_rank_transformation():
    """Test rank transformation creates percentile ranks."""
    print("\n=== Testing rank transformation ===")
    
    df = pd.DataFrame({
        "date": ["2020-01-31"] * 5,
        "ticker": ["A", "B", "C", "D", "E"],
        "sector": ["Tech"] * 5,
        "Mom_12_1": [0.1, 0.2, 0.3, 0.4, 0.5],  # Ascending order
        TARGET: [0.01] * 5,
    })
    
    config = {
        "enable_nonlinear": True,
        "nonlinear_features": ["Mom_12_1"]
    }
    
    result = add_nonlinear_transformations(df.copy(), config)
    
    # Check rank values are percentiles (0 to 1)
    ranks = result["Mom_12_1_rank"].values
    assert all(0 <= r <= 1 for r in ranks)
    assert ranks[0] < ranks[1] < ranks[2] < ranks[3] < ranks[4]
    print("  ✓ Rank transformation creates valid percentile ranks")


def test_add_sector_relative_features_disabled():
    """Test that sector-relative features are not added when disabled."""
    print("\n=== Testing add_sector_relative_features (disabled) ===")
    
    df = create_test_data()
    original_columns = set(df.columns)
    
    config = {"enable_sector_relative": False}
    result = add_sector_relative_features(df.copy(), config)
    
    # Should not add any new columns
    assert set(result.columns) == original_columns
    print("  ✓ No sector-relative features added when disabled")


def test_add_sector_relative_features_enabled():
    """Test that sector-relative features are correctly created."""
    print("\n=== Testing add_sector_relative_features (enabled) ===")
    
    df = create_test_data()
    original_columns = set(df.columns)
    
    config = {
        "enable_sector_relative": True,
        "sector_relative_features": ["Mom_12_1", "Vol_12", "ROE"]
    }
    
    result = add_sector_relative_features(df.copy(), config)
    
    # Should add 2 transformations per feature (z-score, percentile)
    new_columns = set(result.columns) - original_columns
    expected_count = 3 * 2  # 3 features × 2 transformations
    assert len(new_columns) == expected_count
    print(f"  ✓ Added {len(new_columns)} sector-relative features")
    
    # Check specific transformations exist
    assert "Mom_12_1_sector_z" in result.columns
    assert "Mom_12_1_sector_pct" in result.columns
    assert "Vol_12_sector_z" in result.columns
    assert "ROE_sector_pct" in result.columns
    print("  ✓ Expected sector-relative features created")


def test_sector_relative_zscore():
    """Test sector-relative z-score transformation."""
    print("\n=== Testing sector-relative z-score ===")
    
    # Create data with clear sector differences
    df = pd.DataFrame({
        "date": ["2020-01-31"] * 6,
        "ticker": ["A", "B", "C", "D", "E", "F"],
        "sector": ["Tech", "Tech", "Tech", "Finance", "Finance", "Finance"],
        "Mom_12_1": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7],  # Finance has higher momentum
        TARGET: [0.01] * 6,
    })
    
    config = {
        "enable_sector_relative": True,
        "sector_relative_features": ["Mom_12_1"]
    }
    
    result = add_sector_relative_features(df.copy(), config)
    
    # Within each sector, z-scores should have mean ~0
    tech_z = result[result["sector"] == "Tech"]["Mom_12_1_sector_z"]
    finance_z = result[result["sector"] == "Finance"]["Mom_12_1_sector_z"]
    
    assert np.isclose(tech_z.mean(), 0.0, atol=1e-10)
    assert np.isclose(finance_z.mean(), 0.0, atol=1e-10)
    print("  ✓ Sector z-scores have mean ~0 within each sector")


def test_sector_relative_percentile():
    """Test sector-relative percentile transformation."""
    print("\n=== Testing sector-relative percentile ===")
    
    # Create data with clear sector differences
    df = pd.DataFrame({
        "date": ["2020-01-31"] * 6,
        "ticker": ["A", "B", "C", "D", "E", "F"],
        "sector": ["Tech", "Tech", "Tech", "Finance", "Finance", "Finance"],
        "Mom_12_1": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7],
        TARGET: [0.01] * 6,
    })
    
    config = {
        "enable_sector_relative": True,
        "sector_relative_features": ["Mom_12_1"]
    }
    
    result = add_sector_relative_features(df.copy(), config)
    
    # Within each sector, percentiles should be between 0 and 1
    tech_pct = result[result["sector"] == "Tech"]["Mom_12_1_sector_pct"]
    finance_pct = result[result["sector"] == "Finance"]["Mom_12_1_sector_pct"]
    
    assert all(0 <= p <= 1 for p in tech_pct)
    assert all(0 <= p <= 1 for p in finance_pct)
    print("  ✓ Sector percentiles are valid (0 to 1)")


def test_sector_relative_missing_sector_column():
    """Test that sector-relative features handle missing sector column."""
    print("\n=== Testing sector-relative with missing sector column ===")
    
    df = create_test_data()
    df = df.drop(columns=["sector"])  # Remove sector column
    
    original_columns = set(df.columns)
    
    config = {
        "enable_sector_relative": True,
        "sector_relative_features": ["Mom_12_1"]
    }
    
    result = add_sector_relative_features(df.copy(), config)
    
    # Should not add any new columns when sector is missing
    assert set(result.columns) == original_columns
    print("  ✓ Gracefully handles missing sector column")


def test_apply_feature_engineering_all_disabled():
    """Test apply_feature_engineering with all features disabled."""
    print("\n=== Testing apply_feature_engineering (all disabled) ===")
    
    df = create_test_data()
    original_columns = set(df.columns)
    
    config = {
        "enable_interactions": False,
        "enable_nonlinear": False,
        "enable_sector_relative": False,
    }
    
    result = apply_feature_engineering(df.copy(), config)
    
    # Should not add any new columns
    assert set(result.columns) == original_columns
    print("  ✓ No features added when all disabled")


def test_apply_feature_engineering_all_enabled():
    """Test apply_feature_engineering with all features enabled."""
    print("\n=== Testing apply_feature_engineering (all enabled) ===")
    
    df = create_test_data()
    original_columns = set(df.columns)
    
    config = {
        "enable_interactions": True,
        "enable_nonlinear": True,
        "enable_sector_relative": True,
        "interaction_pairs": [("G1 Momentum", "G4 Value")],
        "nonlinear_features": ["Mom_12_1", "Vol_12"],
        "sector_relative_features": ["Mom_12_1", "ROE"],
    }
    
    result = apply_feature_engineering(df.copy(), config)
    
    # Should add many new columns
    new_columns = set(result.columns) - original_columns
    assert len(new_columns) > 0
    print(f"  ✓ Added {len(new_columns)} total engineered features")
    
    # Check that features from all three categories exist
    interaction_features = [c for c in new_columns if "_x_" in c]
    nonlinear_features = [c for c in new_columns if any(x in c for x in ["_log", "_sqrt", "_rank"])]
    sector_features = [c for c in new_columns if "_sector_" in c]
    
    assert len(interaction_features) > 0
    assert len(nonlinear_features) > 0
    assert len(sector_features) > 0
    print(f"  ✓ Interaction features: {len(interaction_features)}")
    print(f"  ✓ Non-linear features: {len(nonlinear_features)}")
    print(f"  ✓ Sector-relative features: {len(sector_features)}")


def test_temporal_integrity_no_future_leakage():
    """Test that feature engineering maintains temporal integrity."""
    print("\n=== Testing temporal integrity (no future leakage) ===")
    
    df = create_test_data()
    
    config = {
        "enable_interactions": True,
        "enable_nonlinear": True,
        "enable_sector_relative": True,
        "interaction_pairs": [("G1 Momentum", "G4 Value")],
        "nonlinear_features": ["Mom_12_1"],
        "sector_relative_features": ["Mom_12_1"],
    }
    
    result = apply_feature_engineering(df.copy(), config)
    
    # All transformations should be computed within each date
    # (no future data should be used)
    
    # For rank transformations, check that ranks are computed per date
    for date in result["date"].unique():
        date_data = result[result["date"] == date]
        
        # Rank should be between 0 and 1
        if "Mom_12_1_rank" in date_data.columns:
            ranks = date_data["Mom_12_1_rank"].dropna()
            assert all(0 <= r <= 1 for r in ranks)
    
    # For sector-relative features, check that they're computed per date+sector
    for date in result["date"].unique():
        for sector in result["sector"].unique():
            sector_data = result[(result["date"] == date) & (result["sector"] == sector)]
            
            if len(sector_data) > 1 and "Mom_12_1_sector_z" in sector_data.columns:
                z_scores = sector_data["Mom_12_1_sector_z"].dropna()
                if len(z_scores) > 1:
                    # Z-scores within sector should have mean ~0
                    assert np.isclose(z_scores.mean(), 0.0, atol=1e-5)
    
    print("  ✓ All transformations maintain temporal integrity")


def test_imputation_applied_to_new_features():
    """Test that cross-sectional imputation is applied to new features."""
    print("\n=== Testing imputation for new features ===")
    
    df = create_test_data()
    
    # Introduce NaN values
    df.loc[0, "Mom_12_1"] = np.nan
    df.loc[1, "Vol_12"] = np.nan
    
    config = {
        "enable_interactions": True,
        "enable_nonlinear": True,
        "interaction_pairs": [("G1 Momentum", "G4 Value")],
        "nonlinear_features": ["Mom_12_1", "Vol_12"],
    }
    
    result = apply_feature_engineering(df.copy(), config)
    
    # Check that new features have imputation applied
    # (missing rate should be low after imputation)
    new_features = [c for c in result.columns 
                    if c not in df.columns and c not in ["date", "ticker", "sector", TARGET]]
    
    for feat in new_features:
        missing_rate = result[feat].isna().mean()
        # After imputation, missing rate should be low (not 100%)
        # Some features may still have NaN if entire date has NaN
        assert missing_rate < 0.5, f"Feature {feat} has high missing rate: {missing_rate}"
    
    print("  ✓ Cross-sectional imputation applied to new features")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  FEATURE ENGINEERING UNIT TESTS")
    print("="*70)
    
    # Run all tests
    test_add_interaction_features_disabled()
    test_add_interaction_features_enabled()
    test_add_interaction_features_handles_nan()
    
    test_add_nonlinear_transformations_disabled()
    test_add_nonlinear_transformations_enabled()
    test_nonlinear_log_transformation()
    test_nonlinear_sqrt_transformation()
    test_nonlinear_rank_transformation()
    
    test_add_sector_relative_features_disabled()
    test_add_sector_relative_features_enabled()
    test_sector_relative_zscore()
    test_sector_relative_percentile()
    test_sector_relative_missing_sector_column()
    
    test_apply_feature_engineering_all_disabled()
    test_apply_feature_engineering_all_enabled()
    
    test_temporal_integrity_no_future_leakage()
    test_imputation_applied_to_new_features()
    
    print("\n" + "="*70)
    print("  ALL TESTS PASSED ✓")
    print("="*70)

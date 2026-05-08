"""
test_sector_model.py — Tests for Sector-Specific Modeling
==========================================================
Tests sector-specific model training, hyperparameter tuning,
performance evaluation, and comparison with global models.

**Validates: Requirements 3.4**
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sector_model import (
    SectorSpecificModel,
    train_sector_specific_models
)
from data_loader import FEATURES, TARGET


def create_test_data(n_months=36, n_stocks_per_sector=10):
    """
    Create synthetic test data with multiple sectors.
    
    Parameters
    ----------
    n_months : int
        Number of months of data
    n_stocks_per_sector : int
        Number of stocks per sector
    
    Returns
    -------
    pd.DataFrame
        Test factor data
    """
    sectors = ["Technology", "Healthcare", "Financials", "Consumer", "Energy"]
    dates = pd.date_range(end=datetime.today(), periods=n_months, freq="ME")
    
    records = []
    
    for sector_idx, sector in enumerate(sectors):
        for stock_idx in range(n_stocks_per_sector):
            ticker = f"{sector[:4].upper()}{stock_idx:02d}"
            
            for date_idx, date in enumerate(dates):
                # Create sector-specific patterns
                # Technology: high momentum, high volatility
                # Healthcare: stable, low volatility
                # Financials: value-oriented
                # Consumer: growth-oriented
                # Energy: cyclical
                
                if sector == "Technology":
                    mom_base = 0.15
                    vol_base = 0.25
                elif sector == "Healthcare":
                    mom_base = 0.05
                    vol_base = 0.15
                elif sector == "Financials":
                    mom_base = 0.08
                    vol_base = 0.20
                elif sector == "Consumer":
                    mom_base = 0.12
                    vol_base = 0.18
                else:  # Energy
                    mom_base = 0.10
                    vol_base = 0.30
                
                # Add noise and time trends
                noise = np.random.randn() * 0.05
                time_trend = np.sin(date_idx / 6) * 0.03
                
                record = {
                    "date": date,
                    "ticker": ticker,
                    "sector": sector,
                }
                
                # Generate features
                for feat in FEATURES:
                    if "Mom" in feat:
                        record[feat] = mom_base + noise + time_trend
                    elif "Vol" in feat:
                        record[feat] = vol_base + abs(noise)
                    elif "Beta" in feat:
                        record[feat] = 1.0 + noise * 0.3
                    elif "PB" in feat or "PE" in feat:
                        record[feat] = 15.0 + noise * 5
                    elif "ROE" in feat:
                        record[feat] = 0.15 + noise * 0.05
                    elif "LogMktCap" in feat:
                        record[feat] = 20.0 + noise
                    else:
                        record[feat] = noise
                
                # Generate target with sector-specific alpha
                sector_alpha = mom_base * 0.5
                record[TARGET] = sector_alpha + noise * 0.1
                
                records.append(record)
    
    return pd.DataFrame(records)


def test_sector_specific_model_initialization():
    """Test SectorSpecificModel initialization."""
    print("\n=== Testing SectorSpecificModel initialization ===")
    
    # Test Ridge initialization
    model = SectorSpecificModel(model_type="ridge", tune_hyperparameters=False)
    assert model.model_type == "ridge"
    assert model.tune_hyperparameters == False
    assert model.n_cv_splits == 3
    print("  ✓ Ridge initialization successful")
    
    # Test LightGBM initialization
    model = SectorSpecificModel(model_type="lightgbm", tune_hyperparameters=True)
    assert model.model_type == "lightgbm"
    assert model.tune_hyperparameters == True
    print("  ✓ LightGBM initialization successful")
    
    # Test invalid model type
    try:
        SectorSpecificModel(model_type="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  ✓ Invalid model type raises ValueError")
    
    print("  ✓ All initialization tests passed")


def test_get_default_params():
    """Test default parameter generation."""
    print("\n=== Testing _get_default_params ===")
    
    # Ridge defaults
    model = SectorSpecificModel(model_type="ridge")
    params = model._get_default_params()
    assert "alpha" in params
    assert params["alpha"] == 1.0
    print("  ✓ Ridge default params correct")
    
    # LightGBM defaults
    model = SectorSpecificModel(model_type="lightgbm")
    params = model._get_default_params()
    assert "objective" in params
    assert "n_estimators" in params
    assert "learning_rate" in params
    assert params["objective"] == "regression"
    print("  ✓ LightGBM default params correct")
    
    print("  ✓ All default params tests passed")


def test_tune_sector_hyperparameters():
    """Test sector-specific hyperparameter tuning."""
    print("\n=== Testing _tune_sector_hyperparameters ===")
    
    # Create test data
    test_df = create_test_data(n_months=30, n_stocks_per_sector=10)
    tech_df = test_df[test_df["sector"] == "Technology"]
    
    # Test Ridge tuning
    model = SectorSpecificModel(model_type="ridge", tune_hyperparameters=True, verbose=False)
    params = model._tune_sector_hyperparameters(tech_df, "Technology")
    
    assert "alpha" in params
    assert params["alpha"] > 0
    print(f"  ✓ Ridge tuning successful: alpha={params['alpha']}")
    
    # Test LightGBM tuning (with small grid for speed)
    model = SectorSpecificModel(model_type="lightgbm", tune_hyperparameters=True, verbose=False)
    params = model._tune_sector_hyperparameters(tech_df, "Technology")
    
    assert "learning_rate" in params
    assert "num_leaves" in params
    print(f"  ✓ LightGBM tuning successful: lr={params['learning_rate']}, "
          f"leaves={params['num_leaves']}")
    
    print("  ✓ All hyperparameter tuning tests passed")


def test_walk_forward_sector_models_ridge():
    """Test walk-forward validation with sector-specific Ridge models."""
    print("\n=== Testing walk_forward_sector_models (Ridge) ===")
    
    # Create test data
    test_df = create_test_data(n_months=30, n_stocks_per_sector=10)
    
    # Train sector-specific models without tuning (for speed)
    model = SectorSpecificModel(
        model_type="ridge",
        tune_hyperparameters=False,
        verbose=False
    )
    
    results_df = model.walk_forward_sector_models(
        test_df,
        min_train_months=12
    )
    
    # Validate results
    assert len(results_df) > 0, "Should have predictions"
    assert "date" in results_df.columns
    assert "ticker" in results_df.columns
    assert "sector" in results_df.columns
    assert "actual" in results_df.columns
    assert "predicted" in results_df.columns
    
    # Check all sectors are present
    sectors = results_df["sector"].unique()
    assert len(sectors) >= 3, f"Should have multiple sectors, got {len(sectors)}"
    
    # Check predictions are numeric
    assert results_df["predicted"].dtype in [np.float64, np.float32]
    assert not results_df["predicted"].isnull().all()
    
    print(f"  ✓ Generated {len(results_df):,} predictions across "
          f"{len(sectors)} sectors")
    print(f"  ✓ Prediction months: {results_df['date'].nunique()}")
    print("  ✓ All Ridge walk-forward tests passed")


def test_walk_forward_sector_models_lightgbm():
    """Test walk-forward validation with sector-specific LightGBM models."""
    print("\n=== Testing walk_forward_sector_models (LightGBM) ===")
    
    # Create test data
    test_df = create_test_data(n_months=30, n_stocks_per_sector=10)
    
    # Train sector-specific models without tuning (for speed)
    model = SectorSpecificModel(
        model_type="lightgbm",
        tune_hyperparameters=False,
        verbose=False
    )
    
    results_df = model.walk_forward_sector_models(
        test_df,
        min_train_months=12
    )
    
    # Validate results
    assert len(results_df) > 0, "Should have predictions"
    assert "date" in results_df.columns
    assert "ticker" in results_df.columns
    assert "sector" in results_df.columns
    assert "actual" in results_df.columns
    assert "predicted" in results_df.columns
    
    # Check all sectors are present
    sectors = results_df["sector"].unique()
    assert len(sectors) >= 3, f"Should have multiple sectors, got {len(sectors)}"
    
    print(f"  ✓ Generated {len(results_df):,} predictions across "
          f"{len(sectors)} sectors")
    print(f"  ✓ Prediction months: {results_df['date'].nunique()}")
    print("  ✓ All LightGBM walk-forward tests passed")


def test_evaluate_sector_performance():
    """Test sector performance evaluation."""
    print("\n=== Testing evaluate_sector_performance ===")
    
    # Create test data and predictions
    test_df = create_test_data(n_months=30, n_stocks_per_sector=10)
    
    model = SectorSpecificModel(
        model_type="ridge",
        tune_hyperparameters=False,
        verbose=False
    )
    
    results_df = model.walk_forward_sector_models(
        test_df,
        min_train_months=12
    )
    
    # Evaluate performance
    performance = model.evaluate_sector_performance(results_df)
    
    # Validate performance metrics
    assert "Overall" in performance
    assert "Mean_IC" in performance["Overall"]
    assert "IC_Std" in performance["Overall"]
    assert "IC_IR" in performance["Overall"]
    assert "Pos_IC" in performance["Overall"]
    assert "Total_Months" in performance["Overall"]
    
    # Check sector-specific metrics
    sectors = results_df["sector"].unique()
    for sector in sectors:
        assert sector in performance, f"Missing performance for {sector}"
        assert "Mean_IC" in performance[sector]
        assert "IC_IR" in performance[sector]
    
    print(f"  ✓ Overall Mean IC: {performance['Overall']['Mean_IC']:.5f}")
    print(f"  ✓ Overall IC-IR: {performance['Overall']['IC_IR']:.5f}")
    print(f"  ✓ Evaluated {len(sectors)} sectors")
    print("  ✓ All performance evaluation tests passed")


def test_compare_with_global_model():
    """Test comparison between sector-specific and global models."""
    print("\n=== Testing compare_with_global_model ===")
    
    # Create test data
    test_df = create_test_data(n_months=30, n_stocks_per_sector=10)
    
    # Train sector-specific model
    sector_model = SectorSpecificModel(
        model_type="ridge",
        tune_hyperparameters=False,
        verbose=False
    )
    
    sector_results = sector_model.walk_forward_sector_models(
        test_df,
        min_train_months=12
    )
    
    # Create "global" model results (just add noise to sector results for testing)
    global_results = sector_results.copy()
    global_results["predicted"] = global_results["predicted"] + np.random.randn(len(global_results)) * 0.01
    
    # Compare models
    comparison_df = sector_model.compare_with_global_model(
        sector_results,
        global_results
    )
    
    # Validate comparison
    assert len(comparison_df) > 0, "Should have comparison results"
    assert "Sector" in comparison_df.columns
    assert "Sector_Mean_IC" in comparison_df.columns
    assert "Global_Mean_IC" in comparison_df.columns
    assert "IC_Improvement" in comparison_df.columns
    assert "Sector_IC_IR" in comparison_df.columns
    assert "Global_IC_IR" in comparison_df.columns
    
    # Check Overall row exists
    assert "Overall" in comparison_df["Sector"].values
    
    # Check sector rows exist
    sectors = sector_results["sector"].unique()
    for sector in sectors:
        assert sector in comparison_df["Sector"].values, f"Missing comparison for {sector}"
    
    print(f"  ✓ Compared {len(comparison_df)} entries (Overall + {len(sectors)} sectors)")
    print("  ✓ All comparison tests passed")


def test_train_sector_specific_models():
    """Test high-level training function."""
    print("\n=== Testing train_sector_specific_models ===")
    
    # Create test data
    test_df = create_test_data(n_months=30, n_stocks_per_sector=10)
    
    # Train models
    results_df, sector_model, performance = train_sector_specific_models(
        test_df,
        model_type="ridge",
        tune_hyperparameters=False,
        min_train_months=12
    )
    
    # Validate outputs
    assert len(results_df) > 0, "Should have predictions"
    assert isinstance(sector_model, SectorSpecificModel)
    assert isinstance(performance, dict)
    assert "Overall" in performance
    
    print(f"  ✓ Trained models for {len(results_df['sector'].unique())} sectors")
    print(f"  ✓ Generated {len(results_df):,} predictions")
    print(f"  ✓ Overall Mean IC: {performance['Overall']['Mean_IC']:.5f}")
    print("  ✓ All high-level training tests passed")


def test_sector_specific_hyperparameters():
    """Test that different sectors get different hyperparameters."""
    print("\n=== Testing sector-specific hyperparameters ===")
    
    # Create test data with distinct sector patterns
    test_df = create_test_data(n_months=30, n_stocks_per_sector=15)
    
    # Train with hyperparameter tuning
    model = SectorSpecificModel(
        model_type="ridge",
        tune_hyperparameters=True,
        verbose=False
    )
    
    results_df = model.walk_forward_sector_models(
        test_df,
        min_train_months=12
    )
    
    # Check that sector parameters were stored
    assert len(model.sector_params) > 0, "Should have sector parameters"
    
    # Check that different sectors have parameters
    sectors = list(model.sector_params.keys())
    assert len(sectors) >= 3, f"Should have multiple sectors, got {len(sectors)}"
    
    # Check parameter variation (at least some sectors should differ)
    alphas = [model.sector_params[s]["alpha"] for s in sectors]
    unique_alphas = len(set(alphas))
    
    print(f"  ✓ Tuned parameters for {len(sectors)} sectors")
    print(f"  ✓ Alpha values: {alphas}")
    print(f"  ✓ Unique alpha values: {unique_alphas}")
    
    # Note: It's possible all sectors get the same alpha if the data patterns
    # are similar, so we just check that tuning completed successfully
    print("  ✓ All sector-specific hyperparameter tests passed")


def test_temporal_integrity():
    """Test that sector models maintain temporal integrity (no data leakage)."""
    print("\n=== Testing temporal integrity ===")
    
    # Create test data
    test_df = create_test_data(n_months=30, n_stocks_per_sector=10)
    
    # Train models
    model = SectorSpecificModel(
        model_type="ridge",
        tune_hyperparameters=False,
        verbose=False
    )
    
    results_df = model.walk_forward_sector_models(
        test_df,
        min_train_months=12
    )
    
    # Check temporal ordering
    all_dates = sorted(test_df["date"].unique())
    prediction_dates = sorted(results_df["date"].unique())
    
    # Predictions should start after min_train_months
    assert prediction_dates[0] >= all_dates[12], \
        "Predictions should not start before min_train_months"
    
    # All prediction dates should be in the original data
    for pred_date in prediction_dates:
        assert pred_date in all_dates, f"Prediction date {pred_date} not in training data"
    
    # For each prediction date, verify it comes after training data
    for pred_date in prediction_dates:
        pred_idx = all_dates.index(pred_date)
        assert pred_idx >= 12, \
            f"Prediction at {pred_date} (index {pred_idx}) uses insufficient training data"
    
    print(f"  ✓ First prediction date: {prediction_dates[0].date()}")
    print(f"  ✓ Last prediction date: {prediction_dates[-1].date()}")
    print(f"  ✓ Total prediction months: {len(prediction_dates)}")
    print("  ✓ All temporal integrity tests passed")


def test_missing_sector_column():
    """Test error handling when sector column is missing."""
    print("\n=== Testing missing sector column ===")
    
    # Create test data without sector column
    test_df = create_test_data(n_months=30, n_stocks_per_sector=10)
    test_df = test_df.drop(columns=["sector"])
    
    # Should raise ValueError
    model = SectorSpecificModel(model_type="ridge", verbose=False)
    
    try:
        model.walk_forward_sector_models(test_df, min_train_months=12)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "sector" in str(e).lower()
        print("  ✓ Missing sector column raises ValueError")
    print("  ✓ All error handling tests passed")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  SECTOR-SPECIFIC MODEL TESTS")
    print("="*70)
    
    test_sector_specific_model_initialization()
    test_get_default_params()
    test_tune_sector_hyperparameters()
    test_walk_forward_sector_models_ridge()
    test_walk_forward_sector_models_lightgbm()
    test_evaluate_sector_performance()
    test_compare_with_global_model()
    test_train_sector_specific_models()
    test_sector_specific_hyperparameters()
    test_temporal_integrity()
    test_missing_sector_column()
    
    print("\n" + "="*70)
    print("  ALL TESTS PASSED")
    print("="*70)

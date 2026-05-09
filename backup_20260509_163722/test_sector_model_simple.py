"""
test_sector_model_simple.py — Quick tests for Sector-Specific Modeling
=======================================================================
Simplified tests that run quickly for validation.

**Validates: Requirements 3.4**
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sector_model import SectorSpecificModel
from data_loader import FEATURES, TARGET


def create_minimal_test_data():
    """Create minimal test data for quick testing."""
    sectors = ["Technology", "Healthcare", "Financials"]
    dates = pd.date_range(end=datetime.today(), periods=18, freq="ME")
    
    records = []
    
    for sector_idx, sector in enumerate(sectors):
        for stock_idx in range(5):  # 5 stocks per sector
            ticker = f"{sector[:4].upper()}{stock_idx:02d}"
            
            for date_idx, date in enumerate(dates):
                record = {
                    "date": date,
                    "ticker": ticker,
                    "sector": sector,
                }
                
                # Generate simple features
                for feat in FEATURES:
                    record[feat] = np.random.randn() * 0.1 + sector_idx * 0.05
                
                # Generate target
                record[TARGET] = np.random.randn() * 0.05 + sector_idx * 0.02
                
                records.append(record)
    
    return pd.DataFrame(records)


def test_basic_functionality():
    """Test basic sector model functionality."""
    print("\n=== Testing Basic Functionality ===")
    
    # Create test data
    test_df = create_minimal_test_data()
    print(f"  Created test data: {len(test_df)} rows, "
          f"{test_df['sector'].nunique()} sectors, "
          f"{test_df['date'].nunique()} months")
    
    # Initialize model
    model = SectorSpecificModel(
        model_type="ridge",
        tune_hyperparameters=False,
        verbose=False
    )
    print("  ✓ Model initialized")
    
    # Train models
    results_df = model.walk_forward_sector_models(
        test_df,
        min_train_months=6
    )
    print(f"  ✓ Generated {len(results_df)} predictions")
    
    # Evaluate performance
    performance = model.evaluate_sector_performance(results_df)
    print(f"  ✓ Overall Mean IC: {performance['Overall']['Mean_IC']:.5f}")
    print(f"  ✓ Evaluated {len(performance) - 1} sectors")
    
    # Verify results structure
    assert "date" in results_df.columns
    assert "ticker" in results_df.columns
    assert "sector" in results_df.columns
    assert "actual" in results_df.columns
    assert "predicted" in results_df.columns
    print("  ✓ Results structure validated")
    
    # Verify performance structure
    assert "Overall" in performance
    assert "Mean_IC" in performance["Overall"]
    assert "IC_IR" in performance["Overall"]
    print("  ✓ Performance structure validated")
    
    print("  ✓ All basic functionality tests passed")
    return True


def test_sector_parameters():
    """Test that sector-specific parameters are stored."""
    print("\n=== Testing Sector Parameters ===")
    
    test_df = create_minimal_test_data()
    
    model = SectorSpecificModel(
        model_type="ridge",
        tune_hyperparameters=False,
        verbose=False
    )
    
    results_df = model.walk_forward_sector_models(
        test_df,
        min_train_months=6
    )
    
    # Check sector parameters
    assert len(model.sector_params) > 0
    print(f"  ✓ Stored parameters for {len(model.sector_params)} sectors")
    
    for sector, params in model.sector_params.items():
        assert "alpha" in params
        print(f"    {sector}: alpha={params['alpha']}")
    
    print("  ✓ All sector parameter tests passed")
    return True


def test_comparison_framework():
    """Test comparison with global model."""
    print("\n=== Testing Comparison Framework ===")
    
    test_df = create_minimal_test_data()
    
    # Train sector model
    model = SectorSpecificModel(
        model_type="ridge",
        tune_hyperparameters=False,
        verbose=False
    )
    
    sector_results = model.walk_forward_sector_models(
        test_df,
        min_train_months=6
    )
    
    # Create mock global results
    global_results = sector_results.copy()
    global_results["predicted"] = global_results["predicted"] + np.random.randn(len(global_results)) * 0.01
    
    # Compare
    comparison_df = model.compare_with_global_model(
        sector_results,
        global_results
    )
    
    assert len(comparison_df) > 0
    assert "Sector" in comparison_df.columns
    assert "IC_Improvement" in comparison_df.columns
    print(f"  ✓ Generated comparison for {len(comparison_df)} entries")
    
    # Check Overall row
    assert "Overall" in comparison_df["Sector"].values
    print("  ✓ Overall comparison included")
    
    print("  ✓ All comparison framework tests passed")
    return True


def test_temporal_integrity():
    """Test temporal integrity."""
    print("\n=== Testing Temporal Integrity ===")
    
    test_df = create_minimal_test_data()
    all_dates = sorted(test_df["date"].unique())
    
    model = SectorSpecificModel(
        model_type="ridge",
        tune_hyperparameters=False,
        verbose=False
    )
    
    results_df = model.walk_forward_sector_models(
        test_df,
        min_train_months=6
    )
    
    prediction_dates = sorted(results_df["date"].unique())
    
    # Predictions should start after min_train_months
    assert prediction_dates[0] >= all_dates[6]
    print(f"  ✓ First prediction after training period: {prediction_dates[0].date()}")
    
    # All prediction dates should be in original data
    for pred_date in prediction_dates:
        assert pred_date in all_dates
    print(f"  ✓ All {len(prediction_dates)} prediction dates valid")
    
    print("  ✓ All temporal integrity tests passed")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  SECTOR-SPECIFIC MODEL TESTS (SIMPLIFIED)")
    print("="*70)
    
    all_passed = True
    
    try:
        all_passed &= test_basic_functionality()
    except Exception as e:
        print(f"  ✗ Basic functionality test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_sector_parameters()
    except Exception as e:
        print(f"  ✗ Sector parameters test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_comparison_framework()
    except Exception as e:
        print(f"  ✗ Comparison framework test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_temporal_integrity()
    except Exception as e:
        print(f"  ✗ Temporal integrity test failed: {e}")
        all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗")
    print("="*70)

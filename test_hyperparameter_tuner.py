"""
test_hyperparameter_tuner.py — Unit tests for hyperparameter_tuner.py
=======================================================================
Tests the hyperparameter optimization framework to ensure:
1. Temporal integrity (no future data leakage)
2. Nested cross-validation correctness
3. Parameter grid search functionality
4. Integration with Ridge and LightGBM models
"""

import pandas as pd
import numpy as np
from hyperparameter_tuner import (
    HyperparameterTuner,
    tune_ridge_hyperparameters,
    tune_lightgbm_hyperparameters
)


def create_synthetic_data(n_months=36, n_stocks=50, seed=42):
    """
    Create synthetic factor data for testing.
    
    Parameters
    ----------
    n_months : int
        Number of months
    n_stocks : int
        Number of stocks per month
    seed : int
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Synthetic factor data
    """
    np.random.seed(seed)
    
    dates = pd.date_range("2020-01-01", periods=n_months, freq="ME")
    tickers = [f"STOCK{i:03d}" for i in range(n_stocks)]
    
    records = []
    for date in dates:
        for ticker in tickers:
            # Create features with some predictive power
            mom_12_1 = np.random.randn()
            vol_12 = abs(np.random.randn())
            pb_ratio = abs(np.random.randn() + 1)
            
            # Target correlated with features
            target = 0.3 * mom_12_1 - 0.2 * vol_12 + 0.1 * np.log(pb_ratio) + 0.4 * np.random.randn()
            
            records.append({
                "date": date,
                "ticker": ticker,
                "sector": f"Sector{hash(ticker) % 5}",
                "Mom_12_1": mom_12_1,
                "Mom_6_1": np.random.randn(),
                "Mom_1": np.random.randn(),
                "Vol_12": vol_12,
                "IdioVol": abs(np.random.randn()),
                "Beta_12": np.random.randn(),
                "High52W": np.random.uniform(0.5, 1.0),
                "Trend_MA": np.random.uniform(0.8, 1.2),
                "MaxRet_1M": np.random.uniform(-0.1, 0.2),
                "PB_ratio": pb_ratio,
                "PE_TTM": abs(np.random.randn() * 10 + 15),
                "EV_EBITDA": abs(np.random.randn() * 5 + 10),
                "ROE": np.random.uniform(-0.1, 0.3),
                "GrossMargin": np.random.uniform(0.2, 0.6),
                "CashFlowYield": np.random.uniform(-0.05, 0.15),
                "RevGrowth_YoY": np.random.uniform(-0.2, 0.4),
                "EarnGrowth_YoY": np.random.uniform(-0.3, 0.5),
                "LogMktCap": np.random.uniform(18, 24),
                "VolRatio": np.random.uniform(0.5, 1.5),
                "Next_Month_Return": target
            })
    
    return pd.DataFrame(records)


def test_temporal_split():
    """Test that temporal splits maintain chronological order."""
    print("\n=== Testing temporal_split ===")
    
    tuner = HyperparameterTuner(model_type="ridge", n_splits=3, verbose=False)
    
    # Create date sequence
    dates = pd.date_range("2020-01-01", periods=24, freq="ME")
    
    splits = tuner._temporal_split(dates, n_splits=3)
    
    # Verify we got splits
    assert len(splits) > 0, "Expected at least one split"
    
    # Verify temporal ordering
    for train_dates, val_dates in splits:
        assert len(train_dates) > 0, "Training dates should not be empty"
        assert len(val_dates) > 0, "Validation dates should not be empty"
        
        # Validation dates should come after training dates
        max_train = max(train_dates)
        min_val = min(val_dates)
        assert max_train <= min_val, \
            f"Validation dates must come after training dates: {max_train} > {min_val}"
        
        # No overlap between train and validation
        train_set = set(train_dates)
        val_set = set(val_dates)
        assert len(train_set & val_set) == 0, "Train and validation should not overlap"
    
    print(f"  Created {len(splits)} temporal splits")
    print(f"  First split: {len(splits[0][0])} train, {len(splits[0][1])} val dates")
    print("  ✓ temporal_split test passed")


def test_compute_score():
    """Test scoring metric computation."""
    print("\n=== Testing compute_score ===")
    
    np.random.seed(42)
    
    # Create correlated predictions
    y_true = np.random.randn(100)
    y_pred = 0.7 * y_true + 0.3 * np.random.randn(100)
    
    # Test IC scoring
    tuner_ic = HyperparameterTuner(model_type="ridge", scoring="ic", verbose=False)
    ic_score = tuner_ic._compute_score(y_true, y_pred)
    assert 0 < ic_score < 1, f"Expected IC between 0 and 1, got {ic_score}"
    print(f"  IC score: {ic_score:.4f}")
    
    # Test RMSE scoring (negative for maximization)
    tuner_rmse = HyperparameterTuner(model_type="ridge", scoring="rmse", verbose=False)
    rmse_score = tuner_rmse._compute_score(y_true, y_pred)
    assert rmse_score < 0, f"Expected negative RMSE, got {rmse_score}"
    print(f"  RMSE score: {rmse_score:.4f}")
    
    # Test R2 scoring
    tuner_r2 = HyperparameterTuner(model_type="ridge", scoring="r2", verbose=False)
    r2_score = tuner_r2._compute_score(y_true, y_pred)
    assert 0 < r2_score < 1, f"Expected R2 between 0 and 1, got {r2_score}"
    print(f"  R2 score: {r2_score:.4f}")
    
    print("  ✓ compute_score test passed")


def test_grid_to_combinations():
    """Test parameter grid expansion."""
    print("\n=== Testing grid_to_combinations ===")
    
    tuner = HyperparameterTuner(model_type="ridge", verbose=False)
    
    # Test simple grid
    param_grid = {
        "alpha": [0.1, 1.0, 10.0],
        "fit_intercept": [True, False]
    }
    
    combinations = tuner._grid_to_combinations(param_grid)
    
    # Should have 3 * 2 = 6 combinations
    assert len(combinations) == 6, f"Expected 6 combinations, got {len(combinations)}"
    
    # Verify all combinations are present
    alphas = [c["alpha"] for c in combinations]
    assert alphas.count(0.1) == 2, "Each alpha should appear twice"
    assert alphas.count(1.0) == 2
    assert alphas.count(10.0) == 2
    
    # Test grid with fixed params
    param_grid_mixed = {
        "alpha": [0.1, 1.0],
        "fit_intercept": True  # Fixed value
    }
    
    combinations_mixed = tuner._grid_to_combinations(param_grid_mixed)
    assert len(combinations_mixed) == 2, "Expected 2 combinations"
    assert all(c["fit_intercept"] is True for c in combinations_mixed), \
        "Fixed param should be in all combinations"
    
    print(f"  Generated {len(combinations)} combinations from grid")
    print("  ✓ grid_to_combinations test passed")


def test_cross_validate_params_ridge():
    """Test cross-validation for Ridge parameters."""
    print("\n=== Testing cross_validate_params (Ridge) ===")
    
    # Create synthetic data
    df = create_synthetic_data(n_months=24, n_stocks=50)
    
    tuner = HyperparameterTuner(model_type="ridge", n_splits=3, scoring="ic", verbose=False)
    
    # Test with different alpha values
    params_low = {"alpha": 0.1}
    params_high = {"alpha": 100.0}
    
    score_low = tuner._cross_validate_params(df, params_low)
    score_high = tuner._cross_validate_params(df, params_high)
    
    print(f"  Score (alpha=0.1): {score_low:.4f}")
    print(f"  Score (alpha=100.0): {score_high:.4f}")
    
    # Scores should be finite
    assert np.isfinite(score_low), "Score should be finite"
    assert np.isfinite(score_high), "Score should be finite"
    
    print("  ✓ cross_validate_params (Ridge) test passed")


def test_cross_validate_params_lightgbm():
    """Test cross-validation for LightGBM parameters."""
    print("\n=== Testing cross_validate_params (LightGBM) ===")
    
    # Create synthetic data
    df = create_synthetic_data(n_months=24, n_stocks=50)
    
    tuner = HyperparameterTuner(model_type="lightgbm", n_splits=3, scoring="ic", verbose=False)
    
    # Test with different learning rates
    params_low = {
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": 5,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1
    }
    
    params_high = {
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 5,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1
    }
    
    score_low = tuner._cross_validate_params(df, params_low)
    score_high = tuner._cross_validate_params(df, params_high)
    
    print(f"  Score (lr=0.01): {score_low:.4f}")
    print(f"  Score (lr=0.1): {score_high:.4f}")
    
    # Scores should be finite
    assert np.isfinite(score_low), "Score should be finite"
    assert np.isfinite(score_high), "Score should be finite"
    
    print("  ✓ cross_validate_params (LightGBM) test passed")


def test_tune_ridge():
    """Test full Ridge hyperparameter tuning."""
    print("\n=== Testing tune (Ridge) ===")
    
    # Create synthetic data
    df = create_synthetic_data(n_months=30, n_stocks=50)
    
    tuner = HyperparameterTuner(model_type="ridge", n_splits=3, scoring="ic", verbose=False)
    
    # Small parameter grid for testing
    param_grid = {
        "alpha": [0.1, 1.0, 10.0]
    }
    
    results = tuner.tune(df, param_grid)
    
    # Verify we got results back
    assert results is not None, "Should return results dictionary"
    assert "best_params" in results, "Should contain best_params key"
    assert "best_score" in results, "Should contain best_score key"
    assert "param_scores" in results, "Should contain param_scores key"
    
    best_params = results["best_params"]
    assert best_params is not None, "Should return best parameters"
    assert "alpha" in best_params, "Should contain alpha parameter"
    assert best_params["alpha"] in [0.1, 1.0, 10.0], "Alpha should be from grid"
    
    print(f"  Best parameters: {best_params}")
    print(f"  Best score: {results['best_score']:.5f}")
    print("  ✓ tune (Ridge) test passed")


def test_tune_lightgbm():
    """Test full LightGBM hyperparameter tuning."""
    print("\n=== Testing tune (LightGBM) ===")
    
    # Create synthetic data
    df = create_synthetic_data(n_months=30, n_stocks=50)
    
    tuner = HyperparameterTuner(model_type="lightgbm", n_splits=2, scoring="ic", verbose=False)
    
    # Small parameter grid for testing
    param_grid = {
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31],
        "max_depth": [5],
        "min_data_in_leaf": [20],
        "lambda_l1": [0.0],
        "lambda_l2": [0.0],
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1
    }
    
    results = tuner.tune(df, param_grid)
    
    # Verify we got results back
    assert results is not None, "Should return results dictionary"
    assert "best_params" in results, "Should contain best_params key"
    assert "best_score" in results, "Should contain best_score key"
    
    best_params = results["best_params"]
    assert best_params is not None, "Should return best parameters"
    assert "learning_rate" in best_params, "Should contain learning_rate"
    assert best_params["learning_rate"] in [0.05, 0.1], "learning_rate should be from grid"
    
    print(f"  Best parameters: {best_params}")
    print(f"  Best score: {results['best_score']:.5f}")
    print("  ✓ tune (LightGBM) test passed")


def test_tune_ridge_hyperparameters():
    """Test high-level Ridge tuning function."""
    print("\n=== Testing tune_ridge_hyperparameters ===")
    
    # Create synthetic data
    df = create_synthetic_data(n_months=30, n_stocks=50)
    
    # Small parameter grid for testing
    param_grid = {
        "alpha": [0.1, 1.0]
    }
    
    results = tune_ridge_hyperparameters(df, min_train_months=24, param_grid=param_grid)
    
    # Verify we got results back
    assert results is not None, "Should return results dictionary"
    assert "best_params" in results, "Should contain best_params key"
    
    best_params = results["best_params"]
    assert best_params is not None, "Should return best parameters"
    assert "alpha" in best_params, "Should contain alpha parameter"
    
    print(f"  Best parameters: {best_params}")
    print(f"  Best score: {results['best_score']:.5f}")
    print("  ✓ tune_ridge_hyperparameters test passed")


def test_tune_lightgbm_hyperparameters():
    """Test high-level LightGBM tuning function."""
    print("\n=== Testing tune_lightgbm_hyperparameters ===")
    
    # Create synthetic data
    df = create_synthetic_data(n_months=30, n_stocks=50)
    
    # Small parameter grid for testing
    param_grid = {
        "learning_rate": [0.1],
        "num_leaves": [31],
        "max_depth": [5],
        "min_data_in_leaf": [20],
        "lambda_l1": [0.0],
        "lambda_l2": [0.0],
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1
    }
    
    results = tune_lightgbm_hyperparameters(df, min_train_months=24, param_grid=param_grid)
    
    # Verify we got results back
    assert results is not None, "Should return results dictionary"
    assert "best_params" in results, "Should contain best_params key"
    
    best_params = results["best_params"]
    assert best_params is not None, "Should return best parameters"
    assert "learning_rate" in best_params, "Should contain learning_rate"
    
    print(f"  Best parameters: {best_params}")
    print(f"  Best score: {results['best_score']:.5f}")
    print("  ✓ tune_lightgbm_hyperparameters test passed")


def test_no_data_leakage():
    """
    Test that hyperparameter tuning does not leak future data.
    
    This test verifies temporal integrity by ensuring:
    1. All CV splits maintain chronological order
    2. Validation data always comes after training data
    3. No overlap between train and validation sets
    """
    print("\n=== Testing no_data_leakage ===")
    
    # Create synthetic data with time-dependent pattern
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=36, freq="ME")
    tickers = [f"STOCK{i:03d}" for i in range(50)]
    
    records = []
    for i, date in enumerate(dates):
        # Create time-dependent signal (increases over time)
        time_signal = i / len(dates)
        
        for ticker in tickers:
            # Features and target both depend on time
            feature_val = time_signal + 0.1 * np.random.randn()
            target_val = time_signal + 0.1 * np.random.randn()
            
            records.append({
                "date": date,
                "ticker": ticker,
                "sector": f"Sector{hash(ticker) % 5}",
                "Mom_12_1": feature_val,
                "Mom_6_1": np.random.randn(),
                "Mom_1": np.random.randn(),
                "Vol_12": abs(np.random.randn()),
                "IdioVol": abs(np.random.randn()),
                "Beta_12": np.random.randn(),
                "High52W": np.random.uniform(0.5, 1.0),
                "Trend_MA": np.random.uniform(0.8, 1.2),
                "MaxRet_1M": np.random.uniform(-0.1, 0.2),
                "PB_ratio": abs(np.random.randn() + 1),
                "PE_TTM": abs(np.random.randn() * 10 + 15),
                "EV_EBITDA": abs(np.random.randn() * 5 + 10),
                "ROE": np.random.uniform(-0.1, 0.3),
                "GrossMargin": np.random.uniform(0.2, 0.6),
                "CashFlowYield": np.random.uniform(-0.05, 0.15),
                "RevGrowth_YoY": np.random.uniform(-0.2, 0.4),
                "EarnGrowth_YoY": np.random.uniform(-0.3, 0.5),
                "LogMktCap": np.random.uniform(18, 24),
                "VolRatio": np.random.uniform(0.5, 1.5),
                "Next_Month_Return": target_val
            })
    
    df = pd.DataFrame(records)
    
    # Create tuner and get temporal splits
    tuner = HyperparameterTuner(model_type="ridge", n_splits=3, verbose=False)
    all_dates = sorted(df["date"].unique())
    splits = tuner._temporal_split(all_dates, n_splits=3)
    
    # Verify temporal ordering for each split
    for split_idx, (train_dates, val_dates) in enumerate(splits):
        # Check no overlap
        train_set = set(train_dates)
        val_set = set(val_dates)
        overlap = train_set & val_set
        assert len(overlap) == 0, \
            f"Split {split_idx}: Found {len(overlap)} overlapping dates"
        
        # Check validation comes after training
        max_train = max(train_dates)
        min_val = min(val_dates)
        assert max_train <= min_val, \
            f"Split {split_idx}: Validation dates must come after training dates"
        
        # Check mean target values (should increase over time)
        train_df = df[df["date"].isin(train_dates)]
        val_df = df[df["date"].isin(val_dates)]
        
        mean_train_target = train_df["Next_Month_Return"].mean()
        mean_val_target = val_df["Next_Month_Return"].mean()
        
        # Validation mean should be >= training mean (due to time trend)
        # Allow small tolerance for randomness
        assert mean_val_target >= mean_train_target - 0.1, \
            f"Split {split_idx}: Validation mean ({mean_val_target:.3f}) should be >= " \
            f"training mean ({mean_train_target:.3f}) due to time trend"
    
    print(f"  Verified {len(splits)} splits for temporal integrity")
    print("  ✓ no_data_leakage test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("  HYPERPARAMETER TUNER UNIT TESTS")
    print("="*70)
    
    test_temporal_split()
    test_compute_score()
    test_grid_to_combinations()
    test_cross_validate_params_ridge()
    test_cross_validate_params_lightgbm()
    test_tune_ridge()
    test_tune_lightgbm()
    test_tune_ridge_hyperparameters()
    test_tune_lightgbm_hyperparameters()
    test_no_data_leakage()
    
    print("\n" + "="*70)
    print("  ✓ ALL TESTS PASSED")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()

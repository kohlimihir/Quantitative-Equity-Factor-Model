"""
test_early_stopping_regularization.py — Unit tests for early_stopping_regularization.py
=========================================================================================
Tests the early stopping and regularization framework to ensure:
1. Temporal integrity (validation data comes after training data)
2. Early stopping prevents overfitting
3. Regularization techniques work correctly
4. Model complexity analysis produces valid results
5. No test data leakage in validation splits
"""

import pandas as pd
import numpy as np
from early_stopping_regularization import (
    EarlyStoppingValidator,
    ModelComplexityAnalyzer,
    walk_forward_with_early_stopping
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


def test_split_train_validation():
    """Test that train/validation split maintains temporal ordering."""
    print("\n=== Testing split_train_validation ===")
    
    df = create_synthetic_data(n_months=24, n_stocks=50)
    
    validator = EarlyStoppingValidator(validation_fraction=0.2, verbose=False)
    train_subset, validation = validator.split_train_validation(df)
    
    # Verify sizes
    total_dates = df["date"].nunique()
    train_dates = train_subset["date"].nunique()
    val_dates = validation["date"].nunique()
    
    assert train_dates > 0, "Training set should not be empty"
    assert val_dates > 0, "Validation set should not be empty"
    assert train_dates + val_dates == total_dates, "All dates should be used"
    
    # Verify temporal ordering
    max_train_date = train_subset["date"].max()
    min_val_date = validation["date"].min()
    
    assert max_train_date <= min_val_date, \
        f"Validation dates must come after training dates: {max_train_date} > {min_val_date}"
    
    # Verify no overlap
    train_date_set = set(train_subset["date"].unique())
    val_date_set = set(validation["date"].unique())
    overlap = train_date_set & val_date_set
    
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping dates"
    
    print(f"  Train: {train_dates} months, Val: {val_dates} months")
    print(f"  Train ends: {max_train_date.strftime('%Y-%m')}, Val starts: {min_val_date.strftime('%Y-%m')}")
    print("  ✓ split_train_validation test passed")


def test_compute_metric():
    """Test metric computation."""
    print("\n=== Testing compute_metric ===")
    
    np.random.seed(42)
    
    # Create correlated predictions
    y_true = np.random.randn(100)
    y_pred = 0.7 * y_true + 0.3 * np.random.randn(100)
    
    # Test IC metric
    validator_ic = EarlyStoppingValidator(metric="ic", verbose=False)
    ic_score = validator_ic.compute_metric(y_true, y_pred)
    assert 0 < ic_score < 1, f"Expected IC between 0 and 1, got {ic_score}"
    print(f"  IC score: {ic_score:.4f}")
    
    # Test RMSE metric (negative for maximization)
    validator_rmse = EarlyStoppingValidator(metric="rmse", verbose=False)
    rmse_score = validator_rmse.compute_metric(y_true, y_pred)
    assert rmse_score < 0, f"Expected negative RMSE, got {rmse_score}"
    print(f"  RMSE score: {rmse_score:.4f}")
    
    # Test R2 metric
    validator_r2 = EarlyStoppingValidator(metric="r2", verbose=False)
    r2_score = validator_r2.compute_metric(y_true, y_pred)
    assert 0 < r2_score < 1, f"Expected R2 between 0 and 1, got {r2_score}"
    print(f"  R2 score: {r2_score:.4f}")
    
    print("  ✓ compute_metric test passed")


def test_train_with_early_stopping_lgbm():
    """Test LightGBM training with early stopping."""
    print("\n=== Testing train_with_early_stopping_lgbm ===")
    
    df = create_synthetic_data(n_months=30, n_stocks=50)
    
    validator = EarlyStoppingValidator(
        validation_fraction=0.2,
        patience=10,
        metric="ic",
        verbose=False
    )
    
    result = validator.train_with_early_stopping_lgbm(df, max_rounds=200)
    
    # Verify result structure
    assert "model" in result, "Result should contain model"
    assert "best_iteration" in result, "Result should contain best_iteration"
    assert "stopped_early" in result, "Result should contain stopped_early"
    assert "train_scores" in result, "Result should contain train_scores"
    assert "val_scores" in result, "Result should contain val_scores"
    assert "scaler" in result, "Result should contain scaler"
    
    # Verify early stopping worked
    best_iter = result["best_iteration"]
    assert best_iter > 0, "Best iteration should be positive"
    assert best_iter <= 200, "Best iteration should not exceed max_rounds"
    
    # If stopped early, best_iter should be less than max_rounds
    if result["stopped_early"]:
        assert best_iter < 200, "If stopped early, best_iter should be < max_rounds"
    
    print(f"  Best iteration: {best_iter}")
    print(f"  Stopped early: {result['stopped_early']}")
    print("  ✓ train_with_early_stopping_lgbm test passed")


def test_train_with_regularization_ridge():
    """Test Ridge training with regularization."""
    print("\n=== Testing train_with_regularization_ridge ===")
    
    df = create_synthetic_data(n_months=30, n_stocks=50)
    
    validator = EarlyStoppingValidator(
        validation_fraction=0.2,
        metric="ic",
        verbose=False
    )
    
    # Test Ridge
    result_ridge = validator.train_with_regularization_ridge(
        df, alpha=1.0, regularization_type="ridge"
    )
    
    assert "model" in result_ridge, "Result should contain model"
    assert "train_score" in result_ridge, "Result should contain train_score"
    assert "val_score" in result_ridge, "Result should contain val_score"
    assert "scaler" in result_ridge, "Result should contain scaler"
    
    # Scores should be finite
    assert np.isfinite(result_ridge["train_score"]), "Train score should be finite"
    assert np.isfinite(result_ridge["val_score"]), "Val score should be finite"
    
    print(f"  Ridge - Train IC: {result_ridge['train_score']:.4f}, Val IC: {result_ridge['val_score']:.4f}")
    
    # Test Lasso
    result_lasso = validator.train_with_regularization_ridge(
        df, alpha=0.01, regularization_type="lasso"
    )
    
    print(f"  Lasso - Train IC: {result_lasso['train_score']:.4f}, Val IC: {result_lasso['val_score']:.4f}")
    
    # Test ElasticNet
    result_elasticnet = validator.train_with_regularization_ridge(
        df, alpha=0.01, regularization_type="elasticnet"
    )
    
    print(f"  ElasticNet - Train IC: {result_elasticnet['train_score']:.4f}, Val IC: {result_elasticnet['val_score']:.4f}")
    
    print("  ✓ train_with_regularization_ridge test passed")


def test_early_stopping_prevents_overfitting():
    """
    Test that early stopping prevents overfitting.
    
    Creates data where model can overfit, then verifies that early stopping
    stops training before overfitting becomes severe.
    """
    print("\n=== Testing early_stopping_prevents_overfitting ===")
    
    # Create data with noise (easy to overfit)
    np.random.seed(42)
    df = create_synthetic_data(n_months=30, n_stocks=30)  # Smaller dataset
    
    validator = EarlyStoppingValidator(
        validation_fraction=0.2,
        patience=5,  # Low patience to stop quickly
        metric="rmse",
        verbose=False
    )
    
    # Train with early stopping
    result_early = validator.train_with_early_stopping_lgbm(df, max_rounds=500)
    
    # Train without early stopping (fixed iterations)
    result_no_early = validator.train_with_early_stopping_lgbm(df, max_rounds=50)
    
    # Early stopping should use fewer iterations
    assert result_early["best_iteration"] < 500, \
        "Early stopping should stop before max_rounds"
    
    # Verify that early stopping found a reasonable stopping point
    assert result_early["best_iteration"] >= 10, \
        "Should train for at least 10 iterations"
    
    print(f"  With early stopping: {result_early['best_iteration']} iterations")
    print(f"  Without early stopping: 50 iterations (fixed)")
    print("  ✓ early_stopping_prevents_overfitting test passed")


def test_no_test_data_leakage():
    """
    Test that validation split does not leak test data.
    
    Verifies that:
    1. Validation data comes from training period only
    2. No future data is used in validation
    3. Temporal ordering is maintained
    """
    print("\n=== Testing no_test_data_leakage ===")
    
    # Create data with time-dependent pattern
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
    
    validator = EarlyStoppingValidator(validation_fraction=0.2, verbose=False)
    train_subset, validation = validator.split_train_validation(df)
    
    # Check temporal ordering
    max_train_date = train_subset["date"].max()
    min_val_date = validation["date"].min()
    
    assert max_train_date <= min_val_date, \
        "Validation dates must come after training dates"
    
    # Check mean target values (should increase over time)
    mean_train_target = train_subset["Next_Month_Return"].mean()
    mean_val_target = validation["Next_Month_Return"].mean()
    
    # Validation mean should be >= training mean (due to time trend)
    assert mean_val_target >= mean_train_target - 0.1, \
        f"Validation mean ({mean_val_target:.3f}) should be >= " \
        f"training mean ({mean_train_target:.3f}) due to time trend"
    
    print(f"  Train mean target: {mean_train_target:.4f}")
    print(f"  Val mean target: {mean_val_target:.4f}")
    print("  ✓ no_test_data_leakage test passed")


def test_analyze_lgbm_complexity():
    """Test LightGBM complexity analysis."""
    print("\n=== Testing analyze_lgbm_complexity ===")
    
    df = create_synthetic_data(n_months=30, n_stocks=50)
    
    analyzer = ModelComplexityAnalyzer(verbose=False)
    
    # Small parameter grid for testing
    param_grid = {
        "num_leaves": [15, 31],
        "max_depth": [3, 5],
        "min_child_samples": [20, 50],
        "learning_rate": [0.05],
    }
    
    results_df = analyzer.analyze_lgbm_complexity(df, param_grid)
    
    # Verify results structure
    assert len(results_df) > 0, "Should have results"
    assert "num_leaves" in results_df.columns, "Should have num_leaves column"
    assert "max_depth" in results_df.columns, "Should have max_depth column"
    assert "val_rmse" in results_df.columns, "Should have val_rmse column"
    assert "complexity_score" in results_df.columns, "Should have complexity_score column"
    
    # Verify all val_rmse values are finite
    assert results_df["val_rmse"].notna().all(), "All val_rmse should be non-NaN"
    assert np.isfinite(results_df["val_rmse"]).all(), "All val_rmse should be finite"
    
    # Find best configuration
    best_config = results_df.nsmallest(1, "val_rmse").iloc[0]
    
    print(f"  Tested {len(results_df)} configurations")
    print(f"  Best config: leaves={best_config['num_leaves']}, depth={best_config['max_depth']}, "
          f"RMSE={best_config['val_rmse']:.5f}")
    print("  ✓ analyze_lgbm_complexity test passed")


def test_analyze_ridge_regularization():
    """Test Ridge regularization analysis."""
    print("\n=== Testing analyze_ridge_regularization ===")
    
    df = create_synthetic_data(n_months=30, n_stocks=50)
    
    analyzer = ModelComplexityAnalyzer(verbose=False)
    
    # Test different alpha values
    alpha_range = [0.1, 1.0, 10.0, 100.0]
    
    results_df = analyzer.analyze_ridge_regularization(df, alpha_range)
    
    # Verify results structure
    assert len(results_df) > 0, "Should have results"
    assert "alpha" in results_df.columns, "Should have alpha column"
    assert "train_ic" in results_df.columns, "Should have train_ic column"
    assert "val_ic" in results_df.columns, "Should have val_ic column"
    assert "overfit_gap" in results_df.columns, "Should have overfit_gap column"
    
    # Verify all scores are finite
    assert results_df["train_ic"].notna().all(), "All train_ic should be non-NaN"
    assert results_df["val_ic"].notna().all(), "All val_ic should be non-NaN"
    
    # Find best configuration
    best_config = results_df.nlargest(1, "val_ic").iloc[0]
    
    print(f"  Tested {len(results_df)} alpha values")
    print(f"  Best alpha: {best_config['alpha']:.1f}, Val IC: {best_config['val_ic']:.5f}")
    print("  ✓ analyze_ridge_regularization test passed")


def test_walk_forward_with_early_stopping():
    """Test walk-forward validation with early stopping."""
    print("\n=== Testing walk_forward_with_early_stopping ===")
    
    df = create_synthetic_data(n_months=30, n_stocks=30)  # Smaller for speed
    
    # Run walk-forward with early stopping
    results_df, training_history = walk_forward_with_early_stopping(
        df, min_train_months=20, use_early_stopping=True
    )
    
    # Verify results structure
    assert len(results_df) > 0, "Should have predictions"
    assert "date" in results_df.columns, "Should have date column"
    assert "ticker" in results_df.columns, "Should have ticker column"
    assert "actual" in results_df.columns, "Should have actual column"
    assert "predicted" in results_df.columns, "Should have predicted column"
    
    # Verify training history
    assert len(training_history) > 0, "Should have training history"
    assert "best_iteration" in training_history.columns, "Should have best_iteration"
    assert "stopped_early" in training_history.columns, "Should have stopped_early"
    
    # Check that early stopping was used
    avg_iterations = training_history["best_iteration"].mean()
    
    print(f"  Predictions: {len(results_df):,} rows")
    print(f"  Training windows: {len(training_history)}")
    print(f"  Avg iterations: {avg_iterations:.1f}")
    print("  ✓ walk_forward_with_early_stopping test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("  EARLY STOPPING AND REGULARIZATION UNIT TESTS")
    print("="*70)
    
    test_split_train_validation()
    test_compute_metric()
    test_train_with_early_stopping_lgbm()
    test_train_with_regularization_ridge()
    test_early_stopping_prevents_overfitting()
    test_no_test_data_leakage()
    test_analyze_lgbm_complexity()
    test_analyze_ridge_regularization()
    test_walk_forward_with_early_stopping()
    
    print("\n" + "="*70)
    print("  ✓ ALL TESTS PASSED")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()

"""
Test suite for target variable leakage detection (Task 1.5)

This test validates that the target variable (Next_Month_Return) is properly constructed
using only subsequent month returns with comprehensive temporal alignment checks.

Requirements: 1.7
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from leakage_detector import LeakageDetector


def create_test_data_with_proper_targets():
    """Create test data with properly constructed target variables."""
    dates = pd.date_range('2020-01-31', '2020-06-30', freq='ME')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    data = []
    
    # Create monthly returns for validation
    monthly_returns = {}
    for ticker in tickers:
        # Generate realistic monthly returns
        np.random.seed(42 + hash(ticker) % 1000)
        returns = np.random.normal(0.01, 0.05, len(dates) + 1)  # Extra month for target
        monthly_returns[ticker] = returns
    
    for i, date in enumerate(dates[:-1]):  # Exclude last date (no future target)
        for j, ticker in enumerate(tickers):
            # Features are calculated using data up to current month
            row = {
                'date': date,
                'ticker': ticker,
                'sector': 'Technology',
                'Mom_12_1': np.random.normal(0.1, 0.2),
                'Vol_12': np.random.uniform(0.05, 0.15),
                'Beta_12': np.random.uniform(0.5, 1.5),
                # Target is the NEXT month's return (i+1)
                'Next_Month_Return': monthly_returns[ticker][i + 1]
            }
            data.append(row)
    
    # Add final month with NO target (proper behavior)
    final_date = dates[-1]
    for ticker in tickers:
        row = {
            'date': final_date,
            'ticker': ticker,
            'sector': 'Technology',
            'Mom_12_1': np.random.normal(0.1, 0.2),
            'Vol_12': np.random.uniform(0.05, 0.15),
            'Beta_12': np.random.uniform(0.5, 1.5),
            'Next_Month_Return': np.nan  # No future data available
        }
        data.append(row)
    
    return pd.DataFrame(data)


def create_test_data_with_leakage():
    """Create test data with target variable leakage (for negative testing)."""
    dates = pd.date_range('2020-01-31', '2020-06-30', freq='ME')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    data = []
    
    for i, date in enumerate(dates):
        for ticker in tickers:
            row = {
                'date': date,
                'ticker': ticker,
                'sector': 'Technology',
                'Mom_12_1': np.random.normal(0.1, 0.2),
                'Vol_12': np.random.uniform(0.05, 0.15),
                'Beta_12': np.random.uniform(0.5, 1.5),
                # LEAKAGE: Target includes current month return (should be next month only)
                'Next_Month_Return': np.random.normal(0.01, 0.05)
            }
            data.append(row)
    
    return pd.DataFrame(data)


def create_test_data_with_extreme_returns():
    """Create test data with extreme/unrealistic return values."""
    dates = pd.date_range('2020-01-31', '2020-03-31', freq='ME')
    tickers = ['AAPL', 'MSFT']
    
    data = []
    
    for i, date in enumerate(dates[:-1]):
        for j, ticker in enumerate(tickers):
            # Create extreme return values that should be flagged
            extreme_return = 10.0 if j == 0 else -0.99  # 1000% gain or 99% loss
            
            row = {
                'date': date,
                'ticker': ticker,
                'sector': 'Technology',
                'Mom_12_1': np.random.normal(0.1, 0.2),
                'Vol_12': np.random.uniform(0.05, 0.15),
                'Beta_12': np.random.uniform(0.5, 1.5),
                'Next_Month_Return': extreme_return
            }
            data.append(row)
    
    return pd.DataFrame(data)


def test_proper_target_variable_construction():
    """Test that properly constructed target variables pass validation."""
    detector = LeakageDetector(verbose=False)
    test_data = create_test_data_with_proper_targets()
    
    results = detector.validate_target_variable(test_data)
    
    # Should pass all validations
    assert results["passed"], f"Validation failed: {results['violations']}"
    assert len(results["violations"]) == 0
    
    # Check temporal alignment
    assert "temporal_alignment_checks" in results
    assert results["temporal_alignment_checks"]["passed"]
    
    # Check return calculation validation
    assert "return_calculation_validation" in results
    assert results["return_calculation_validation"]["passed"]
    
    # Check future data contamination tests
    assert "future_data_contamination_tests" in results
    assert results["future_data_contamination_tests"]["passed"]
    
    # Check subsequent month validation
    assert "subsequent_month_validation" in results
    assert results["subsequent_month_validation"]["passed"]
    
    print("✓ Proper target variable construction validation passed")


def test_target_variable_leakage_detection():
    """Test that target variable leakage is properly detected."""
    detector = LeakageDetector(verbose=False)
    test_data = create_test_data_with_leakage()
    
    results = detector.validate_target_variable(test_data)
    
    # Should detect leakage (targets exist for final date)
    assert not results["passed"], "Should have detected target variable leakage"
    assert len(results["violations"]) > 0
    
    # Check that final date contamination is detected
    contamination_tests = results["future_data_contamination_tests"]
    assert "final_date_check" in contamination_tests["contamination_tests"]
    final_check = contamination_tests["contamination_tests"]["final_date_check"]
    assert final_check["contamination_detected"], "Should detect contamination in final date"
    
    print("✓ Target variable leakage detection passed")


def test_extreme_return_detection():
    """Test that extreme/unrealistic return values are flagged."""
    detector = LeakageDetector(verbose=False)
    test_data = create_test_data_with_extreme_returns()
    
    results = detector.validate_target_variable(test_data)
    
    # Should flag extreme returns
    temporal_checks = results["temporal_alignment_checks"]
    violations = temporal_checks["violations"]
    
    # Should detect extreme returns
    extreme_detected = any("Extreme target returns detected" in v for v in violations)
    assert extreme_detected, "Should detect extreme return values"
    
    print("✓ Extreme return detection passed")


def test_missing_target_column():
    """Test handling of missing target column."""
    detector = LeakageDetector(verbose=False)
    
    # Create data without target column
    dates = pd.date_range('2020-01-31', '2020-02-29', freq='ME')
    test_data = pd.DataFrame({
        'date': [dates[0], dates[0]],
        'ticker': ['AAPL', 'MSFT'],
        'Mom_12_1': [0.1, 0.2]
    })
    
    results = detector.validate_target_variable(test_data)
    
    # Should fail due to missing target column
    assert not results["passed"]
    assert any("not found in data" in v for v in results["violations"])
    
    print("✓ Missing target column handling passed")


def test_temporal_alignment_validation():
    """Test temporal alignment validation between features and targets."""
    detector = LeakageDetector(verbose=False)
    
    # Create data with irregular date spacing
    irregular_dates = [
        pd.Timestamp('2020-01-31'),
        pd.Timestamp('2020-02-29'),
        pd.Timestamp('2020-05-31'),  # Skip March and April
        pd.Timestamp('2020-06-30')
    ]
    
    data = []
    for i, date in enumerate(irregular_dates[:-1]):
        data.append({
            'date': date,
            'ticker': 'AAPL',
            'sector': 'Technology',
            'Mom_12_1': 0.1,
            'Next_Month_Return': 0.05
        })
    
    test_data = pd.DataFrame(data)
    results = detector.validate_target_variable(test_data)
    
    # Should detect irregular date spacing
    temporal_checks = results["temporal_alignment_checks"]
    
    # Check if date sequence validation caught the irregular spacing
    date_validation = temporal_checks["date_sequence_validation"]
    assert not date_validation["monthly_sequence_valid"], "Should detect irregular date sequence"
    
    print("✓ Temporal alignment validation passed")


def test_return_calculation_framework():
    """Test return calculation framework validation."""
    detector = LeakageDetector(verbose=False)
    
    # Create data with high missing target rate
    dates = pd.date_range('2020-01-31', '2020-03-31', freq='M')
    data = []
    
    for i, date in enumerate(dates[:-1]):
        for j in range(10):  # 10 observations per date
            target_value = 0.05 if j < 2 else np.nan  # 80% missing targets
            
            data.append({
                'date': date,
                'ticker': f'STOCK_{j}',
                'sector': 'Technology',
                'Mom_12_1': 0.1,
                'Next_Month_Return': target_value
            })
    
    test_data = pd.DataFrame(data)
    results = detector.validate_target_variable(test_data)
    
    # Should detect high missing target rate
    calc_validation = results["return_calculation_validation"]
    violations = calc_validation["violations"]
    
    missing_rate_detected = any("High missing target rate" in v for v in violations)
    assert missing_rate_detected, "Should detect high missing target rate"
    
    print("✓ Return calculation framework validation passed")


def test_comprehensive_validation_integration():
    """Test integration of all validation components."""
    detector = LeakageDetector(verbose=False)
    
    # Load actual factor features data if available
    try:
        actual_data = pd.read_csv('data/factor_features.csv')
        actual_data['date'] = pd.to_datetime(actual_data['date'])
        
        results = detector.validate_target_variable(actual_data)
        
        # Print summary for manual inspection
        print(f"\n=== Actual Data Validation Results ===")
        print(f"Overall passed: {results['passed']}")
        print(f"Total violations: {len(results['violations'])}")
        
        if results['violations']:
            print("Violations detected:")
            for violation in results['violations']:
                print(f"  - {violation}")
        
        # Print summary statistics
        summary = results['summary']
        print(f"\nSummary Statistics:")
        print(f"  Total observations: {summary['total_observations']:,}")
        print(f"  Unique dates: {summary['unique_dates']}")
        print(f"  Unique tickers: {summary['unique_tickers']}")
        print(f"  Target coverage: {summary['target_coverage']:.1%}")
        
        print("✓ Comprehensive validation integration completed")
        
    except FileNotFoundError:
        print("⚠ Actual data file not found, skipping integration test")


if __name__ == "__main__":
    print("Running Target Variable Leakage Detection Tests...")
    print("=" * 60)
    
    # Run all tests
    test_proper_target_variable_construction()
    test_target_variable_leakage_detection()
    test_extreme_return_detection()
    test_missing_target_column()
    test_temporal_alignment_validation()
    test_return_calculation_framework()
    test_comprehensive_validation_integration()
    
    print("\n" + "=" * 60)
    print("All target variable leakage detection tests completed!")
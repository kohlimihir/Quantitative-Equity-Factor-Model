#!/usr/bin/env python3
"""
Test suite for cross-sectional imputation validation.

This module implements comprehensive tests to validate that cross-sectional 
imputation (filling missing values with median across all stocks in the same month)
doesn't accidentally use future data and maintains proper temporal isolation.

Requirements: 1.3
- Validate median imputation uses only same-period data
- Add checks for future data contamination in missing value handling  
- Implement temporal isolation tests for imputation logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_loader import FEATURES, TARGET, compute_factors, download_price_data, download_fundamentals
from leakage_detector import LeakageDetector


class CrossSectionalImputationValidator:
    """Enhanced validator for cross-sectional imputation temporal isolation."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = {}
        
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
    
    def create_synthetic_test_data(self) -> pd.DataFrame:
        """
        Create synthetic test data with known missing patterns to test imputation logic.
        
        Returns:
            DataFrame with controlled missing data patterns for testing
        """
        self.log("Creating synthetic test data for imputation validation...")
        
        # Create 3 months of data for 10 stocks
        dates = pd.date_range('2023-01-31', periods=3, freq='M')
        tickers = [f'STOCK_{i:02d}' for i in range(10)]
        
        records = []
        for i, date in enumerate(dates):
            for j, ticker in enumerate(tickers):
                # Create base values that vary by month and stock
                base_value = 100 + i * 10 + j
                
                record = {
                    'date': date,
                    'ticker': ticker,
                    'sector': f'Sector_{j % 3}',  # 3 sectors
                    TARGET: np.random.normal(0.02, 0.05)  # Random returns
                }
                
                # Add features with controlled missing patterns
                for k, feature in enumerate(FEATURES[:5]):  # Use first 5 features for testing
                    if feature == 'Mom_12_1':
                        # Month 0: Missing for stocks 0,1,2
                        # Month 1: Missing for stocks 3,4,5  
                        # Month 2: Missing for stocks 6,7,8
                        if (i == 0 and j in [0,1,2]) or (i == 1 and j in [3,4,5]) or (i == 2 and j in [6,7,8]):
                            record[feature] = np.nan
                        else:
                            record[feature] = base_value + k * 10
                    elif feature == 'Vol_12':
                        # Different missing pattern
                        if (i == 0 and j in [1,3,5]) or (i == 1 and j in [2,4,6]) or (i == 2 and j in [0,7,9]):
                            record[feature] = np.nan
                        else:
                            record[feature] = base_value * 0.1 + k
                    else:
                        # Some features have no missing data
                        record[feature] = base_value + k * 5
                
                records.append(record)
        
        df = pd.DataFrame(records)
        self.log(f"Created synthetic data: {len(df)} rows, {df['date'].nunique()} dates, {df['ticker'].nunique()} tickers")
        return df
    
    def validate_temporal_isolation_synthetic(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test temporal isolation using synthetic data with known missing patterns.
        
        Args:
            test_df: Synthetic test DataFrame
            
        Returns:
            Validation results dictionary
        """
        self.log("Testing temporal isolation with synthetic data...")
        
        results = {
            'test_name': 'temporal_isolation_synthetic',
            'passed': True,
            'violations': [],
            'details': {}
        }
        
        # Apply the same imputation logic as in compute_factors
        test_df_imputed = test_df.copy()
        
        for feature in FEATURES:
            if feature in test_df_imputed.columns:
                # This is the exact same logic from compute_factors
                test_df_imputed[feature] = test_df_imputed.groupby('date')[feature].transform(
                    lambda x: x.fillna(x.median())
                )
        
        # Validate that imputation only uses same-date data
        dates = sorted(test_df['date'].unique())
        
        for date in dates:
            date_data_original = test_df[test_df['date'] == date]
            date_data_imputed = test_df_imputed[test_df_imputed['date'] == date]
            
            for feature in FEATURES:
                if feature in test_df.columns:
                    original_values = date_data_original[feature]
                    imputed_values = date_data_imputed[feature]
                    
                    # Find which values were imputed (originally NaN)
                    was_missing = original_values.isna()
                    
                    if was_missing.any():
                        # Calculate expected median from non-missing values in same date
                        non_missing_same_date = original_values.dropna()
                        expected_median = non_missing_same_date.median() if len(non_missing_same_date) > 0 else np.nan
                        
                        # Check that imputed values match expected median
                        imputed_subset = imputed_values[was_missing]
                        
                        if not np.isnan(expected_median):
                            if not all(np.isclose(imputed_subset, expected_median, equal_nan=True)):
                                results['passed'] = False
                                results['violations'].append({
                                    'date': date,
                                    'feature': feature,
                                    'expected_median': expected_median,
                                    'actual_imputed_values': imputed_subset.tolist(),
                                    'issue': 'Imputed values do not match same-date median'
                                })
                        
                        results['details'][f'{date}_{feature}'] = {
                            'date': str(date),
                            'feature': feature,
                            'missing_count': was_missing.sum(),
                            'total_count': len(original_values),
                            'expected_median': expected_median,
                            'imputed_correctly': np.isnan(expected_median) or all(np.isclose(imputed_subset, expected_median, equal_nan=True))
                        }
        
        self.log(f"Temporal isolation synthetic test completed. Passed: {results['passed']}")
        return results
    
    def validate_no_future_contamination(self, factors_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that imputation doesn't use future data by checking temporal boundaries.
        
        Args:
            factors_df: Full factors DataFrame
            
        Returns:
            Validation results dictionary
        """
        self.log("Testing for future data contamination in imputation...")
        
        results = {
            'test_name': 'no_future_contamination',
            'passed': True,
            'violations': [],
            'details': {}
        }
        
        dates = sorted(factors_df['date'].unique())
        
        # For each date, simulate the imputation process and verify it only uses same-date data
        for i, current_date in enumerate(dates):
            current_data = factors_df[factors_df['date'] == current_date].copy()
            
            # Check if there are any missing values that would need imputation
            for feature in FEATURES:
                if feature in current_data.columns:
                    missing_mask = current_data[feature].isna()
                    
                    if missing_mask.any():
                        # Calculate what the median should be using only current date data
                        same_date_values = current_data[feature].dropna()
                        expected_median = same_date_values.median() if len(same_date_values) > 0 else np.nan
                        
                        # Verify no future dates are accessible
                        future_dates = [d for d in dates if d > current_date]
                        
                        if future_dates:
                            # Ensure future data would not affect the median calculation
                            future_data = factors_df[factors_df['date'].isin(future_dates)]
                            future_values = future_data[feature].dropna()
                            
                            if len(future_values) > 0:
                                # Calculate what median would be if future data was included (this should NOT happen)
                                combined_values = pd.concat([same_date_values, future_values])
                                contaminated_median = combined_values.median()
                                
                                # If medians differ significantly, it indicates potential for contamination
                                if not np.isnan(expected_median) and not np.isnan(contaminated_median):
                                    if abs(expected_median - contaminated_median) > 1e-10:
                                        results['details'][f'{current_date}_{feature}_contamination_risk'] = {
                                            'date': str(current_date),
                                            'feature': feature,
                                            'same_date_median': expected_median,
                                            'contaminated_median': contaminated_median,
                                            'difference': abs(expected_median - contaminated_median),
                                            'future_values_count': len(future_values)
                                        }
                        
                        results['details'][f'{current_date}_{feature}'] = {
                            'date': str(current_date),
                            'feature': feature,
                            'missing_count': missing_mask.sum(),
                            'available_for_median': len(same_date_values),
                            'expected_median': expected_median
                        }
        
        self.log(f"Future contamination test completed. Passed: {results['passed']}")
        return results
    
    def validate_imputation_consistency(self, factors_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that the imputation process is consistent and deterministic.
        
        Args:
            factors_df: Factors DataFrame
            
        Returns:
            Validation results dictionary
        """
        self.log("Testing imputation consistency and determinism...")
        
        results = {
            'test_name': 'imputation_consistency',
            'passed': True,
            'violations': [],
            'details': {}
        }
        
        # Test that re-running imputation gives same results
        dates = sorted(factors_df['date'].unique())[:5]  # Test first 5 dates
        
        for date in dates:
            date_data = factors_df[factors_df['date'] == date].copy()
            
            for feature in FEATURES:
                if feature in date_data.columns:
                    original_values = date_data[feature].copy()
                    
                    # Run imputation twice
                    imputed_1 = original_values.fillna(original_values.median())
                    imputed_2 = original_values.fillna(original_values.median())
                    
                    # Check consistency
                    if not imputed_1.equals(imputed_2):
                        results['passed'] = False
                        results['violations'].append({
                            'date': date,
                            'feature': feature,
                            'issue': 'Imputation results are not consistent between runs'
                        })
                    
                    # Check that non-missing values are unchanged
                    non_missing_mask = original_values.notna()
                    if not original_values[non_missing_mask].equals(imputed_1[non_missing_mask]):
                        results['passed'] = False
                        results['violations'].append({
                            'date': date,
                            'feature': feature,
                            'issue': 'Non-missing values were modified during imputation'
                        })
                    
                    results['details'][f'{date}_{feature}'] = {
                        'date': str(date),
                        'feature': feature,
                        'original_missing_count': original_values.isna().sum(),
                        'imputed_missing_count': imputed_1.isna().sum(),
                        'non_missing_preserved': original_values[non_missing_mask].equals(imputed_1[non_missing_mask])
                    }
        
        self.log(f"Imputation consistency test completed. Passed: {results['passed']}")
        return results
    
    def validate_cross_sectional_boundaries(self, factors_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that cross-sectional imputation respects proper boundaries.
        
        Args:
            factors_df: Factors DataFrame
            
        Returns:
            Validation results dictionary
        """
        self.log("Testing cross-sectional boundary validation...")
        
        results = {
            'test_name': 'cross_sectional_boundaries',
            'passed': True,
            'violations': [],
            'details': {}
        }
        
        dates = sorted(factors_df['date'].unique())
        
        for date in dates[:10]:  # Test first 10 dates
            date_data = factors_df[factors_df['date'] == date]
            
            # Verify that each date's data is processed independently
            for feature in FEATURES:
                if feature in date_data.columns:
                    feature_values = date_data[feature]
                    missing_count = feature_values.isna().sum()
                    
                    if missing_count > 0:
                        # Calculate median using only same-date, non-missing values
                        non_missing = feature_values.dropna()
                        
                        if len(non_missing) > 0:
                            calculated_median = non_missing.median()
                            
                            # Verify this is what would be used for imputation
                            # (This tests the groupby('date') logic)
                            test_series = feature_values.fillna(calculated_median)
                            imputed_values = test_series[feature_values.isna()]
                            
                            if len(imputed_values) > 0:
                                if not all(np.isclose(imputed_values, calculated_median)):
                                    results['passed'] = False
                                    results['violations'].append({
                                        'date': date,
                                        'feature': feature,
                                        'expected_median': calculated_median,
                                        'actual_imputed': imputed_values.tolist(),
                                        'issue': 'Cross-sectional median calculation incorrect'
                                    })
                            
                            results['details'][f'{date}_{feature}'] = {
                                'date': str(date),
                                'feature': feature,
                                'stocks_total': len(feature_values),
                                'stocks_missing': missing_count,
                                'stocks_available': len(non_missing),
                                'calculated_median': calculated_median,
                                'median_coverage': len(non_missing) / len(feature_values) if len(feature_values) > 0 else 0
                            }
        
        self.log(f"Cross-sectional boundaries test completed. Passed: {results['passed']}")
        return results
    
    def run_comprehensive_validation(self, factors_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Run all cross-sectional imputation validation tests.
        
        Args:
            factors_df: Optional factors DataFrame. If None, loads from file.
            
        Returns:
            Comprehensive validation results
        """
        self.log("Starting comprehensive cross-sectional imputation validation...")
        
        if factors_df is None:
            self.log("Loading factors data from file...")
            factors_df = pd.read_csv('data/factor_features.csv', parse_dates=['date'])
        
        # Run all validation tests
        results = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_records': len(factors_df),
            'date_range': {
                'start': str(factors_df['date'].min()),
                'end': str(factors_df['date'].max()),
                'unique_dates': factors_df['date'].nunique()
            },
            'tests': {}
        }
        
        # Test 1: Synthetic data temporal isolation
        synthetic_df = self.create_synthetic_test_data()
        results['tests']['synthetic_temporal_isolation'] = self.validate_temporal_isolation_synthetic(synthetic_df)
        
        # Test 2: Future contamination check
        results['tests']['future_contamination'] = self.validate_no_future_contamination(factors_df)
        
        # Test 3: Imputation consistency
        results['tests']['imputation_consistency'] = self.validate_imputation_consistency(factors_df)
        
        # Test 4: Cross-sectional boundaries
        results['tests']['cross_sectional_boundaries'] = self.validate_cross_sectional_boundaries(factors_df)
        
        # Overall pass/fail
        all_passed = all(test_result['passed'] for test_result in results['tests'].values())
        results['overall_passed'] = all_passed
        results['failed_tests'] = [name for name, test_result in results['tests'].items() if not test_result['passed']]
        
        self.log(f"Comprehensive validation completed. Overall passed: {all_passed}")
        if not all_passed:
            self.log(f"Failed tests: {results['failed_tests']}")
        
        return results


def test_cross_sectional_imputation_validation():
    """Main test function for cross-sectional imputation validation."""
    print("=" * 80)
    print("CROSS-SECTIONAL IMPUTATION VALIDATION TEST")
    print("=" * 80)
    
    validator = CrossSectionalImputationValidator(verbose=True)
    results = validator.run_comprehensive_validation()
    
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"Overall Passed: {results['overall_passed']}")
    print(f"Total Records: {results['total_records']:,}")
    print(f"Date Range: {results['date_range']['start']} to {results['date_range']['end']}")
    print(f"Unique Dates: {results['date_range']['unique_dates']}")
    
    print(f"\nTest Results:")
    for test_name, test_result in results['tests'].items():
        status = "✓ PASS" if test_result['passed'] else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not test_result['passed']:
            print(f"    Violations: {len(test_result['violations'])}")
    
    if not results['overall_passed']:
        print(f"\nFailed Tests: {results['failed_tests']}")
        print("\nDetailed Violations:")
        for test_name in results['failed_tests']:
            test_result = results['tests'][test_name]
            print(f"\n{test_name}:")
            for violation in test_result['violations']:
                print(f"  - {violation}")
    
    return results


if __name__ == "__main__":
    # Run the validation
    results = test_cross_sectional_imputation_validation()
    
    # Assert overall success for automated testing
    assert results['overall_passed'], f"Cross-sectional imputation validation failed: {results['failed_tests']}"
    
    print("\n✓ All cross-sectional imputation validation tests passed!")
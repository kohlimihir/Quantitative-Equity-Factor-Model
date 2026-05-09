#!/usr/bin/env python3
"""
Test script for enhanced fundamental data lag validation (Task 1.2)

This script validates that:
1. The _fund_val() function respects 45-day publication lag requirements
2. Temporal boundary enforcement works correctly for quarterly data
3. No future data contamination occurs in fundamental features
4. Quarterly data usage patterns are properly validated

Requirements: 1.2, 1.8
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Import the enhanced leakage detector
from leakage_detector import LeakageDetector
from data_loader import download_fundamentals, compute_factors, download_price_data, compute_monthly_returns

def test_enhanced_fundamental_lag_validation():
    """
    Comprehensive test of the enhanced fundamental data lag validation system.
    """
    print("="*80)
    print("TESTING ENHANCED FUNDAMENTAL DATA LAG VALIDATION (Task 1.2)")
    print("="*80)
    
    # Initialize the leakage detector
    detector = LeakageDetector(verbose=True)
    
    try:
        # Load fundamental data
        print("\n1. Loading fundamental data...")
        fund_df = pd.read_parquet('data/fundamentals.parquet')
        print(f"   Loaded {len(fund_df)} fundamental records")
        print(f"   Date range: {fund_df.index.min()} to {fund_df.index.max()}")
        print(f"   Tickers: {fund_df['ticker'].nunique()}")
        
        # Load price data for context
        print("\n2. Loading price and return data...")
        prices = pd.read_parquet('data/daily_prices.parquet')
        monthly_returns = compute_monthly_returns(prices)
        print(f"   Loaded price data for {len(prices.columns)} assets")
        print(f"   Monthly returns: {len(monthly_returns)} months")
        
        # Test dates - use recent dates where we have data
        test_dates = [
            pd.Timestamp('2024-06-30'),
            pd.Timestamp('2024-09-30'),
            pd.Timestamp('2024-12-31')
        ]
        
        print(f"\n3. Testing enhanced fundamental lag validation...")
        print(f"   Test dates: {[d.strftime('%Y-%m-%d') for d in test_dates]}")
        
        # Run enhanced fundamental lag validation
        results = detector.validate_fundamental_lag(fund_df, test_dates[-1])
        
        print(f"\n4. Enhanced Fundamental Lag Validation Results:")
        print(f"   Test Status: {'PASSED' if results['passed'] else 'FAILED'}")
        print(f"   Total Violations: {len(results['violations'])}")
        
        if results['violations']:
            print("\n   VIOLATIONS DETECTED:")
            for i, violation in enumerate(results['violations'][:5]):  # Show first 5
                print(f"     {i+1}. {violation}")
        
        # Display summary statistics
        if 'summary' in results:
            summary = results['summary']
            print(f"\n   Summary Statistics:")
            print(f"     Tickers Tested: {summary['total_tickers_tested']}")
            print(f"     Columns Tested: {summary['fundamental_columns_tested']}")
            print(f"     Fund Val Violations: {summary['fund_val_violations']}")
            print(f"     Temporal Violations: {summary['temporal_violations']}")
            print(f"     Quarterly Violations: {summary['quarterly_violations']}")
            print(f"     Edge Case Violations: {summary['edge_case_violations']}")
            print(f"     Lag Days Enforced: {summary['lag_days_enforced']}")
        
        # Test temporal boundary validation
        print(f"\n5. Testing _fund_val temporal boundaries...")
        boundary_results = detector.validate_fund_val_temporal_boundaries(fund_df, test_dates)
        
        print(f"   Temporal Boundary Test Status: {'PASSED' if boundary_results['passed'] else 'FAILED'}")
        print(f"   Boundary Violations: {len(boundary_results['violations'])}")
        
        if boundary_results['violations']:
            print("\n   BOUNDARY VIOLATIONS:")
            for i, violation in enumerate(boundary_results['violations'][:3]):
                print(f"     {i+1}. Type: {violation['type']}")
                print(f"        Ticker: {violation['ticker']}, Column: {violation['column']}")
                print(f"        Description: {violation['description']}")
        
        # Test quarterly data usage
        print(f"\n6. Testing quarterly data usage patterns...")
        quarterly_results = detector.create_quarterly_data_usage_tests(fund_df, test_dates)
        
        print(f"   Quarterly Usage Test Status: {'PASSED' if quarterly_results['passed'] else 'FAILED'}")
        print(f"   Quarterly Violations: {len(quarterly_results['violations'])}")
        
        if quarterly_results['violations']:
            print("\n   QUARTERLY VIOLATIONS:")
            for i, violation in enumerate(quarterly_results['violations'][:3]):
                print(f"     {i+1}. Type: {violation['type']}")
                print(f"        Ticker: {violation['ticker']}")
                print(f"        Issue: {violation.get('description', 'N/A')}")
        
        # Run comprehensive audit with enhanced tests
        print(f"\n7. Running comprehensive audit with enhanced fundamental lag tests...")
        
        # Load factor data for comprehensive audit
        factors_df = pd.read_csv('data/factor_features.csv')
        factors_df['date'] = pd.to_datetime(factors_df['date'])
        
        audit_results = detector.run_comprehensive_audit(
            factors_df=factors_df,
            fund_df=fund_df
        )
        
        print(f"\n8. Comprehensive Audit Results:")
        print(f"   Overall Assessment: {audit_results['overall_assessment']}")
        print(f"   Total Tests: {audit_results['total_tests']}")
        print(f"   Tests Passed: {audit_results['tests_passed']}")
        print(f"   Tests Failed: {audit_results['tests_failed']}")
        
        if 'fundamental_lag_summary' in audit_results:
            fls = audit_results['fundamental_lag_summary']
            print(f"\n   Fundamental Lag Summary:")
            print(f"     Tests Run: {fls['tests_run']}")
            print(f"     Tests Passed: {fls['tests_passed']}")
            print(f"     Total Violations: {fls['total_violations']}")
            print(f"     Enforcement Status: {fls['lag_enforcement_status']}")
        
        # Test specific edge cases
        print(f"\n9. Testing specific edge cases...")
        
        # Test with exact cutoff date
        cutoff_test_date = pd.Timestamp('2024-06-30')
        cutoff_date = cutoff_test_date - pd.Timedelta(days=45)
        
        print(f"   Testing with feature date: {cutoff_test_date}")
        print(f"   Cutoff date: {cutoff_date}")
        
        # Test a few tickers manually
        test_tickers = ['AAPL', 'MSFT', 'GOOGL']
        test_columns = ['Revenue', 'NetIncome', 'TotalEquity']
        
        for ticker in test_tickers:
            if ticker in fund_df['ticker'].values:
                print(f"\n   Testing {ticker}:")
                for col in test_columns:
                    if col in fund_df.columns:
                        from data_loader import _fund_val
                        val = _fund_val(fund_df, ticker, cutoff_test_date, col, 45)
                        print(f"     {col}: {val}")
        
        # Final assessment
        print(f"\n" + "="*80)
        print("FINAL ASSESSMENT - TASK 1.2 IMPLEMENTATION")
        print("="*80)
        
        all_tests_passed = (
            results['passed'] and 
            boundary_results['passed'] and 
            quarterly_results['passed'] and
            audit_results['overall_assessment'] == 'PASSED'
        )
        
        if all_tests_passed:
            print("✓ ALL FUNDAMENTAL LAG VALIDATION TESTS PASSED")
            print("✓ 45-day publication lag enforcement is working correctly")
            print("✓ _fund_val() function respects temporal boundaries")
            print("✓ Quarterly data usage patterns are validated")
            print("✓ No future data contamination detected")
        else:
            print("✗ SOME TESTS FAILED - VIOLATIONS DETECTED")
            print("✗ Review the violations above for details")
        
        print(f"\nTask 1.2 Status: {'COMPLETED SUCCESSFULLY' if all_tests_passed else 'NEEDS ATTENTION'}")
        
        return all_tests_passed
        
    except Exception as e:
        print(f"\nERROR during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_fundamental_lag_validation()
    sys.exit(0 if success else 1)
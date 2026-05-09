"""
Integration test for EWM smoothing leakage validation with actual sector neutralization.

This test runs the sector neutralization pipeline and validates that
the EWM smoothing operations don't have data leakage.

Requirements: 1.6
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from leakage_detector import LeakageDetector
from sector_neutralisation import walk_forward_sector_neutral, add_sector_features
from data_loader import FEATURES, TARGET, SECTOR_MAP


def test_ewm_leakage_with_real_sector_neutralization():
    """Test EWM leakage validation with real sector neutralization data."""
    print("="*60)
    print("TESTING EWM LEAKAGE WITH REAL SECTOR NEUTRALIZATION")
    print("="*60)
    
    # Check if we have the required data
    if not os.path.exists("data/factor_features.csv"):
        print("⚠️  factor_features.csv not found. Creating synthetic data for testing...")
        
        # Create synthetic factor data for testing
        dates = pd.date_range('2024-01-01', periods=6, freq='MS')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'PG']
        
        factors_data = []
        for date in dates:
            for ticker in tickers:
                # Create synthetic feature data
                row = {'date': date, 'ticker': ticker}
                
                # Add synthetic features
                for feature in FEATURES:
                    row[feature] = np.random.normal(0, 1)
                
                # Add target variable
                row[TARGET] = np.random.normal(0.01, 0.05)
                
                factors_data.append(row)
        
        factors_df = pd.DataFrame(factors_data)
        
        # Save synthetic data
        os.makedirs("data", exist_ok=True)
        factors_df.to_csv("data/factor_features.csv", index=False)
        print("✓ Created synthetic factor_features.csv for testing")
    else:
        # Load existing data
        factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
        print(f"✓ Loaded existing factor_features.csv with {len(factors_df)} rows")
    
    try:
        # Run sector neutralization
        print("\n1. Running sector neutralization...")
        factors_df, z_features = add_sector_features(factors_df)
        results_df = walk_forward_sector_neutral(factors_df, z_features, min_train_months=2)
        
        print(f"✓ Sector neutralization completed. Generated {len(results_df)} predictions")
        
        # Initialize leakage detector
        print("\n2. Running EWM leakage validation...")
        detector = LeakageDetector(verbose=True)
        
        # Run comprehensive EWM validation
        ewm_results = detector.run_ewm_leakage_validation(results_df)
        
        # Print results
        print("\n" + "="*60)
        print("EWM LEAKAGE VALIDATION RESULTS")
        print("="*60)
        print(f"Total EWM Tests: {ewm_results['total_ewm_tests']}")
        print(f"Tests Passed: {ewm_results['ewm_tests_passed']}")
        print(f"Tests Failed: {ewm_results['ewm_tests_failed']}")
        print(f"Overall Assessment: {ewm_results['ewm_overall_assessment']}")
        
        if ewm_results['ewm_violations']:
            print("\nVIOLATIONS DETECTED:")
            for violation in ewm_results['ewm_violations']:
                print(f"  - {violation}")
        else:
            print("\n✓ No EWM leakage violations detected!")
        
        # Test individual EWM validation methods
        print("\n3. Testing individual EWM validation methods...")
        
        # Test temporal sequence validation
        temporal_result = detector.validate_ewm_temporal_sequence(results_df)
        print(f"✓ Temporal sequence validation: {'PASSED' if temporal_result['passed'] else 'FAILED'}")
        
        # Test ranking calculations validation
        ranking_result = detector.validate_ewm_ranking_calculations(results_df)
        print(f"✓ Ranking calculations validation: {'PASSED' if ranking_result['passed'] else 'FAILED'}")
        
        # Test future data prevention
        prevention_result = detector.validate_ewm_future_data_prevention(results_df)
        print(f"✓ Future data prevention validation: {'PASSED' if prevention_result['passed'] else 'FAILED'}")
        
        # Overall assessment
        all_passed = (ewm_results['ewm_overall_assessment'] == 'PASSED' and
                     temporal_result['passed'] and
                     ranking_result['passed'] and
                     prevention_result['passed'])
        
        print("\n" + "="*60)
        if all_passed:
            print("🎉 ALL EWM LEAKAGE VALIDATION TESTS PASSED!")
            print("The EWM smoothing implementation is free from data leakage.")
        else:
            print("⚠️  SOME EWM LEAKAGE VALIDATION TESTS FAILED!")
            print("Please review the EWM smoothing implementation for potential data leakage.")
        print("="*60)
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ Error during EWM leakage validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_ewm_validation_with_comprehensive_audit():
    """Test EWM validation integration with comprehensive leakage audit."""
    print("\n" + "="*60)
    print("TESTING EWM VALIDATION WITH COMPREHENSIVE AUDIT")
    print("="*60)
    
    try:
        # Load factor data
        if os.path.exists("data/factor_features.csv"):
            factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
        else:
            print("⚠️  No factor data available for comprehensive audit test")
            return False
        
        # Initialize detector
        detector = LeakageDetector(verbose=True)
        
        # Run comprehensive audit (includes EWM future data prevention test)
        audit_results = detector.run_comprehensive_audit(factors_df)
        
        # Check results
        print(f"\nComprehensive audit results:")
        print(f"Total tests: {audit_results['total_tests']}")
        print(f"Tests passed: {audit_results['tests_passed']}")
        print(f"Tests failed: {audit_results['tests_failed']}")
        print(f"Overall assessment: {audit_results['overall_assessment']}")
        
        # Check if EWM test was included
        ewm_test_included = "ewm_future_data_prevention" in audit_results["test_results"]
        if ewm_test_included:
            ewm_test_result = audit_results["test_results"]["ewm_future_data_prevention"]
            print(f"EWM future data prevention test: {'PASSED' if ewm_test_result['passed'] else 'FAILED'}")
        else:
            print("⚠️  EWM future data prevention test was not included in comprehensive audit")
        
        return audit_results['overall_assessment'] == 'PASSED' and ewm_test_included
        
    except Exception as e:
        print(f"\n❌ Error during comprehensive audit: {str(e)}")
        return False


if __name__ == "__main__":
    print("Starting EWM leakage validation integration tests...")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Test 1: EWM validation with real sector neutralization
    test1_passed = test_ewm_leakage_with_real_sector_neutralization()
    
    # Test 2: EWM validation with comprehensive audit
    test2_passed = test_ewm_validation_with_comprehensive_audit()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"EWM with Sector Neutralization: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"EWM with Comprehensive Audit: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("EWM smoothing leakage validation is working correctly with the real implementation.")
    else:
        print("\n⚠️  SOME INTEGRATION TESTS FAILED!")
        print("Please review the EWM leakage validation implementation.")
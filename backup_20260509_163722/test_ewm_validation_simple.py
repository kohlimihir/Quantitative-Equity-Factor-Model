"""
Simple test for EWM smoothing leakage validation using existing sector predictions.

Requirements: 1.6
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from leakage_detector import LeakageDetector


def test_ewm_validation_with_existing_data():
    """Test EWM leakage validation with existing sector predictions."""
    print("="*60)
    print("TESTING EWM LEAKAGE WITH EXISTING SECTOR PREDICTIONS")
    print("="*60)
    
    # Load existing sector predictions
    if not os.path.exists("data/sector_predictions.csv"):
        print("❌ sector_predictions.csv not found. Please run sector neutralization first.")
        return False
    
    try:
        # Load sector predictions
        results_df = pd.read_csv("data/sector_predictions.csv", parse_dates=["date"])
        print(f"✓ Loaded sector predictions: {len(results_df)} rows, {results_df['date'].nunique()} months")
        
        # Display data info
        print(f"Date range: {results_df['date'].min()} to {results_df['date'].max()}")
        print(f"Unique tickers: {results_df['ticker'].nunique()}")
        print(f"Columns: {list(results_df.columns)}")
        
        # Initialize leakage detector
        detector = LeakageDetector(verbose=True)
        
        # Run comprehensive EWM validation
        print("\n1. Running comprehensive EWM leakage validation...")
        ewm_results = detector.run_ewm_leakage_validation(results_df)
        
        # Print detailed results
        print("\n" + "="*60)
        print("EWM LEAKAGE VALIDATION RESULTS")
        print("="*60)
        print(f"Total EWM Tests: {ewm_results['total_ewm_tests']}")
        print(f"Tests Passed: {ewm_results['ewm_tests_passed']}")
        print(f"Tests Failed: {ewm_results['ewm_tests_failed']}")
        print(f"Overall Assessment: {ewm_results['ewm_overall_assessment']}")
        
        # Show individual test results
        for test_name, test_result in ewm_results['ewm_test_results'].items():
            status = "PASSED" if test_result['passed'] else "FAILED"
            print(f"  {test_name}: {status}")
            if not test_result['passed'] and test_result['violations']:
                for violation in test_result['violations'][:3]:  # Show first 3 violations
                    print(f"    - {violation}")
        
        if ewm_results['ewm_violations']:
            print(f"\nTotal violations: {len(ewm_results['ewm_violations'])}")
            print("First few violations:")
            for violation in ewm_results['ewm_violations'][:5]:
                print(f"  - {violation}")
        else:
            print("\n✓ No EWM leakage violations detected!")
        
        # Test with factors data for comprehensive audit
        print("\n2. Testing EWM validation in comprehensive audit...")
        if os.path.exists("data/factor_features.csv"):
            factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
            audit_results = detector.run_comprehensive_audit(factors_df)
            
            ewm_test_included = "ewm_future_data_prevention" in audit_results["test_results"]
            if ewm_test_included:
                ewm_audit_result = audit_results["test_results"]["ewm_future_data_prevention"]
                print(f"✓ EWM test in comprehensive audit: {'PASSED' if ewm_audit_result['passed'] else 'FAILED'}")
            else:
                print("⚠️  EWM test not included in comprehensive audit")
        
        # Overall assessment
        success = ewm_results['ewm_overall_assessment'] == 'PASSED'
        
        print("\n" + "="*60)
        if success:
            print("🎉 EWM LEAKAGE VALIDATION PASSED!")
            print("The EWM smoothing implementation appears to be free from data leakage.")
        else:
            print("⚠️  EWM LEAKAGE VALIDATION FAILED!")
            print("Potential data leakage detected in EWM smoothing implementation.")
        print("="*60)
        
        return success
        
    except Exception as e:
        print(f"\n❌ Error during EWM validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_ewm_validation_methods_individually():
    """Test individual EWM validation methods."""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL EWM VALIDATION METHODS")
    print("="*60)
    
    if not os.path.exists("data/sector_predictions.csv"):
        print("❌ sector_predictions.csv not found.")
        return False
    
    try:
        # Load data
        results_df = pd.read_csv("data/sector_predictions.csv", parse_dates=["date"])
        detector = LeakageDetector(verbose=False)  # Less verbose for individual tests
        
        # Test 1: Temporal sequence validation
        print("1. Testing temporal sequence validation...")
        temporal_result = detector.validate_ewm_temporal_sequence(results_df)
        print(f"   Result: {'PASSED' if temporal_result['passed'] else 'FAILED'}")
        if not temporal_result['passed']:
            print(f"   Violations: {len(temporal_result['violations'])}")
        
        # Test 2: Future data prevention validation
        print("2. Testing future data prevention validation...")
        prevention_result = detector.validate_ewm_future_data_prevention(results_df)
        print(f"   Result: {'PASSED' if prevention_result['passed'] else 'FAILED'}")
        if not prevention_result['passed']:
            print(f"   Violations: {len(prevention_result['violations'])}")
        
        # Test 3: Ranking calculations validation
        print("3. Testing ranking calculations validation...")
        ranking_result = detector.validate_ewm_ranking_calculations(results_df)
        print(f"   Result: {'PASSED' if ranking_result['passed'] else 'FAILED'}")
        if not ranking_result['passed']:
            print(f"   Violations: {len(ranking_result['violations'])}")
        
        # Summary
        all_passed = (temporal_result['passed'] and 
                     prevention_result['passed'] and 
                     ranking_result['passed'])
        
        print(f"\nIndividual tests summary: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        return all_passed
        
    except Exception as e:
        print(f"❌ Error during individual tests: {str(e)}")
        return False


if __name__ == "__main__":
    print("Starting EWM leakage validation tests with existing data...")
    
    # Test 1: Comprehensive EWM validation
    test1_passed = test_ewm_validation_with_existing_data()
    
    # Test 2: Individual method validation
    test2_passed = test_ewm_validation_methods_individually()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    print(f"Comprehensive EWM Validation: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Individual Methods Validation: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL EWM LEAKAGE VALIDATION TESTS PASSED!")
        print("Task 1.4 - EWM smoothing leakage checks implementation is complete and working correctly.")
    else:
        print("\n⚠️  SOME EWM LEAKAGE VALIDATION TESTS FAILED!")
        print("Please review the implementation for potential issues.")
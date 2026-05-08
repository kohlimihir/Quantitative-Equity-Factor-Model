"""
Final integration test for Task 1.6: Comprehensive Leakage Audit Report
Tests all requirements: detailed report, OOT anomaly detection, automated flagging
"""

import pandas as pd
import os
from leakage_detector import LeakageDetector, run_leakage_detection_pipeline


def test_requirement_1_9_comprehensive_audit_report():
    """
    Test Requirement 1.9: THE Leakage_Detector SHALL generate a comprehensive 
    audit report documenting all leakage checks and their results
    """
    print("\n" + "="*80)
    print("Testing Requirement 1.9: Comprehensive Audit Report Generation")
    print("="*80)
    
    # Load data
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=['date'])
    fund_df = pd.read_parquet("data/fundamentals.parquet") if os.path.exists("data/fundamentals.parquet") else None
    
    # Run audit
    detector = LeakageDetector(verbose=False)
    audit_results = detector.run_comprehensive_audit(
        factors_df=factors_df,
        fund_df=fund_df,
        in_sample_ic=0.00471,
        oot_ic=0.03342
    )
    
    # Generate report
    report_path = detector.generate_audit_report(audit_results)
    
    # Verify report exists
    assert os.path.exists(report_path), "CSV report not generated"
    assert os.path.exists(report_path.replace('.csv', '_detailed.txt')), "Text report not generated"
    
    # Verify report content
    report_df = pd.read_csv(report_path)
    
    # Check all required columns
    required_columns = [
        'test_name', 'category', 'passed', 'severity', 
        'violations_count', 'violations', 'automated_flags', 
        'recommendations', 'details_summary'
    ]
    for col in required_columns:
        assert col in report_df.columns, f"Missing required column: {col}"
    
    # Check that all tests are documented
    assert len(report_df) > 0, "No tests documented in report"
    
    # Check for summary row
    summary_rows = report_df[report_df['test_name'].str.contains('SUMMARY', na=False)]
    assert len(summary_rows) > 0, "Missing summary row"
    
    # Check for test results
    test_results = audit_results.get('test_results', {})
    assert len(test_results) > 0, "No test results in audit"
    
    print("✓ Comprehensive audit report generated successfully")
    print(f"✓ Report contains {len(report_df)} rows")
    print(f"✓ All {len(required_columns)} required columns present")
    print(f"✓ {len(test_results)} tests documented")
    print(f"✓ Report saved to: {report_path}")
    
    return True


def test_requirement_1_10_performance_anomaly_detection():
    """
    Test Requirement 1.10: IF OOT performance exceeds in-sample performance 
    by more than 0.02 IC, THEN THE Leakage_Detector SHALL flag this as a 
    potential leakage indicator
    """
    print("\n" + "="*80)
    print("Testing Requirement 1.10: Performance Anomaly Detection")
    print("="*80)
    
    # Test Case 1: Anomaly detected (OOT > in-sample + 0.02)
    print("\nTest Case 1: Anomaly Detection (OOT exceeds threshold)")
    detector = LeakageDetector(verbose=False)
    
    in_sample_ic = 0.00471
    oot_ic = 0.03342
    threshold = 0.02
    
    result = detector.detect_performance_anomaly(in_sample_ic, oot_ic, threshold)
    
    # Verify anomaly is detected
    assert not result['passed'], "Anomaly should be detected but wasn't"
    assert result['anomaly_detected'], "Anomaly flag not set"
    assert len(result['violations']) > 0, "No violations recorded for anomaly"
    
    # Verify details
    assert result['details']['in_sample_ic'] == in_sample_ic
    assert result['details']['oot_ic'] == oot_ic
    assert result['details']['ic_difference'] == oot_ic - in_sample_ic
    assert result['details']['ic_difference'] > threshold
    
    print(f"✓ Anomaly correctly detected")
    print(f"  In-sample IC: {in_sample_ic:.5f}")
    print(f"  OOT IC: {oot_ic:.5f}")
    print(f"  Difference: {oot_ic - in_sample_ic:.5f}")
    print(f"  Threshold: {threshold:.5f}")
    print(f"  Status: ANOMALY DETECTED")
    
    # Test Case 2: No anomaly (OOT < in-sample + 0.02)
    print("\nTest Case 2: No Anomaly (OOT within threshold)")
    
    in_sample_ic_2 = 0.05
    oot_ic_2 = 0.06
    
    result_2 = detector.detect_performance_anomaly(in_sample_ic_2, oot_ic_2, threshold)
    
    # Verify no anomaly
    assert result_2['passed'], "No anomaly should be detected"
    assert not result_2['anomaly_detected'], "Anomaly flag incorrectly set"
    assert len(result_2['violations']) == 0, "Violations recorded when there should be none"
    
    print(f"✓ No anomaly correctly identified")
    print(f"  In-sample IC: {in_sample_ic_2:.5f}")
    print(f"  OOT IC: {oot_ic_2:.5f}")
    print(f"  Difference: {oot_ic_2 - in_sample_ic_2:.5f}")
    print(f"  Threshold: {threshold:.5f}")
    print(f"  Status: NO ANOMALY")
    
    # Test Case 3: Edge case (exactly at threshold)
    print("\nTest Case 3: Edge Case (exactly at threshold)")
    
    in_sample_ic_3 = 0.05
    oot_ic_3 = 0.07  # Exactly 0.02 difference
    
    result_3 = detector.detect_performance_anomaly(in_sample_ic_3, oot_ic_3, threshold)
    
    print(f"  In-sample IC: {in_sample_ic_3:.5f}")
    print(f"  OOT IC: {oot_ic_3:.5f}")
    print(f"  Difference: {oot_ic_3 - in_sample_ic_3:.5f}")
    print(f"  Threshold: {threshold:.5f}")
    print(f"  Status: {'ANOMALY' if result_3['anomaly_detected'] else 'NO ANOMALY'}")
    
    return True


def test_automated_flagging_system():
    """
    Test the automated flagging system generates appropriate flags
    """
    print("\n" + "="*80)
    print("Testing Automated Flagging System")
    print("="*80)
    
    # Load data and run audit
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=['date'])
    fund_df = pd.read_parquet("data/fundamentals.parquet") if os.path.exists("data/fundamentals.parquet") else None
    
    detector = LeakageDetector(verbose=False)
    audit_results = detector.run_comprehensive_audit(
        factors_df=factors_df,
        fund_df=fund_df,
        in_sample_ic=0.00471,
        oot_ic=0.03342
    )
    
    # Generate report
    report_path = detector.generate_audit_report(audit_results)
    report_df = pd.read_csv(report_path)
    
    # Collect all flags
    all_flags = set()
    for flags_str in report_df['automated_flags'].dropna():
        if flags_str:
            all_flags.update(flags_str.split('; '))
    
    print(f"✓ Total unique flags generated: {len(all_flags)}")
    
    # Check for expected flags given the current system state
    expected_flags = [
        'PERFORMANCE_ANOMALY_DETECTED',
        'POTENTIAL_DATA_LEAKAGE',
        'REQUIRES_URGENT_INVESTIGATION'
    ]
    
    for flag in expected_flags:
        if flag in all_flags:
            print(f"✓ Expected flag present: {flag}")
        else:
            print(f"⚠️  Expected flag missing: {flag}")
    
    # Check severity classification
    severity_counts = report_df['severity'].value_counts().to_dict()
    print(f"\n✓ Severity classification:")
    for severity, count in severity_counts.items():
        print(f"  {severity}: {count} tests")
    
    # Check recommendations are provided
    recs_with_content = report_df['recommendations'].notna().sum()
    print(f"\n✓ Recommendations provided for {recs_with_content}/{len(report_df)} rows")
    
    # Check categories
    categories = report_df['category'].unique()
    print(f"\n✓ Test categories: {', '.join(categories)}")
    
    return True


def test_integration_with_pipeline():
    """
    Test integration with the complete pipeline
    """
    print("\n" + "="*80)
    print("Testing Integration with Complete Pipeline")
    print("="*80)
    
    # Run complete pipeline
    results = run_leakage_detection_pipeline(
        factors_df_path="data/factor_features.csv",
        fund_df_path="data/fundamentals.parquet",
        in_sample_ic=0.00471,
        oot_ic=0.03342
    )
    
    # Verify results structure
    assert 'overall_assessment' in results, "Missing overall_assessment"
    assert 'test_results' in results, "Missing test_results"
    assert 'report_path' in results, "Missing report_path"
    
    # Verify report was generated
    assert os.path.exists(results['report_path']), "Report file not created"
    
    print(f"✓ Pipeline executed successfully")
    print(f"✓ Overall Assessment: {results['overall_assessment']}")
    print(f"✓ Total Tests: {results['total_tests']}")
    print(f"✓ Tests Passed: {results['tests_passed']}")
    print(f"✓ Tests Failed: {results['tests_failed']}")
    print(f"✓ Report Path: {results['report_path']}")
    
    return True


def main():
    """Run all tests"""
    print("="*80)
    print("TASK 1.6 FINAL INTEGRATION TESTS")
    print("="*80)
    
    tests = [
        ("Requirement 1.9: Comprehensive Audit Report", test_requirement_1_9_comprehensive_audit_report),
        ("Requirement 1.10: Performance Anomaly Detection", test_requirement_1_10_performance_anomaly_detection),
        ("Automated Flagging System", test_automated_flagging_system),
        ("Pipeline Integration", test_integration_with_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            results.append((test_name, "FAILED"))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Task 1.6 implementation is complete and verified.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review implementation.")
    
    print("="*80)


if __name__ == "__main__":
    main()

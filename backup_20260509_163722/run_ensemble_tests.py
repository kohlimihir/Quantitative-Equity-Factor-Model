"""Simple test runner for ensemble model tests."""

import sys
import traceback

# Import all test functions
from test_ensemble_model import (
    test_ensemble_model_initialization,
    test_ensemble_predict,
    test_equal_weighting,
    test_grid_search_weights,
    test_optimize_weights,
    test_walk_forward_ensemble,
    test_temporal_integrity,
    test_evaluate_ensemble,
    test_ensemble_methods_comparison,
    test_weight_stability
)

def run_tests():
    """Run all tests and report results."""
    tests = [
        ("Ensemble Model Initialization", test_ensemble_model_initialization),
        ("Ensemble Predict", test_ensemble_predict),
        ("Equal Weighting", test_equal_weighting),
        ("Grid Search Weights", test_grid_search_weights),
        ("Optimize Weights", test_optimize_weights),
        ("Walk Forward Ensemble", test_walk_forward_ensemble),
        ("Temporal Integrity", test_temporal_integrity),
        ("Evaluate Ensemble", test_evaluate_ensemble),
        ("Ensemble Methods Comparison", test_ensemble_methods_comparison),
        ("Weight Stability", test_weight_stability)
    ]
    
    print("="*70)
    print("  ENSEMBLE MODEL TEST SUITE")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ FAILED: {name}")
            print(f"   Error: {str(e)}")
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"  TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

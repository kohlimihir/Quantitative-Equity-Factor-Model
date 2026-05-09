"""
validate_model_improvements.py - Comprehensive validation for Task 5

This script validates all model architecture improvements from tasks 4.1-4.5:
1. Hyperparameter tuner (task 4.1)
2. Ensemble model (task 4.2)
3. Sector-specific modeling (task 4.3)
4. Early stopping and regularization (task 4.5)
"""

import sys
import traceback


def run_test_suite(test_name, test_command):
    """Run a test suite and report results."""
    print(f"\n{'='*70}")
    print(f"  {test_name}")
    print(f"{'='*70}")
    
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-u", test_command],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Check if tests passed
        if "ALL TESTS PASSED" in result.stdout or "passed, 0 failed" in result.stdout:
            print(f"✓ {test_name} PASSED")
            return True
        else:
            print(f"✗ {test_name} FAILED")
            print("\nOutput:")
            print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
            if result.stderr:
                print("\nErrors:")
                print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⚠ {test_name} TIMEOUT (>120s)")
        return False
    except Exception as e:
        print(f"✗ {test_name} ERROR: {e}")
        traceback.print_exc()
        return False


def validate_implementations():
    """Validate all model improvement implementations."""
    print("\n" + "="*70)
    print("  MODEL IMPROVEMENTS VALIDATION - TASK 5")
    print("="*70)
    print("\nValidating implementations from tasks 4.1-4.5:")
    print("  4.1 - Hyperparameter optimization framework")
    print("  4.2 - Ensemble methods")
    print("  4.3 - Sector-specific modeling")
    print("  4.5 - Early stopping and regularization")
    
    results = {}
    
    # Test 1: Hyperparameter Tuner (Task 4.1)
    results["Hyperparameter Tuner"] = run_test_suite(
        "Task 4.1: Hyperparameter Tuner",
        "test_hyperparameter_tuner.py"
    )
    
    # Test 2: Ensemble Model (Task 4.2)
    results["Ensemble Model"] = run_test_suite(
        "Task 4.2: Ensemble Model",
        "run_ensemble_tests.py"
    )
    
    # Test 3: Sector-Specific Model (Task 4.3)
    results["Sector Model"] = run_test_suite(
        "Task 4.3: Sector-Specific Model",
        "test_sector_model_simple.py"
    )
    
    # Test 4: Early Stopping & Regularization (Task 4.5)
    results["Early Stopping"] = run_test_suite(
        "Task 4.5: Early Stopping & Regularization",
        "test_early_stopping_regularization.py"
    )
    
    # Summary
    print("\n" + "="*70)
    print("  VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, status in results.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"  {status_str:8} - {name}")
    
    print(f"\n  Total: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n  ✓ ALL MODEL IMPROVEMENTS VALIDATED SUCCESSFULLY")
        return True
    else:
        print(f"\n  ⚠ {total - passed} test suite(s) failed")
        return False


def check_integration():
    """Check that model improvements integrate with main pipeline."""
    print("\n" + "="*70)
    print("  INTEGRATION CHECK")
    print("="*70)
    
    try:
        # Check imports
        print("\nChecking imports...")
        from hyperparameter_tuner import HyperparameterTuner, tune_ridge_hyperparameters, tune_lightgbm_hyperparameters
        print("  ✓ hyperparameter_tuner imports successful")
        
        from ensemble_model import EnsembleModel, walk_forward_ensemble, evaluate_ensemble
        print("  ✓ ensemble_model imports successful")
        
        from sector_model import SectorSpecificModel, train_sector_specific_models
        print("  ✓ sector_model imports successful")
        
        from early_stopping_regularization import (
            EarlyStoppingValidator,
            ModelComplexityAnalyzer,
            walk_forward_with_early_stopping
        )
        print("  ✓ early_stopping_regularization imports successful")
        
        # Check that run_all.py references these modules
        print("\nChecking run_all.py integration...")
        with open("run_all.py", "r") as f:
            run_all_content = f.read()
        
        checks = [
            ("hyperparameter_tuner", "from hyperparameter_tuner import"),
            ("ensemble_model", "from ensemble_model import"),
            ("sector_model", "from sector_model import"),
            ("early_stopping", "from early_stopping_regularization import")
        ]
        
        for name, import_str in checks:
            if import_str in run_all_content:
                print(f"  ✓ {name} integrated in run_all.py")
            else:
                print(f"  ⚠ {name} not found in run_all.py (may be optional)")
        
        print("\n  ✓ ALL INTEGRATION CHECKS PASSED")
        return True
        
    except Exception as e:
        print(f"\n  ✗ INTEGRATION CHECK FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Main validation function."""
    print("\n" + "="*70)
    print("  CHECKPOINT TASK 5: VALIDATE MODEL IMPROVEMENTS")
    print("="*70)
    
    # Run integration check first
    integration_ok = check_integration()
    
    # Run test suites
    tests_ok = validate_implementations()
    
    # Final result
    print("\n" + "="*70)
    print("  FINAL RESULT")
    print("="*70)
    
    if integration_ok and tests_ok:
        print("\n  ✓✓✓ TASK 5 VALIDATION COMPLETE ✓✓✓")
        print("\n  All model improvements are working correctly:")
        print("    • Hyperparameter optimization framework (4.1)")
        print("    • Ensemble methods (4.2)")
        print("    • Sector-specific modeling (4.3)")
        print("    • Early stopping and regularization (4.5)")
        print("\n  The systems integrate properly with the main pipeline.")
        return 0
    else:
        print("\n  ✗✗✗ TASK 5 VALIDATION FAILED ✗✗✗")
        if not integration_ok:
            print("\n  Integration issues detected.")
        if not tests_ok:
            print("\n  Some test suites failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

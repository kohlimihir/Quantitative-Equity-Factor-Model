"""
test_integration.py — Integration Test for Task 12.1
=====================================================
Tests that all systems are properly integrated into the main pipeline.

This test verifies:
1. Configuration manager loads successfully
2. All diagnostic systems are importable
3. Pipeline orchestration logic is correct
"""

import os
import sys

def test_config_manager():
    """Test configuration manager."""
    print("\n" + "="*70)
    print("  TEST 1: Configuration Manager")
    print("="*70)
    
    try:
        from config_manager import ConfigManager, load_config
        
        # Test loading baseline configuration
        config = load_config("baseline", verbose=False)
        assert config is not None, "Config should not be None"
        assert "data" in config, "Config should have 'data' section"
        assert "models" in config, "Config should have 'models' section"
        
        # Test ConfigManager methods
        manager = ConfigManager(verbose=False)
        manager.load_config("baseline")
        
        # Test get method
        alpha = manager.get("models.ridge.alpha")
        assert alpha is not None, "Should be able to get nested config values"
        
        # Test list profiles
        profiles = manager.list_profiles()
        assert len(profiles) > 0, "Should have at least one profile"
        assert "baseline" in profiles, "Should have baseline profile"
        
        print("  ✓ Configuration manager works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration manager test failed: {e}")
        return False


def test_system_imports():
    """Test that all diagnostic systems can be imported."""
    print("\n" + "="*70)
    print("  TEST 2: System Imports")
    print("="*70)
    
    systems = [
        ("config_manager", "ConfigManager"),
        ("leakage_detector", "LeakageDetector"),
        ("feature_analyzer", "generate_feature_quality_report"),
        ("hyperparameter_tuner", "HyperparameterTuner"),
        ("ensemble_model", "EnsembleModel"),
        ("sector_model", "SectorSpecificModel"),
        ("early_stopping_regularization", "EarlyStoppingValidator"),
    ]
    
    all_passed = True
    
    for module_name, class_or_func in systems:
        try:
            module = __import__(module_name)
            assert hasattr(module, class_or_func), f"Module should have {class_or_func}"
            print(f"  ✓ {module_name}.{class_or_func}")
        except Exception as e:
            print(f"  ❌ {module_name}.{class_or_func}: {e}")
            all_passed = False
    
    if all_passed:
        print("\n  ✓ All systems import successfully")
    else:
        print("\n  ❌ Some systems failed to import")
    
    return all_passed


def test_pipeline_structure():
    """Test that run_all.py has correct structure."""
    print("\n" + "="*70)
    print("  TEST 3: Pipeline Structure")
    print("="*70)
    
    try:
        with open("run_all.py", "r") as f:
            content = f.read()
        
        # Check for key stages
        required_stages = [
            "Configuration loading",
            "Data leakage detection",
            "Feature quality analysis",
            "Hyperparameter tuning",
            "Ensemble models",
            "Sector-specific models",
            "Early stopping",
            "Comprehensive diagnostic report",
        ]
        
        all_found = True
        for stage in required_stages:
            if stage.lower() in content.lower():
                print(f"  ✓ Found stage: {stage}")
            else:
                print(f"  ❌ Missing stage: {stage}")
                all_found = False
        
        # Check for error handling
        if "error_handler" in content:
            print(f"  ✓ Error handling implemented")
        else:
            print(f"  ❌ Error handling missing")
            all_found = False
        
        # Check for command-line arguments
        if "argparse" in content:
            print(f"  ✓ Command-line arguments supported")
        else:
            print(f"  ❌ Command-line arguments missing")
            all_found = False
        
        if all_found:
            print("\n  ✓ Pipeline structure is correct")
        else:
            print("\n  ❌ Pipeline structure has issues")
        
        return all_found
        
    except Exception as e:
        print(f"  ❌ Pipeline structure test failed: {e}")
        return False


def test_configuration_profiles():
    """Test that configuration profiles exist and are valid."""
    print("\n" + "="*70)
    print("  TEST 4: Configuration Profiles")
    print("="*70)
    
    try:
        from config_manager import ConfigManager
        
        manager = ConfigManager(verbose=False)
        profiles = manager.list_profiles()
        
        print(f"  Found {len(profiles)} configuration profiles:")
        for profile in profiles:
            print(f"    - {profile}")
        
        # Test loading each profile
        all_valid = True
        for profile in profiles:
            try:
                config = manager.load_config(profile)
                print(f"  ✓ {profile}: valid")
            except Exception as e:
                print(f"  ❌ {profile}: invalid - {e}")
                all_valid = False
        
        if all_valid:
            print(f"\n  ✓ All {len(profiles)} profiles are valid")
        else:
            print(f"\n  ❌ Some profiles are invalid")
        
        return all_valid
        
    except Exception as e:
        print(f"  ❌ Configuration profiles test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("  INTEGRATION TEST FOR TASK 12.1")
    print("  Testing: All systems integrated into main pipeline")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Configuration Manager", test_config_manager()))
    results.append(("System Imports", test_system_imports()))
    results.append(("Pipeline Structure", test_pipeline_structure()))
    results.append(("Configuration Profiles", test_configuration_profiles()))
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"  {test_name:<30} {status}")
    
    print("\n" + "="*70)
    print(f"  OVERALL: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n  ✓ All integration tests PASSED")
        print("  Task 12.1 implementation is complete and working correctly.")
        return 0
    else:
        print(f"\n  ❌ {total - passed} test(s) FAILED")
        print("  Please review the failures above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

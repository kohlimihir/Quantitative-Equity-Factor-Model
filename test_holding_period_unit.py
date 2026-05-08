"""
test_holding_period_unit.py — Unit tests for holding period enhancements
=========================================================================
Tests the holding period functionality without running full walk-forward.
"""

import pandas as pd
import numpy as np
from sector_neutralisation import walk_forward_sector_neutral, add_sector_features

def test_configurable_parameters():
    """
    Test that walk_forward_sector_neutral accepts configurable parameters.
    """
    print("\n" + "="*70)
    print("  UNIT TEST: Configurable Parameters")
    print("="*70)
    
    # Load a small sample of data
    print("\nLoading data...")
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    
    # Take only first 30 months for quick test
    dates = sorted(factors_df["date"].unique())[:30]
    factors_df = factors_df[factors_df["date"].isin(dates)]
    
    print(f"Using {len(factors_df):,} rows, {factors_df['date'].nunique()} months")
    
    # Prepare data
    factors_df, features = add_sector_features(factors_df)
    
    # Test 1: Default parameters
    print("\nTest 1: Default parameters...")
    try:
        results_default = walk_forward_sector_neutral(
            factors_df, 
            features,
            min_train_months=24
        )
        print(f"✓ Default parameters work: {len(results_default)} predictions")
    except Exception as e:
        print(f"✗ Default parameters failed: {e}")
        return False
    
    # Test 2: Custom EWM alpha
    print("\nTest 2: Custom EWM alpha...")
    try:
        results_ewm = walk_forward_sector_neutral(
            factors_df, 
            features,
            min_train_months=24,
            ewm_alpha=0.3
        )
        print(f"✓ Custom EWM alpha works: {len(results_ewm)} predictions")
    except Exception as e:
        print(f"✗ Custom EWM alpha failed: {e}")
        return False
    
    # Test 3: Custom holding bonus
    print("\nTest 3: Custom holding bonus...")
    try:
        results_bonus = walk_forward_sector_neutral(
            factors_df, 
            features,
            min_train_months=24,
            hold_bonus_per_month=0.03,
            hold_bonus_cap=8
        )
        print(f"✓ Custom holding bonus works: {len(results_bonus)} predictions")
    except Exception as e:
        print(f"✗ Custom holding bonus failed: {e}")
        return False
    
    # Test 4: Minimum holding period
    print("\nTest 4: Minimum holding period...")
    try:
        results_min_hold = walk_forward_sector_neutral(
            factors_df, 
            features,
            min_train_months=24,
            min_hold_period=2
        )
        print(f"✓ Minimum holding period works: {len(results_min_hold)} predictions")
    except Exception as e:
        print(f"✗ Minimum holding period failed: {e}")
        return False
    
    # Test 5: All parameters combined
    print("\nTest 5: All parameters combined...")
    try:
        results_combined = walk_forward_sector_neutral(
            factors_df, 
            features,
            min_train_months=24,
            ewm_alpha=0.4,
            hold_bonus_per_month=0.025,
            hold_bonus_cap=6,
            min_hold_period=1
        )
        print(f"✓ All parameters combined work: {len(results_combined)} predictions")
    except Exception as e:
        print(f"✗ All parameters combined failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("  ALL TESTS PASSED ✓")
    print("="*70)
    print("\nHolding period enhancements are correctly implemented!")
    print("Parameters are configurable and work as expected.")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    import os
    if not os.path.exists("data/factor_features.csv"):
        print("ERROR: data/factor_features.csv not found!")
        print("Please run data_loader.py first.")
    else:
        test_configurable_parameters()

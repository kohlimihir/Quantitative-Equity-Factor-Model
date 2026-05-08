"""
test_regime_detection.py — Test script for regime detection functionality
==========================================================================
Verifies that the regime detection system works correctly.
"""

import pandas as pd
import numpy as np
from stability_monitor import StabilityMonitor


def test_regime_detection():
    """Test regime detection functionality."""
    print("\n" + "="*70)
    print("   TESTING REGIME DETECTION SYSTEM")
    print("="*70)
    
    # Load predictions
    predictions_df = pd.read_csv("data/sector_predictions.csv", parse_dates=["date"])
    print(f"\nLoaded {len(predictions_df):,} predictions")
    
    # Create monitor
    monitor = StabilityMonitor()
    
    # Test 1: Detect market regimes
    print("\n--- Test 1: Detect Market Regimes ---")
    regime_df = monitor.detect_market_regimes(
        predictions_df=predictions_df
    )
    
    assert len(regime_df) > 0, "Regime detection should return data"
    assert "regime" in regime_df.columns, "Should have regime column"
    assert "volatility" in regime_df.columns, "Should have volatility column"
    assert "trend_strength" in regime_df.columns, "Should have trend_strength column"
    assert "mean_reversion_score" in regime_df.columns, "Should have mean_reversion_score column"
    
    # Check regime types
    regimes = regime_df["regime"].unique()
    print(f"  ✓ Found {len(regimes)} regime types: {sorted(regimes)}")
    
    expected_regimes = {"HIGH_VOLATILITY", "TRENDING", "MEAN_REVERTING"}
    assert set(regimes).issubset(expected_regimes), f"Unexpected regimes: {regimes}"
    
    # Test 2: Analyze performance by regime
    print("\n--- Test 2: Analyze Performance by Regime ---")
    regime_performance = monitor.analyze_performance_by_regime(
        predictions_df,
        regime_df=regime_df
    )
    
    assert len(regime_performance) > 0, "Should have regime performance data"
    
    for regime, perf in regime_performance.items():
        assert "mean" in perf, f"Regime {regime} should have mean IC"
        assert "std" in perf, f"Regime {regime} should have std IC"
        assert "win_rate" in perf, f"Regime {regime} should have win rate"
        assert "n_months" in perf, f"Regime {regime} should have n_months"
        print(f"  ✓ {regime}: Mean IC = {perf['mean']:+.5f}, Win Rate = {perf['win_rate']:.1%}")
    
    # Test 3: Analyze regime transitions
    print("\n--- Test 3: Analyze Regime Transitions ---")
    transitions_df = monitor.analyze_regime_transitions(
        predictions_df,
        regime_df=regime_df
    )
    
    if len(transitions_df) > 0:
        assert "transition_type" in transitions_df.columns, "Should have transition_type column"
        assert "mean_ic" in transitions_df.columns, "Should have mean_ic column"
        print(f"  ✓ Found {len(transitions_df)} transition types")
        print(f"  ✓ Transition types: {transitions_df['transition_type'].tolist()}")
    else:
        print("  ⚠ No transition data (may be insufficient data)")
    
    # Test 4: Full stability report with regime detection
    print("\n--- Test 4: Full Stability Report with Regime Detection ---")
    report = monitor.generate_stability_report(
        predictions_df,
        enable_regime_detection=True,
        save_path=None  # Don't save during test
    )
    
    assert "regime_data" in report, "Report should include regime data"
    assert "regime_performance" in report, "Report should include regime performance"
    assert "regime_transitions" in report, "Report should include regime transitions"
    
    print(f"  ✓ Report includes regime data: {len(report['regime_data'])} months")
    print(f"  ✓ Report includes regime performance: {len(report['regime_performance'])} regimes")
    
    # Test 5: Verify regime detection can be disabled
    print("\n--- Test 5: Verify Regime Detection Can Be Disabled ---")
    monitor2 = StabilityMonitor()
    report_no_regime = monitor2.generate_stability_report(
        predictions_df,
        enable_regime_detection=False,
        save_path=None
    )
    
    assert len(report_no_regime.get("regime_data", [])) == 0, "Should have no regime data when disabled"
    print("  ✓ Regime detection can be disabled")
    
    print("\n" + "="*70)
    print("   ALL TESTS PASSED!")
    print("="*70)
    
    return True


def test_regime_characteristics():
    """Test that regime characteristics are computed correctly."""
    print("\n" + "="*70)
    print("   TESTING REGIME CHARACTERISTICS")
    print("="*70)
    
    # Load predictions
    predictions_df = pd.read_csv("data/sector_predictions.csv", parse_dates=["date"])
    
    # Create monitor and detect regimes
    monitor = StabilityMonitor()
    regime_df = monitor.detect_market_regimes(predictions_df=predictions_df)
    
    # Verify regime characteristics
    print("\n--- Regime Characteristics ---")
    
    # Check volatility values are reasonable
    vol_values = regime_df["volatility"].values
    assert np.all(vol_values >= 0), "Volatility should be non-negative"
    assert np.all(vol_values < 2.0), "Volatility should be reasonable (< 200%)"
    print(f"  ✓ Volatility range: {vol_values.min():.4f} to {vol_values.max():.4f}")
    
    # Check trend strength values
    trend_values = regime_df["trend_strength"].values
    assert np.all(trend_values >= 0), "Trend strength should be non-negative"
    print(f"  ✓ Trend strength range: {trend_values.min():.4f} to {trend_values.max():.4f}")
    
    # Check mean reversion scores
    mr_values = regime_df["mean_reversion_score"].values
    print(f"  ✓ Mean reversion score range: {mr_values.min():.4f} to {mr_values.max():.4f}")
    
    # Verify regime distribution is reasonable
    regime_counts = regime_df["regime"].value_counts()
    print(f"\n--- Regime Distribution ---")
    for regime, count in regime_counts.items():
        pct = count / len(regime_df) * 100
        print(f"  {regime}: {count} months ({pct:.1f}%)")
        assert count > 0, f"Regime {regime} should have at least one month"
    
    print("\n" + "="*70)
    print("   REGIME CHARACTERISTICS TESTS PASSED!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    try:
        # Run tests
        test_regime_detection()
        test_regime_characteristics()
        
        print("\n✓ All regime detection tests passed successfully!")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise

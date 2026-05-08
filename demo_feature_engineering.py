"""
demo_feature_engineering.py — Demonstration of Feature Engineering Enhancements
================================================================================
Shows how to use the new feature engineering capabilities added in Task 3.3.

This script demonstrates:
1. Interaction features between factor groups
2. Non-linear transformations (log, sqrt, rank)
3. Sector-relative feature transformations

Requirements: 2.5, 2.6, 2.9
"""

import pandas as pd
import numpy as np
from data_loader import (
    apply_feature_engineering,
    FEATURE_ENGINEERING_CONFIG,
    FEATURES,
    TARGET
)


def demo_basic_usage():
    """Demonstrate basic usage with synthetic data."""
    print("\n" + "="*70)
    print("  DEMO 1: Basic Usage with Synthetic Data")
    print("="*70)
    
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=12, freq="ME")
    tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "BAC", "GS"]
    sectors = ["Technology", "Technology", "Technology", "Financials", "Financials", "Financials"]
    
    data = []
    for date in dates:
        for ticker, sector in zip(tickers, sectors):
            data.append({
                "date": date,
                "ticker": ticker,
                "sector": sector,
                "Mom_12_1": np.random.uniform(-0.2, 0.3),
                "Mom_6_1": np.random.uniform(-0.15, 0.25),
                "Mom_1": np.random.uniform(-0.1, 0.15),
                "Vol_12": np.random.uniform(0.1, 0.4),
                "IdioVol": np.random.uniform(0.05, 0.3),
                "Beta_12": np.random.uniform(0.5, 1.5),
                "High52W": np.random.uniform(0.7, 1.0),
                "Trend_MA": np.random.uniform(0.9, 1.1),
                "MaxRet_1M": np.random.uniform(0.01, 0.05),
                "PB_ratio": np.random.uniform(1.0, 5.0),
                "PE_TTM": np.random.uniform(10.0, 30.0),
                "EV_EBITDA": np.random.uniform(8.0, 20.0),
                "ROE": np.random.uniform(0.05, 0.25),
                "GrossMargin": np.random.uniform(0.3, 0.7),
                "CashFlowYield": np.random.uniform(0.02, 0.08),
                "RevGrowth_YoY": np.random.uniform(-0.1, 0.2),
                "EarnGrowth_YoY": np.random.uniform(-0.15, 0.25),
                "LogMktCap": np.random.uniform(20.0, 25.0),
                "VolRatio": np.random.uniform(0.8, 1.2),
                TARGET: np.random.uniform(-0.05, 0.05),
            })
    
    df = pd.DataFrame(data)
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Original features: {len([c for c in df.columns if c not in ['date', 'ticker', 'sector', TARGET]])}")
    
    # Apply feature engineering with default config (all disabled)
    result = apply_feature_engineering(df.copy())
    
    print(f"\nResult shape: {result.shape}")
    print(f"Total features: {len([c for c in result.columns if c not in ['date', 'ticker', 'sector', TARGET]])}")


def demo_interaction_features():
    """Demonstrate interaction features."""
    print("\n" + "="*70)
    print("  DEMO 2: Interaction Features")
    print("="*70)
    
    # Create synthetic data
    np.random.seed(42)
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=6, freq="ME").repeat(3),
        "ticker": ["AAPL", "MSFT", "GOOGL"] * 6,
        "sector": ["Technology"] * 18,
        "Mom_12_1": np.random.uniform(-0.2, 0.3, 18),
        "Mom_6_1": np.random.uniform(-0.15, 0.25, 18),
        "Mom_1": np.random.uniform(-0.1, 0.15, 18),
        "Vol_12": np.random.uniform(0.1, 0.4, 18),
        "PB_ratio": np.random.uniform(1.0, 5.0, 18),
        "PE_TTM": np.random.uniform(10.0, 30.0, 18),
        "LogMktCap": np.random.uniform(20.0, 25.0, 18),
        TARGET: np.random.uniform(-0.05, 0.05, 18),
    })
    
    # Enable only interaction features
    config = {
        "enable_interactions": True,
        "enable_nonlinear": False,
        "enable_sector_relative": False,
        "interaction_pairs": [
            ("G1 Momentum", "G4 Value"),  # Momentum × Value
            ("G2 Risk", "G7 Size"),       # Risk × Size
        ],
    }
    
    result = apply_feature_engineering(df.copy(), config)
    
    # Show sample interaction features
    print("\nSample interaction features created:")
    interaction_cols = [c for c in result.columns if "_x_" in c]
    for col in interaction_cols[:5]:
        print(f"  - {col}")
    
    print(f"\nTotal interaction features: {len(interaction_cols)}")
    
    # Show example values
    print("\nExample values for AAPL on first date:")
    sample = result[(result["ticker"] == "AAPL") & (result["date"] == result["date"].min())].iloc[0]
    print(f"  Mom_12_1: {sample['Mom_12_1']:.4f}")
    print(f"  PB_ratio: {sample['PB_ratio']:.4f}")
    print(f"  Mom_12_1_x_PB_ratio: {sample['Mom_12_1_x_PB_ratio']:.4f}")
    print(f"  (Product: {sample['Mom_12_1'] * sample['PB_ratio']:.4f})")


def demo_nonlinear_transformations():
    """Demonstrate non-linear transformations."""
    print("\n" + "="*70)
    print("  DEMO 3: Non-Linear Transformations")
    print("="*70)
    
    # Create synthetic data with specific values to show transformations
    df = pd.DataFrame({
        "date": ["2020-01-31"] * 5,
        "ticker": ["A", "B", "C", "D", "E"],
        "sector": ["Tech"] * 5,
        "Mom_12_1": [0.25, -0.25, 0.0, 0.5, -0.5],
        "Vol_12": [0.1, 0.2, 0.3, 0.4, 0.5],
        "LogMktCap": [20.0, 21.0, 22.0, 23.0, 24.0],
        TARGET: [0.01] * 5,
    })
    
    # Enable only non-linear transformations
    config = {
        "enable_interactions": False,
        "enable_nonlinear": True,
        "enable_sector_relative": False,
        "nonlinear_features": ["Mom_12_1", "Vol_12"],
    }
    
    result = apply_feature_engineering(df.copy(), config)
    
    # Show transformations
    print("\nNon-linear transformations for Mom_12_1:")
    print(f"{'Ticker':<8} {'Original':<12} {'Log':<12} {'Sqrt':<12} {'Rank':<12}")
    print("-" * 60)
    for _, row in result.iterrows():
        print(f"{row['ticker']:<8} "
              f"{row['Mom_12_1']:>11.4f} "
              f"{row['Mom_12_1_log']:>11.4f} "
              f"{row['Mom_12_1_sqrt']:>11.4f} "
              f"{row['Mom_12_1_rank']:>11.4f}")
    
    print("\nNon-linear transformations for Vol_12:")
    print(f"{'Ticker':<8} {'Original':<12} {'Log':<12} {'Sqrt':<12} {'Rank':<12}")
    print("-" * 60)
    for _, row in result.iterrows():
        print(f"{row['ticker']:<8} "
              f"{row['Vol_12']:>11.4f} "
              f"{row['Vol_12_log']:>11.4f} "
              f"{row['Vol_12_sqrt']:>11.4f} "
              f"{row['Vol_12_rank']:>11.4f}")


def demo_sector_relative_features():
    """Demonstrate sector-relative features."""
    print("\n" + "="*70)
    print("  DEMO 4: Sector-Relative Features")
    print("="*70)
    
    # Create data with clear sector differences
    df = pd.DataFrame({
        "date": ["2020-01-31"] * 6,
        "ticker": ["AAPL", "MSFT", "GOOGL", "JPM", "BAC", "GS"],
        "sector": ["Technology", "Technology", "Technology", "Financials", "Financials", "Financials"],
        "Mom_12_1": [0.10, 0.15, 0.20, 0.30, 0.35, 0.40],  # Financials have higher momentum
        "Vol_12": [0.20, 0.25, 0.30, 0.15, 0.18, 0.22],    # Tech has higher volatility
        "ROE": [0.15, 0.18, 0.20, 0.10, 0.12, 0.14],       # Tech has higher ROE
        TARGET: [0.01] * 6,
    })
    
    # Enable only sector-relative features
    config = {
        "enable_interactions": False,
        "enable_nonlinear": False,
        "enable_sector_relative": True,
        "sector_relative_features": ["Mom_12_1", "Vol_12", "ROE"],
    }
    
    result = apply_feature_engineering(df.copy(), config)
    
    # Show sector-relative transformations
    print("\nSector-relative transformations for Mom_12_1:")
    print(f"{'Ticker':<8} {'Sector':<12} {'Original':<12} {'Z-Score':<12} {'Percentile':<12}")
    print("-" * 70)
    for _, row in result.iterrows():
        print(f"{row['ticker']:<8} "
              f"{row['sector']:<12} "
              f"{row['Mom_12_1']:>11.4f} "
              f"{row['Mom_12_1_sector_z']:>11.4f} "
              f"{row['Mom_12_1_sector_pct']:>11.4f}")
    
    print("\nNote: Within each sector, z-scores have mean ~0 and percentiles range 0-1")
    print("This creates sector-neutral signals that capture relative strength within sectors")


def demo_all_features_enabled():
    """Demonstrate all features enabled together."""
    print("\n" + "="*70)
    print("  DEMO 5: All Features Enabled")
    print("="*70)
    
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=6, freq="ME")
    tickers = ["AAPL", "MSFT", "JPM", "BAC"]
    sectors = ["Technology", "Technology", "Financials", "Financials"]
    
    data = []
    for date in dates:
        for ticker, sector in zip(tickers, sectors):
            data.append({
                "date": date,
                "ticker": ticker,
                "sector": sector,
                "Mom_12_1": np.random.uniform(-0.2, 0.3),
                "Mom_6_1": np.random.uniform(-0.15, 0.25),
                "Mom_1": np.random.uniform(-0.1, 0.15),
                "Vol_12": np.random.uniform(0.1, 0.4),
                "IdioVol": np.random.uniform(0.05, 0.3),
                "Beta_12": np.random.uniform(0.5, 1.5),
                "PB_ratio": np.random.uniform(1.0, 5.0),
                "PE_TTM": np.random.uniform(10.0, 30.0),
                "ROE": np.random.uniform(0.05, 0.25),
                "LogMktCap": np.random.uniform(20.0, 25.0),
                TARGET: np.random.uniform(-0.05, 0.05),
            })
    
    df = pd.DataFrame(data)
    
    # Enable all features
    config = {
        "enable_interactions": True,
        "enable_nonlinear": True,
        "enable_sector_relative": True,
        "interaction_pairs": [("G1 Momentum", "G4 Value")],
        "nonlinear_features": ["Mom_12_1", "Vol_12"],
        "sector_relative_features": ["Mom_12_1", "ROE"],
    }
    
    result = apply_feature_engineering(df.copy(), config)
    
    # Categorize new features
    all_features = [c for c in result.columns if c not in ["date", "ticker", "sector", TARGET]]
    interaction_features = [c for c in all_features if "_x_" in c]
    nonlinear_features = [c for c in all_features if any(x in c for x in ["_log", "_sqrt", "_rank"])]
    sector_features = [c for c in all_features if "_sector_" in c]
    base_features = [c for c in all_features if c not in interaction_features + nonlinear_features + sector_features]
    
    print(f"\nFeature breakdown:")
    print(f"  Base features: {len(base_features)}")
    print(f"  Interaction features: {len(interaction_features)}")
    print(f"  Non-linear features: {len(nonlinear_features)}")
    print(f"  Sector-relative features: {len(sector_features)}")
    print(f"  Total features: {len(all_features)}")
    
    print(f"\nSample interaction features:")
    for feat in interaction_features[:3]:
        print(f"  - {feat}")
    
    print(f"\nSample non-linear features:")
    for feat in nonlinear_features[:3]:
        print(f"  - {feat}")
    
    print(f"\nSample sector-relative features:")
    for feat in sector_features[:3]:
        print(f"  - {feat}")


def demo_usage_with_real_data():
    """Show how to use with real data pipeline."""
    print("\n" + "="*70)
    print("  DEMO 6: Usage with Real Data Pipeline")
    print("="*70)
    
    print("\nTo use feature engineering in the real pipeline:")
    print("\n1. Edit FEATURE_ENGINEERING_CONFIG in data_loader.py:")
    print("   FEATURE_ENGINEERING_CONFIG = {")
    print("       'enable_interactions': True,")
    print("       'enable_nonlinear': True,")
    print("       'enable_sector_relative': True,")
    print("       'interaction_pairs': [")
    print("           ('G1 Momentum', 'G4 Value'),")
    print("           ('G2 Risk', 'G7 Size'),")
    print("       ],")
    print("       'nonlinear_features': ['Mom_12_1', 'Vol_12', 'PB_ratio'],")
    print("       'sector_relative_features': ['Mom_12_1', 'ROE', 'LogMktCap'],")
    print("   }")
    print("\n2. Run data_loader.py to generate features:")
    print("   python data_loader.py")
    print("\n3. The enhanced features will be saved to data/factor_features.csv")
    print("\n4. Use feature_analyzer.py to evaluate new features:")
    print("   python feature_analyzer.py")
    print("\n5. Train models with the enhanced feature set:")
    print("   python model.py")
    
    print("\nBenefits:")
    print("  • Interaction features capture non-linear relationships")
    print("  • Non-linear transforms improve signal quality")
    print("  • Sector-relative features create sector-neutral signals")
    print("  • All transformations maintain temporal integrity (no leakage)")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  FEATURE ENGINEERING ENHANCEMENTS DEMONSTRATION")
    print("  Task 3.3: Interaction, Non-Linear, and Sector-Relative Features")
    print("="*70)
    
    demo_basic_usage()
    demo_interaction_features()
    demo_nonlinear_transformations()
    demo_sector_relative_features()
    demo_all_features_enabled()
    demo_usage_with_real_data()
    
    print("\n" + "="*70)
    print("  DEMONSTRATION COMPLETE")
    print("="*70)

"""
demo_sector_model.py — Demonstration of Sector-Specific Modeling
=================================================================
Demonstrates the sector-specific modeling capability including:
1. Training separate models per sector
2. Sector-specific hyperparameter tuning
3. Performance comparison with global model
4. Sector performance analysis

**Validates: Requirements 3.4**
"""

import pandas as pd
import numpy as np
import os

from sector_model import train_sector_specific_models, SectorSpecificModel


def main():
    """Run sector-specific modeling demonstration."""
    
    print("\n" + "="*70)
    print("  SECTOR-SPECIFIC MODELING DEMONSTRATION")
    print("="*70)
    
    # Check if factor data exists
    if not os.path.exists("data/factor_features.csv"):
        print("\n⚠  Warning: data/factor_features.csv not found")
        print("  Please run data_loader.py first to generate factor data")
        return
    
    # Load factor data
    print("\nLoading factor data...")
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    print(f"  {len(factors_df):,} rows")
    print(f"  {factors_df['date'].nunique()} months")
    print(f"  {factors_df['ticker'].nunique()} stocks")
    print(f"  {factors_df['sector'].nunique()} sectors")
    
    # Display sector distribution
    print("\n  Sector distribution:")
    sector_counts = factors_df.groupby("sector")["ticker"].nunique()
    for sector, count in sector_counts.items():
        print(f"    {sector:<14}: {count:>3} stocks")
    
    # Train sector-specific models
    print("\n" + "="*70)
    print("  TRAINING SECTOR-SPECIFIC MODELS")
    print("="*70)
    print("\n  Configuration:")
    print("    Model type: LightGBM")
    print("    Hyperparameter tuning: Enabled")
    print("    Min training months: 24")
    
    sector_results, sector_model, sector_performance = train_sector_specific_models(
        factors_df,
        model_type="lightgbm",
        tune_hyperparameters=True,
        min_train_months=24
    )
    
    # Save sector-specific predictions
    os.makedirs("data", exist_ok=True)
    sector_results.to_csv("data/sector_specific_predictions.csv", index=False)
    print(f"\n✓ Saved: data/sector_specific_predictions.csv")
    
    # Display sector-specific hyperparameters
    print("\n" + "="*70)
    print("  SECTOR-SPECIFIC HYPERPARAMETERS")
    print("="*70)
    
    for sector in sorted(sector_model.sector_params.keys()):
        params = sector_model.sector_params[sector]
        print(f"\n  {sector}:")
        print(f"    Learning rate: {params.get('learning_rate', 'N/A')}")
        print(f"    Num leaves: {params.get('num_leaves', 'N/A')}")
        print(f"    Max depth: {params.get('max_depth', 'N/A')}")
        print(f"    Min data in leaf: {params.get('min_data_in_leaf', 'N/A')}")
        print(f"    Lambda L1: {params.get('lambda_l1', 'N/A')}")
        print(f"    Lambda L2: {params.get('lambda_l2', 'N/A')}")
    
    # Compare with global model if available
    if os.path.exists("data/sector_predictions.csv"):
        print("\n" + "="*70)
        print("  COMPARING WITH GLOBAL MODEL")
        print("="*70)
        
        global_results = pd.read_csv("data/sector_predictions.csv", parse_dates=["date"])
        
        comparison_df = sector_model.compare_with_global_model(
            sector_results,
            global_results
        )
        
        # Save comparison
        os.makedirs("reports", exist_ok=True)
        comparison_df.to_csv("reports/sector_model_comparison.csv", index=False)
        print(f"\n✓ Saved: reports/sector_model_comparison.csv")
        
        # Display key insights
        print("\n" + "="*70)
        print("  KEY INSIGHTS")
        print("="*70)
        
        overall = comparison_df[comparison_df["Sector"] == "Overall"].iloc[0]
        print(f"\n  Overall Performance:")
        print(f"    Sector-specific IC: {overall['Sector_Mean_IC']:+.5f}")
        print(f"    Global model IC: {overall['Global_Mean_IC']:+.5f}")
        print(f"    IC improvement: {overall['IC_Improvement']:+.5f}")
        print(f"    Sector-specific IR: {overall['Sector_IC_IR']:+.5f}")
        print(f"    Global model IR: {overall['Global_IC_IR']:+.5f}")
        print(f"    IR improvement: {overall['IR_Improvement']:+.5f}")
        
        # Find best and worst performing sectors
        sector_rows = comparison_df[comparison_df["Sector"] != "Overall"]
        best_sector = sector_rows.loc[sector_rows["IC_Improvement"].idxmax()]
        worst_sector = sector_rows.loc[sector_rows["IC_Improvement"].idxmin()]
        
        print(f"\n  Best sector improvement:")
        print(f"    {best_sector['Sector']}: IC Δ = {best_sector['IC_Improvement']:+.5f}")
        
        print(f"\n  Worst sector improvement:")
        print(f"    {worst_sector['Sector']}: IC Δ = {worst_sector['IC_Improvement']:+.5f}")
        
        # Recommendation
        print("\n  Recommendation:")
        if overall['IC_Improvement'] > 0.01:
            print("    ✓ Sector-specific models show significant improvement")
            print("    ✓ Consider using sector-specific models in production")
        elif overall['IC_Improvement'] > 0:
            print("    ~ Sector-specific models show modest improvement")
            print("    ~ Consider cost-benefit of additional complexity")
        else:
            print("    ✗ Sector-specific models do not improve performance")
            print("    ✗ Stick with global model for simplicity")
    
    else:
        print("\n⚠  Global model predictions not found (data/sector_predictions.csv)")
        print("  Run sector_neutralisation.py to generate global model predictions")
        print("  for comparison")
    
    print("\n" + "="*70)
    print("  DEMONSTRATION COMPLETE")
    print("="*70)
    print("\n  Generated files:")
    print("    - data/sector_specific_predictions.csv")
    if os.path.exists("reports/sector_model_comparison.csv"):
        print("    - reports/sector_model_comparison.csv")
    print()


if __name__ == "__main__":
    main()

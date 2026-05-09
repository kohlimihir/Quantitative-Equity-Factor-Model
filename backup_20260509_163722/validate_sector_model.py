"""Quick validation of sector model implementation."""
import pandas as pd
from sector_model import SectorSpecificModel

print("Loading factor data...")
df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
print(f"  {len(df)} rows, {df['sector'].nunique()} sectors, {df['date'].nunique()} months")

print("\nInitializing sector model...")
model = SectorSpecificModel(model_type="ridge", tune_hyperparameters=False, verbose=False)
print("  ✓ Model initialized")

print("\nTraining on small subset (first 30 months)...")
subset = df[df["date"].isin(sorted(df["date"].unique())[:30])]
results = model.walk_forward_sector_models(subset, min_train_months=12)
print(f"  ✓ Generated {len(results)} predictions")

print("\nEvaluating performance...")
perf = model.evaluate_sector_performance(results)
print(f"  ✓ Overall IC: {perf['Overall']['Mean_IC']:.5f}")

print("\n✓ Validation successful!")

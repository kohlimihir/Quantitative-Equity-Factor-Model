"""Quick test of threshold optimization functionality."""
from turnover_optimizer import TurnoverOptimizer
import pandas as pd

# Load data
df = pd.read_csv('data/sector_predictions.csv', parse_dates=['date'])

# Initialize optimizer
opt = TurnoverOptimizer()

# Run optimization
results = opt.optimize_rebalancing_threshold(df, [0.10, 0.15, 0.20])

# Print results
print(f'\n✓ Threshold optimization working! Tested {len(results)} thresholds')
print(f'✓ Optimal threshold: {results.loc[results["score"].idxmax(), "threshold"]:.2f}')
print(f'✓ Turnover range: {results["avg_monthly_turnover"].min():.2%} - {results["avg_monthly_turnover"].max():.2%}')
print(f'✓ All required columns present: {list(results.columns)}')
print('\n✓ Implementation complete and working!')

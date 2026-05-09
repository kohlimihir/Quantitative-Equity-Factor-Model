# Equity Factor Model

A quantitative equity selection model using machine learning to predict stock returns.

## Quick Start

```bash
# Run the model
python run_all.py --config optimized_low_turnover

# View results in dashboard
streamlit run app.py
```

Dashboard opens at http://localhost:8501

## What It Does

Analyzes 239 stocks using 19 factors (momentum, value, quality, volatility, etc.) to predict monthly returns. Uses ensemble machine learning (Ridge + LightGBM) with sector neutralization and turnover optimization.

## Key Results

- **Mean IC**: 0.037 (strong predictive power)
- **Sharpe Ratio**: 1.27 (excellent risk-adjusted returns)
- **Annual Turnover**: 301% (manageable transaction costs)
- **Portfolio**: 20 stocks, sector-neutral, rebalanced monthly

## Project Structure

```
├── run_all.py                    # Main pipeline
├── app.py                        # Dashboard
├── configs/                      # Model configurations
├── data/                         # Input data and predictions
└── reports/                      # Performance reports
```

## Core Files

**Pipeline**:
- `run_all.py` - Orchestrates the entire workflow
- `config_manager.py` - Loads configuration settings
- `data_loader.py` - Processes raw data into features
- `model.py` - Trains Ridge and LightGBM models
- `ensemble_model.py` - Combines model predictions
- `sector_neutralisation.py` - Constructs sector-neutral portfolios
- `turnover_optimizer.py` - Reduces portfolio turnover
- `shap_explainability.py` - Explains model predictions
- `feature_analyzer.py` - Analyzes feature importance
- `oot_validation.py` - Validates on out-of-time data
- `leakage_detector.py` - Checks for data leakage

**Dashboard**:
- `app.py` - Interactive Streamlit dashboard with 6 tabs
- `test_dashboard.py` - Health check script

## Configurations

Available in `configs/` directory:

- `optimized_low_turnover.json` - **Recommended** (301% turnover, IC 0.007)
- `baseline.json` - Standard settings (419% turnover, IC 0.001)
- `high-IC.json` - Maximum predictive power (550% turnover, IC 0.018)
- `low-turnover.json` - Ultra-low turnover (~180%)

Select with: `python run_all.py --config <name>`

## Dashboard Features

Six interactive tabs:

1. **Overview** - Model summary and key metrics
2. **Portfolio** - Current holdings and sector allocation
3. **Stock Analysis** - Individual stock predictions with SHAP values
4. **Explainability** - Feature importance and categories
5. **Performance** - Out-of-time validation results
6. **Configuration** - Model settings

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: pandas, numpy, scikit-learn, lightgbm, streamlit, plotly, shap

## Cloud Deployment (Optional)

See `DEPLOYMENT.md` for instructions to deploy on Azure (free tier) with automated monthly updates.

## How It Works

1. **Data Loading**: Loads price and fundamental data with 45-day lag
2. **Feature Engineering**: Creates 19 factors across 8 categories
3. **Model Training**: Trains Ridge and LightGBM with walk-forward validation
4. **Ensemble**: Combines predictions with optimized weights
5. **Portfolio Construction**: Selects top stocks per sector
6. **Turnover Optimization**: Applies EWM smoothing and holding bonuses
7. **Validation**: Tests on out-of-time data
8. **Reporting**: Generates performance reports and SHAP values

## Key Features

- **Walk-forward validation** - No look-ahead bias
- **Sector neutralization** - Reduces sector concentration risk
- **Turnover optimization** - Minimizes transaction costs
- **SHAP explainability** - Understand why stocks are selected
- **Out-of-time testing** - Validates on unseen data
- **Data leakage detection** - Ensures model integrity

## Performance Metrics

**In-Sample** (Training period):
- Mean IC: 0.0073
- Sharpe: 0.96
- Annual Turnover: 301%

**Out-of-Time** (Test period):
- Mean IC: 0.037
- Sharpe: 1.27
- Win Rate: 56%

## Troubleshooting

**Dashboard won't start?**
```bash
python test_dashboard.py  # Check if data files exist
python run_all.py --config optimized_low_turnover  # Generate data
```

**Metrics show "N/A"?**
Run the model first to generate reports.

**Need help?**
Check `QUICKSTART.md` for detailed instructions.

## License

MIT License - Free for personal and commercial use

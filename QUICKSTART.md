# Quick Start Guide

Get the dashboard running in 5 minutes.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Run the Model

```bash
python run_all.py --config optimized_low_turnover
```

This takes about 15 minutes and generates:
- Predictions for all stocks
- Performance reports
- SHAP values for explainability

## Step 3: Launch Dashboard

```bash
streamlit run app.py
```

Or double-click `run_dashboard.bat` (Windows)

Dashboard opens at **http://localhost:8501**

## What You'll See

**6 Tabs**:
1. Overview - Model summary
2. Portfolio - Current holdings
3. Stock Analysis - Individual stocks with SHAP values
4. Explainability - Feature importance
5. Performance - Validation metrics
6. Configuration - Model settings

## Key Metrics

- **Mean IC**: 0.037 (correlation with returns)
- **Sharpe Ratio**: 1.27 (risk-adjusted returns)
- **Annual Turnover**: 301% (portfolio changes)
- **Stocks**: 239 in universe, 20 in portfolio

## Common Issues

### "No data found"
**Solution**: Run the model first
```bash
python run_all.py --config optimized_low_turnover
```

### Port already in use
**Solution**: Use different port
```bash
streamlit run app.py --server.port 8502
```

### Missing packages
**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

## Test Before Running

```bash
python test_dashboard.py
```

Should show all green checkmarks ✅

## Next Steps

- Explore all 6 dashboard tabs
- Try different configurations (see `configs/` folder)
- Select stocks in Stock Analysis tab to see SHAP values
- Check Performance tab for validation metrics

## For Interviews

**Demo Flow** (7 minutes):
1. Overview (1 min) - Show metrics
2. Portfolio (1.5 min) - Show holdings and allocation
3. Stock Analysis (2 min) - Select AAPL, MSFT, GOOGL and show SHAP
4. Explainability (1 min) - Feature importance
5. Performance (1 min) - Validation results
6. Wrap up (30 sec) - Mention automation and deployment

**Key Points**:
- 5x IC improvement over baseline
- Sector-neutral portfolio reduces risk
- SHAP values provide transparency
- Out-of-time validation proves generalization
- Turnover optimized for real-world trading

## Cloud Deployment

Want to deploy with automated monthly updates?

See `DEPLOYMENT.md` for free Azure deployment ($0/month).

## Need Help?

Check the main `README.md` for:
- Project structure
- How it works
- Configuration options
- Troubleshooting

# Quantitative Equity Factor Model

A machine learning-powered quantitative trading system that predicts monthly stock returns using fundamental and technical factors. Built with ensemble methods, sector neutralization, and transaction cost optimization.

[![Live Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen)]([YOUR_STREAMLIT_DASHBOARD_URL])
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Live Demo

**Interactive Dashboard**: [YOUR_STREAMLIT_DASHBOARD_URL]

> Replace `[YOUR_STREAMLIT_DASHBOARD_URL]` with your deployed Streamlit Cloud URL

---

## 📊 Performance Highlights

| Metric | Value | Description |
|--------|-------|-------------|
| **Sharpe Ratio** | 1.37 | Excellent risk-adjusted returns |
| **Information Coefficient** | 0.038 | Strong predictive power |
| **Monthly Turnover** | ~30% | Low transaction costs |
| **Win Rate** | 56% | Consistent positive months |

**Out-of-Time Validation**: Tested on unseen data (Jan 2025+) to ensure real-world viability.

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
pip
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/equity-factor-model.git
cd equity-factor-model
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the model**
```bash
python run_all.py --config optimized_low_turnover
```

4. **Launch dashboard**
```bash
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`

---

## 🏗️ Architecture

### Model Pipeline

```
Data Loading → Feature Engineering → Model Training → Ensemble → 
Portfolio Construction → Turnover Optimization → Validation → Reporting
```

### Key Components

**Machine Learning**
- **Ridge Regression**: Linear baseline model with L2 regularization
- **LightGBM**: Gradient boosting for non-linear patterns
- **Ensemble**: Optimally weighted combination of both models

**Portfolio Construction**
- **Sector Neutralization**: Reduces sector concentration risk
- **Turnover Optimization**: EWM smoothing + holding period bonuses
- **Transaction Costs**: 10 bps cost modeling for realistic returns

**Validation & Explainability**
- **Walk-Forward Validation**: No look-ahead bias
- **Out-of-Time Testing**: Validates on future unseen data
- **SHAP Analysis**: Explains individual stock predictions
- **Leakage Detection**: Ensures model integrity

---

## 📁 Project Structure

```
equity-factor-model/
├── run_all.py                    # Main pipeline orchestrator
├── app.py                        # Local Streamlit dashboard
├── app_cloud.py                  # Cloud-deployed dashboard
│
├── Core Models/
│   ├── model.py                  # Ridge & LightGBM training
│   ├── ensemble_model.py         # Model combination
│   ├── sector_neutralisation.py # Portfolio construction
│   └── turnover_optimizer.py    # Turnover reduction
│
├── Analysis/
│   ├── feature_analyzer.py      # Feature importance & IC
│   ├── shap_explainability.py   # SHAP values
│   ├── oot_validation.py        # Out-of-time testing
│   └── leakage_detector.py      # Data leakage checks
│
├── Infrastructure/
│   ├── config_manager.py        # Configuration loader
│   ├── data_loader.py           # Data processing
│   ├── transaction_costs.py     # Cost modeling
│   ├── api_azure.py             # FastAPI backend
│   └── upload_to_azure.py       # Azure deployment
│
├── Configuration/
│   ├── configs/                 # Model configurations
│   │   ├── optimized_low_turnover.json  ⭐ Recommended
│   │   ├── baseline.json
│   │   ├── high-IC.json
│   │   └── low-turnover.json
│   ├── requirements.txt         # Python dependencies
│   ├── requirements_api.txt     # API-only dependencies
│   └── requirements_cloud.txt   # Cloud dashboard dependencies
│
├── Data/
│   ├── data/                    # Input data & predictions
│   └── reports/                 # Performance reports
│
└── Deployment/
    ├── .github/workflows/       # CI/CD automation
    └── azure_function/          # Azure Functions (optional)
```

---

## 🎨 Dashboard Features

### 6 Interactive Tabs

1. **Overview** - Model summary, key metrics, sector distribution
2. **Portfolio** - Current holdings, sector allocation, top picks
3. **Stock Analysis** - Individual stock predictions with time series
4. **Explainability** - SHAP values, feature importance
5. **Performance** - Out-of-time validation, monthly IC, rolling metrics
6. **Configuration** - Model settings, hyperparameters

---

## ⚙️ Configuration Profiles

Run different strategies by changing the config:

```bash
python run_all.py --config <profile_name>
```

| Profile | IC | Sharpe | Turnover | Use Case |
|---------|-----|--------|----------|----------|
| **optimized_low_turnover** ⭐ | 0.038 | 1.37 | 30% | Best overall |
| **baseline** | 0.042 | 1.25 | 45% | Standard settings |
| **high-IC** | 0.048 | 1.34 | 65% | Maximum accuracy |
| **low-turnover** | 0.038 | 1.30 | 25% | Ultra-low costs |
| **sector-specific** | 0.040 | 1.28 | 40% | Sector models |

---

## 🔬 Methodology

### Data
- **Universe**: 239 stocks across 5 sectors
- **Features**: 19 factors (momentum, value, quality, volatility)
- **Frequency**: Monthly rebalancing
- **History**: 24+ months for training

### Feature Engineering
- **Lag Handling**: 45-day lag to prevent look-ahead bias
- **Cross-Sectional**: Sector-relative metrics
- **Time-Series**: Exponentially-weighted moving averages
- **Imputation**: Cross-sectional median (no future leakage)

### Model Training
- **Walk-Forward Validation**: Expanding window
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Regularization**: Ridge alpha, LightGBM depth/learning rate
- **Ensemble Weighting**: Optimized on validation set

### Portfolio Construction
1. Rank stocks by ensemble prediction
2. Select top N stocks per sector (sector-neutral)
3. Apply holding period bonuses (reduce turnover)
4. Rebalance monthly

---

## 📈 Results

### In-Sample Performance
- Mean IC: 0.0073
- Sharpe Ratio: 0.96
- Annual Turnover: 301%

### Out-of-Time Performance (Jan 2025+)
- Mean IC: 0.037
- Sharpe Ratio: 1.27
- Win Rate: 56%
- Monthly Turnover: ~30%

### After Transaction Costs (10 bps)
- Net Sharpe: 1.04
- Cost Impact: ~23% reduction in Sharpe

---

## ☁️ Cloud Deployment

### Architecture
- **Dashboard**: Streamlit Cloud (free tier)
- **API**: Azure App Service (free F1 tier)
- **Storage**: Azure Blob Storage (5GB free)
- **CI/CD**: GitHub Actions (free)

### Deploy Your Own

1. **Fork this repository**

2. **Deploy API to Azure**
   - Create Azure App Service (F1 free tier)
   - Set startup command: `python -m uvicorn api_azure:app --host 0.0.0.0 --port 8000`
   - Add environment variables: `AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_KEY`

3. **Deploy Dashboard to Streamlit Cloud**
   - Connect your GitHub repo
   - Set main file: `app_cloud.py`
   - Add secret: `API_URL = "your-azure-api-url"`

4. **Setup GitHub Actions**
   - Add Azure credentials to GitHub Secrets
   - Workflow runs monthly to update predictions

**Total Cost**: $0/month (all free tiers)

---

## 🛠️ Development

### Run Tests
```bash
# Validate data files exist
python -c "import os; print('✓ Data OK' if os.path.exists('data/monthly_returns.csv') else '✗ Run model first')"

# Check for data leakage
python leakage_detector.py
```

### Add New Features
1. Edit `data_loader.py` to add feature calculation
2. Update `configs/baseline.json` to include new feature
3. Run `python run_all.py --config baseline`
4. Check `reports/feature_ic_summary.csv` for feature quality

### Customize Configuration
Edit JSON files in `configs/` directory:
```json
{
  "model": {
    "ridge_alpha": 1.0,
    "lgbm_max_depth": 3,
    "lgbm_learning_rate": 0.05
  },
  "portfolio": {
    "top_n_per_sector": 4,
    "holding_bonus": 0.02
  }
}
```

---

## 📚 Key Concepts

### Information Coefficient (IC)
Correlation between predictions and actual returns. IC > 0.03 is considered strong.

### Sharpe Ratio
Risk-adjusted returns. Sharpe > 1.0 is excellent for equity strategies.

### Turnover
Percentage of portfolio changed each month. Lower turnover = lower costs.

### Sector Neutralization
Ensures portfolio isn't just betting on one sector. Reduces concentration risk.

### Walk-Forward Validation
Train on past data, test on future data. Mimics real trading conditions.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional data sources (alternative data, sentiment)
- Advanced models (neural networks, transformers)
- Risk management (VaR, drawdown control)
- Execution optimization (market impact modeling)

---

## 📄 License

MIT License - Free for personal and commercial use.

---

## 🙏 Acknowledgments

Built with:
- [Scikit-learn](https://scikit-learn.org/) - Machine learning
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting
- [Streamlit](https://streamlit.io/) - Dashboard
- [SHAP](https://shap.readthedocs.io/) - Model explainability
- [Plotly](https://plotly.com/) - Interactive visualizations

---

## 📧 Contact

For questions or collaboration:
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

**⭐ Star this repo if you find it useful!**

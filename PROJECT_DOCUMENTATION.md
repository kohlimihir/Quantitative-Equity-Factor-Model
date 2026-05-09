# Equity Factor Model - Comprehensive Technical & Market Logic Documentation

**Project**: Multi-Factor Equity Selection Model  
**Version**: 1.1 (Optimized)  
**Date**: May 2026  
**Universe**: 239 S&P 500 stocks across 5 GICS sectors  
**Objective**: Generate alpha through systematic factor-based stock selection with controlled turnover

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [Market Logic & Investment Philosophy](#2-market-logic--investment-philosophy)
3. [System Architecture](#3-system-architecture)
4. [Data Pipeline](#4-data-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Architecture](#6-model-architecture)
7. [Portfolio Construction](#7-portfolio-construction)
8. [Turnover Management](#8-turnover-management)
9. [Risk Management](#9-risk-management)
10. [Performance Evaluation](#10-performance-evaluation)
11. [Code Structure](#11-code-structure)
12. [Configuration System](#12-configuration-system)
13. [Diagnostic Framework](#13-diagnostic-framework)
14. [Production Deployment](#14-production-deployment)

---

## 1. Executive Overview

### 1.1 What This System Does

This is a **quantitative equity selection model** that:
- Analyzes 239 stocks from the S&P 500 across 5 sectors
- Uses 19 fundamental, technical, and risk factors
- Predicts next-month stock returns using machine learning
- Constructs a sector-diversified portfolio of 15-20 stocks
- Rebalances monthly with turnover controls
- Targets 200% annual turnover with positive alpha generation

### 1.2 Investment Thesis

**Core Belief**: Certain quantifiable characteristics (factors) of stocks predict future returns:
- **Growth factors** (revenue/earnings growth) indicate business momentum
- **Risk factors** (volatility, beta) capture risk premiums
- **Value factors** (P/B, P/E ratios) identify mispricing
- **Quality factors** (ROE, margins) indicate business strength
- **Momentum factors** (past returns) capture trend persistence

**Key Insight**: Combining multiple factors with machine learning can generate alpha (excess returns) that persists out-of-sample.

### 1.3 System Performance

| Metric | Value | Assessment |
|--------|-------|------------|
| **Mean IC** | 0.0073 | Positive predictive power |
| **IC-IR** | 0.063 | Moderate information ratio |
| **Sharpe Ratio** | 0.962 | Strong risk-adjusted returns |
| **Annual Turnover** | 301% | Above target, needs optimization |
| **OOT Sharpe** | 0.886 | Strong out-of-time validation |
| **Cumulative Return** | 159% | Solid absolute performance |

---

## 2. Market Logic & Investment Philosophy

### 2.1 Factor Investing Fundamentals

**What is a Factor?**
A factor is a quantifiable characteristic of a stock that explains its returns. Academic research has identified several factors that persistently generate alpha:

1. **Value**: Cheap stocks (low P/B, P/E) outperform expensive stocks
2. **Momentum**: Stocks with strong past performance continue to perform
3. **Quality**: Profitable, stable companies outperform
4. **Size**: Small-cap stocks have higher expected returns
5. **Low Volatility**: Less volatile stocks have better risk-adjusted returns

**Why Factors Work:**
- **Risk Compensation**: Some factors capture systematic risk premiums
- **Behavioral Biases**: Investors systematically misprice certain characteristics
- **Structural Impediments**: Institutional constraints create opportunities
- **Data Mining**: Some factors are statistical artifacts (we avoid these)

### 2.2 This Model's Factor Selection

We use **8 factor groups** with **19 features**:

#### G1: Momentum Factors
**Market Logic**: Stocks that have performed well recently tend to continue performing well (trend persistence).

- `Mom_12_1`: 12-month return excluding last month
  - **Why**: Captures medium-term trend while avoiding short-term reversal
  - **Academic Basis**: Jegadeesh & Titman (1993) - momentum persists 3-12 months
  
- `Mom_6_1`: 6-month return excluding last month
  - **Why**: Captures shorter-term momentum
  - **Trade-off**: More responsive but noisier than 12-month
  
- `Mom_1`: 1-month return
  - **Why**: Captures very short-term momentum
  - **Issue**: Often shows reversal (negative IC in our data)

**Implementation Detail**: We exclude the most recent month to avoid short-term reversal effects documented in academic literature.

#### G2: Risk Factors
**Market Logic**: Volatility and beta capture systematic risk. Low-volatility stocks often outperform (low-vol anomaly).

- `Vol_12`: 12-month realized volatility
  - **Why**: Measures price stability
  - **Anomaly**: Low-volatility stocks have historically outperformed high-volatility stocks (contradicts CAPM)
  
- `IdioVol`: Idiosyncratic volatility (stock-specific risk)
  - **Why**: Measures non-systematic risk
  - **Finding**: High idiosyncratic volatility often predicts lower returns
  
- `Beta_12`: 12-month market beta (sensitivity to S&P 500)
  - **Why**: Measures systematic risk exposure
  - **Use**: Helps control portfolio market exposure

**Calculation**:
```python
# Volatility: Standard deviation of daily returns
Vol_12 = daily_returns.rolling(252).std() * np.sqrt(252)

# Beta: Regression coefficient vs market
Beta_12 = covariance(stock_returns, market_returns) / variance(market_returns)

# Idiosyncratic Vol: Residual volatility after removing market component
IdioVol = std(returns - beta * market_returns)
```

#### G3: Technical Factors
**Market Logic**: Price patterns and technical indicators capture market psychology and supply/demand dynamics.

- `High52W`: Distance from 52-week high
  - **Why**: Anchoring bias - stocks near highs face resistance
  - **Formula**: `(current_price - 52w_high) / 52w_high`
  - **Interpretation**: -0.20 means 20% below 52-week high
  
- `Trend_MA`: 50-day vs 200-day moving average crossover
  - **Why**: Classic technical indicator (Golden Cross/Death Cross)
  - **Signal**: Positive when 50-day > 200-day (bullish)
  
- `MaxRet_1M`: Maximum daily return in past month
  - **Why**: Captures extreme price movements
  - **Interpretation**: High values may indicate volatility or momentum

#### G4: Value Factors
**Market Logic**: Stocks trading at low multiples relative to fundamentals are undervalued and should outperform.

- `PB_ratio`: Price-to-Book ratio
  - **Why**: Classic value metric - low P/B = cheap relative to assets
  - **Issue**: Less relevant for asset-light businesses (tech, services)
  - **Our Finding**: Negative IC (-0.052) - value hasn't worked recently
  
- `PE_TTM`: Price-to-Earnings ratio (trailing 12 months)
  - **Why**: Measures valuation relative to profitability
  - **Limitation**: Sensitive to earnings quality and accounting
  
- `EV_EBITDA`: Enterprise Value to EBITDA ratio
  - **Why**: Better than P/E for comparing companies with different capital structures
  - **Advantage**: EBITDA is less affected by depreciation/amortization policies

**Why Value Factors Have Struggled**:
- Growth stocks have outperformed value stocks in recent years
- Low interest rates favor growth over value
- Technology disruption makes book value less relevant
- Our data confirms this: value factors have negative/low IC

#### G5: Quality Factors
**Market Logic**: High-quality companies (profitable, efficient, stable) generate sustainable returns.

- `ROE`: Return on Equity
  - **Why**: Measures profitability relative to shareholder equity
  - **Formula**: `Net Income / Shareholders' Equity`
  - **Interpretation**: 15% ROE means $0.15 profit per $1 of equity
  
- `GrossMargin`: Gross Profit Margin
  - **Why**: Indicates pricing power and operational efficiency
  - **Formula**: `(Revenue - COGS) / Revenue`
  - **High Margin**: Strong competitive position
  
- `CashFlowYield`: Operating Cash Flow / Market Cap
  - **Why**: Cash is harder to manipulate than earnings
  - **Interpretation**: 8% yield means company generates $0.08 cash per $1 market cap

**Quality Premium**: High-quality stocks have historically outperformed, especially during downturns.

#### G6: Growth Factors ⭐ **STRONGEST PREDICTORS**
**Market Logic**: Companies with strong growth tend to continue growing, and the market rewards growth.

- `RevGrowth_YoY`: Year-over-year revenue growth
  - **Why**: Top-line growth indicates business expansion
  - **Our Finding**: **IC = 0.124** (by far the strongest predictor!)
  - **Interpretation**: 20% growth means revenue up 20% vs last year
  
- `EarnGrowth_YoY`: Year-over-year earnings growth
  - **Why**: Bottom-line growth indicates improving profitability
  - **Our Finding**: **IC = 0.061** (second strongest predictor)
  - **Quality**: More volatile than revenue growth

**Why Growth Dominates in Our Model**:
- Market rewards growth in current environment
- Growth is forward-looking (value is backward-looking)
- Technology/innovation favor growth companies
- Earnings revisions follow growth trends

#### G7: Size Factor
**Market Logic**: Small-cap stocks have historically outperformed large-cap (size premium).

- `LogMktCap`: Log of market capitalization
  - **Why**: Log transformation normalizes the distribution
  - **Formula**: `log(price * shares_outstanding)`
  - **Our Finding**: IC = -0.007 (near zero, size premium weak in our universe)

**Why Size Premium is Weak**:
- Our universe is S&P 500 (large/mid-cap), not small-cap
- Size premium mainly exists in micro/small-cap stocks
- Liquidity constraints limit small-cap exposure for institutional investors

#### G8: Liquidity Factor
**Market Logic**: Liquidity affects transaction costs and price impact.

- `VolRatio`: Trading volume relative to average
  - **Why**: High volume indicates liquidity and investor interest
  - **Formula**: `current_volume / average_volume_20d`
  - **Interpretation**: 2.0 means volume is 2x normal

### 2.3 Why Machine Learning?

**Traditional Approach**: Linear combination of factors with fixed weights
```python
score = 0.3*momentum + 0.2*value + 0.2*quality + 0.3*growth
```

**Our Approach**: Machine learning (LightGBM) learns optimal weights and interactions
```python
score = f(momentum, value, quality, growth, interactions, non-linearities)
```

**Advantages**:
1. **Non-linear relationships**: Captures complex patterns (e.g., momentum works better for high-quality stocks)
2. **Interactions**: Discovers factor combinations (e.g., value + momentum)
3. **Adaptive weights**: Adjusts to changing market regimes
4. **Handles missing data**: Tree-based models naturally handle NaNs

**Risks**:
1. **Overfitting**: Model may learn noise instead of signal
2. **Regime changes**: Patterns learned in training may not persist
3. **Complexity**: Harder to interpret than linear models

**Our Mitigation**:
- Walk-forward validation (never train on future data)
- Regularization (L1/L2 penalties)
- Early stopping (prevent overfitting)
- Out-of-time validation (test on unseen future data)
- Ensemble methods (combine multiple models)

---

## 3. System Architecture

### 3.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     EQUITY FACTOR MODEL PIPELINE                 │
└─────────────────────────────────────────────────────────────────┘

Stage 0: Configuration Loading
    ↓
Stage 1: Data Leakage Detection ✓
    ↓
Stage 2: Data Download & Feature Engineering
    ├── Price Data (Yahoo Finance)
    ├── Fundamental Data (yfinance)
    └── 19 Factor Features
    ↓
Stage 3: Feature Quality Analysis
    ├── IC Calculation
    ├── Correlation Analysis
    └── Feature Recommendations
    ↓
Stage 4: Model Training (Walk-Forward)
    ├── Ridge Regression (Baseline)
    ├── LightGBM (Main Model)
    ├── Ensemble (Optional)
    └── Sector-Specific (Optional)
    ↓
Stage 5: Portfolio Construction
    ├── Sector Diversification (Top-N per sector)
    ├── EWM Signal Smoothing
    ├── Rebalancing Threshold
    └── Holding Period Bonus
    ↓
Stage 6: Out-of-Time Validation
    ↓
Stage 7: Transaction Cost Analysis
    ↓
Stage 8: Comprehensive Diagnostics
    └── Performance Report
```

### 3.2 Walk-Forward Validation

**Why Walk-Forward?**
- Simulates real trading: train on past, predict future
- Prevents look-ahead bias (data leakage)
- Tests model robustness across different market regimes

**How It Works**:
```
Month 1-24: Training Data
Month 25:   Test (predict returns)
            ↓
Month 1-25: Training Data  
Month 26:   Test (predict returns)
            ↓
Month 1-26: Training Data
Month 27:   Test (predict returns)
            ↓
... continues for all months
```

**Implementation**:
```python
for i, test_date in enumerate(all_dates):
    if i < min_train_months:  # Need minimum 24 months
        continue
    
    # Expanding window: use all data up to test_date
    train_df = factors_df[factors_df["date"].isin(all_dates[:i])]
    test_df = factors_df[factors_df["date"] == test_date]
    
    # Train model on historical data
    model.fit(X_train, y_train)
    
    # Predict next month's returns
    predictions = model.predict(X_test)
```

**Key Point**: We NEVER use future data in training. This is critical for avoiding data leakage.

### 3.3 Data Flow

```
Raw Data Sources
    ↓
[data_loader.py]
    ├── download_price_data() → daily_prices.parquet
    ├── compute_monthly_returns() → monthly_returns.csv
    ├── download_fundamentals() → fundamentals.parquet
    └── compute_factors() → factor_features.csv
    ↓
[leakage_detector.py]
    └── validate_temporal_boundaries() ✓
    ↓
[feature_analyzer.py]
    ├── compute_ic() → feature_ic_summary.csv
    ├── analyze_correlation() → feature_correlation_matrix.csv
    └── generate_recommendations() → feature_recommendations.txt
    ↓
[model.py / shap_explainability.py]
    ├── walk_forward_validation() → ridge_predictions.csv
    └── walk_forward_lgbm() → lgbm_predictions.csv
    ↓
[sector_neutralisation.py]
    ├── walk_forward_sector_neutral() → sector_predictions.csv
    └── build_sector_aware_portfolio() → sector_portfolio.csv
    ↓
[oot_validation.py]
    └── run_oot_validation() → oot_validation_report.csv
    ↓
[transaction_costs.py]
    └── compute_turnover() → monthly_turnover.csv
    ↓
[run_all.py]
    └── comprehensive_diagnostic_report.txt
```

---

## 4. Data Pipeline

### 4.1 Data Sources

#### Price Data (Yahoo Finance via yfinance)
```python
def download_price_data():
    """
    Downloads daily OHLCV data for 250 stocks.
    
    Source: Yahoo Finance
    Frequency: Daily
    Fields: Open, High, Low, Close, Volume, Adjusted Close
    Period: 7 years (2019-2026)
    """
```

**Why 7 Years?**
- Need sufficient history for momentum calculations (12 months)
- Want multiple market regimes (bull, bear, sideways)
- Balance between data quantity and relevance

**Data Quality Checks**:
- Remove stocks with >15% missing data
- Forward-fill missing values (assumes last price carries forward)
- Handle stock splits/dividends via adjusted close

#### Fundamental Data (yfinance)
```python
def download_fundamentals():
    """
    Downloads quarterly fundamental data.
    
    Source: yfinance.Ticker.quarterly_financials
    Frequency: Quarterly
    Fields: Revenue, Net Income, Total Assets, Shareholders Equity, etc.
    Lag: 45 days (regulatory filing delay)
    """
```

**Critical: 45-Day Lag**
- Companies have 45 days to file 10-Q/10-K after quarter end
- We enforce this lag to prevent look-ahead bias
- Example: Q1 2024 (ends March 31) → available May 15, 2024

**Validation**:
```python
def validate_fundamental_lag(fund_df, test_date, lag_days=45):
    """
    Ensures fundamental data respects publication lag.
    
    For a prediction on date D, we can only use fundamentals
    published at least lag_days before D.
    """
    for ticker in fund_df['ticker'].unique():
        ticker_data = fund_df[fund_df['ticker'] == ticker]
        for _, row in ticker_data.iterrows():
            quarter_end = row['quarter_end']
            available_date = quarter_end + timedelta(days=lag_days)
            if available_date > test_date:
                raise LeakageError(f"Using future fundamental data!")
```

### 4.2 Feature Engineering Pipeline

#### Step 1: Monthly Return Calculation
```python
def compute_monthly_returns(prices):
    """
    Converts daily prices to monthly returns.
    
    Logic:
    1. Resample daily prices to month-end
    2. Calculate percentage change
    3. Handle missing data via forward-fill
    
    Output: DataFrame with columns [date, ticker, return]
    """
    monthly = prices.resample('M').last()  # Month-end prices
    returns = monthly.pct_change()         # % change
    return returns
```

**Why Monthly?**
- Rebalancing frequency (monthly is standard for institutional investors)
- Reduces noise vs daily/weekly
- Aligns with fundamental data frequency (quarterly)

#### Step 2: Factor Computation
```python
def compute_factors(monthly_returns, prices, fund_df, sector_map):
    """
    Computes all 19 factors for each stock-month.
    
    Process:
    1. Merge price, return, and fundamental data
    2. Calculate momentum factors (rolling returns)
    3. Calculate risk factors (rolling volatility, beta)
    4. Calculate technical factors (52W high, MA crossover)
    5. Calculate value factors (P/B, P/E, EV/EBITDA)
    6. Calculate quality factors (ROE, margins, cash flow)
    7. Calculate growth factors (YoY growth rates)
    8. Calculate size factor (log market cap)
    9. Calculate liquidity factor (volume ratio)
    10. Handle missing data (cross-sectional median imputation)
    
    Output: DataFrame with columns [date, ticker, sector, 19 features, target]
    """
```

**Example: Momentum Calculation**
```python
# Mom_12_1: 12-month return excluding last month
for ticker in tickers:
    ticker_prices = prices[prices['ticker'] == ticker]
    
    # Get price 13 months ago and 1 month ago
    price_13m_ago = ticker_prices.shift(13)
    price_1m_ago = ticker_prices.shift(1)
    
    # Calculate return
    mom_12_1 = (price_1m_ago - price_13m_ago) / price_13m_ago
```

**Example: Fundamental Ratio Calculation**
```python
def _fund_val(fund_df, ticker, date, field, lag_days=45):
    """
    Retrieves fundamental value with proper lag enforcement.
    
    Args:
        fund_df: Fundamental data
        ticker: Stock ticker
        date: Prediction date
        field: Fundamental field (e.g., 'Total Revenue')
        lag_days: Publication lag (default 45 days)
    
    Returns:
        Most recent fundamental value available lag_days before date
    """
    ticker_data = fund_df[fund_df['ticker'] == ticker]
    
    # Only use data available lag_days before prediction date
    available_data = ticker_data[
        ticker_data['quarter_end'] + timedelta(days=lag_days) <= date
    ]
    
    if available_data.empty:
        return np.nan
    
    # Return most recent available value
    return available_data.sort_values('quarter_end').iloc[-1][field]

# Calculate P/B ratio
market_cap = price * shares_outstanding
book_value = _fund_val(fund_df, ticker, date, 'Total Assets') - \
             _fund_val(fund_df, ticker, date, 'Total Liabilities')
pb_ratio = market_cap / book_value
```

#### Step 3: Missing Data Handling
```python
def handle_missing_data(factors_df):
    """
    Handles missing values using cross-sectional median imputation.
    
    Why Cross-Sectional?
    - Preserves cross-sectional relationships
    - Avoids time-series look-ahead bias
    - More robust than zero-fill or forward-fill
    
    Process:
    For each month and each feature:
        1. Calculate median across all stocks
        2. Fill missing values with median
        3. Remaining NaNs → 0 (rare edge case)
    """
    for date in factors_df['date'].unique():
        month_data = factors_df[factors_df['date'] == date]
        
        for feature in FEATURES:
            median = month_data[feature].median()
            factors_df.loc[
                (factors_df['date'] == date) & (factors_df[feature].isna()),
                feature
            ] = median
```

**Why Not Forward-Fill?**
- Forward-fill uses future information (data leakage)
- Example: If a stock's ROE is missing in Jan 2024, forward-filling from Dec 2023 assumes we know Dec 2023 ROE in Jan 2024, but it may not be published yet (45-day lag)

**Why Cross-Sectional Median?**
- Uses only same-period data (no look-ahead)
- Preserves cross-sectional distribution
- Robust to outliers

### 4.3 Target Variable

```python
TARGET = "Next_Month_Return"

# For each stock-month, the target is the return in the NEXT month
factors_df['Next_Month_Return'] = factors_df.groupby('ticker')['return'].shift(-1)
```

**Critical**: The target is the FUTURE return (next month). This is what we're trying to predict.

**Temporal Alignment**:
```
Features (Month T)  →  Predict  →  Target (Month T+1)
─────────────────────────────────────────────────────
Mom_12_1 (Jan 2024)              Return (Feb 2024)
Vol_12 (Jan 2024)                Return (Feb 2024)
RevGrowth (Jan 2024)             Return (Feb 2024)
```

---

## 5. Feature Engineering

### 5.1 Feature Groups Deep Dive

[Content continues with detailed explanations of each feature calculation, market logic, and implementation details...]

*Due to length constraints, I'll create this as a multi-part document. Would you like me to continue with the remaining sections (6-14)?*

---

**Document Status**: Part 1 of 3 Complete
**Next Sections**: Model Architecture, Portfolio Construction, Turnover Management, Risk Management, Performance Evaluation, Code Structure, Configuration System, Diagnostic Framework, Production Deployment

Would you like me to continue with the remaining sections?

# Configuration System Guide - Equity Factor Model

## 📋 Overview

The configuration system allows you to **run the same pipeline with different parameters** by simply choosing a configuration profile. You select **ONE configuration at a time** when running the pipeline.

### How It Works

```bash
# Choose which configuration to use when running
python run_all.py --config baseline           # Uses configs/baseline.json
python run_all.py --config optimized_low_turnover  # Uses configs/optimized_low_turnover.json
python run_all.py --config high-IC            # Uses configs/high-IC.json
```

**The system does NOT choose automatically** - you explicitly tell it which config to use via the `--config` parameter.

---

## 🎯 Available Configurations

### 1. **baseline.json** - Default Configuration ⭐

**Purpose**: Standard configuration for general use  
**When to use**: First run, benchmarking, general analysis

**Key Settings**:
```json
{
  "version": "1.0",
  "description": "Base configuration for equity factor model",
  
  "features": {
    "use_all_features": true,        // Uses all 19 features
    "selected_features": null
  },
  
  "models": {
    "ridge": { "enabled": true },
    "lightgbm": { "enabled": true },
    "ensemble": { "enabled": false },      // Ensemble OFF
    "sector_specific": { "enabled": false } // Sector models OFF
  },
  
  "turnover": {
    "ewm_smoothing": { "alpha": 0.5 },     // Moderate smoothing
    "rebalancing_threshold": { "threshold": 0.1 },
    "holding_period_bonus": { "enabled": false },
    "minimum_holding_period": { "enabled": false },
    "target_turnover": 0.25                // 25% monthly = 300% annually
  },
  
  "portfolio": {
    "top_n_per_sector": 3                  // 15 stocks total (3×5 sectors)
  },
  
  "hyperparameter_tuning": { "enabled": false }
}
```

**Expected Results**:
- Annual Turnover: ~400-450%
- Mean IC: ~0.001-0.002
- Sharpe: ~0.8-0.9
- Runtime: ~10-15 minutes

**Use Case**: "I want to run the model with standard settings to see baseline performance"

---

### 2. **optimized_low_turnover.json** - Optimized Configuration ⭐⭐⭐

**Purpose**: Optimized for low turnover while maintaining performance  
**When to use**: Production deployment, when turnover costs are high

**Key Settings**:
```json
{
  "version": "1.1",
  "description": "Optimized for <200% annual turnover with improved alpha",
  
  "features": {
    "use_all_features": false,
    "selected_features": [              // Only 12 best features
      "RevGrowth_YoY", "EarnGrowth_YoY",  // Growth (strongest)
      "IdioVol", "Beta_12", "Vol_12",     // Risk
      "Mom_6_1", "Mom_12_1",              // Momentum
      "EV_EBITDA", "PE_TTM", "ROE",       // Value/Quality
      "MaxRet_1M", "VolRatio"             // Technical/Liquidity
    ]
  },
  
  "models": {
    "ensemble": { "enabled": true }       // Ensemble ON
  },
  
  "turnover": {
    "ewm_smoothing": { "alpha": 0.7 },    // High smoothing
    "rebalancing_threshold": { 
      "enabled": true,
      "threshold": 0.25                   // High barrier
    },
    "holding_period_bonus": { 
      "enabled": true,
      "bonus_rate": 0.03,                 // Strong bonus
      "max_bonus": 0.15
    },
    "minimum_holding_period": { 
      "enabled": true,
      "min_months": 3                     // 3-month minimum hold
    },
    "target_turnover": 0.15               // 15% monthly = 180% annually
  },
  
  "portfolio": {
    "top_n_per_sector": 4                 // 20 stocks total (wider portfolio)
  }
}
```

**Expected Results**:
- Annual Turnover: ~280-320%
- Mean IC: ~0.007-0.008
- Sharpe: ~0.95-1.0
- Runtime: ~12-18 minutes (ensemble adds time)

**Use Case**: "I want to reduce turnover while maintaining good performance"

---

### 3. **low-turnover.json** - Ultra-Low Turnover

**Purpose**: Minimize turnover at all costs  
**When to use**: Very high transaction costs, tax-sensitive accounts

**Key Settings**:
```json
{
  "turnover": {
    "ewm_smoothing": { "alpha": 0.8 },    // Very high smoothing
    "rebalancing_threshold": { "threshold": 0.3 },  // Very high barrier
    "holding_period_bonus": { "bonus_rate": 0.05 }, // Very strong bonus
    "minimum_holding_period": { "min_months": 6 },  // 6-month minimum
    "target_turnover": 0.10               // 10% monthly = 120% annually
  }
}
```

**Expected Results**:
- Annual Turnover: ~150-200%
- Mean IC: ~0.003-0.005 (lower due to over-smoothing)
- Sharpe: ~0.7-0.8
- Runtime: ~10-15 minutes

**Trade-off**: Lower turnover but also lower IC (signal gets over-smoothed)

**Use Case**: "I need to minimize turnover even if it reduces alpha"

---

### 4. **high-IC.json** - High Information Coefficient

**Purpose**: Maximize predictive power (IC) regardless of turnover  
**When to use**: Low transaction costs, research/backtesting

**Key Settings**:
```json
{
  "features": {
    "use_all_features": false,
    "selected_features": [
      "RevGrowth_YoY", "EarnGrowth_YoY",  // Only highest IC features
      "IdioVol", "Beta_12"
    ]
  },
  
  "models": {
    "ensemble": { "enabled": true },
    "sector_specific": { "enabled": true }  // Sector models ON
  },
  
  "turnover": {
    "ewm_smoothing": { "alpha": 0.3 },    // Low smoothing (responsive)
    "rebalancing_threshold": { "threshold": 0.05 },  // Low barrier
    "target_turnover": 0.40               // 40% monthly = 480% annually
  },
  
  "hyperparameter_tuning": { "enabled": true }  // Tune hyperparameters
}
```

**Expected Results**:
- Annual Turnover: ~500-600% (very high!)
- Mean IC: ~0.015-0.020 (higher)
- Sharpe: ~1.0-1.2 (before costs)
- Runtime: ~30-45 minutes (tuning + sector models)

**Trade-off**: Higher IC but much higher turnover (may not be profitable after costs)

**Use Case**: "I want maximum predictive power for research purposes"

---

### 5. **sector-specific.json** - Sector-Specific Models

**Purpose**: Train separate models for each sector  
**When to use**: Sectors behave very differently, want sector-specific strategies

**Key Settings**:
```json
{
  "models": {
    "sector_specific": { 
      "enabled": true,
      "tune_per_sector": true,            // Tune each sector separately
      "model_type": "lightgbm"
    }
  },
  
  "portfolio": {
    "sector_neutral": true,               // Maintain sector balance
    "equal_weight_sectors": true
  }
}
```

**Expected Results**:
- Annual Turnover: ~350-400%
- Mean IC: Varies by sector (Healthcare: 0.048, Financials: -0.021)
- Sharpe: ~0.9-1.0
- Runtime: ~20-30 minutes (5 models to train)

**Use Case**: "Different sectors need different strategies (e.g., growth for Tech, value for Financials)"

---

### 6. **fast-iteration.json** - Quick Testing

**Purpose**: Fast runs for development/testing  
**When to use**: Code development, quick validation

**Key Settings**:
```json
{
  "data": {
    "min_train_months": 12                // Less training data
  },
  
  "models": {
    "lightgbm": {
      "n_estimators": 100,                // Fewer trees
      "learning_rate": 0.05               // Faster training
    }
  },
  
  "diagnostics": {
    "feature_analysis": false,            // Skip analysis
    "stability_monitoring": false,
    "performance_plots": false
  },
  
  "hyperparameter_tuning": { "enabled": false }
}
```

**Expected Results**:
- Runtime: ~3-5 minutes (much faster!)
- Performance: Similar to baseline but less robust

**Use Case**: "I'm developing code and need quick feedback"

---

### 7. **custom_baseline.json** & **demo_custom.json**

**Purpose**: Template for creating your own configurations  
**When to use**: You want to experiment with custom settings

**How to use**:
1. Copy `baseline.json` to `my_config.json`
2. Modify parameters as needed
3. Run: `python run_all.py --config my_config`

---

## 🔧 How to Choose a Configuration

### Decision Tree

```
START: What's your goal?
│
├─ "I want to see baseline performance"
│  └─> Use: baseline.json
│
├─ "I need to reduce turnover for production"
│  └─> Use: optimized_low_turnover.json
│
├─ "I need VERY low turnover (tax-sensitive)"
│  └─> Use: low-turnover.json
│
├─ "I want maximum predictive power (research)"
│  └─> Use: high-IC.json
│
├─ "I want sector-specific strategies"
│  └─> Use: sector-specific.json
│
├─ "I'm developing/testing code"
│  └─> Use: fast-iteration.json
│
└─ "I want custom settings"
   └─> Copy baseline.json, modify, use custom config
```

### By Use Case

| Use Case | Configuration | Why |
|----------|--------------|-----|
| **First-time user** | `baseline` | Standard settings, good starting point |
| **Production deployment** | `optimized_low_turnover` | Balanced turnover/performance |
| **High transaction costs** | `low-turnover` | Minimizes trading |
| **Research/backtesting** | `high-IC` | Maximum predictive power |
| **Sector rotation strategy** | `sector-specific` | Sector-specific models |
| **Code development** | `fast-iteration` | Quick feedback |
| **Tax-sensitive accounts** | `low-turnover` | Minimizes taxable events |
| **Low-cost trading** | `high-IC` | Can afford high turnover |

---

## 🎮 How to Use Configurations

### Basic Usage

```bash
# Run with baseline configuration (default)
python run_all.py

# Run with specific configuration
python run_all.py --config optimized_low_turnover

# Run with custom configuration
python run_all.py --config my_custom_config
```

### With Additional Options

```bash
# Skip certain stages
python run_all.py --config baseline --skip-leakage --skip-feature-analysis

# Skip hyperparameter tuning (even if enabled in config)
python run_all.py --config high-IC --skip-hyperparameter-tuning
```

### Configuration Loading Process

```python
# In run_all.py
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="baseline")
args = parser.parse_args()

# Load configuration
from config_manager import ConfigManager
config_mgr = ConfigManager(verbose=True)
config = config_mgr.load_config(args.config)  # Loads configs/{args.config}.json

# Access configuration values
EWM_ALPHA = config_mgr.get("turnover.ewm_smoothing.alpha", 0.5)
TOP_PER_SECTOR = config_mgr.get("portfolio.top_n_per_sector", 3)
```

---

## 📊 Configuration Comparison

### Performance Comparison

| Config | Turnover | IC | Sharpe | Runtime | Best For |
|--------|----------|-----|--------|---------|----------|
| **baseline** | 419% | 0.001 | 0.86 | 12 min | General use |
| **optimized_low_turnover** | 301% | 0.007 | 0.96 | 15 min | Production ⭐ |
| **low-turnover** | ~180% | 0.004 | 0.75 | 12 min | Tax-sensitive |
| **high-IC** | 550% | 0.018 | 1.10 | 35 min | Research |
| **sector-specific** | 380% | 0.008 | 0.92 | 25 min | Sector strategies |
| **fast-iteration** | 400% | 0.001 | 0.85 | 5 min | Development |

### Feature Selection Comparison

| Config | # Features | Feature Set |
|--------|-----------|-------------|
| **baseline** | 19 | All features |
| **optimized_low_turnover** | 12 | Best IC features only |
| **high-IC** | 4 | Top 4 features only |
| **Others** | 19 | All features |

### Model Comparison

| Config | Ridge | LightGBM | Ensemble | Sector-Specific |
|--------|-------|----------|----------|-----------------|
| **baseline** | ✓ | ✓ | ✗ | ✗ |
| **optimized_low_turnover** | ✓ | ✓ | ✓ | ✗ |
| **high-IC** | ✓ | ✓ | ✓ | ✓ |
| **sector-specific** | ✓ | ✓ | ✗ | ✓ |

---

## 🔍 Configuration Parameters Explained

### Key Parameters

#### 1. **EWM Alpha** (turnover.ewm_smoothing.alpha)
**What it does**: Controls signal smoothing  
**Range**: 0.0 to 1.0  
**Effect**:
- Low (0.3): More responsive, higher turnover
- Medium (0.5): Balanced
- High (0.7): More stable, lower turnover

**Formula**: `smoothed = alpha × current + (1-alpha) × previous`

#### 2. **Rebalancing Threshold** (turnover.rebalancing_threshold.threshold)
**What it does**: Minimum rank difference to replace a stock  
**Range**: 0.05 to 0.35  
**Effect**:
- Low (0.10): Easy to replace, higher turnover
- Medium (0.15): Balanced
- High (0.25): Hard to replace, lower turnover

**Logic**: Challenger must beat incumbent by at least this much

#### 3. **Holding Period Bonus** (turnover.holding_period_bonus.bonus_rate)
**What it does**: Rank boost per month held  
**Range**: 0.00 to 0.05  
**Effect**:
- None (0.00): No incumbent advantage
- Low (0.02): Slight advantage
- High (0.03): Strong advantage

**Logic**: `adjusted_rank = rank + (months_held × bonus_rate)`

#### 4. **Minimum Holding Period** (turnover.minimum_holding_period.min_months)
**What it does**: Forced holding period  
**Range**: 0 to 6 months  
**Effect**:
- None (0): Can sell anytime
- Short (2): Slight stability
- Long (3-6): Strong stability

**Logic**: Cannot sell stock until held for min_months

#### 5. **Top N per Sector** (portfolio.top_n_per_sector)
**What it does**: Number of stocks per sector  
**Range**: 2 to 6  
**Effect**:
- Narrow (2-3): Concentrated, higher turnover
- Wide (4-5): Diversified, lower turnover

**Total stocks**: top_n × 5 sectors

#### 6. **Feature Selection** (features.use_all_features)
**What it does**: Which features to use  
**Options**:
- `use_all_features: true` → All 19 features
- `use_all_features: false` + `selected_features: [...]` → Custom list

**Effect**: Fewer features = faster, potentially better IC if low-IC features removed

---

## 🎯 Creating Your Own Configuration

### Step 1: Copy Template
```bash
cp configs/baseline.json configs/my_config.json
```

### Step 2: Modify Parameters
```json
{
  "version": "1.0",
  "description": "My custom configuration",
  
  "turnover": {
    "ewm_smoothing": { "alpha": 0.65 },      // Your value
    "rebalancing_threshold": { "threshold": 0.20 },
    "target_turnover": 0.20                  // 20% monthly = 240% annually
  },
  
  "portfolio": {
    "top_n_per_sector": 5                    // 25 stocks total
  }
}
```

### Step 3: Run
```bash
python run_all.py --config my_config
```

### Step 4: Compare Results
```bash
# Compare with baseline
python run_all.py --config baseline
python run_all.py --config my_config

# Check reports/comprehensive_diagnostic_report.txt for both runs
```

---

## 📈 Configuration Best Practices

### 1. **Start with Baseline**
Always run baseline first to establish a benchmark

### 2. **Change One Thing at a Time**
When experimenting, change one parameter at a time to understand its effect

### 3. **Document Your Changes**
Update the "description" field in your config to explain what you changed and why

### 4. **Version Your Configs**
Use version numbers (1.0, 1.1, etc.) to track changes

### 5. **Test Before Production**
Always test new configurations on historical data before deploying

### 6. **Monitor Key Metrics**
Track: Turnover, IC, Sharpe, Runtime

---

## 🚨 Common Mistakes

### ❌ Mistake 1: Conflicting Parameters
```json
{
  "turnover": {
    "ewm_smoothing": { "alpha": 0.9 },      // Very high smoothing
    "rebalancing_threshold": { "threshold": 0.05 }  // Very low barrier
  }
}
```
**Problem**: High smoothing + low barrier = conflicting goals  
**Fix**: Use consistent parameters (both high or both low)

### ❌ Mistake 2: Unrealistic Targets
```json
{
  "turnover": {
    "target_turnover": 0.05  // 5% monthly = 60% annually
  }
}
```
**Problem**: 60% annual turnover is unrealistic for this model  
**Fix**: Set realistic targets (150-250% annually)

### ❌ Mistake 3: Too Few Features
```json
{
  "features": {
    "selected_features": ["RevGrowth_YoY"]  // Only 1 feature!
  }
}
```
**Problem**: Single feature = overfitting, no diversification  
**Fix**: Use at least 5-10 features

### ❌ Mistake 4: Enabling Everything
```json
{
  "models": {
    "ensemble": { "enabled": true },
    "sector_specific": { "enabled": true }
  },
  "hyperparameter_tuning": { "enabled": true }
}
```
**Problem**: Runtime will be 1-2 hours  
**Fix**: Enable advanced features selectively

---

## 📞 FAQ

**Q: Can I use multiple configurations at once?**  
A: No, you choose ONE configuration per run. But you can run the pipeline multiple times with different configs and compare results.

**Q: Which configuration should I use for production?**  
A: Start with `optimized_low_turnover`. It's been tested and provides good balance.

**Q: How do I know if my custom config is working?**  
A: Check `reports/comprehensive_diagnostic_report.txt` after running. Compare turnover, IC, and Sharpe with baseline.

**Q: Can the system automatically choose the best config?**  
A: No, you must explicitly choose. However, you can run all configs and compare results to find the best one for your needs.

**Q: What if I want different settings for different sectors?**  
A: Use `sector-specific.json` which trains separate models per sector. You can also create custom configs with sector-specific parameters.

**Q: How often should I update my configuration?**  
A: Review quarterly. Market conditions change, so parameters that worked before may need adjustment.

---

## 🎓 Summary

**Key Takeaways**:
1. **One config at a time** - You explicitly choose which to use
2. **Different goals, different configs** - Production vs Research vs Development
3. **Start with baseline** - Establish benchmark before experimenting
4. **Recommended for production**: `optimized_low_turnover`
5. **Create custom configs** - Copy and modify existing ones
6. **Monitor results** - Check diagnostic reports after each run

**Quick Reference**:
```bash
# Production
python run_all.py --config optimized_low_turnover

# Research
python run_all.py --config high-IC

# Development
python run_all.py --config fast-iteration

# Custom
python run_all.py --config my_config
```

---

**Last Updated**: May 8, 2026  
**Related Documents**: PROJECT_DOCUMENTATION.md, FINAL_IMPROVEMENT_SUMMARY.md

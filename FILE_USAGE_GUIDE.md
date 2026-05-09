# File Usage Guide - Which Files Are Actually Used?

## 📋 TL;DR - Quick Answer

**YES, there are many files, but they serve different purposes:**

- **Core Pipeline Files** (13 files) - Used by `run_all.py` ✅
- **Test/Demo Files** (40+ files) - For testing and examples ⚠️
- **Documentation Files** (30+ files) - Summaries and guides 📄
- **Configuration Files** (8 files) - Different settings 🔧
- **Output Files** (100+ files) - Generated results 📊

**Most files are NOT used in the main pipeline** - they're for testing, examples, or documentation.

---

## 🎯 Core Pipeline Files (ACTUALLY USED)

These **13 files** are used when you run `python run_all.py`:

### 1. **run_all.py** ⭐ **MAIN ORCHESTRATOR**
**Purpose**: Master pipeline that runs everything  
**Used**: YES - This is what you run  
**Dependencies**: Calls all other core files

```python
# What it does:
python run_all.py --config baseline

# Internally calls:
# 1. config_manager.py
# 2. leakage_detector.py
# 3. data_loader.py
# 4. feature_analyzer.py
# 5. model.py
# 6. shap_explainability.py
# 7. hyperparameter_tuner.py (optional)
# 8. ensemble_model.py (optional)
# 9. sector_model.py (optional)
# 10. early_stopping_regularization.py
# 11. sector_neutralisation.py
# 12. oot_validation.py
# 13. transaction_costs.py
```

### 2. **config_manager.py** ⭐
**Purpose**: Loads and manages configuration files  
**Used**: YES - Always used  
**Called by**: run_all.py (first thing)

```python
# Loads configs/baseline.json or other configs
config_mgr = ConfigManager()
config = config_mgr.load_config("baseline")
```

### 3. **data_loader.py** ⭐
**Purpose**: Downloads data and engineers 19 features  
**Used**: YES - Always used  
**Called by**: run_all.py (Stage 2)

```python
# Downloads price data, fundamentals, computes features
prices = download_price_data()
factors_df = compute_factors(monthly_returns, prices, fund_df)
```

### 4. **leakage_detector.py** ⭐
**Purpose**: Detects data leakage (future data in training)  
**Used**: YES - Unless you use `--skip-leakage`  
**Called by**: run_all.py (Stage 1)

```python
# Validates no future data is used
detector = LeakageDetector()
detector.validate_temporal_boundaries(factors_df)
```

### 5. **feature_analyzer.py** ⭐
**Purpose**: Analyzes feature quality (IC, correlation, stability)  
**Used**: YES - Unless you use `--skip-feature-analysis`  
**Called by**: run_all.py (Stage 3)

```python
# Generates feature quality reports
feature_report = generate_feature_quality_report(factors_df)
```

### 6. **model.py** ⭐
**Purpose**: Ridge regression baseline model  
**Used**: YES - Always used  
**Called by**: run_all.py (Stage 4)

```python
# Trains Ridge model with walk-forward validation
ridge_results, coef_df = walk_forward_validation(factors_df)
```

### 7. **shap_explainability.py** ⭐
**Purpose**: LightGBM model + SHAP feature importance  
**Used**: YES - Always used  
**Called by**: run_all.py (Stage 5)

```python
# Trains LightGBM and generates SHAP values
lgbm_results, shap_df = walk_forward_lgbm(factors_df)
```

### 8. **hyperparameter_tuner.py**
**Purpose**: Optimizes model hyperparameters  
**Used**: OPTIONAL - Only if enabled in config  
**Called by**: run_all.py (Stage 6)

```python
# Only runs if config has: "hyperparameter_tuning": {"enabled": true}
if ENABLE_HYPERPARAMETER_TUNING:
    ridge_tuning_results = tune_ridge_hyperparameters(factors_df)
```

### 9. **ensemble_model.py**
**Purpose**: Combines Ridge + LightGBM predictions  
**Used**: OPTIONAL - Only if enabled in config  
**Called by**: run_all.py (Stage 7)

```python
# Only runs if config has: "ensemble": {"enabled": true}
if ENABLE_ENSEMBLE:
    ensemble_results = walk_forward_ensemble(ridge_results, lgbm_results)
```

### 10. **sector_model.py**
**Purpose**: Trains separate models per sector  
**Used**: OPTIONAL - Only if enabled in config  
**Called by**: run_all.py (Stage 8)

```python
# Only runs if config has: "sector_specific": {"enabled": true}
if ENABLE_SECTOR_SPECIFIC:
    sector_results = train_sector_specific_models(factors_df)
```

### 11. **early_stopping_regularization.py** ⭐
**Purpose**: Analyzes model complexity vs performance  
**Used**: YES - Always used  
**Called by**: run_all.py (Stage 9)

```python
# Analyzes LightGBM complexity
analyzer = ModelComplexityAnalyzer()
lgbm_complexity = analyzer.analyze_lgbm_complexity(train_df)
```

### 12. **sector_neutralisation.py** ⭐
**Purpose**: Builds sector-diversified portfolio with turnover controls  
**Used**: YES - Always used  
**Called by**: run_all.py (Stage 10)

```python
# Builds portfolio with EWM smoothing, thresholds, holding bonuses
sector_results = walk_forward_sector_neutral(factors_df_sector)
sector_port = build_sector_aware_portfolio(sector_results)
```

### 13. **oot_validation.py** ⭐
**Purpose**: Out-of-time validation on unseen data  
**Used**: YES - Always used  
**Called by**: run_all.py (Stage 11)

```python
# Tests model on future data not used in training
oot_metrics = run_oot_validation(factors_df_sector, sector_results_full)
```

### 14. **transaction_costs.py** ⭐
**Purpose**: Models transaction costs and turnover  
**Used**: YES - Always used  
**Called by**: run_all.py (Stage 12)

```python
# Calculates turnover and applies transaction costs
turnover_df = compute_turnover(SECTOR_PRED)
plot_cost_scenarios(PORTFOLIO, turnover_df)
```

### 15. **stability_monitor.py**
**Purpose**: Monitors IC stability and regime detection  
**Used**: OPTIONAL - If enabled in config  
**Called by**: run_all.py (Stage 13, if enabled)

```python
# Only runs if config has: "stability_monitoring": true
if ENABLE_STABILITY_MONITORING:
    stability_report = generate_stability_report(results_df)
```

### 16. **turnover_optimizer.py**
**Purpose**: Turnover optimization utilities  
**Used**: INDIRECTLY - Used by sector_neutralisation.py  
**Called by**: sector_neutralisation.py

```python
# Provides turnover calculation functions
from turnover_optimizer import TurnoverOptimizer
```

---

## ⚠️ Test/Demo Files (NOT USED IN MAIN PIPELINE)

These **40+ files** are for testing, examples, and validation. They are **NOT called by run_all.py**.

### Demo Files (5 files) - Examples/Tutorials
**Purpose**: Show how to use specific features  
**Used**: NO - Run manually for learning

- `demo_feature_engineering.py` - Example of feature engineering
- `demo_holding_period_integration.py` - Example of holding period logic
- `demo_sector_model.py` - Example of sector-specific models
- `demo_stability_monitoring.py` - Example of stability monitoring
- `demo_turnover_monitoring.py` - Example of turnover monitoring

**How to use**: Run manually to learn
```bash
python demo_feature_engineering.py  # Learn about feature engineering
```

### Example Files (3 files) - Optimization Examples
**Purpose**: Show how to optimize parameters  
**Used**: NO - Run manually for optimization

- `example_ewm_integration.py` - EWM parameter optimization
- `example_holding_period_optimization.py` - Holding period optimization
- `example_threshold_optimization.py` - Threshold optimization

**How to use**: Run manually to find optimal parameters
```bash
python example_ewm_integration.py  # Find optimal EWM alpha
```

### Quick Test Files (3 files) - Quick Validation
**Purpose**: Quick tests of specific features  
**Used**: NO - Run manually for quick checks

- `quick_ewm_validation.py` - Quick EWM test
- `quick_holding_period_test.py` - Quick holding period test
- `quick_threshold_test.py` - Quick threshold test

### Test Files (30+ files) - Unit/Integration Tests
**Purpose**: Automated testing of individual components  
**Used**: NO - Run manually or in CI/CD

All files starting with `test_*`:
- `test_comprehensive_leakage_audit.py`
- `test_cross_sectional_imputation_validation.py`
- `test_early_stopping_regularization.py`
- `test_ensemble_model.py`
- `test_ewm_optimization.py`
- `test_ewm_simple.py`
- `test_ewm_smoothing_leakage_validation.py`
- `test_ewm_validation_simple.py`
- `test_ewm_with_sector_neutralization.py`
- `test_feature_analyzer.py`
- `test_feature_engineering_integration.py`
- `test_feature_engineering.py`
- `test_feature_quality_report.py`
- `test_fundamental_lag_validation.py`
- `test_holding_period_unit.py`
- `test_hyperparameter_tuner.py`
- `test_integration.py`
- `test_optimized_turnover.py` ⭐ (Used for optimization testing)
- `test_regime_detection.py`
- `test_sector_model_simple.py`
- `test_sector_model.py`
- `test_stability_monitor.py`
- `test_target_variable_leakage_validation.py`
- `test_task_1_6_final.py`
- `test_threshold_optimization.py`
- `test_turnover_monitoring.py`
- `test_turnover_optimizer.py`
- `test_ultra_low_turnover.py` ⭐ (Used for optimization testing)

**How to use**: Run manually to test specific components
```bash
python test_feature_analyzer.py  # Test feature analyzer
```

### Validation Files (3 files) - Validation Scripts
**Purpose**: Validate specific improvements  
**Used**: NO - Run manually to validate

- `validate_model_improvements.py`
- `validate_reports.py`
- `validate_sector_model.py`

### Run Files (1 file) - Alternative Runners
**Purpose**: Alternative ways to run the pipeline  
**Used**: NO - Alternative to run_all.py

- `run_ensemble_tests.py` - Runs only ensemble tests

---

## 📄 Documentation Files (NOT USED IN PIPELINE)

These **30+ files** are documentation, summaries, and guides. They are **NOT code**.

### Main Documentation (5 files)
- `COMPREHENSIVE_PROJECT_DOCUMENTATION.md` - Empty placeholder
- `PROJECT_DOCUMENTATION.md` - Main technical documentation
- `FINAL_IMPROVEMENT_SUMMARY.md` - Improvement analysis
- `IMPROVEMENT_RESULTS.md` - Detailed improvement results
- `QUICK_RESULTS.txt` - Executive summary

### Guides (2 files)
- `CONFIGURATION_GUIDE.md` - Configuration system guide
- `DOCUMENTATION_INDEX.md` - Index of all documentation
- `FILE_USAGE_GUIDE.md` - This file!

### Task Summaries (20+ files)
All files starting with `task_*`:
- `task_1_2_implementation_summary.md`
- `task_1_3_cross_sectional_imputation_validation_summary.md`
- `task_1_4_ewm_smoothing_leakage_validation_summary.md`
- `task_1_5_target_variable_leakage_validation_summary.md`
- `task_1_6_comprehensive_leakage_audit_report_summary.md`
- `task_12_1_integration_summary.md`
- `task_3_1_feature_analyzer_summary.md`
- `task_3_2_feature_quality_metrics_summary.md`
- `task_3_3_feature_engineering_summary.md`
- `task_3_4_feature_quality_report_summary.md`
- `task_4_1_hyperparameter_tuner_summary.md`
- `task_4_2_ensemble_methods_summary.md`
- `task_4_3_sector_specific_modeling_summary.md`
- `task_4_5_early_stopping_regularization_summary.md`
- `task_5_validation_summary.md`
- `task_6_2_ewm_optimization_summary.md`
- `task_6_3_threshold_optimization_summary.md`
- `task_6_4_holding_period_implementation_summary.md`
- `task_6_5_summary.md`

**Purpose**: Document implementation of each task  
**Used**: NO - For reference only

---

## 🔧 Configuration Files (USED)

These **8 files** define different parameter sets. **ONE is used per run**.

- `configs/baseline.json` ⭐ - Default configuration
- `configs/optimized_low_turnover.json` ⭐⭐⭐ - Optimized for production
- `configs/low-turnover.json` - Ultra-low turnover
- `configs/high-IC.json` - High predictive power
- `configs/sector-specific.json` - Sector-specific models
- `configs/fast-iteration.json` - Quick testing
- `configs/custom_baseline.json` - Custom template
- `configs/demo_custom.json` - Demo template

**How to use**: Choose one when running
```bash
python run_all.py --config optimized_low_turnover
```

---

## 📊 Output Files (GENERATED)

These **100+ files** are generated by the pipeline. They are **NOT code**.

### Data Files (in `data/` folder)
- `daily_prices.parquet` - Cached price data
- `fundamentals.parquet` - Cached fundamental data
- `monthly_returns.csv` - Monthly returns
- `factor_features.csv` - Engineered features
- `ridge_predictions.csv` - Ridge predictions
- `lgbm_predictions.csv` - LightGBM predictions
- `sector_predictions.csv` - Sector-neutral predictions
- `sector_predictions_optimized.csv` - Optimized predictions
- `shap_values.csv` - SHAP feature importance
- `ensemble_predictions.csv` - Ensemble predictions (if enabled)
- `ensemble_weights.csv` - Ensemble weights (if enabled)

### Report Files (in `reports/` folder)
- `comprehensive_diagnostic_report.txt` - Main diagnostic report
- `comprehensive_diagnostic_report.json` - Machine-readable diagnostics
- `feature_ic_summary.csv` - Feature IC statistics
- `feature_ic_monthly.csv` - Monthly IC by feature
- `feature_stability.csv` - Feature stability metrics
- `feature_coverage.csv` - Feature coverage analysis
- `feature_ranking.csv` - Overall feature ranking
- `feature_recommendations.txt` - Feature recommendations
- `feature_correlation_matrix.csv` - Feature correlations
- `monthly_turnover.csv` - Monthly turnover analysis
- `turnover_optimized.csv` - Optimized turnover analysis
- `oot_validation_report.csv` - Out-of-time validation
- `cost_adjusted_returns.csv` - Transaction cost analysis
- `leakage_audit_report.txt` - Leakage detection report
- `sector_portfolio.csv` - Portfolio holdings
- And 50+ more...

### Output Files (in `outputs/` folder)
- Various PNG charts and visualizations

---

## 🎯 What You Actually Need to Run

### Minimum Required Files (Core Pipeline)

To run the basic pipeline, you need these **16 files**:

```
run_all.py                      # Main orchestrator
config_manager.py               # Configuration loader
data_loader.py                  # Data download & features
leakage_detector.py             # Leakage detection
feature_analyzer.py             # Feature analysis
model.py                        # Ridge model
shap_explainability.py          # LightGBM model
hyperparameter_tuner.py         # Hyperparameter tuning (optional)
ensemble_model.py               # Ensemble (optional)
sector_model.py                 # Sector models (optional)
early_stopping_regularization.py # Regularization analysis
sector_neutralisation.py        # Portfolio construction
oot_validation.py               # OOT validation
transaction_costs.py            # Transaction costs
stability_monitor.py            # Stability monitoring (optional)
turnover_optimizer.py           # Turnover utilities

+ configs/baseline.json         # At least one config file
```

**Everything else is optional** (tests, demos, documentation, examples).

---

## 📋 File Categories Summary

| Category | Count | Used in Pipeline? | Purpose |
|----------|-------|-------------------|---------|
| **Core Pipeline** | 16 | ✅ YES | Main functionality |
| **Test Files** | 30+ | ❌ NO | Testing components |
| **Demo Files** | 5 | ❌ NO | Learning examples |
| **Example Files** | 3 | ❌ NO | Optimization examples |
| **Quick Test Files** | 3 | ❌ NO | Quick validation |
| **Validation Files** | 3 | ❌ NO | Validation scripts |
| **Documentation** | 30+ | ❌ NO | Guides and summaries |
| **Configuration** | 8 | ✅ ONE per run | Parameter sets |
| **Output Files** | 100+ | ✅ Generated | Results |

**Total Files**: ~200  
**Actually Used in Pipeline**: ~16 core + 1 config = **17 files**  
**Everything Else**: Tests, demos, documentation, examples

---

## 🤔 Why So Many Files?

### 1. **Modular Design**
Each file has a specific purpose. This makes the code:
- Easier to understand
- Easier to test
- Easier to modify
- Easier to maintain

### 2. **Testing & Validation**
The 30+ test files ensure each component works correctly:
- Unit tests (test individual functions)
- Integration tests (test components together)
- Validation tests (validate improvements)

### 3. **Examples & Learning**
The demo and example files help you:
- Learn how to use specific features
- Optimize parameters
- Understand the system

### 4. **Documentation**
The 30+ documentation files provide:
- Technical explanations
- Market logic
- Implementation details
- Usage guides

### 5. **Flexibility**
Multiple configuration files allow:
- Different use cases (production, research, development)
- Different parameter sets
- Easy experimentation

---

## 🎯 What Should You Focus On?

### If You're a User (Just Want to Run the Model):
**Focus on these 3 files**:
1. `run_all.py` - Run this
2. `configs/optimized_low_turnover.json` - Use this config
3. `reports/comprehensive_diagnostic_report.txt` - Read results

**Ignore**: All test files, demo files, documentation files

### If You're a Developer (Want to Modify the Code):
**Focus on these 16 core files**:
1. `run_all.py` - Main orchestrator
2. `data_loader.py` - Data & features
3. `model.py` - Ridge model
4. `shap_explainability.py` - LightGBM model
5. `sector_neutralisation.py` - Portfolio construction
6. `oot_validation.py` - Validation
7. `transaction_costs.py` - Costs
8. Plus the other core files as needed

**Use**: Test files to validate your changes

### If You're a Researcher (Want to Understand the System):
**Focus on documentation**:
1. `PROJECT_DOCUMENTATION.md` - Technical details
2. `FINAL_IMPROVEMENT_SUMMARY.md` - Results analysis
3. `CONFIGURATION_GUIDE.md` - Configuration system
4. Task summaries - Implementation details

**Run**: Demo and example files to learn

---

## 🧹 Can You Delete Files?

### Safe to Delete (Won't Break Pipeline):
- All `test_*.py` files (30+ files)
- All `demo_*.py` files (5 files)
- All `example_*.py` files (3 files)
- All `quick_*.py` files (3 files)
- All `validate_*.py` files (3 files)
- All `task_*.md` files (20+ files)
- All documentation `.md` files except guides
- `run_ensemble_tests.py`

**Total**: ~70 files can be deleted without affecting the main pipeline

### DO NOT Delete (Will Break Pipeline):
- `run_all.py` ⭐
- `config_manager.py`
- `data_loader.py`
- `leakage_detector.py`
- `feature_analyzer.py`
- `model.py`
- `shap_explainability.py`
- `hyperparameter_tuner.py`
- `ensemble_model.py`
- `sector_model.py`
- `early_stopping_regularization.py`
- `sector_neutralisation.py`
- `oot_validation.py`
- `transaction_costs.py`
- `stability_monitor.py`
- `turnover_optimizer.py`
- `configs/*.json` (at least one)

**Total**: 16 core files + configs must be kept

---

## 📊 Visual Summary

```
Project Files (~200 total)
│
├─ Core Pipeline (16 files) ✅ USED
│  ├─ run_all.py (orchestrator)
│  ├─ config_manager.py
│  ├─ data_loader.py
│  ├─ leakage_detector.py
│  ├─ feature_analyzer.py
│  ├─ model.py
│  ├─ shap_explainability.py
│  ├─ hyperparameter_tuner.py (optional)
│  ├─ ensemble_model.py (optional)
│  ├─ sector_model.py (optional)
│  ├─ early_stopping_regularization.py
│  ├─ sector_neutralisation.py
│  ├─ oot_validation.py
│  ├─ transaction_costs.py
│  ├─ stability_monitor.py (optional)
│  └─ turnover_optimizer.py
│
├─ Configuration (8 files) ✅ ONE per run
│  └─ configs/*.json
│
├─ Test Files (30+ files) ❌ NOT USED
│  └─ test_*.py
│
├─ Demo Files (5 files) ❌ NOT USED
│  └─ demo_*.py
│
├─ Example Files (3 files) ❌ NOT USED
│  └─ example_*.py
│
├─ Quick Test Files (3 files) ❌ NOT USED
│  └─ quick_*.py
│
├─ Validation Files (3 files) ❌ NOT USED
│  └─ validate_*.py
│
├─ Documentation (30+ files) ❌ NOT CODE
│  ├─ *.md (guides)
│  └─ task_*.md (summaries)
│
└─ Output Files (100+ files) ✅ GENERATED
   ├─ data/*.csv, *.parquet
   ├─ reports/*.csv, *.txt
   └─ outputs/*.png
```

---

## 🎓 Summary

**Key Takeaways**:

1. **Only 16 core files are used** in the main pipeline
2. **40+ test/demo files** are for testing and learning (not used in pipeline)
3. **30+ documentation files** are guides and summaries (not code)
4. **8 configuration files** - you choose ONE per run
5. **100+ output files** are generated results

**To run the model, you only need**:
```bash
python run_all.py --config optimized_low_turnover
```

This uses 16 core files + 1 config file = **17 files total**.

**Everything else** (180+ files) is for:
- Testing
- Examples
- Documentation
- Validation
- Generated outputs

**Bottom Line**: The project looks complex because it includes comprehensive testing, documentation, and examples. But the actual pipeline is just 16 core files!

---

**Last Updated**: May 8, 2026  
**Related Documents**: DOCUMENTATION_INDEX.md, PROJECT_DOCUMENTATION.md

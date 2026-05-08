# Design Document: Model Performance Improvement

## Overview

This design addresses critical issues in the equity factor model: potential data leakage (OOT IC 0.0334 > in-sample IC 0.0002), high turnover (397% annually), and inconsistent monthly performance. The solution comprises six integrated systems that work together to detect and prevent leakage, improve feature quality, optimize model architecture, reduce turnover, enhance stability, and provide comprehensive diagnostics.

### Current System Analysis

The existing codebase implements a 250-stock equity factor model with:
- **Data Pipeline** (`data_loader.py`): Downloads price/fundamental data, engineers 19 features across 8 groups
- **Ridge Baseline** (`model.py`): Walk-forward validation with expanding windows
- **LightGBM + SHAP** (`shap_explainability.py`): Non-linear model with feature attribution
- **Sector Neutralization** (`sector_neutralisation.py`): Top-3 per sector portfolio with EWM smoothing
- **OOT Validation** (`oot_validation.py`): Held-out period testing (Jan 2025+)
- **Transaction Costs** (`transaction_costs.py`): Cost modeling and turnover analysis

### Problem Statement

The model exhibits three critical issues:

1. **Suspected Data Leakage**: OOT performance significantly exceeds in-sample performance, violating the fundamental expectation that models perform worse on unseen data
2. **High Turnover**: 397% annual turnover generates excessive transaction costs that erode alpha
3. **Inconsistent Performance**: Monthly IC varies widely, indicating instability across market regimes

### Design Goals

1. **Eliminate Data Leakage**: Implement comprehensive detection and prevention mechanisms
2. **Improve Feature Quality**: Enhance predictive power through better feature engineering and selection
3. **Optimize Model Architecture**: Tune hyperparameters and ensemble methods without overfitting
4. **Reduce Turnover**: Cut annual turnover below 200% while maintaining IC
5. **Enhance Stability**: Achieve consistent positive monthly IC across market conditions
6. **Provide Diagnostics**: Enable rapid identification and debugging of model issues

## Architecture

### System Components

```mermaid
graph TB
    subgraph "Data Layer"
        A[Raw Data] --> B[Feature Engineering]
        B --> C[Leakage Detector]
        C --> D[Clean Features]
    end
    
    subgraph "Analysis Layer"
        D --> E[Feature Analyzer]
        E --> F[Feature Selection]
        F --> G[Model Training]
    end
    
    subgraph "Model Layer"
        G --> H[Hyperparameter Tuner]
        H --> I[Ensemble Builder]
        I --> J[Predictions]
    end
    
    subgraph "Portfolio Layer"
        J --> K[Turnover Optimizer]
        K --> L[Portfolio Constructor]
        L --> M[Portfolio Returns]
    end
    
    subgraph "Monitoring Layer"
        M --> N[Stability Monitor]
        N --> O[Diagnostic Framework]
        O --> P[Reports & Alerts]
    end
    
    C -.feedback.-> B
    E -.feedback.-> B
    N -.feedback.-> H
    O -.feedback.-> E

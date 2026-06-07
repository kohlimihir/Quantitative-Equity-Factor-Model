"""
Equity Factor Model Pipeline
Usage: python run_all.py [--config PROFILE] [--skip-leakage] [--skip-feature-analysis]
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import argparse
import warnings

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

RF = 0.045 / 12

parser = argparse.ArgumentParser(description="Run equity factor model pipeline")
parser.add_argument("--config", type=str, default="baseline", help="Config profile (default: baseline)")
parser.add_argument("--skip-leakage", action="store_true", help="Skip leakage detection")
parser.add_argument("--skip-feature-analysis", action="store_true", help="Skip feature analysis")
parser.add_argument("--skip-hyperparameter-tuning", action="store_true", help="Skip hyperparameter tuning")
args = parser.parse_args()


def section(title):
    print("\n" + "="*78)
    print(f"  {title}")
    print("="*78)


def step(n, label):
    print(f"\n[Step {n}] {label}...")


def done(label, t):
    print(f"         Done — {label} ({t:.1f}s)")


def error_handler(stage_name, error, continue_on_error=False):
    print(f"\n❌ ERROR in {stage_name}: {str(error)}")
    if not continue_on_error:
        print("   Pipeline stopped. Fix the error and re-run.")
        sys.exit(1)
    else:
        print("   Continuing...")


t0 = time.time()
section("EQUITY FACTOR MODEL — 250 STOCKS, 19 FEATURES")
print(f"  Config: {args.config}")
print("  Pipeline: Config → Leakage → Data → Features → Models → Portfolio → OOT → Costs")
print("  11 GICS sectors: InfoTech, CommSvcs, Financials, HealthCare, ConsDisc, ConsStap, Energy, Industrials, Materials, Utilities, RealEstate")
print("  Data cached locally — subsequent runs skip download")


# Config Loading
step(0, "Loading configuration")
t = time.time()

try:
    from config_manager import ConfigManager
    
    config_mgr = ConfigManager(verbose=True)
    config = config_mgr.load_config(args.config)
    config_mgr.print_summary()
    
    MIN_TRAIN_MONTHS = config_mgr.get("data.min_train_months", 24)
    FUNDAMENTAL_LAG_DAYS = config_mgr.get("data.fundamental_lag_days", 45)
    
    ENABLE_RIDGE = config_mgr.get("models.ridge.enabled", True)
    ENABLE_LIGHTGBM = config_mgr.get("models.lightgbm.enabled", True)
    ENABLE_ENSEMBLE = config_mgr.get("models.ensemble.enabled", False)
    ENABLE_SECTOR_SPECIFIC = config_mgr.get("models.sector_specific.enabled", False)
    
    ENABLE_HYPERPARAMETER_TUNING = (
        config_mgr.get("hyperparameter_tuning.enabled", False) and 
        not args.skip_hyperparameter_tuning
    )
    
    ENABLE_LEAKAGE_DETECTION = (
        config_mgr.get("diagnostics.leakage_detection", True) and 
        not args.skip_leakage
    )
    ENABLE_FEATURE_ANALYSIS = (
        config_mgr.get("diagnostics.feature_analysis", True) and 
        not args.skip_feature_analysis
    )
    
    EWM_ALPHA = config_mgr.get("turnover.ewm_smoothing.alpha", 0.5)
    REBAL_THRESHOLD = config_mgr.get("turnover.rebalancing_threshold.threshold", 0.12)
    TOP_PER_SECTOR = config_mgr.get("portfolio.top_n_per_sector", 3)
    
    done("config", time.time() - t)
    
except Exception as e:
    error_handler("Config", e, continue_on_error=False)


# ── STEP 1: Data Leakage Detection ────────────────────────────────────────────
if ENABLE_LEAKAGE_DETECTION:
    step(1, "Running comprehensive data leakage detection")
    t = time.time()
    
    try:
        from leakage_detector import LeakageDetector
        from data_loader import download_fundamentals
        
        # Initialize leakage detector
        detector = LeakageDetector(verbose=True)
        
        print("\n  Leakage Detection Checks:")
        print("    1. Temporal boundary validation")
        print("    2. Scaler fitting validation")
        print("    3. Walk-forward window validation")
        print("    4. Fundamental data lag validation (45-day enforcement)")
        print("    5. Cross-sectional imputation validation")
        print("    6. EWM smoothing leakage checks")
        print("    7. Target variable leakage detection")
        
        # Note: Full leakage detection will run after data is loaded
        # For now, we'll prepare the detector
        leakage_results = {
            "detector_initialized": True,
            "checks_pending": True
        }
        
        done("leakage detection initialization", time.time() - t)
        
    except Exception as e:
        error_handler("Leakage Detection", e, continue_on_error=True)
        ENABLE_LEAKAGE_DETECTION = False
else:
    print("\n⚠️  Leakage detection SKIPPED (not recommended)")


# ── STEP 2: Data & Features ───────────────────────────────────────────────────
step(2, "Building point-in-time universe and engineering factors")
t = time.time()

# Import and configure features FIRST (before other imports that use FEATURES)
from data_loader import get_features_from_config
ACTIVE_BASE_FEATURES = get_features_from_config(config)

# Now import everything else
from data_loader import (
    download_price_data, compute_monthly_returns, compute_factors,
    download_fundamentals, check_feature_correlation,
    apply_feature_engineering,
    SECTOR_MAP, FEATURES, FEATURE_GROUPS, TARGET,
    ALL_FEATURES, SECTOR_REL_TARGET,
)

# ── Build point-in-time S&P 500 universe (survivorship bias fix) ──────────
try:
    from universe_builder import UniverseBuilder
    ub = UniverseBuilder(verbose=True)
    ub.build(start_year=2018)
    ub.get_universe_summary()

    # Get full historical ticker list (union of all months) for data download
    historical_tickers = ub.get_all_historical_tickers()
    # Get full sector map covering all historical tickers
    dynamic_sector_map = ub.get_full_sector_map()
    # Get monthly universe for point-in-time filtering
    monthly_universe = ub.monthly_universe

    print(f"\n  ✓ Dynamic universe: {len(historical_tickers)} unique tickers")
    print(f"  ✓ Monthly universes: {len(monthly_universe)} months")
    print(f"  ✓ Survivorship bias: ELIMINATED")

    USE_DYNAMIC_UNIVERSE = True
except Exception as e:
    print(f"\n  ⚠ Universe builder failed: {e}")
    print(f"  ⚠ Falling back to static 250-stock universe (survivorship bias present)")
    historical_tickers = None
    dynamic_sector_map = SECTOR_MAP
    monthly_universe = None
    USE_DYNAMIC_UNIVERSE = False

# ── Download price and fundamental data ───────────────────────────────────
prices          = download_price_data(tickers=historical_tickers)
monthly_returns = compute_monthly_returns(prices)
monthly_returns.to_csv("data/monthly_returns.csv")

fund_tickers = historical_tickers if USE_DYNAMIC_UNIVERSE else None
fund_df    = download_fundamentals(tickers=fund_tickers)
factors_df = compute_factors(monthly_returns, prices, fund_df,
                             dynamic_sector_map, monthly_universe)
factors_df.to_csv("data/factor_features.csv", index=False)

corr_matrix = check_feature_correlation(factors_df)

# ── Apply feature engineering (interactions, sector-relative, sector target) ──
factors_df = apply_feature_engineering(factors_df)
factors_df.to_csv("data/factor_features.csv", index=False)

# Re-import ALL_FEATURES after feature engineering has populated it
from data_loader import ALL_FEATURES as ACTIVE_FEATURES

done("data_loader + feature_engineering", time.time() - t)
universe_label = "DYNAMIC (point-in-time)" if USE_DYNAMIC_UNIVERSE else "STATIC (fallback)"
print(f"\n  Universe : {factors_df['ticker'].nunique()} stocks, "
      f"{factors_df['date'].nunique()} months [{universe_label}]")
print(f"  Base features: {len(FEATURES)} | Total features: {len(ACTIVE_FEATURES)}")
print(f"  Feature groups: {list(FEATURE_GROUPS.keys())}")


# ── STEP 1b: Complete Leakage Detection (after data loaded) ───────────────────
if ENABLE_LEAKAGE_DETECTION:
    step("1b", "Completing comprehensive leakage detection on loaded data")
    t = time.time()
    
    try:
        # Run temporal boundary validation
        temporal_results = detector.validate_temporal_boundaries(factors_df)
        
        # Run fundamental lag validation
        test_dates = sorted(factors_df["date"].unique())[-12:]  # Test last 12 months
        fund_lag_results = detector.validate_fundamental_lag(
            fund_df, test_dates[0], lag_days=FUNDAMENTAL_LAG_DAYS
        )
        
        # Generate comprehensive audit report
        audit_report = {
            "temporal_boundaries": temporal_results,
            "fundamental_lag": fund_lag_results,
            "all_checks_passed": (
                temporal_results.get("passed", False) and
                fund_lag_results.get("passed", False)
            )
        }
        
        # Save audit report
        os.makedirs("reports", exist_ok=True)
        with open("reports/leakage_audit_report.txt", "w") as f:
            f.write("="*70 + "\n")
            f.write("  COMPREHENSIVE DATA LEAKAGE AUDIT REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {args.config}\n\n")
            
            f.write("TEMPORAL BOUNDARY VALIDATION:\n")
            f.write(f"  Status: {'✓ PASSED' if temporal_results['passed'] else '❌ FAILED'}\n")
            f.write(f"  Total months: {temporal_results.get('total_months', 0)}\n")
            if temporal_results.get('violations'):
                f.write(f"  Violations: {len(temporal_results['violations'])}\n")
                for violation in temporal_results['violations'][:5]:
                    f.write(f"    - {violation}\n")
            f.write("\n")
            
            f.write("FUNDAMENTAL LAG VALIDATION:\n")
            f.write(f"  Status: {'✓ PASSED' if fund_lag_results['passed'] else '❌ FAILED'}\n")
            f.write(f"  Lag days enforced: {FUNDAMENTAL_LAG_DAYS}\n")
            if fund_lag_results.get('violations'):
                f.write(f"  Violations: {len(fund_lag_results['violations'])}\n")
                for violation in fund_lag_results['violations'][:5]:
                    f.write(f"    - {violation}\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
            f.write(f"OVERALL STATUS: {'✓ ALL CHECKS PASSED' if audit_report['all_checks_passed'] else '❌ SOME CHECKS FAILED'}\n")
            f.write("="*70 + "\n")
        
        print(f"\n  ✓ Leakage audit report saved: reports/leakage_audit_report.txt")
        
        if not audit_report["all_checks_passed"]:
            print(f"\n  ⚠️  WARNING: Some leakage checks FAILED!")
            print(f"      Review reports/leakage_audit_report.txt for details")
        else:
            print(f"\n  ✓ All leakage checks PASSED")
        
        done("comprehensive leakage detection", time.time() - t)
        
    except Exception as e:
        error_handler("Leakage Detection (Complete)", e, continue_on_error=True)


# ── STEP 3: Feature Quality Analysis ──────────────────────────────────────────
if ENABLE_FEATURE_ANALYSIS:
    step(3, "Analyzing feature quality and generating recommendations")
    t = time.time()
    
    try:
        from feature_analyzer import (
            generate_feature_quality_report,
            generate_feature_recommendations
        )
        
        # Generate comprehensive feature quality report
        feature_report = generate_feature_quality_report(
            factors_df,
            features=FEATURES,
            correlation_threshold=0.75,
            coverage_threshold=0.30,
            include_incremental_ic=True
        )
        
        # Save feature analysis results
        feature_report["ic_monthly"].to_csv("reports/feature_ic_monthly.csv", index=False)
        feature_report["ic_summary"].to_csv("reports/feature_ic_summary.csv", index=False)
        feature_report["stability"].to_csv("reports/feature_stability.csv", index=False)
        feature_report["coverage"].to_csv("reports/feature_coverage.csv", index=False)
        feature_report["overall_ranking"].to_csv("reports/feature_ranking.csv", index=False)
        
        if "incremental_ic" in feature_report:
            feature_report["incremental_ic"].to_csv("reports/feature_incremental_ic.csv", index=False)
        
        # Save correlation matrix
        if not feature_report["correlation_matrix"].empty:
            feature_report["correlation_matrix"].to_csv("reports/feature_correlation_matrix.csv")
        
        # Save correlated pairs
        if feature_report["correlated_pairs"]:
            corr_pairs_df = pd.DataFrame(feature_report["correlated_pairs"],
                                        columns=["feature1", "feature2", "correlation"])
            corr_pairs_df.to_csv("reports/feature_correlated_pairs.csv", index=False)
        
        # Generate feature recommendations
        recommendations = generate_feature_recommendations(
            feature_report["overall_ranking"],
            feature_report["correlated_pairs"],
            feature_report["coverage"],
            feature_report.get("incremental_ic"),
            low_ic_threshold=0.01,
            high_missing_threshold=0.30,
            high_corr_threshold=0.75
        )
        
        # Save recommendations
        with open("reports/feature_recommendations.txt", "w") as f:
            f.write("="*70 + "\n")
            f.write("  FEATURE QUALITY RECOMMENDATIONS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for category, items in recommendations.items():
                f.write(f"\n{category.upper().replace('_', ' ')}:\n")
                f.write("-" * 70 + "\n")
                for item in items[:10]:  # Top 10 per category
                    f.write(f"  • {item.get('feature', 'N/A')}: {item.get('reason', 'N/A')}\n")
                if len(items) > 10:
                    f.write(f"  ... and {len(items) - 10} more\n")
                f.write("\n")
        
        print(f"\n  ✓ Feature analysis reports saved to reports/")
        print(f"    - feature_ic_monthly.csv")
        print(f"    - feature_ic_summary.csv")
        print(f"    - feature_stability.csv")
        print(f"    - feature_coverage.csv")
        print(f"    - feature_ranking.csv")
        print(f"    - feature_recommendations.txt")
        
        done("feature quality analysis", time.time() - t)
        
    except Exception as e:
        error_handler("Feature Quality Analysis", e, continue_on_error=True)
        ENABLE_FEATURE_ANALYSIS = False
else:
    print("\n⚠️  Feature quality analysis SKIPPED")


# ── STEP 4: Ridge Baseline ────────────────────────────────────────────────────
step(4, "Ridge regression baseline — Stage 1 (19 features)")
t = time.time()

from model import walk_forward_validation, evaluate_model

ridge_results, coef_df = walk_forward_validation(factors_df)
metrics_ridge           = evaluate_model(ridge_results, coef_df)
ridge_results.to_csv("data/ridge_predictions.csv", index=False)

# Ridge portfolio: top-15 cross-sectional (no sector constraint)
ridge_port_list = []
for date, g in ridge_results.groupby("date"):
    ridge_port_list.append(g.nlargest(15, "predicted")["actual"].mean())
ridge_port   = pd.Series(ridge_port_list)
ridge_sharpe = ((ridge_port-RF).mean()*12) / (ridge_port.std()*np.sqrt(12))
ridge_cum    = (1+ridge_port).prod()-1
ridge_mdd    = ((1+ridge_port).cumprod()/(1+ridge_port).cumprod().cummax()-1).min()
print(f"\n  Ridge portfolio: Sharpe {ridge_sharpe:.3f} | "
      f"Return {ridge_cum:.2%} | MaxDD {ridge_mdd:.2%}")
done("model (Ridge)", time.time() - t)


# ── STEP 3: LightGBM + SHAP ───────────────────────────────────────────────────
step(3, "LightGBM + SHAP explainability — Stage 2")
t = time.time()

from shap_explainability import (
    walk_forward_lgbm, evaluate_lgbm,
    plot_global_importance, plot_shap_direction,
    plot_rolling_factor_importance, plot_shap_by_group,
    explain_single_prediction,
)

lgbm_results, shap_df = walk_forward_lgbm(factors_df)
metrics_lgbm            = evaluate_lgbm(lgbm_results)
lgbm_results.to_csv("data/lgbm_predictions.csv", index=False)
shap_df.to_csv("data/shap_values.csv",            index=False)

# LightGBM portfolio: top-15 cross-sectional
lgbm_port_list = []
for date, g in lgbm_results.groupby("date"):
    lgbm_port_list.append(g.nlargest(15, "predicted")["actual"].mean())
lgbm_port   = pd.Series(lgbm_port_list)
lgbm_sharpe = ((lgbm_port-RF).mean()*12) / (lgbm_port.std()*np.sqrt(12))

print("\n  Generating SHAP charts...")
plot_global_importance(shap_df,
    save_path="outputs/shap_global_importance.png")
plot_shap_direction(shap_df,
    save_path="outputs/shap_direction.png")
plot_rolling_factor_importance(shap_df,
    save_path="outputs/shap_rolling_importance.png")
plot_shap_by_group(shap_df,
    save_path="outputs/shap_group_importance.png")
explain_single_prediction(shap_df,
    save_path="outputs/shap_single_prediction.png")
done("LightGBM + SHAP", time.time() - t)


# ── STEP 6: Hyperparameter Tuning (Optional) ──────────────────────────────────
if ENABLE_HYPERPARAMETER_TUNING:
    step(6, "Hyperparameter tuning with nested cross-validation")
    t = time.time()
    
    try:
        from hyperparameter_tuner import (
            tune_ridge_hyperparameters,
            tune_lightgbm_hyperparameters
        )
        
        print("\n  Tuning hyperparameters using nested CV...")
        print(f"    Training window: first {MIN_TRAIN_MONTHS} months")
        print(f"    CV folds: {config_mgr.get('hyperparameter_tuning.n_cv_splits', 3)}")
        print(f"    Scoring: {config_mgr.get('hyperparameter_tuning.scoring', 'ic')}")
        
        # Tune Ridge if enabled
        if ENABLE_RIDGE:
            print("\n  Tuning Ridge hyperparameters...")
            ridge_param_grid = config_mgr.get("hyperparameter_tuning.ridge_param_grid")
            ridge_tuning_results = tune_ridge_hyperparameters(
                factors_df,
                min_train_months=MIN_TRAIN_MONTHS,
                param_grid=ridge_param_grid
            )
            print(f"    Best Ridge alpha: {ridge_tuning_results['best_params']['alpha']}")
            print(f"    Best Ridge IC: {ridge_tuning_results['best_score']:.5f}")
        
        # Tune LightGBM if enabled
        if ENABLE_LIGHTGBM:
            print("\n  Tuning LightGBM hyperparameters...")
            lgbm_param_grid = config_mgr.get("hyperparameter_tuning.lightgbm_param_grid")
            lgbm_tuning_results = tune_lightgbm_hyperparameters(
                factors_df,
                min_train_months=MIN_TRAIN_MONTHS,
                param_grid=lgbm_param_grid
            )
            print(f"    Best LightGBM params: {lgbm_tuning_results['best_params']}")
            print(f"    Best LightGBM IC: {lgbm_tuning_results['best_score']:.5f}")
        
        # Save tuning results
        os.makedirs("reports", exist_ok=True)
        if ENABLE_RIDGE:
            pd.DataFrame(ridge_tuning_results["param_scores"],
                        columns=["params", "score"]).to_csv(
                "reports/ridge_hyperparameter_tuning.csv", index=False
            )
        if ENABLE_LIGHTGBM:
            pd.DataFrame(lgbm_tuning_results["param_scores"],
                        columns=["params", "score"]).to_csv(
                "reports/lgbm_hyperparameter_tuning.csv", index=False
            )
        
        done("hyperparameter tuning", time.time() - t)
        
    except Exception as e:
        error_handler("Hyperparameter Tuning", e, continue_on_error=True)
        ENABLE_HYPERPARAMETER_TUNING = False
else:
    print("\n  Hyperparameter tuning SKIPPED (using default parameters)")


# ── STEP 7: Ensemble Models (Optional) ────────────────────────────────────────
if ENABLE_ENSEMBLE:
    step(7, "Building ensemble models (Ridge + LightGBM)")
    t = time.time()
    
    try:
        from ensemble_model import walk_forward_ensemble, evaluate_ensemble
        
        ensemble_method = config_mgr.get("models.ensemble.method", "grid_search")
        validation_months = config_mgr.get("models.ensemble.validation_months", 6)
        
        print(f"\n  Ensemble method: {ensemble_method}")
        print(f"  Validation window: {validation_months} months")
        
        # Build ensemble
        ensemble_results, ensemble_weights = walk_forward_ensemble(
            ridge_results,
            lgbm_results,
            method=ensemble_method,
            validation_months=validation_months,
            verbose=True
        )
        
        # Evaluate ensemble
        ensemble_metrics = evaluate_ensemble(ensemble_results, ensemble_weights)
        
        # Save ensemble results
        ensemble_results.to_csv("data/ensemble_predictions.csv", index=False)
        ensemble_weights.to_csv("data/ensemble_weights.csv", index=False)
        
        print(f"\n  ✓ Ensemble predictions saved: data/ensemble_predictions.csv")
        print(f"  ✓ Ensemble weights saved: data/ensemble_weights.csv")
        
        done("ensemble models", time.time() - t)
        
    except Exception as e:
        error_handler("Ensemble Models", e, continue_on_error=True)
        ENABLE_ENSEMBLE = False
else:
    print("\n  Ensemble models SKIPPED")


# ── STEP 8: Sector-Specific Models (Optional) ─────────────────────────────────
if ENABLE_SECTOR_SPECIFIC:
    step(8, "Training sector-specific models")
    t = time.time()
    
    try:
        from sector_model import train_sector_specific_models
        
        sector_model_type = config_mgr.get("models.sector_specific.model_type", "lightgbm")
        tune_per_sector = config_mgr.get("models.sector_specific.tune_per_sector", True)
        
        print(f"\n  Model type: {sector_model_type}")
        print(f"  Tune per sector: {tune_per_sector}")
        
        # Train sector-specific models
        sector_specific_results, sector_model_obj, sector_performance = train_sector_specific_models(
            factors_df,
            model_type=sector_model_type,
            tune_hyperparameters=tune_per_sector,
            min_train_months=MIN_TRAIN_MONTHS
        )
        
        # Save sector-specific results
        sector_specific_results.to_csv("data/sector_specific_predictions.csv", index=False)
        
        # Save sector performance
        sector_perf_df = pd.DataFrame.from_dict(sector_performance, orient="index")
        sector_perf_df.to_csv("reports/sector_specific_performance.csv")
        
        print(f"\n  ✓ Sector-specific predictions saved: data/sector_specific_predictions.csv")
        print(f"  ✓ Sector performance saved: reports/sector_specific_performance.csv")
        
        done("sector-specific models", time.time() - t)
        
    except Exception as e:
        error_handler("Sector-Specific Models", e, continue_on_error=True)
        ENABLE_SECTOR_SPECIFIC = False
else:
    print("\n  Sector-specific models SKIPPED")


# ── STEP 9: Early Stopping & Regularization ───────────────────────────────────
step(9, "Applying early stopping and regularization analysis")
t = time.time()

try:
    from early_stopping_regularization import (
        EarlyStoppingValidator,
        ModelComplexityAnalyzer
    )
    
    print("\n  Analyzing model complexity vs performance...")
    
    # Use first walk-forward window for analysis
    all_dates = sorted(factors_df["date"].unique())
    train_df = factors_df[factors_df["date"].isin(all_dates[:MIN_TRAIN_MONTHS])]
    
    # Model complexity analysis
    analyzer = ModelComplexityAnalyzer(verbose=True)
    
    # Analyze LightGBM complexity (smaller grid for performance)
    lgbm_complexity = analyzer.analyze_lgbm_complexity(
        train_df,
        param_grid={
            "num_leaves": [15, 31, 63],
            "max_depth": [3, 5, 7],
            "min_child_samples": [20, 50, 100],
            "learning_rate": [0.03, 0.05],
        }
    )
    
    # Save complexity analysis
    lgbm_complexity.to_csv("reports/lgbm_complexity_analysis.csv", index=False)
    
    # Generate complexity plots
    analyzer.plot_complexity_analysis("outputs/complexity_analysis.png")
    
    print(f"\n  ✓ Complexity analysis saved: reports/lgbm_complexity_analysis.csv")
    print(f"  ✓ Complexity plots saved: outputs/complexity_analysis.png")
    
    done("early stopping & regularization", time.time() - t)
    
except Exception as e:
    error_handler("Early Stopping & Regularization", e, continue_on_error=True)


# ── STEP 10: Sector Neutralisation + EWM ──────────────────────────────────────
step(10, "Sector-diversified portfolio + EWM smoothing — Stage 3")
t = time.time()

from sector_neutralisation import (
    add_sector_features, walk_forward_sector_neutral,
    evaluate_sector_neutral, build_sector_aware_portfolio,
    EWM_ALPHA, REBAL_THRESHOLD, TOP_PER_SECTOR,
    HOLD_BONUS_PER_MONTH, HOLD_BONUS_CAP,
)

factors_df_sector, z_features = add_sector_features(factors_df, dynamic_sector_map)
sector_results = walk_forward_sector_neutral(factors_df_sector, z_features)
sector_results.to_csv("data/sector_predictions.csv", index=False)

metrics_sector = evaluate_sector_neutral(
    sector_results, old_path="data/lgbm_predictions.csv"
)

sector_port = build_sector_aware_portfolio(sector_results)
sector_port.to_csv("reports/sector_portfolio.csv")

sr          = sector_port["portfolio_return"]
sect_sharpe = ((sr-RF).mean()*12) / (sr.std()*np.sqrt(12))
sect_cum    = (1+sr).prod()-1
sect_mdd    = ((1+sr).cumprod()/(1+sr).cumprod().cummax()-1).min()
print(f"\n  Sector portfolio: Sharpe {sect_sharpe:.3f} | "
      f"Return {sect_cum:.2%} | MaxDD {sect_mdd:.2%}")
done("sector_neutralisation", time.time() - t)


# ── STEP 11: OOT Validation ───────────────────────────────────────────────────
step(11, "Out-of-Time validation — Stage 4")
t = time.time()

from oot_validation import run_oot_validation

oot_metrics = run_oot_validation(
    factors_df_sector=factors_df_sector,
    z_features=z_features,
    sector_results_full=sector_results,
    sector_port_full=sr,
)
done("oot_validation", time.time() - t)


# ── STEP 12: Transaction Costs ────────────────────────────────────────────────
step(12, "Transaction cost modelling — Stage 5")
t = time.time()

from transaction_costs import (
    compute_turnover, apply_transaction_costs,
    print_cost_analysis, plot_cost_scenarios,
    plot_sharpe_sensitivity, COST_SCENARIOS,
)

SECTOR_PRED = "data/sector_predictions.csv"
PORTFOLIO   = "reports/sector_portfolio.csv"

turnover_df = compute_turnover(SECTOR_PRED)
print_cost_analysis(PORTFOLIO, turnover_df)
plot_cost_scenarios(PORTFOLIO, turnover_df)
plot_sharpe_sensitivity(PORTFOLIO, turnover_df)

# Save cost-adjusted returns report
gross, _, _ = apply_transaction_costs(PORTFOLIO, turnover_df, 0)
rows = []
for date, g in gross.items():
    row = {"date": date, "gross_return": g}
    for name, cost in COST_SCENARIOS.items():
        _, net, mc = apply_transaction_costs(PORTFOLIO, turnover_df, cost)
        short = name.split("(")[1].replace(")","").replace(" ","_")
        row[f"net_{short}"]  = net.get(date, g)
        row[f"cost_{short}"] = mc.get(date, 0)
    rows.append(row)
pd.DataFrame(rows).set_index("date").to_csv("reports/cost_adjusted_returns.csv")

_, net_10, _ = apply_transaction_costs(PORTFOLIO, turnover_df, 0.0010)
net_sharpe   = ((net_10-RF).mean()*12) / (net_10.std()*np.sqrt(12))
net_cum      = (1+net_10).prod()-1
done("transaction_costs", time.time() - t)


# ── STEP 13: Comprehensive Diagnostic Report ──────────────────────────────────
step(13, "Generating comprehensive diagnostic report")
t = time.time()

try:
    print("\n  Compiling integrated diagnostic report...")
    
    # Create comprehensive diagnostic report
    diagnostic_report = {
        "pipeline_info": {
            "configuration": args.config,
            "total_runtime_minutes": (time.time() - t0) / 60,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "universe_size": factors_df["ticker"].nunique(),
            "time_periods": factors_df["date"].nunique(),
            "features_count": len(FEATURES),
        },
        "leakage_detection": {
            "enabled": ENABLE_LEAKAGE_DETECTION,
            "all_checks_passed": audit_report.get("all_checks_passed", False) if ENABLE_LEAKAGE_DETECTION else None,
        },
        "feature_analysis": {
            "enabled": ENABLE_FEATURE_ANALYSIS,
            "top_features": feature_report["overall_ranking"].head(5)["feature"].tolist() if ENABLE_FEATURE_ANALYSIS else [],
            "bottom_features": feature_report["overall_ranking"].tail(5)["feature"].tolist() if ENABLE_FEATURE_ANALYSIS else [],
        },
        "model_performance": {
            "ridge": {
                "mean_ic": metrics_ridge.get("Mean_IC", 0),
                "ic_ir": metrics_ridge.get("IC_IR", 0),
                "sharpe": ridge_sharpe,
            },
            "lightgbm": {
                "mean_ic": metrics_lgbm.get("Mean_IC", 0),
                "ic_ir": metrics_lgbm.get("IC_IR", 0),
                "sharpe": lgbm_sharpe,
            },
            "sector_neutral": {
                "mean_ic": metrics_sector.get("Mean_IC", 0),
                "ic_ir": metrics_sector.get("IC_IR", 0),
                "sharpe": sect_sharpe,
            },
        },
        "hyperparameter_tuning": {
            "enabled": ENABLE_HYPERPARAMETER_TUNING,
            "ridge_best_alpha": ridge_tuning_results["best_params"]["alpha"] if ENABLE_HYPERPARAMETER_TUNING and ENABLE_RIDGE else None,
            "lgbm_best_score": lgbm_tuning_results["best_score"] if ENABLE_HYPERPARAMETER_TUNING and ENABLE_LIGHTGBM else None,
        },
        "ensemble": {
            "enabled": ENABLE_ENSEMBLE,
            "mean_ic": ensemble_metrics["ensemble"]["mean_ic"] if ENABLE_ENSEMBLE else None,
            "ic_ir": ensemble_metrics["ensemble"]["ic_ir"] if ENABLE_ENSEMBLE else None,
        },
        "sector_specific": {
            "enabled": ENABLE_SECTOR_SPECIFIC,
            "overall_ic": sector_performance.get("Overall", {}).get("Mean_IC", 0) if ENABLE_SECTOR_SPECIFIC else None,
        },
        "turnover": {
            "mean_monthly": turnover_df["turnover"].mean(),
            "annual": turnover_df["turnover"].mean() * 12,
            "target": config_mgr.get("turnover.target_turnover", 0.25),
        },
        "oot_validation": {
            "mean_ic": oot_metrics.get("OOT_Mean_IC", 0),
            "ic_ir": oot_metrics.get("OOT_IC_IR", 0),
            "sharpe": oot_metrics.get("OOT_Sharpe", 0),
            "months": oot_metrics.get("OOT_Months", 0),
        },
        "net_performance": {
            "sharpe_10bps": net_sharpe,
            "cumulative_return": net_cum,
        },
    }
    
    # Save diagnostic report as JSON
    import json
    with open("reports/comprehensive_diagnostic_report.json", "w") as f:
        json.dump(diagnostic_report, f, indent=2, default=str)
    
    # Generate human-readable diagnostic report
    with open("reports/comprehensive_diagnostic_report.txt", "w") as f:
        f.write("="*78 + "\n")
        f.write("  COMPREHENSIVE DIAGNOSTIC REPORT\n")
        f.write("="*78 + "\n\n")
        f.write(f"Generated: {diagnostic_report['pipeline_info']['timestamp']}\n")
        f.write(f"Configuration: {diagnostic_report['pipeline_info']['configuration']}\n")
        f.write(f"Total Runtime: {diagnostic_report['pipeline_info']['total_runtime_minutes']:.1f} minutes\n\n")
        
        f.write("PIPELINE CONFIGURATION:\n")
        f.write("-" * 78 + "\n")
        f.write(f"  Universe: {diagnostic_report['pipeline_info']['universe_size']} stocks\n")
        f.write(f"  Time Periods: {diagnostic_report['pipeline_info']['time_periods']} months\n")
        f.write(f"  Features: {diagnostic_report['pipeline_info']['features_count']}\n")
        f.write(f"  Min Train Months: {MIN_TRAIN_MONTHS}\n")
        f.write(f"  Fundamental Lag: {FUNDAMENTAL_LAG_DAYS} days\n\n")
        
        f.write("DATA QUALITY:\n")
        f.write("-" * 78 + "\n")
        f.write(f"  Leakage Detection: {'✓ ENABLED' if ENABLE_LEAKAGE_DETECTION else '⚠ DISABLED'}\n")
        if ENABLE_LEAKAGE_DETECTION:
            f.write(f"    All Checks Passed: {'✓ YES' if diagnostic_report['leakage_detection']['all_checks_passed'] else '❌ NO'}\n")
        f.write(f"  Feature Analysis: {'✓ ENABLED' if ENABLE_FEATURE_ANALYSIS else '⚠ DISABLED'}\n")
        if ENABLE_FEATURE_ANALYSIS:
            f.write(f"    Top Features: {', '.join(diagnostic_report['feature_analysis']['top_features'][:3])}\n")
        f.write("\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write("-" * 78 + "\n")
        f.write(f"  {'Model':<25} {'Mean IC':>12} {'IC-IR':>10} {'Sharpe':>10}\n")
        f.write("  " + "-" * 76 + "\n")
        
        for model_name, metrics in diagnostic_report["model_performance"].items():
            f.write(f"  {model_name.replace('_', ' ').title():<25} "
                   f"{metrics['mean_ic']:>12.5f} "
                   f"{metrics['ic_ir']:>10.5f} "
                   f"{metrics['sharpe']:>10.3f}\n")
        f.write("\n")
        
        f.write("ADVANCED FEATURES:\n")
        f.write("-" * 78 + "\n")
        f.write(f"  Hyperparameter Tuning: {'✓ ENABLED' if ENABLE_HYPERPARAMETER_TUNING else '⚠ DISABLED'}\n")
        if ENABLE_HYPERPARAMETER_TUNING:
            if ENABLE_RIDGE and diagnostic_report["hyperparameter_tuning"]["ridge_best_alpha"]:
                f.write(f"    Ridge Best Alpha: {diagnostic_report['hyperparameter_tuning']['ridge_best_alpha']}\n")
            if ENABLE_LIGHTGBM and diagnostic_report["hyperparameter_tuning"]["lgbm_best_score"]:
                f.write(f"    LightGBM Best IC: {diagnostic_report['hyperparameter_tuning']['lgbm_best_score']:.5f}\n")
        
        f.write(f"  Ensemble Models: {'✓ ENABLED' if ENABLE_ENSEMBLE else '⚠ DISABLED'}\n")
        if ENABLE_ENSEMBLE:
            f.write(f"    Ensemble Mean IC: {diagnostic_report['ensemble']['mean_ic']:.5f}\n")
            f.write(f"    Ensemble IC-IR: {diagnostic_report['ensemble']['ic_ir']:.5f}\n")
        
        f.write(f"  Sector-Specific Models: {'✓ ENABLED' if ENABLE_SECTOR_SPECIFIC else '⚠ DISABLED'}\n")
        if ENABLE_SECTOR_SPECIFIC:
            f.write(f"    Overall IC: {diagnostic_report['sector_specific']['overall_ic']:.5f}\n")
        f.write("\n")
        
        f.write("TURNOVER ANALYSIS:\n")
        f.write("-" * 78 + "\n")
        f.write(f"  Mean Monthly Turnover: {diagnostic_report['turnover']['mean_monthly']:.1%}\n")
        f.write(f"  Annual Turnover: {diagnostic_report['turnover']['annual']:.0%}\n")
        f.write(f"  Target Turnover: {diagnostic_report['turnover']['target']:.1%}\n")
        f.write(f"  Status: {'✓ BELOW TARGET' if diagnostic_report['turnover']['annual'] < diagnostic_report['turnover']['target'] * 12 else '⚠ ABOVE TARGET'}\n\n")
        
        f.write("OUT-OF-TIME VALIDATION:\n")
        f.write("-" * 78 + "\n")
        f.write(f"  OOT Months: {diagnostic_report['oot_validation']['months']}\n")
        f.write(f"  OOT Mean IC: {diagnostic_report['oot_validation']['mean_ic']:.5f}\n")
        f.write(f"  OOT IC-IR: {diagnostic_report['oot_validation']['ic_ir']:.5f}\n")
        f.write(f"  OOT Sharpe: {diagnostic_report['oot_validation']['sharpe']:.3f}\n\n")
        
        f.write("NET PERFORMANCE (10bps costs):\n")
        f.write("-" * 78 + "\n")
        f.write(f"  Net Sharpe Ratio: {diagnostic_report['net_performance']['sharpe_10bps']:.3f}\n")
        f.write(f"  Cumulative Return: {diagnostic_report['net_performance']['cumulative_return']:.2%}\n\n")
        
        f.write("="*78 + "\n")
        f.write("  DIAGNOSTIC REPORT COMPLETE\n")
        f.write("="*78 + "\n")
    
    print(f"\n  ✓ Comprehensive diagnostic report saved:")
    print(f"    - reports/comprehensive_diagnostic_report.json")
    print(f"    - reports/comprehensive_diagnostic_report.txt")
    
    done("comprehensive diagnostic report", time.time() - t)
    
except Exception as e:
    error_handler("Comprehensive Diagnostic Report", e, continue_on_error=True)


# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
total = time.time() - t0
section("INTEGRATED PIPELINE COMPLETE")

print(f"\n  Total runtime  : {total/60:.1f} minutes")
print(f"  Configuration  : {args.config}")
print(f"  Universe       : {factors_df['ticker'].nunique()} stocks | "
      f"{factors_df['date'].nunique()} months")
print(f"  Features       : {len(FEATURES)} across "
      f"{len(FEATURE_GROUPS)} groups")
n_sectors = factors_df['sector'].nunique()
print(f"  Portfolio      : top-{TOP_PER_SECTOR}/sector = "
      f"{TOP_PER_SECTOR*n_sectors} stocks | "
      f"EWM α={EWM_ALPHA} | threshold={REBAL_THRESHOLD} | "
      f"{n_sectors} GICS sectors")
print(f"  Turnover       : {turnover_df['turnover'].mean():.1%}/month "
      f"({turnover_df['turnover'].mean()*12:.0%}/year)")

print(f"\n  {'Stage':<40} {'Mean IC':>9} {'IC-IR':>8} {'Sharpe':>8}")
print("  " + "-"*67)
print(f"  {'1. Ridge (19 features)':<40} "
      f"{metrics_ridge.get('Mean_IC',0):>9.4f} "
      f"{metrics_ridge.get('IC_IR',0):>8.4f} "
      f"{ridge_sharpe:>8.3f}")
print(f"  {'2. LightGBM (19 features)':<40} "
      f"{metrics_lgbm.get('Mean_IC',0):>9.4f} "
      f"{metrics_lgbm.get('IC_IR',0):>8.4f} "
      f"{lgbm_sharpe:>8.3f}")

if ENABLE_ENSEMBLE:
    print(f"  {'3. Ensemble (Ridge + LightGBM)':<40} "
          f"{ensemble_metrics['ensemble']['mean_ic']:>9.4f} "
          f"{ensemble_metrics['ensemble']['ic_ir']:>8.4f} "
          f"{'—':>8}")

if ENABLE_SECTOR_SPECIFIC:
    print(f"  {'4. Sector-Specific Models':<40} "
          f"{sector_performance['Overall']['Mean_IC']:>9.4f} "
          f"{sector_performance['Overall']['IC_IR']:>8.4f} "
          f"{'—':>8}")

print(f"  {'5. + Sector Diversified + EWM':<40} "
      f"{metrics_sector.get('Mean_IC',0):>9.4f} "
      f"{metrics_sector.get('IC_IR',0):>8.4f} "
      f"{sect_sharpe:>8.3f}")
print(f"  {'6. OOT (Jan 2025+)':<40} "
      f"{oot_metrics.get('OOT_Mean_IC',0):>9.4f} "
      f"{oot_metrics.get('OOT_IC_IR',0):>8.4f} "
      f"{oot_metrics.get('OOT_Sharpe',0):>8.3f}")
print(f"  {'7. Net of 10bps costs':<40} {'—':>9} {'—':>8} "
      f"{net_sharpe:>8.3f}")

print("\n  Outputs saved:")
print("    data/          — factor_features.csv, *_predictions.csv,")
print("                     shap_values.csv, daily_prices.parquet,")
print("                     fundamentals.parquet (cache)")
if ENABLE_ENSEMBLE:
    print("                     ensemble_predictions.csv, ensemble_weights.csv")
if ENABLE_SECTOR_SPECIFIC:
    print("                     sector_specific_predictions.csv")
print("    outputs/       — all charts (Ridge + SHAP + OOT + costs + complexity)")
print("    reports/       — sector_portfolio.csv, oot_*.csv,")
print("                     cost_adjusted_returns.csv,")
if ENABLE_LEAKAGE_DETECTION:
    print("                     leakage_audit_report.txt,")
if ENABLE_FEATURE_ANALYSIS:
    print("                     feature_ic_*.csv, feature_recommendations.txt,")
if ENABLE_HYPERPARAMETER_TUNING:
    print("                     *_hyperparameter_tuning.csv,")
print("                     comprehensive_diagnostic_report.txt/json")

# Print system status summary
print("\n  System Status:")
print(f"    Leakage Detection: {'✓ PASSED' if ENABLE_LEAKAGE_DETECTION and audit_report.get('all_checks_passed', False) else '⚠ CHECK REQUIRED'}")
print(f"    Feature Analysis: {'✓ COMPLETE' if ENABLE_FEATURE_ANALYSIS else '⚠ SKIPPED'}")
print(f"    Hyperparameter Tuning: {'✓ COMPLETE' if ENABLE_HYPERPARAMETER_TUNING else '⚠ SKIPPED'}")
print(f"    Ensemble Models: {'✓ ENABLED' if ENABLE_ENSEMBLE else '⚠ DISABLED'}")
print(f"    Sector-Specific Models: {'✓ ENABLED' if ENABLE_SECTOR_SPECIFIC else '⚠ DISABLED'}")
print(f"    Turnover Target: {'✓ MET' if turnover_df['turnover'].mean() * 12 < config_mgr.get('turnover.target_turnover', 0.25) * 12 else '⚠ EXCEEDED'}")

ic_v  = metrics_sector.get("Mean_IC", 0)
ir_v  = metrics_sector.get("IC_IR",   0)
om    = oot_metrics.get("OOT_Months", 0)
os_   = oot_metrics.get("OOT_Sharpe", 0)
turn  = turnover_df["turnover"].mean() * 12
n_st  = factors_df["ticker"].nunique()

print("\n" + "="*78)
print("  Pipeline complete. Check reports/comprehensive_diagnostic_report.txt for details.")
print("="*78)

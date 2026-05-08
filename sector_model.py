"""
sector_model.py — Sector-Specific Modeling Framework
====================================================
Implements separate models per sector with sector-specific hyperparameter
tuning and performance tracking. Each sector (Technology, Healthcare, 
Financials, Consumer, Energy) gets its own model trained on sector-specific
data patterns.

ARCHITECTURE:
  - Separate model instance per sector
  - Sector-specific hyperparameter tuning
  - Sector-specific performance tracking
  - Comparison framework for sector vs global models

TEMPORAL INTEGRITY:
  - All sector models use walk-forward validation
  - No future data leakage
  - Sector-specific CV splits maintain temporal order

SUPPORTED MODELS:
  - Ridge: sector-specific regularization
  - LightGBM: sector-specific tree parameters

**Validates: Requirements 3.4**
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from data_loader import FEATURES, TARGET, SECTOR_MAP
from hyperparameter_tuner import HyperparameterTuner


class SectorSpecificModel:
    """
    Sector-specific modeling framework.
    
    Trains separate models for each sector with sector-specific
    hyperparameters and tracks performance by sector.
    """
    
    def __init__(self, model_type="lightgbm", tune_hyperparameters=True, 
                 n_cv_splits=3, verbose=True):
        """
        Initialize sector-specific model framework.
        
        Parameters
        ----------
        model_type : str
            Model type: "ridge" or "lightgbm"
        tune_hyperparameters : bool
            Whether to tune hyperparameters per sector
        n_cv_splits : int
            Number of CV folds for hyperparameter tuning
        verbose : bool
            Print progress messages
        """
        self.model_type = model_type.lower()
        self.tune_hyperparameters = tune_hyperparameters
        self.n_cv_splits = n_cv_splits
        self.verbose = verbose
        
        # Store sector models and hyperparameters
        self.sector_models = {}
        self.sector_params = {}
        self.sector_scalers = {}
        
        # Performance tracking
        self.sector_performance = {}
        
        if self.model_type not in ["ridge", "lightgbm"]:
            raise ValueError(f"Unsupported model_type: {model_type}")
    
    def _get_default_params(self, sector=None):
        """Get default hyperparameters for a sector."""
        if self.model_type == "ridge":
            return {"alpha": 1.0}
        else:  # lightgbm
            return {
                "objective": "regression",
                "metric": "rmse",
                "n_estimators": 300,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.2,
                "reg_lambda": 0.2,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }
    
    def _tune_sector_hyperparameters(self, sector_df, sector):
        """
        Tune hyperparameters for a specific sector.
        
        Parameters
        ----------
        sector_df : pd.DataFrame
            Training data for the sector
        sector : str
            Sector name
        
        Returns
        -------
        dict
            Best hyperparameters for the sector
        """
        if self.verbose:
            print(f"\n  Tuning hyperparameters for {sector}...")
            print(f"    Training data: {len(sector_df):,} rows, "
                  f"{sector_df['date'].nunique()} months")
        
        tuner = HyperparameterTuner(
            model_type=self.model_type,
            n_splits=self.n_cv_splits,
            scoring="ic",
            verbose=False
        )
        
        # Define sector-specific parameter grids
        if self.model_type == "ridge":
            param_grid = {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
            }
        else:  # lightgbm
            param_grid = {
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "max_depth": [3, 5, 7],
                "min_data_in_leaf": [10, 20, 50],
                "lambda_l1": [0.0, 0.1, 1.0],
                "lambda_l2": [0.0, 0.1, 1.0],
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
                "n_estimators": 300,
                "random_state": 42,
                "n_jobs": -1
            }
        
        results = tuner.tune(sector_df, param_grid)
        best_params = results["best_params"]
        
        if self.verbose:
            print(f"    Best params: {best_params}")
            print(f"    Best IC: {results['best_score']:.5f}")
        
        return best_params
    
    def walk_forward_sector_models(self, factors_df, min_train_months=24):
        """
        Walk-forward validation with separate models per sector.
        
        For each month t:
          1. For each sector:
             - Train sector-specific model on sector data from months [0..t-1]
             - Predict for sector stocks at month t
          2. Combine predictions across all sectors
        
        Parameters
        ----------
        factors_df : pd.DataFrame
            Factor data with columns: date, ticker, sector, features, target
        min_train_months : int
            Minimum number of months for initial training window
        
        Returns
        -------
        pd.DataFrame
            Predictions with columns: date, ticker, sector, actual, predicted
        """
        if "sector" not in factors_df.columns:
            raise ValueError("factors_df must have 'sector' column")
        
        all_dates = sorted(factors_df["date"].unique())
        sectors = sorted(factors_df["sector"].unique())
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  SECTOR-SPECIFIC MODEL ({self.model_type.upper()})")
            print(f"{'='*70}")
            print(f"  Sectors: {sectors}")
            print(f"  Total months: {len(all_dates)}")
            print(f"  Min training months: {min_train_months}")
            print(f"  Hyperparameter tuning: {self.tune_hyperparameters}")
        
        # Tune hyperparameters per sector using initial training window
        if self.tune_hyperparameters:
            print(f"\n  Tuning hyperparameters per sector...")
            initial_train = factors_df[
                factors_df["date"].isin(all_dates[:min_train_months])
            ]
            
            for sector in sectors:
                sector_train = initial_train[initial_train["sector"] == sector]
                
                if len(sector_train) < 100:
                    if self.verbose:
                        print(f"    {sector}: insufficient data, using defaults")
                    self.sector_params[sector] = self._get_default_params(sector)
                else:
                    self.sector_params[sector] = self._tune_sector_hyperparameters(
                        sector_train, sector
                    )
        else:
            # Use default parameters for all sectors
            for sector in sectors:
                self.sector_params[sector] = self._get_default_params(sector)
        
        # Walk-forward validation
        results = []
        
        if self.verbose:
            print(f"\n  Walk-forward validation...")
        
        for i, test_date in enumerate(all_dates):
            if i < min_train_months:
                continue
            
            train_df = factors_df[factors_df["date"].isin(all_dates[:i])]
            test_df = factors_df[factors_df["date"] == test_date]
            
            if len(train_df) < 200 or len(test_df) == 0:
                continue
            
            if self.verbose and (i - min_train_months) % 6 == 0:
                print(f"    Month {i - min_train_months + 1}/"
                      f"{len(all_dates) - min_train_months}: {test_date.date()}")
            
            # Train and predict for each sector
            for sector in sectors:
                sector_train = train_df[train_df["sector"] == sector]
                sector_test = test_df[test_df["sector"] == sector]
                
                if len(sector_train) < 50 or len(sector_test) == 0:
                    continue
                
                # Ensure sector has parameters (use defaults if missing)
                if sector not in self.sector_params:
                    self.sector_params[sector] = self._get_default_params(sector)
                
                # Prepare features with cross-sectional median imputation
                X_train_df = sector_train[FEATURES].copy()
                for col in FEATURES:
                    med = X_train_df[col].median()
                    X_train_df[col] = X_train_df[col].fillna(med)
                X_train_df = X_train_df.fillna(0)
                
                X_test_df = sector_test[FEATURES].copy()
                for col in FEATURES:
                    med = X_test_df[col].median()
                    X_test_df[col] = X_test_df[col].fillna(med)
                X_test_df = X_test_df.fillna(0)
                
                X_train = X_train_df.values
                y_train = sector_train[TARGET].values
                X_test = X_test_df.values
                y_test = sector_test[TARGET].values
                
                # Train sector-specific model
                if self.model_type == "ridge":
                    # Scale features for Ridge
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    
                    model = Ridge(**self.sector_params[sector])
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                else:  # lightgbm
                    model = lgb.LGBMRegressor(**self.sector_params[sector])
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Store predictions
                for j, ticker in enumerate(sector_test["ticker"].values):
                    results.append({
                        "date": test_date,
                        "ticker": ticker,
                        "sector": sector,
                        "actual": y_test[j],
                        "predicted": y_pred[j],
                    })
        
        results_df = pd.DataFrame(results)
        
        if self.verbose:
            print(f"\n  Walk-forward complete: {len(results_df):,} predictions, "
                  f"{results_df['date'].nunique()} months")
        
        return results_df
    
    def evaluate_sector_performance(self, results_df):
        """
        Evaluate performance by sector and overall.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Predictions with columns: date, ticker, sector, actual, predicted
        
        Returns
        -------
        dict
            Performance metrics by sector and overall
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  SECTOR-SPECIFIC MODEL PERFORMANCE")
            print(f"{'='*70}")
        
        performance = {}
        
        # Overall performance
        overall_ic = results_df.groupby("date").apply(
            lambda g: g["actual"].corr(g["predicted"], method="spearman")
        )
        
        performance["Overall"] = {
            "Mean_IC": overall_ic.mean(),
            "IC_Std": overall_ic.std(),
            "IC_IR": overall_ic.mean() / overall_ic.std() if overall_ic.std() > 0 else 0,
            "Pos_IC": (overall_ic > 0).sum(),
            "Total_Months": len(overall_ic),
            "N_Predictions": len(results_df)
        }
        
        if self.verbose:
            print(f"\n  Overall Performance:")
            print(f"    Mean IC: {performance['Overall']['Mean_IC']:.5f}")
            print(f"    IC Std: {performance['Overall']['IC_Std']:.5f}")
            print(f"    IC-IR: {performance['Overall']['IC_IR']:.5f}")
            print(f"    Positive IC: {performance['Overall']['Pos_IC']}/"
                  f"{performance['Overall']['Total_Months']} "
                  f"({performance['Overall']['Pos_IC']/performance['Overall']['Total_Months']*100:.1f}%)")
        
        # Sector-specific performance
        if self.verbose:
            print(f"\n  Sector Performance:")
        
        for sector in sorted(results_df["sector"].unique()):
            sector_df = results_df[results_df["sector"] == sector]
            
            sector_ic = sector_df.groupby("date").apply(
                lambda g: g["actual"].corr(g["predicted"], method="spearman")
            )
            
            performance[sector] = {
                "Mean_IC": sector_ic.mean(),
                "IC_Std": sector_ic.std(),
                "IC_IR": sector_ic.mean() / sector_ic.std() if sector_ic.std() > 0 else 0,
                "Pos_IC": (sector_ic > 0).sum(),
                "Total_Months": len(sector_ic),
                "N_Predictions": len(sector_df)
            }
            
            if self.verbose:
                print(f"    {sector:<14}: IC={performance[sector]['Mean_IC']:+.5f}, "
                      f"IR={performance[sector]['IC_IR']:+.5f}, "
                      f"Pos={performance[sector]['Pos_IC']}/{performance[sector]['Total_Months']}")
        
        if self.verbose:
            print(f"{'='*70}")
        
        self.sector_performance = performance
        return performance
    
    def compare_with_global_model(self, sector_results_df, global_results_df):
        """
        Compare sector-specific models with global model.
        
        Parameters
        ----------
        sector_results_df : pd.DataFrame
            Predictions from sector-specific models
        global_results_df : pd.DataFrame
            Predictions from global model
        
        Returns
        -------
        pd.DataFrame
            Comparison metrics
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  SECTOR vs GLOBAL MODEL COMPARISON")
            print(f"{'='*70}")
        
        comparison = []
        
        # Overall comparison
        sector_ic = sector_results_df.groupby("date").apply(
            lambda g: g["actual"].corr(g["predicted"], method="spearman")
        )
        global_ic = global_results_df.groupby("date").apply(
            lambda g: g["actual"].corr(g["predicted"], method="spearman")
        )
        
        comparison.append({
            "Sector": "Overall",
            "Sector_Mean_IC": sector_ic.mean(),
            "Global_Mean_IC": global_ic.mean(),
            "IC_Improvement": sector_ic.mean() - global_ic.mean(),
            "Sector_IC_IR": sector_ic.mean() / sector_ic.std() if sector_ic.std() > 0 else 0,
            "Global_IC_IR": global_ic.mean() / global_ic.std() if global_ic.std() > 0 else 0,
            "IR_Improvement": (sector_ic.mean() / sector_ic.std() if sector_ic.std() > 0 else 0) - 
                             (global_ic.mean() / global_ic.std() if global_ic.std() > 0 else 0),
        })
        
        # Sector-by-sector comparison
        for sector in sorted(sector_results_df["sector"].unique()):
            sector_df = sector_results_df[sector_results_df["sector"] == sector]
            global_sector_df = global_results_df[global_results_df["sector"] == sector]
            
            if len(global_sector_df) == 0:
                continue
            
            sector_ic = sector_df.groupby("date").apply(
                lambda g: g["actual"].corr(g["predicted"], method="spearman")
            )
            global_ic = global_sector_df.groupby("date").apply(
                lambda g: g["actual"].corr(g["predicted"], method="spearman")
            )
            
            comparison.append({
                "Sector": sector,
                "Sector_Mean_IC": sector_ic.mean(),
                "Global_Mean_IC": global_ic.mean(),
                "IC_Improvement": sector_ic.mean() - global_ic.mean(),
                "Sector_IC_IR": sector_ic.mean() / sector_ic.std() if sector_ic.std() > 0 else 0,
                "Global_IC_IR": global_ic.mean() / global_ic.std() if global_ic.std() > 0 else 0,
                "IR_Improvement": (sector_ic.mean() / sector_ic.std() if sector_ic.std() > 0 else 0) - 
                                 (global_ic.mean() / global_ic.std() if global_ic.std() > 0 else 0),
            })
        
        comparison_df = pd.DataFrame(comparison)
        
        if self.verbose:
            print(f"\n  Comparison Results:")
            print(f"\n  {'Sector':<14} {'Sector IC':>10} {'Global IC':>10} "
                  f"{'IC Δ':>8} {'Sector IR':>10} {'Global IR':>10} {'IR Δ':>8}")
            print(f"  {'-'*78}")
            
            for _, row in comparison_df.iterrows():
                print(f"  {row['Sector']:<14} {row['Sector_Mean_IC']:>10.5f} "
                      f"{row['Global_Mean_IC']:>10.5f} {row['IC_Improvement']:>+8.5f} "
                      f"{row['Sector_IC_IR']:>10.5f} {row['Global_IC_IR']:>10.5f} "
                      f"{row['IR_Improvement']:>+8.5f}")
            
            print(f"{'='*70}")
        
        return comparison_df


def train_sector_specific_models(factors_df, model_type="lightgbm", 
                                  tune_hyperparameters=True, min_train_months=24):
    """
    Train sector-specific models with walk-forward validation.
    
    Parameters
    ----------
    factors_df : pd.DataFrame
        Factor data with columns: date, ticker, sector, features, target
    model_type : str
        Model type: "ridge" or "lightgbm"
    tune_hyperparameters : bool
        Whether to tune hyperparameters per sector
    min_train_months : int
        Minimum number of months for initial training window
    
    Returns
    -------
    tuple
        (results_df, sector_model, performance_dict)
    """
    sector_model = SectorSpecificModel(
        model_type=model_type,
        tune_hyperparameters=tune_hyperparameters,
        n_cv_splits=3,
        verbose=True
    )
    
    results_df = sector_model.walk_forward_sector_models(
        factors_df, 
        min_train_months=min_train_months
    )
    
    performance = sector_model.evaluate_sector_performance(results_df)
    
    return results_df, sector_model, performance


if __name__ == "__main__":
    import os
    
    # Load factor data
    print("Loading factor data...")
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    print(f"  {len(factors_df):,} rows, {factors_df['date'].nunique()} months, "
          f"{factors_df['ticker'].nunique()} stocks")
    
    # Train sector-specific LightGBM models
    print("\n" + "="*70)
    print("  TRAINING SECTOR-SPECIFIC LIGHTGBM MODELS")
    print("="*70)
    
    sector_results, sector_model, sector_performance = train_sector_specific_models(
        factors_df,
        model_type="lightgbm",
        tune_hyperparameters=True,
        min_train_months=24
    )
    
    # Save sector-specific predictions
    os.makedirs("data", exist_ok=True)
    sector_results.to_csv("data/sector_specific_predictions.csv", index=False)
    print(f"\nSaved: data/sector_specific_predictions.csv")
    
    # Compare with global model if available
    if os.path.exists("data/sector_predictions.csv"):
        print("\n" + "="*70)
        print("  COMPARING WITH GLOBAL MODEL")
        print("="*70)
        
        global_results = pd.read_csv("data/sector_predictions.csv", parse_dates=["date"])
        
        comparison_df = sector_model.compare_with_global_model(
            sector_results,
            global_results
        )
        
        # Save comparison
        os.makedirs("reports", exist_ok=True)
        comparison_df.to_csv("reports/sector_model_comparison.csv", index=False)
        print(f"\nSaved: reports/sector_model_comparison.csv")

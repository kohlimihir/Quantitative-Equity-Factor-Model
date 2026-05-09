"""
hyperparameter_tuner.py — Hyperparameter Optimization Framework
================================================================
Implements nested cross-validation for hyperparameter tuning that is
compatible with walk-forward validation. Prevents overfitting by using
only training data and maintaining temporal ordering.

NESTED CROSS-VALIDATION STRUCTURE:
  Outer loop: Walk-forward validation (expanding windows)
  Inner loop: Time-series cross-validation for hyperparameter tuning
  
  For each walk-forward window [0..i]:
    - Split training data into K temporal folds
    - For each hyperparameter combination:
      - Evaluate on K folds (respecting temporal order)
      - Compute mean validation score
    - Select best hyperparameters
    - Retrain on full training window with best params
    - Predict on test window [i]

TEMPORAL INTEGRITY:
  - All CV splits maintain chronological order
  - No future data leaks into training or validation
  - Validation folds are always after training folds

SUPPORTED MODELS:
  - Ridge: alpha (regularization strength)
  - LightGBM: learning_rate, num_leaves, max_depth, min_data_in_leaf,
              lambda_l1, lambda_l2

**Validates: Requirements 3.1, 3.2, 3.6**
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

from data_loader import FEATURES, TARGET


class HyperparameterTuner:
    """
    Hyperparameter optimization with nested cross-validation.
    
    Maintains temporal ordering and prevents data leakage by:
    1. Using only training data for tuning
    2. Splitting training data into temporal folds
    3. Always validating on data after training data
    """
    
    def __init__(self, model_type="ridge", n_splits=3, scoring="ic", verbose=True):
        """
        Initialize hyperparameter tuner.
        
        Parameters
        ----------
        model_type : str
            Model type: "ridge" or "lightgbm"
        n_splits : int
            Number of temporal folds for inner CV (default: 3)
        scoring : str
            Scoring metric: "ic" (Information Coefficient), "rmse", or "r2"
        verbose : bool
            Print progress messages
        """
        self.model_type = model_type.lower()
        self.n_splits = n_splits
        self.scoring = scoring.lower()
        self.verbose = verbose
        
        if self.model_type not in ["ridge", "lightgbm"]:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        if self.scoring not in ["ic", "rmse", "r2"]:
            raise ValueError(f"Unsupported scoring: {scoring}")
    
    def _temporal_split(self, dates, n_splits):
        """
        Create temporal train/validation splits.
        
        Splits dates into n_splits folds where each validation fold
        comes after its training fold (expanding window within training data).
        
        Parameters
        ----------
        dates : array-like
            Sorted array of unique dates
        n_splits : int
            Number of folds
        
        Returns
        -------
        list of tuples
            Each tuple contains (train_dates, val_dates)
        """
        dates = sorted(dates)
        n_dates = len(dates)
        
        # Minimum training size: 60% of data
        min_train_size = int(n_dates * 0.6)
        
        # Create expanding window splits
        splits = []
        for i in range(n_splits):
            # Validation fold size
            val_size = (n_dates - min_train_size) // n_splits
            
            # Training end index
            train_end = min_train_size + i * val_size
            
            # Validation start and end
            val_start = train_end
            val_end = min(val_start + val_size, n_dates)
            
            if val_end <= val_start:
                continue
            
            train_dates = dates[:train_end]
            val_dates = dates[val_start:val_end]
            
            splits.append((train_dates, val_dates))
        
        return splits
    
    def _compute_score(self, y_true, y_pred):
        """
        Compute scoring metric.
        
        Parameters
        ----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted values
        
        Returns
        -------
        float
            Score (higher is better for IC and R2, lower is better for RMSE)
        """
        if self.scoring == "ic":
            # Information Coefficient (Spearman correlation)
            ic, _ = spearmanr(y_true, y_pred)
            return ic if not np.isnan(ic) else 0.0
        elif self.scoring == "rmse":
            # RMSE (negate so higher is better)
            return -np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.scoring == "r2":
            # R-squared
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-10))
    
    def _cross_validate_params(self, train_df, params):
        """
        Evaluate hyperparameters using temporal cross-validation.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with features and target
        params : dict
            Hyperparameters to evaluate
        
        Returns
        -------
        float
            Mean validation score across folds
        """
        dates = sorted(train_df["date"].unique())
        
        if len(dates) < self.n_splits + 1:
            # Not enough data for CV, use simple train/val split
            split_idx = int(len(dates) * 0.8)
            splits = [(dates[:split_idx], dates[split_idx:])]
        else:
            splits = self._temporal_split(dates, self.n_splits)
        
        fold_scores = []
        
        for fold_idx, (train_dates, val_dates) in enumerate(splits):
            # Split data
            fold_train = train_df[train_df["date"].isin(train_dates)]
            fold_val = train_df[train_df["date"].isin(val_dates)]
            
            if len(fold_train) < 100 or len(fold_val) < 20:
                continue
            
            # Prepare features
            X_train = fold_train[FEATURES].fillna(0).values
            y_train = fold_train[TARGET].values
            X_val = fold_val[FEATURES].fillna(0).values
            y_val = fold_val[TARGET].values
            
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            
            # Train model
            if self.model_type == "ridge":
                model = Ridge(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            else:  # lightgbm
                train_data = lgb.Dataset(X_train, label=y_train)
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[train_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                )
                y_pred = model.predict(X_val)
            
            # Compute score
            score = self._compute_score(y_val, y_pred)
            fold_scores.append(score)
        
        if len(fold_scores) == 0:
            return -np.inf
        
        return np.mean(fold_scores)
    
    def tune(self, train_df, param_grid=None):
        """
        Tune hyperparameters using nested cross-validation.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with columns: date, ticker, features, target
        param_grid : dict or list of dict, optional
            Hyperparameter grid to search. If None, uses default grids.
        
        Returns
        -------
        dict
            Dictionary containing:
            - "best_params": Best hyperparameters found
            - "best_score": Best validation score
            - "param_scores": List of (params, score) tuples for all combinations
            - "cv_results": Detailed cross-validation results
            
        Validates: Requirements 3.5
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        # Convert param_grid to list of parameter combinations
        if isinstance(param_grid, dict):
            param_combinations = self._grid_to_combinations(param_grid)
        else:
            param_combinations = param_grid
        
        if self.verbose:
            print(f"\nTuning {self.model_type} hyperparameters...")
            print(f"  Training data: {len(train_df):,} rows, "
                  f"{train_df['date'].nunique()} months")
            print(f"  Parameter combinations: {len(param_combinations)}")
            print(f"  CV folds: {self.n_splits}")
            print(f"  Scoring: {self.scoring}")
        
        best_score = -np.inf
        best_params = None
        param_scores = []
        
        for idx, params in enumerate(param_combinations):
            score = self._cross_validate_params(train_df, params)
            param_scores.append((params.copy(), score))
            
            if self.verbose and (idx + 1) % max(1, len(param_combinations) // 10) == 0:
                print(f"    Progress: {idx + 1}/{len(param_combinations)} "
                      f"(best {self.scoring}: {best_score:.5f})")
            
            if score > best_score:
                best_score = score
                best_params = params
        
        if self.verbose:
            print(f"  Best {self.scoring}: {best_score:.5f}")
            print(f"  Best parameters: {best_params}")
        
        # Create detailed results
        results = {
            "best_params": best_params,
            "best_score": best_score,
            "param_scores": param_scores,
            "cv_results": {
                "n_splits": self.n_splits,
                "scoring": self.scoring,
                "n_combinations": len(param_combinations)
            }
        }
        
        return results
    
    def _get_default_param_grid(self):
        """Get default hyperparameter grid for the model type."""
        if self.model_type == "ridge":
            return {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
            }
        else:  # lightgbm
            return {
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "max_depth": [3, 5, 7],
                "min_data_in_leaf": [10, 20, 50],
                "lambda_l1": [0.0, 0.1, 1.0],
                "lambda_l2": [0.0, 0.1, 1.0],
                "objective": ["regression"],
                "metric": ["rmse"],
                "verbosity": [-1]
            }
    
    def _grid_to_combinations(self, param_grid):
        """
        Convert parameter grid to list of parameter combinations.
        
        Parameters
        ----------
        param_grid : dict
            Dictionary mapping parameter names to lists of values
        
        Returns
        -------
        list of dict
            All combinations of parameters
        """
        # Separate fixed params from grid params
        fixed_params = {}
        grid_params = {}
        
        for key, value in param_grid.items():
            if isinstance(value, list):
                grid_params[key] = value
            else:
                fixed_params[key] = value
        
        if len(grid_params) == 0:
            return [fixed_params]
        
        # Generate all combinations
        keys = list(grid_params.keys())
        values = [grid_params[k] for k in keys]
        
        combinations = []
        self._generate_combinations(keys, values, 0, {}, combinations, fixed_params)
        
        return combinations
    
    def _generate_combinations(self, keys, values, idx, current, combinations, fixed):
        """Recursively generate all parameter combinations."""
        if idx == len(keys):
            combinations.append({**fixed, **current})
            return
        
        for value in values[idx]:
            current[keys[idx]] = value
            self._generate_combinations(keys, values, idx + 1, current, combinations, fixed)


def tune_ridge_hyperparameters(factors_df, min_train_months=24, param_grid=None):
    """
    Tune Ridge regression hyperparameters using walk-forward validation.
    
    Parameters
    ----------
    factors_df : pd.DataFrame
        Factor data with columns: date, ticker, features, target
    min_train_months : int
        Minimum number of months for initial training window
    param_grid : dict, optional
        Hyperparameter grid. If None, uses default: alpha=[0.01, 0.1, 1.0, 10.0, 100.0]
    
    Returns
    -------
    dict
        Dictionary containing best_params, best_score, param_scores, and cv_results
    """
    tuner = HyperparameterTuner(model_type="ridge", n_splits=3, scoring="ic")
    
    # Use first walk-forward window for tuning
    all_dates = sorted(factors_df["date"].unique())
    if len(all_dates) <= min_train_months:
        raise ValueError(f"Not enough data: {len(all_dates)} months < {min_train_months}")
    
    train_df = factors_df[factors_df["date"].isin(all_dates[:min_train_months])]
    
    results = tuner.tune(train_df, param_grid)
    
    return results


def tune_lightgbm_hyperparameters(factors_df, min_train_months=24, param_grid=None):
    """
    Tune LightGBM hyperparameters using walk-forward validation.
    
    Parameters
    ----------
    factors_df : pd.DataFrame
        Factor data with columns: date, ticker, features, target
    min_train_months : int
        Minimum number of months for initial training window
    param_grid : dict, optional
        Hyperparameter grid. If None, uses default grid with learning_rate,
        num_leaves, max_depth, min_data_in_leaf, lambda_l1, lambda_l2
    
    Returns
    -------
    dict
        Dictionary containing best_params, best_score, param_scores, and cv_results
    """
    tuner = HyperparameterTuner(model_type="lightgbm", n_splits=3, scoring="ic")
    
    # Use first walk-forward window for tuning
    all_dates = sorted(factors_df["date"].unique())
    if len(all_dates) <= min_train_months:
        raise ValueError(f"Not enough data: {len(all_dates)} months < {min_train_months}")
    
    train_df = factors_df[factors_df["date"].isin(all_dates[:min_train_months])]
    
    results = tuner.tune(train_df, param_grid)
    
    return results


if __name__ == "__main__":
    import os
    
    # Load factor data
    print("Loading factor data...")
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    print(f"  {len(factors_df):,} rows, {factors_df['date'].nunique()} months, "
          f"{factors_df['ticker'].nunique()} stocks")
    
    # Tune Ridge hyperparameters
    print("\n" + "="*70)
    print("  RIDGE HYPERPARAMETER TUNING")
    print("="*70)
    
    ridge_results = tune_ridge_hyperparameters(factors_df)
    
    print("\n" + "="*70)
    print("  LIGHTGBM HYPERPARAMETER TUNING")
    print("="*70)
    
    # Tune LightGBM hyperparameters (with smaller grid for demo)
    lgbm_param_grid = {
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
        "max_depth": [5, 7],
        "min_data_in_leaf": [20, 50],
        "lambda_l1": [0.0, 0.1],
        "lambda_l2": [0.0, 0.1],
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1
    }
    
    lgbm_results = tune_lightgbm_hyperparameters(factors_df, param_grid=lgbm_param_grid)
    
    print("\n" + "="*70)
    print("  TUNING COMPLETE")
    print("="*70)
    print(f"\nBest Ridge parameters: {ridge_results['best_params']}")
    print(f"Best Ridge score: {ridge_results['best_score']:.5f}")
    print(f"\nBest LightGBM parameters: {lgbm_results['best_params']}")
    print(f"Best LightGBM score: {lgbm_results['best_score']:.5f}")

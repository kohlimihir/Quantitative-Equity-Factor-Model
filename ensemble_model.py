"""
ensemble_model.py — Ensemble Framework for Ridge and LightGBM
==============================================================
Combines Ridge and LightGBM predictions using learned weights optimized
on validation data. Implements walk-forward ensemble training that prevents
overfitting by learning weights only from training periods.

ENSEMBLE ARCHITECTURE:
  - Base models: Ridge (linear) + LightGBM (non-linear)
  - Weight optimization: Learned on validation data within training window
  - Prediction: weighted_pred = w1 * ridge_pred + w2 * lgbm_pred
  - Constraint: w1 + w2 = 1, w1, w2 >= 0

TEMPORAL INTEGRITY:
  - Weights learned using nested validation within training window
  - No future data used in weight optimization
  - Walk-forward validation maintains temporal ordering

OPTIMIZATION METHODS:
  1. Grid search: Test fixed weight combinations [0.0, 0.1, ..., 1.0]
  2. Gradient-based: Optimize weights to maximize validation IC
  3. Equal weighting: Simple baseline (w1=0.5, w2=0.5)

**Validates: Requirements 3.3**
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")


class EnsembleModel:
    """
    Ensemble model combining Ridge and LightGBM predictions.
    
    Learns optimal weights on validation data to maximize IC while
    maintaining temporal integrity in walk-forward validation.
    """
    
    def __init__(self, method="grid_search", verbose=True):
        """
        Initialize ensemble model.
        
        Parameters
        ----------
        method : str
            Weight optimization method: "grid_search", "optimize", or "equal"
        verbose : bool
            Print progress messages
        """
        self.method = method.lower()
        self.verbose = verbose
        self.weights_history = []
        
        if self.method not in ["grid_search", "optimize", "equal"]:
            raise ValueError(f"Unsupported method: {method}")
    
    def _compute_ic(self, y_true, y_pred):
        """
        Compute Information Coefficient (Spearman correlation).
        
        Parameters
        ----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted values
        
        Returns
        -------
        float
            IC value (Spearman correlation)
        """
        ic, _ = spearmanr(y_true, y_pred)
        return ic if not np.isnan(ic) else 0.0
    
    def _ensemble_predict(self, ridge_pred, lgbm_pred, w1, w2):
        """
        Compute ensemble predictions with given weights.
        
        Parameters
        ----------
        ridge_pred : array-like
            Ridge predictions
        lgbm_pred : array-like
            LightGBM predictions
        w1 : float
            Weight for Ridge (0 to 1)
        w2 : float
            Weight for LightGBM (0 to 1)
        
        Returns
        -------
        array-like
            Ensemble predictions
        """
        return w1 * ridge_pred + w2 * lgbm_pred
    
    def _grid_search_weights(self, ridge_pred, lgbm_pred, y_true):
        """
        Find optimal weights using grid search.
        
        Tests all combinations of weights in [0.0, 0.1, ..., 1.0]
        that sum to 1.0.
        
        Parameters
        ----------
        ridge_pred : array-like
            Ridge predictions on validation data
        lgbm_pred : array-like
            LightGBM predictions on validation data
        y_true : array-like
            True target values
        
        Returns
        -------
        tuple
            (best_w1, best_w2, best_ic)
        """
        best_ic = -np.inf
        best_w1, best_w2 = 0.5, 0.5
        
        # Grid search over weights [0.0, 0.1, ..., 1.0]
        for w1 in np.linspace(0, 1, 11):
            w2 = 1 - w1
            
            # Compute ensemble predictions
            ensemble_pred = self._ensemble_predict(ridge_pred, lgbm_pred, w1, w2)
            
            # Compute IC
            ic = self._compute_ic(y_true, ensemble_pred)
            
            if ic > best_ic:
                best_ic = ic
                best_w1, best_w2 = w1, w2
        
        return best_w1, best_w2, best_ic
    
    def _optimize_weights(self, ridge_pred, lgbm_pred, y_true):
        """
        Find optimal weights using gradient-based optimization.
        
        Maximizes IC by minimizing negative IC with constraint w1 + w2 = 1.
        
        Parameters
        ----------
        ridge_pred : array-like
            Ridge predictions on validation data
        lgbm_pred : array-like
            LightGBM predictions on validation data
        y_true : array-like
            True target values
        
        Returns
        -------
        tuple
            (best_w1, best_w2, best_ic)
        """
        def objective(weights):
            """Objective function: negative IC (to minimize)."""
            w1, w2 = weights
            ensemble_pred = self._ensemble_predict(ridge_pred, lgbm_pred, w1, w2)
            ic = self._compute_ic(y_true, ensemble_pred)
            return -ic  # Minimize negative IC = maximize IC
        
        # Constraints: w1 + w2 = 1, w1 >= 0, w2 >= 0
        constraints = [
            {"type": "eq", "fun": lambda w: w[0] + w[1] - 1}  # w1 + w2 = 1
        ]
        bounds = [(0, 1), (0, 1)]  # 0 <= w1, w2 <= 1
        
        # Initial guess: equal weights
        x0 = [0.5, 0.5]
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100}
        )
        
        if result.success:
            w1, w2 = result.x
            best_ic = -result.fun
        else:
            # Fallback to equal weights if optimization fails
            w1, w2 = 0.5, 0.5
            ensemble_pred = self._ensemble_predict(ridge_pred, lgbm_pred, w1, w2)
            best_ic = self._compute_ic(y_true, ensemble_pred)
        
        return w1, w2, best_ic
    
    def learn_weights(self, ridge_pred, lgbm_pred, y_true):
        """
        Learn optimal ensemble weights on validation data.
        
        Parameters
        ----------
        ridge_pred : array-like
            Ridge predictions on validation data
        lgbm_pred : array-like
            LightGBM predictions on validation data
        y_true : array-like
            True target values
        
        Returns
        -------
        tuple
            (w1, w2, validation_ic)
        """
        if self.method == "equal":
            # Equal weighting baseline
            w1, w2 = 0.5, 0.5
            ensemble_pred = self._ensemble_predict(ridge_pred, lgbm_pred, w1, w2)
            ic = self._compute_ic(y_true, ensemble_pred)
            return w1, w2, ic
        
        elif self.method == "grid_search":
            return self._grid_search_weights(ridge_pred, lgbm_pred, y_true)
        
        elif self.method == "optimize":
            return self._optimize_weights(ridge_pred, lgbm_pred, y_true)
    
    def predict(self, ridge_pred, lgbm_pred, w1, w2):
        """
        Generate ensemble predictions with given weights.
        
        Parameters
        ----------
        ridge_pred : array-like
            Ridge predictions
        lgbm_pred : array-like
            LightGBM predictions
        w1 : float
            Weight for Ridge
        w2 : float
            Weight for LightGBM
        
        Returns
        -------
        array-like
            Ensemble predictions
        """
        return self._ensemble_predict(ridge_pred, lgbm_pred, w1, w2)


def walk_forward_ensemble(ridge_df, lgbm_df, method="grid_search", 
                         validation_months=6, verbose=True):
    """
    Walk-forward ensemble validation with learned weights.
    
    For each test month:
      1. Use previous validation_months as weight optimization window
      2. Learn optimal weights on validation window
      3. Apply weights to generate ensemble predictions for test month
      4. Track weight evolution over time
    
    Parameters
    ----------
    ridge_df : pd.DataFrame
        Ridge predictions with columns: date, ticker, sector, actual, predicted
    lgbm_df : pd.DataFrame
        LightGBM predictions with columns: date, ticker, sector, actual, predicted
    method : str
        Weight optimization method: "grid_search", "optimize", or "equal"
    validation_months : int
        Number of months to use for weight optimization (default: 6)
    verbose : bool
        Print progress messages
    
    Returns
    -------
    tuple
        (ensemble_results_df, weights_df)
        - ensemble_results_df: predictions with ensemble weights
        - weights_df: weight evolution over time
    """
    # Merge Ridge and LightGBM predictions
    merged = ridge_df.merge(
        lgbm_df[["date", "ticker", "predicted"]],
        on=["date", "ticker"],
        suffixes=("", "_lgbm")
    )
    merged = merged.rename(columns={"predicted": "ridge_pred", "predicted_lgbm": "lgbm_pred"})
    
    # Get sorted dates
    all_dates = sorted(merged["date"].unique())
    
    if verbose:
        print(f"\nEnsemble walk-forward validation ({method}):")
        print(f"  Total months: {len(all_dates)}")
        print(f"  Validation window: {validation_months} months")
        print(f"  Test months: {len(all_dates) - validation_months}")
    
    ensemble = EnsembleModel(method=method, verbose=verbose)
    results = []
    weights_history = []
    
    for i, test_date in enumerate(all_dates):
        # Need at least validation_months for weight learning
        if i < validation_months:
            continue
        
        # Validation window: previous validation_months
        val_dates = all_dates[i - validation_months:i]
        val_df = merged[merged["date"].isin(val_dates)]
        
        # Test data: current month
        test_df = merged[merged["date"] == test_date]
        
        if len(val_df) < 50 or len(test_df) == 0:
            continue
        
        # Learn weights on validation window
        w1, w2, val_ic = ensemble.learn_weights(
            val_df["ridge_pred"].values,
            val_df["lgbm_pred"].values,
            val_df["actual"].values
        )
        
        # Generate ensemble predictions for test month
        ensemble_pred = ensemble.predict(
            test_df["ridge_pred"].values,
            test_df["lgbm_pred"].values,
            w1, w2
        )
        
        # Store results
        for j, ticker in enumerate(test_df["ticker"].values):
            results.append({
                "date": test_date,
                "ticker": ticker,
                "sector": test_df["sector"].values[j],
                "actual": test_df["actual"].values[j],
                "ridge_pred": test_df["ridge_pred"].values[j],
                "lgbm_pred": test_df["lgbm_pred"].values[j],
                "ensemble_pred": ensemble_pred[j],
                "w1_ridge": w1,
                "w2_lgbm": w2
            })
        
        # Track weights
        weights_history.append({
            "date": test_date,
            "w1_ridge": w1,
            "w2_lgbm": w2,
            "validation_ic": val_ic
        })
        
        if verbose and (i + 1) % 5 == 0:
            print(f"    Month {i + 1}/{len(all_dates)}: w1={w1:.3f}, w2={w2:.3f}, val_IC={val_ic:.4f}")
    
    results_df = pd.DataFrame(results)
    weights_df = pd.DataFrame(weights_history)
    
    if verbose:
        print(f"\nComplete: {len(results_df):,} ensemble predictions, "
              f"{results_df['date'].nunique()} months")
    
    return results_df, weights_df


def evaluate_ensemble(results_df, weights_df):
    """
    Evaluate ensemble performance and compare with base models.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Ensemble predictions with columns: date, ticker, actual, ridge_pred,
        lgbm_pred, ensemble_pred
    weights_df : pd.DataFrame
        Weight evolution with columns: date, w1_ridge, w2_lgbm, validation_ic
    
    Returns
    -------
    dict
        Performance metrics for Ridge, LightGBM, and Ensemble
    """
    # Compute monthly IC for each model
    monthly_ic = results_df.groupby("date").apply(
        lambda g: pd.Series({
            "ridge_ic": spearmanr(g["actual"], g["ridge_pred"])[0],
            "lgbm_ic": spearmanr(g["actual"], g["lgbm_pred"])[0],
            "ensemble_ic": spearmanr(g["actual"], g["ensemble_pred"])[0]
        })
    )
    
    # Overall metrics
    ridge_ic = results_df["actual"].corr(results_df["ridge_pred"], method="spearman")
    lgbm_ic = results_df["actual"].corr(results_df["lgbm_pred"], method="spearman")
    ensemble_ic = results_df["actual"].corr(results_df["ensemble_pred"], method="spearman")
    
    ridge_ic_mean = monthly_ic["ridge_ic"].mean()
    lgbm_ic_mean = monthly_ic["lgbm_ic"].mean()
    ensemble_ic_mean = monthly_ic["ensemble_ic"].mean()
    
    ridge_ic_std = monthly_ic["ridge_ic"].std()
    lgbm_ic_std = monthly_ic["lgbm_ic"].std()
    ensemble_ic_std = monthly_ic["ensemble_ic"].std()
    
    ridge_ic_ir = ridge_ic_mean / ridge_ic_std if ridge_ic_std > 0 else 0
    lgbm_ic_ir = lgbm_ic_mean / lgbm_ic_std if lgbm_ic_std > 0 else 0
    ensemble_ic_ir = ensemble_ic_mean / ensemble_ic_std if ensemble_ic_std > 0 else 0
    
    # Weight statistics
    mean_w1 = weights_df["w1_ridge"].mean()
    mean_w2 = weights_df["w2_lgbm"].mean()
    std_w1 = weights_df["w1_ridge"].std()
    std_w2 = weights_df["w2_lgbm"].std()
    
    print(f"\n{'='*70}")
    print(f"   ENSEMBLE MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"\n  Model Performance:")
    print(f"    {'Model':<15} {'Mean IC':<12} {'IC Std':<12} {'IC-IR':<12} {'Overall IC':<12}")
    print(f"    {'-'*66}")
    print(f"    {'Ridge':<15} {ridge_ic_mean:>11.5f} {ridge_ic_std:>11.5f} "
          f"{ridge_ic_ir:>11.5f} {ridge_ic:>11.5f}")
    print(f"    {'LightGBM':<15} {lgbm_ic_mean:>11.5f} {lgbm_ic_std:>11.5f} "
          f"{lgbm_ic_ir:>11.5f} {lgbm_ic:>11.5f}")
    print(f"    {'Ensemble':<15} {ensemble_ic_mean:>11.5f} {ensemble_ic_std:>11.5f} "
          f"{ensemble_ic_ir:>11.5f} {ensemble_ic:>11.5f}")
    
    print(f"\n  Ensemble Weights:")
    print(f"    Ridge:    {mean_w1:.3f} ± {std_w1:.3f}")
    print(f"    LightGBM: {mean_w2:.3f} ± {std_w2:.3f}")
    
    print(f"\n  Performance Improvement:")
    ridge_improvement = ((ensemble_ic_mean - ridge_ic_mean) / abs(ridge_ic_mean) * 100 
                        if ridge_ic_mean != 0 else 0)
    lgbm_improvement = ((ensemble_ic_mean - lgbm_ic_mean) / abs(lgbm_ic_mean) * 100 
                       if lgbm_ic_mean != 0 else 0)
    print(f"    vs Ridge:    {ridge_improvement:+.2f}%")
    print(f"    vs LightGBM: {lgbm_improvement:+.2f}%")
    
    print(f"{'='*70}\n")
    
    return {
        "ridge": {
            "mean_ic": ridge_ic_mean,
            "ic_std": ridge_ic_std,
            "ic_ir": ridge_ic_ir,
            "overall_ic": ridge_ic
        },
        "lgbm": {
            "mean_ic": lgbm_ic_mean,
            "ic_std": lgbm_ic_std,
            "ic_ir": lgbm_ic_ir,
            "overall_ic": lgbm_ic
        },
        "ensemble": {
            "mean_ic": ensemble_ic_mean,
            "ic_std": ensemble_ic_std,
            "ic_ir": ensemble_ic_ir,
            "overall_ic": ensemble_ic
        },
        "weights": {
            "mean_w1_ridge": mean_w1,
            "mean_w2_lgbm": mean_w2,
            "std_w1_ridge": std_w1,
            "std_w2_lgbm": std_w2
        }
    }


if __name__ == "__main__":
    import os
    
    # Load Ridge and LightGBM predictions
    print("Loading predictions...")
    ridge_df = pd.read_csv("data/ridge_predictions.csv", parse_dates=["date"])
    lgbm_df = pd.read_csv("data/lgbm_predictions.csv", parse_dates=["date"])
    
    print(f"  Ridge: {len(ridge_df):,} predictions, {ridge_df['date'].nunique()} months")
    print(f"  LightGBM: {len(lgbm_df):,} predictions, {lgbm_df['date'].nunique()} months")
    
    # Test all ensemble methods
    methods = ["equal", "grid_search", "optimize"]
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"  TESTING METHOD: {method.upper()}")
        print(f"{'='*70}")
        
        # Run ensemble
        results_df, weights_df = walk_forward_ensemble(
            ridge_df, lgbm_df, 
            method=method,
            validation_months=6,
            verbose=True
        )
        
        # Evaluate
        metrics = evaluate_ensemble(results_df, weights_df)
        
        # Save results
        results_df.to_csv(f"data/ensemble_{method}_predictions.csv", index=False)
        weights_df.to_csv(f"data/ensemble_{method}_weights.csv", index=False)
        print(f"Saved: data/ensemble_{method}_predictions.csv")
        print(f"Saved: data/ensemble_{method}_weights.csv")
    
    print(f"\n{'='*70}")
    print("  ENSEMBLE FRAMEWORK COMPLETE")
    print(f"{'='*70}")

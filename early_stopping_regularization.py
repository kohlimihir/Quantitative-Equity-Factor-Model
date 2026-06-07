"""
early_stopping_regularization.py — Early Stopping and Regularization Framework
================================================================================
Implements proper early stopping using validation data only and advanced
regularization techniques to prevent overfitting while maintaining model
performance.

EARLY STOPPING:
  - Uses validation data from training period only (no test data leakage)
  - Monitors validation IC/RMSE to stop training when performance plateaus
  - Maintains temporal ordering in validation splits

ADVANCED REGULARIZATION:
  - L1/L2 regularization (Ridge, Lasso, ElasticNet)
  - Dropout-like feature subsampling (LightGBM colsample_bytree)
  - Min samples per leaf constraints
  - Learning rate decay schedules

MODEL COMPLEXITY ANALYSIS:
  - Tracks model complexity metrics (num_leaves, max_depth, n_estimators)
  - Analyzes complexity vs performance tradeoffs
  - Generates diagnostic plots and reports

**Validates: Requirements 3.7**
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from data_loader import FEATURES, TARGET


class EarlyStoppingValidator:
    """
    Early stopping framework using validation data only.
    
    Splits training data into train/validation sets while maintaining
    temporal ordering. Monitors validation performance to determine
    optimal stopping point.
    """
    
    def __init__(self, validation_fraction=0.2, patience=10, min_delta=0.0001,
                 metric="ic", verbose=True):
        """
        Initialize early stopping validator.
        
        Parameters
        ----------
        validation_fraction : float
            Fraction of training data to use for validation (default: 0.2)
        patience : int
            Number of iterations to wait for improvement (default: 10)
        min_delta : float
            Minimum change to qualify as improvement (default: 0.0001)
        metric : str
            Metric to monitor: "ic", "rmse", or "r2" (default: "ic")
        verbose : bool
            Print progress messages (default: True)
        """
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric.lower()
        self.verbose = verbose
        
        if self.metric not in ["ic", "rmse", "r2"]:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def split_train_validation(self, train_df):
        """
        Split training data into train/validation sets.
        
        Maintains temporal ordering: validation data comes after training data.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with date column
        
        Returns
        -------
        tuple
            (train_subset_df, validation_df)
        """
        dates = sorted(train_df["date"].unique())
        n_dates = len(dates)
        
        # Split point: last validation_fraction of dates
        split_idx = int(n_dates * (1 - self.validation_fraction))
        
        train_dates = dates[:split_idx]
        val_dates = dates[split_idx:]
        
        train_subset = train_df[train_df["date"].isin(train_dates)]
        validation = train_df[train_df["date"].isin(val_dates)]
        
        return train_subset, validation
    
    def compute_metric(self, y_true, y_pred):
        """
        Compute evaluation metric.
        
        Parameters
        ----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted values
        
        Returns
        -------
        float
            Metric value (higher is better for IC and R2, lower is better for RMSE)
        """
        if self.metric == "ic":
            ic, _ = spearmanr(y_true, y_pred)
            return ic if not np.isnan(ic) else 0.0
        elif self.metric == "rmse":
            return -np.sqrt(np.mean((y_true - y_pred) ** 2))  # Negate for maximization
        elif self.metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-10))
    
    def train_with_early_stopping_lgbm(self, train_df, lgbm_params=None,
                                       max_rounds=1000):
        """
        Train LightGBM model with early stopping.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with features and target
        lgbm_params : dict, optional
            LightGBM parameters (default: None, uses default params)
        max_rounds : int
            Maximum number of boosting rounds (default: 1000)
        
        Returns
        -------
        dict
            Dictionary containing:
            - "model": Trained LightGBM model
            - "best_iteration": Optimal number of iterations
            - "train_scores": Training scores by iteration
            - "val_scores": Validation scores by iteration
            - "stopped_early": Whether early stopping was triggered
        """
        # Split into train/validation
        train_subset, validation = self.split_train_validation(train_df)
        
        if len(train_subset) < 100 or len(validation) < 20:
            raise ValueError("Insufficient data for train/validation split")
        
        # Prepare features
        X_train = train_subset[FEATURES].fillna(0).values
        y_train = train_subset[TARGET].values
        X_val = validation[FEATURES].fillna(0).values
        y_val = validation[TARGET].values
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Default LightGBM parameters
        if lgbm_params is None:
            lgbm_params = {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 80,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.2,
                "reg_lambda": 0.2,
                "random_state": 42,
                "verbosity": -1,
            }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train with early stopping
        if self.verbose:
            print(f"  Training with early stopping (patience={self.patience})...")
            print(f"    Train: {len(train_subset):,} rows, Val: {len(validation):,} rows")
        
        evals_result = {}
        model = lgb.train(
            lgbm_params,
            train_data,
            num_boost_round=max_rounds,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.patience, verbose=False),
                lgb.log_evaluation(period=0),  # Disable default logging
                lgb.record_evaluation(evals_result)
            ]
        )
        
        best_iteration = model.best_iteration
        stopped_early = best_iteration < max_rounds
        
        # Get training history from evals_result
        train_scores = evals_result.get("train", {})
        val_scores = evals_result.get("val", {})
        
        if self.verbose:
            print(f"    Best iteration: {best_iteration}")
            print(f"    Stopped early: {stopped_early}")
            if val_scores:
                for metric_name, scores in val_scores.items():
                    # Get the score at best iteration
                    if len(scores) > 0:
                        best_score = scores[min(best_iteration - 1, len(scores) - 1)]
                        print(f"    Val {metric_name}: {best_score:.5f}")
        
        return {
            "model": model,
            "best_iteration": best_iteration,
            "train_scores": train_scores,
            "val_scores": val_scores,
            "stopped_early": stopped_early,
            "scaler": scaler
        }
    
    def train_with_regularization_ridge(self, train_df, alpha=1.0,
                                       regularization_type="ridge"):
        """
        Train Ridge/Lasso/ElasticNet model with regularization.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with features and target
        alpha : float
            Regularization strength (default: 1.0)
        regularization_type : str
            Type of regularization: "ridge", "lasso", or "elasticnet" (default: "ridge")
        
        Returns
        -------
        dict
            Dictionary containing:
            - "model": Trained model
            - "train_score": Training score
            - "val_score": Validation score
            - "scaler": Feature scaler
        """
        # Split into train/validation
        train_subset, validation = self.split_train_validation(train_df)
        
        if len(train_subset) < 100 or len(validation) < 20:
            raise ValueError("Insufficient data for train/validation split")
        
        # Prepare features
        X_train = train_subset[FEATURES].fillna(0).values
        y_train = train_subset[TARGET].values
        X_val = validation[FEATURES].fillna(0).values
        y_val = validation[TARGET].values
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Select model type
        if regularization_type.lower() == "ridge":
            model = Ridge(alpha=alpha, random_state=42)
        elif regularization_type.lower() == "lasso":
            model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        elif regularization_type.lower() == "elasticnet":
            model = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42, max_iter=10000)
        else:
            raise ValueError(f"Unsupported regularization_type: {regularization_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        train_score = self.compute_metric(y_train, y_train_pred)
        val_score = self.compute_metric(y_val, y_val_pred)
        
        if self.verbose:
            print(f"  {regularization_type.capitalize()} (alpha={alpha}):")
            print(f"    Train {self.metric}: {train_score:.5f}")
            print(f"    Val {self.metric}: {val_score:.5f}")
        
        return {
            "model": model,
            "train_score": train_score,
            "val_score": val_score,
            "scaler": scaler
        }


class ModelComplexityAnalyzer:
    """
    Analyzes model complexity vs performance tradeoffs.
    
    Tracks complexity metrics and generates diagnostic reports showing
    how model complexity affects performance.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize model complexity analyzer.
        
        Parameters
        ----------
        verbose : bool
            Print progress messages (default: True)
        """
        self.verbose = verbose
        self.results = []
    
    def analyze_lgbm_complexity(self, train_df, param_grid=None):
        """
        Analyze LightGBM complexity vs performance.
        
        Tests different complexity configurations and measures their
        impact on training and validation performance.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with features and target
        param_grid : dict, optional
            Parameter grid to test (default: None, uses default grid)
        
        Returns
        -------
        pd.DataFrame
            Results with complexity metrics and performance scores
        """
        if param_grid is None:
            param_grid = {
                "num_leaves": [15, 31, 63, 127],
                "max_depth": [3, 5, 7, 10],
                "min_child_samples": [10, 20, 50, 100],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
            }
        
        validator = EarlyStoppingValidator(verbose=False)
        
        if self.verbose:
            print("\nAnalyzing LightGBM complexity vs performance...")
        
        results = []
        
        # Test different complexity configurations
        for num_leaves in param_grid.get("num_leaves", [31]):
            for max_depth in param_grid.get("max_depth", [5]):
                for min_samples in param_grid.get("min_child_samples", [80]):
                    for lr in param_grid.get("learning_rate", [0.03]):
                        params = {
                            "objective": "regression",
                            "metric": "rmse",
                            "learning_rate": lr,
                            "num_leaves": num_leaves,
                            "max_depth": max_depth,
                            "min_child_samples": min_samples,
                            "subsample": 0.7,
                            "colsample_bytree": 0.7,
                            "reg_alpha": 0.2,
                            "reg_lambda": 0.2,
                            "random_state": 42,
                            "verbosity": -1,
                        }
                        
                        try:
                            result = validator.train_with_early_stopping_lgbm(
                                train_df, lgbm_params=params, max_rounds=500
                            )
                            
                            # Compute complexity score
                            complexity_score = (
                                num_leaves * max_depth / (min_samples + 1)
                            )
                            
                            # Extract validation RMSE from result
                            val_rmse = np.nan
                            if result["val_scores"] and "rmse" in result["val_scores"]:
                                rmse_scores = result["val_scores"]["rmse"]
                                if len(rmse_scores) > 0:
                                    best_iter = result["best_iteration"]
                                    val_rmse = rmse_scores[min(best_iter - 1, len(rmse_scores) - 1)]
                            
                            results.append({
                                "num_leaves": num_leaves,
                                "max_depth": max_depth,
                                "min_child_samples": min_samples,
                                "learning_rate": lr,
                                "best_iteration": result["best_iteration"],
                                "stopped_early": result["stopped_early"],
                                "complexity_score": complexity_score,
                                "val_rmse": val_rmse,
                            })
                            
                            if self.verbose and len(results) % 5 == 0:
                                print(f"  Tested {len(results)} configurations...")
                        
                        except Exception as e:
                            if self.verbose:
                                print(f"  Warning: Configuration failed: {e}")
                            continue
        
        results_df = pd.DataFrame(results)
        self.results = results_df
        
        if self.verbose:
            print(f"  Completed: {len(results_df)} configurations tested")
            print(f"\n  Best configurations by validation RMSE:")
            best_configs = results_df.nsmallest(5, "val_rmse")
            for idx, row in best_configs.iterrows():
                print(f"    leaves={row['num_leaves']}, depth={row['max_depth']}, "
                      f"min_samples={row['min_child_samples']}, lr={row['learning_rate']:.3f}")
                print(f"      Val RMSE: {row['val_rmse']:.5f}, "
                      f"Iterations: {row['best_iteration']}, "
                      f"Complexity: {row['complexity_score']:.2f}")
        
        return results_df
    
    def analyze_ridge_regularization(self, train_df, alpha_range=None):
        """
        Analyze Ridge regularization strength vs performance.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with features and target
        alpha_range : list, optional
            List of alpha values to test (default: None, uses default range)
        
        Returns
        -------
        pd.DataFrame
            Results with regularization strength and performance scores
        """
        if alpha_range is None:
            alpha_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        
        validator = EarlyStoppingValidator(verbose=False)
        
        if self.verbose:
            print("\nAnalyzing Ridge regularization strength vs performance...")
        
        results = []
        
        for alpha in alpha_range:
            try:
                result = validator.train_with_regularization_ridge(
                    train_df, alpha=alpha, regularization_type="ridge"
                )
                
                results.append({
                    "alpha": alpha,
                    "train_ic": result["train_score"],
                    "val_ic": result["val_score"],
                    "overfit_gap": result["train_score"] - result["val_score"],
                })
                
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Alpha {alpha} failed: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        if self.verbose:
            print(f"  Completed: {len(results_df)} alpha values tested")
            print(f"\n  Best configurations by validation IC:")
            best_configs = results_df.nlargest(5, "val_ic")
            for idx, row in best_configs.iterrows():
                print(f"    alpha={row['alpha']:.3f}: "
                      f"Train IC={row['train_ic']:.5f}, "
                      f"Val IC={row['val_ic']:.5f}, "
                      f"Gap={row['overfit_gap']:.5f}")
        
        return results_df
    
    def plot_complexity_analysis(self, output_path="outputs/complexity_analysis.png"):
        """
        Generate complexity vs performance plots.
        
        Parameters
        ----------
        output_path : str
            Path to save the plot (default: "outputs/complexity_analysis.png")
        """
        if len(self.results) == 0:
            print("  Warning: No results to plot. Run analyze_lgbm_complexity first.")
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("  Warning: matplotlib not installed. Skipping plot generation.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("white")
        
        NAVY = "#1f4e79"
        TEAL = "#0f6e56"
        ORANGE = "#c55a11"
        
        # Plot 1: Complexity score vs validation RMSE
        ax1 = axes[0, 0]
        ax1.scatter(self.results["complexity_score"], self.results["val_rmse"],
                   alpha=0.6, color=NAVY, s=50)
        ax1.set_xlabel("Complexity Score")
        ax1.set_ylabel("Validation RMSE")
        ax1.set_title("Model Complexity vs Performance", fontweight="bold")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Number of leaves vs validation RMSE
        ax2 = axes[0, 1]
        for depth in sorted(self.results["max_depth"].unique()):
            subset = self.results[self.results["max_depth"] == depth]
            ax2.plot(subset["num_leaves"], subset["val_rmse"],
                    marker="o", label=f"depth={depth}", linewidth=2)
        ax2.set_xlabel("Number of Leaves")
        ax2.set_ylabel("Validation RMSE")
        ax2.set_title("Num Leaves vs Performance (by depth)", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning rate vs iterations
        ax3 = axes[1, 0]
        for lr in sorted(self.results["learning_rate"].unique()):
            subset = self.results[self.results["learning_rate"] == lr]
            ax3.scatter(subset["complexity_score"], subset["best_iteration"],
                       label=f"lr={lr:.3f}", alpha=0.6, s=50)
        ax3.set_xlabel("Complexity Score")
        ax3.set_ylabel("Best Iteration (Early Stopping)")
        ax3.set_title("Complexity vs Training Iterations", fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Min samples vs validation RMSE
        ax4 = axes[1, 1]
        for leaves in sorted(self.results["num_leaves"].unique()):
            subset = self.results[self.results["num_leaves"] == leaves]
            ax4.plot(subset["min_child_samples"], subset["val_rmse"],
                    marker="s", label=f"leaves={leaves}", linewidth=2)
        ax4.set_xlabel("Min Child Samples")
        ax4.set_ylabel("Validation RMSE")
        ax4.set_title("Min Samples vs Performance (by leaves)", fontweight="bold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"  ✓ Complexity analysis plot saved to {output_path}")


def walk_forward_with_early_stopping(factors_df, min_train_months=24,
                                     use_early_stopping=True):
    """
    Walk-forward validation with early stopping.
    
    Parameters
    ----------
    factors_df : pd.DataFrame
        Factor data with columns: date, ticker, features, target
    min_train_months : int
        Minimum number of months for initial training window (default: 24)
    use_early_stopping : bool
        Whether to use early stopping (default: True)
    
    Returns
    -------
    tuple
        (results_df, training_history)
    """
    all_dates = sorted(factors_df["date"].unique())
    results = []
    training_history = []
    
    print(f"\nWalk-forward validation with early stopping: {len(all_dates)} months")
    
    validator = EarlyStoppingValidator(validation_fraction=0.2, patience=10,
                                      metric="ic", verbose=False)
    
    for i, test_date in enumerate(all_dates):
        if i < min_train_months:
            continue
        
        train_df = factors_df[factors_df["date"].isin(all_dates[:i])]
        test_df = factors_df[factors_df["date"] == test_date]
        
        if len(train_df) < 200 or len(test_df) == 0:
            continue
        
        # Train with early stopping
        if use_early_stopping:
            result = validator.train_with_early_stopping_lgbm(train_df, max_rounds=500)
            model = result["model"]
            scaler = result["scaler"]
            
            training_history.append({
                "date": test_date,
                "best_iteration": result["best_iteration"],
                "stopped_early": result["stopped_early"],
            })
        else:
            # Train without early stopping (fixed iterations)
            X_train = train_df[FEATURES].fillna(0).values
            y_train = train_df[TARGET].values
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            
            params = {
                "objective": "regression",
                "metric": "rmse",
                "n_estimators": 300,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 80,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.2,
                "reg_lambda": 0.2,
                "random_state": 42,
                "verbosity": -1,
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
        
        # Predict on test set
        X_test = test_df[FEATURES].fillna(0).values
        y_test = test_df[TARGET].values
        X_test = scaler.transform(X_test)
        
        y_pred = model.predict(X_test)
        
        for j, ticker in enumerate(test_df["ticker"].values):
            results.append({
                "date": test_date,
                "ticker": ticker,
                "sector": test_df["sector"].values[j],
                "actual": y_test[j],
                "predicted": y_pred[j],
            })
        
        if (i - min_train_months + 1) % 5 == 0:
            print(f"  Progress: {i - min_train_months + 1}/{len(all_dates) - min_train_months} months")
    
    results_df = pd.DataFrame(results)
    training_history_df = pd.DataFrame(training_history)
    
    print(f"  Complete: {len(results_df):,} predictions, {results_df['date'].nunique()} months")
    
    return results_df, training_history_df


if __name__ == "__main__":
    import os
    
    # Load factor data
    print("Loading factor data...")
    factors_df = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
    print(f"  {len(factors_df):,} rows, {factors_df['date'].nunique()} months, "
          f"{factors_df['ticker'].nunique()} stocks")
    
    # Test early stopping on first walk-forward window
    print("\n" + "="*70)
    print("  EARLY STOPPING VALIDATION TEST")
    print("="*70)
    
    all_dates = sorted(factors_df["date"].unique())
    train_df = factors_df[factors_df["date"].isin(all_dates[:24])]
    
    validator = EarlyStoppingValidator(validation_fraction=0.2, patience=10,
                                      metric="ic", verbose=True)
    
    print("\n1. LightGBM with early stopping:")
    lgbm_result = validator.train_with_early_stopping_lgbm(train_df, max_rounds=500)
    
    print("\n2. Ridge with regularization:")
    ridge_result = validator.train_with_regularization_ridge(train_df, alpha=1.0)
    
    print("\n3. Lasso with regularization:")
    lasso_result = validator.train_with_regularization_ridge(
        train_df, alpha=0.1, regularization_type="lasso"
    )
    
    print("\n4. ElasticNet with regularization:")
    elasticnet_result = validator.train_with_regularization_ridge(
        train_df, alpha=0.1, regularization_type="elasticnet"
    )
    
    # Model complexity analysis
    print("\n" + "="*70)
    print("  MODEL COMPLEXITY ANALYSIS")
    print("="*70)
    
    analyzer = ModelComplexityAnalyzer(verbose=True)
    
    # Analyze LightGBM complexity (smaller grid for demo)
    lgbm_complexity = analyzer.analyze_lgbm_complexity(
        train_df,
        param_grid={
            "num_leaves": [15, 31, 63],
            "max_depth": [3, 5, 7],
            "min_child_samples": [20, 50, 100],
            "learning_rate": [0.03, 0.05],
        }
    )
    
    # Save results
    os.makedirs("outputs", exist_ok=True)
    lgbm_complexity.to_csv("outputs/lgbm_complexity_analysis.csv", index=False)
    print(f"\n  ✓ Saved: outputs/lgbm_complexity_analysis.csv")
    
    # Generate plots
    analyzer.plot_complexity_analysis()
    
    # Analyze Ridge regularization
    ridge_complexity = analyzer.analyze_ridge_regularization(train_df)
    ridge_complexity.to_csv("outputs/ridge_regularization_analysis.csv", index=False)
    print(f"  ✓ Saved: outputs/ridge_regularization_analysis.csv")
    
    print("\n" + "="*70)
    print("  EARLY STOPPING AND REGULARIZATION COMPLETE")
    print("="*70)

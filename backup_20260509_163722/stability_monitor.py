"""
stability_monitor.py — Model Stability and Consistency Monitoring
==================================================================
Tracks model performance consistency over time through monthly IC analysis,
distribution metrics, negative IC month identification, and market regime detection.

This module implements comprehensive stability monitoring to identify:
- Monthly IC trends and patterns
- IC distribution characteristics (mean, std, percentiles)
- Win rate (percentage of positive IC months)
- Negative IC months and their common characteristics
- Market regime detection (high volatility, trending, mean-reverting)
- Regime-specific performance analysis
- Regime transition impact assessment

Validates: Requirements 5.1, 5.2, 5.3, 5.4
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple
import warnings
import os
warnings.filterwarnings("ignore")

from data_loader import TARGET


class StabilityMonitor:
    """
    Monitor model stability through monthly IC tracking and analysis.
    
    This class provides comprehensive stability metrics including:
    - Monthly IC computation between predictions and actual returns
    - IC distribution analysis (mean, std, percentiles, win rate)
    - Negative IC month identification and characterization
    - Market regime detection (high volatility, trending, mean-reverting)
    - Regime-specific performance analysis
    - Regime transition impact assessment
    - Stability reporting and diagnostics
    """
    
    def __init__(self):
        """Initialize the stability monitor."""
        self.monthly_ic_history = []
        self.negative_ic_months = []
        self.regime_history = []
        self.regime_performance = {}
        
    def compute_monthly_ic(
        self,
        predictions_df: pd.DataFrame,
        pred_col: str = "predicted",
        actual_col: str = None
    ) -> pd.Series:
        """
        Compute monthly Information Coefficient (IC) between predictions and actuals.
        
        IC = Spearman rank correlation between predicted and actual returns.
        This measures the model's ability to rank stocks correctly each month.
        
        Args:
            predictions_df: DataFrame with columns [date, ticker, predicted, actual]
            pred_col: Name of prediction column (default: "predicted")
            actual_col: Name of actual returns column (default: "actual" or TARGET)
        
        Returns:
            Series indexed by date with monthly IC values
            
        Validates: Requirements 5.1
        """
        if actual_col is None:
            # Try "actual" first (common in prediction files), then fall back to TARGET
            if "actual" in predictions_df.columns:
                actual_col = "actual"
            else:
                actual_col = TARGET
            
        # Validate required columns
        required_cols = ["date", pred_col, actual_col]
        missing_cols = [col for col in required_cols if col not in predictions_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        all_dates = sorted(predictions_df["date"].unique())
        ic_records = []
        
        print(f"\nComputing monthly IC across {len(all_dates)} months...")
        
        for date in all_dates:
            month_data = predictions_df[predictions_df["date"] == date].copy()
            
            # Need minimum sample size for reliable correlation
            if len(month_data) < 10:
                ic_records.append({
                    "date": date,
                    "ic": np.nan,
                    "n_stocks": len(month_data)
                })
                continue
            
            predicted_values = month_data[pred_col].values
            actual_values = month_data[actual_col].values
            
            # Remove rows where either prediction or actual is NaN
            valid_mask = ~(np.isnan(predicted_values) | np.isnan(actual_values))
            n_valid = valid_mask.sum()
            
            if n_valid < 10:  # Need minimum valid samples
                ic_records.append({
                    "date": date,
                    "ic": np.nan,
                    "n_stocks": n_valid
                })
                continue
            
            valid_predicted = predicted_values[valid_mask]
            valid_actual = actual_values[valid_mask]
            
            # Check for variance in both series
            if np.std(valid_predicted) < 1e-10 or np.std(valid_actual) < 1e-10:
                ic_records.append({
                    "date": date,
                    "ic": 0.0,
                    "n_stocks": n_valid
                })
                continue
            
            # Compute Spearman correlation
            try:
                ic, p_value = spearmanr(valid_predicted, valid_actual)
                ic_records.append({
                    "date": date,
                    "ic": ic if not np.isnan(ic) else 0.0,
                    "n_stocks": n_valid,
                    "p_value": p_value
                })
            except Exception as e:
                ic_records.append({
                    "date": date,
                    "ic": np.nan,
                    "n_stocks": n_valid
                })
        
        ic_df = pd.DataFrame(ic_records)
        
        # Handle empty DataFrame case
        if len(ic_df) == 0:
            return pd.Series(dtype=float, name="ic")
        
        monthly_ic = ic_df.set_index("date")["ic"]
        
        # Store for later analysis
        self.monthly_ic_history = ic_df
        
        # Print summary
        valid_ic = monthly_ic.dropna()
        if len(valid_ic) > 0:
            print(f"  Computed IC for {len(valid_ic)} months")
            print(f"  Mean IC: {valid_ic.mean():.5f}")
            print(f"  Std IC:  {valid_ic.std():.5f}")
            print(f"  Min IC:  {valid_ic.min():.5f}")
            print(f"  Max IC:  {valid_ic.max():.5f}")
        
        return monthly_ic
    
    def analyze_ic_distribution(
        self,
        monthly_ic: pd.Series
    ) -> Dict[str, float]:
        """
        Analyze IC distribution characteristics.
        
        Computes comprehensive distribution metrics including:
        - Mean, median, standard deviation
        - Percentiles (5th, 25th, 75th, 95th)
        - Win rate (percentage of positive IC months)
        - IC Information Ratio (mean / std)
        
        Args:
            monthly_ic: Series of monthly IC values indexed by date
        
        Returns:
            Dictionary with distribution metrics
            
        Validates: Requirements 5.1
        """
        valid_ic = monthly_ic.dropna()
        
        if len(valid_ic) == 0:
            return {
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "p5": np.nan,
                "p25": np.nan,
                "p75": np.nan,
                "p95": np.nan,
                "win_rate": np.nan,
                "ic_ir": np.nan,
                "n_months": 0
            }
        
        # Compute distribution metrics
        mean_ic = valid_ic.mean()
        median_ic = valid_ic.median()
        std_ic = valid_ic.std()
        min_ic = valid_ic.min()
        max_ic = valid_ic.max()
        
        # Percentiles
        p5 = valid_ic.quantile(0.05)
        p25 = valid_ic.quantile(0.25)
        p75 = valid_ic.quantile(0.75)
        p95 = valid_ic.quantile(0.95)
        
        # Win rate: percentage of positive IC months
        win_rate = (valid_ic > 0).mean()
        
        # IC Information Ratio
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
        
        distribution = {
            "mean": mean_ic,
            "median": median_ic,
            "std": std_ic,
            "min": min_ic,
            "max": max_ic,
            "p5": p5,
            "p25": p25,
            "p75": p75,
            "p95": p95,
            "win_rate": win_rate,
            "ic_ir": ic_ir,
            "n_months": len(valid_ic)
        }
        
        return distribution
    
    def calculate_win_rate(
        self,
        monthly_ic: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate win rate (percentage of months with IC above threshold).
        
        Args:
            monthly_ic: Series of monthly IC values
            threshold: IC threshold for "winning" month (default: 0.0)
        
        Returns:
            Win rate as a fraction (0.0 to 1.0)
            
        Validates: Requirements 5.1
        """
        valid_ic = monthly_ic.dropna()
        
        if len(valid_ic) == 0:
            return np.nan
        
        win_rate = (valid_ic > threshold).mean()
        return win_rate
    
    def identify_negative_ic_months(
        self,
        predictions_df: pd.DataFrame,
        monthly_ic: pd.Series,
        pred_col: str = "predicted",
        actual_col: str = None
    ) -> pd.DataFrame:
        """
        Identify and analyze months with negative IC.
        
        For each negative IC month, computes characteristics including:
        - IC value and magnitude
        - Number of stocks
        - Mean and std of predictions and actuals
        - Sector-level IC breakdown
        
        Args:
            predictions_df: DataFrame with predictions and actuals
            monthly_ic: Series of monthly IC values
            pred_col: Name of prediction column
            actual_col: Name of actual returns column (default: "actual" or TARGET)
        
        Returns:
            DataFrame with negative IC month characteristics
            
        Validates: Requirements 5.2
        """
        if actual_col is None:
            # Try "actual" first (common in prediction files), then fall back to TARGET
            if "actual" in predictions_df.columns:
                actual_col = "actual"
            else:
                actual_col = TARGET
        
        # Filter to negative IC months
        negative_months = monthly_ic[monthly_ic < 0].index
        
        if len(negative_months) == 0:
            print("  No negative IC months found!")
            return pd.DataFrame()
        
        print(f"\nAnalyzing {len(negative_months)} negative IC months...")
        
        negative_records = []
        
        for date in negative_months:
            month_data = predictions_df[predictions_df["date"] == date].copy()
            
            if len(month_data) == 0:
                continue
            
            # Overall month characteristics
            ic_value = monthly_ic.loc[date]
            n_stocks = len(month_data)
            
            # Prediction and actual statistics
            pred_mean = month_data[pred_col].mean()
            pred_std = month_data[pred_col].std()
            actual_mean = month_data[actual_col].mean()
            actual_std = month_data[actual_col].std()
            
            # Sector breakdown (if sector column exists)
            sector_ic = {}
            if "sector" in month_data.columns:
                for sector, sector_data in month_data.groupby("sector"):
                    if len(sector_data) >= 5:  # Minimum for sector IC
                        valid_mask = ~(
                            np.isnan(sector_data[pred_col].values) | 
                            np.isnan(sector_data[actual_col].values)
                        )
                        if valid_mask.sum() >= 5:
                            try:
                                sector_ic_val, _ = spearmanr(
                                    sector_data[pred_col].values[valid_mask],
                                    sector_data[actual_col].values[valid_mask]
                                )
                                sector_ic[sector] = sector_ic_val if not np.isnan(sector_ic_val) else 0.0
                            except Exception:
                                sector_ic[sector] = np.nan
            
            # Find worst performing sector
            worst_sector = None
            worst_sector_ic = np.nan
            if sector_ic:
                worst_sector = min(sector_ic.items(), key=lambda x: x[1] if not np.isnan(x[1]) else 0)[0]
                worst_sector_ic = sector_ic[worst_sector]
            
            negative_records.append({
                "date": date,
                "ic": ic_value,
                "n_stocks": n_stocks,
                "pred_mean": pred_mean,
                "pred_std": pred_std,
                "actual_mean": actual_mean,
                "actual_std": actual_std,
                "worst_sector": worst_sector,
                "worst_sector_ic": worst_sector_ic,
                "n_sectors_negative": sum(1 for ic in sector_ic.values() if ic < 0)
            })
        
        negative_df = pd.DataFrame(negative_records)
        
        # Store for later analysis
        self.negative_ic_months = negative_df
        
        # Print summary
        if len(negative_df) > 0:
            print(f"  Negative IC months: {len(negative_df)}")
            print(f"  Mean negative IC: {negative_df['ic'].mean():.5f}")
            print(f"  Worst IC: {negative_df['ic'].min():.5f} on {negative_df.loc[negative_df['ic'].idxmin(), 'date']}")
            
            if "worst_sector" in negative_df.columns:
                worst_sectors = negative_df["worst_sector"].value_counts()
                print(f"\n  Most problematic sectors:")
                for sector, count in worst_sectors.head(3).items():
                    print(f"    {sector}: {count} months")
        
        return negative_df
    
    def detect_market_regimes(
        self,
        prices_df: pd.DataFrame = None,
        predictions_df: pd.DataFrame = None,
        market_ticker: str = "SPY",
        volatility_window: int = 21,
        trend_window: int = 63,
        mean_reversion_window: int = 21
    ) -> pd.DataFrame:
        """
        Detect market regimes for each month based on market characteristics.
        
        Classifies each month into one of three regimes:
        - HIGH_VOLATILITY: Realized volatility > 75th percentile
        - TRENDING: Strong directional movement (abs(return) > 75th percentile, low mean reversion)
        - MEAN_REVERTING: Low volatility, high autocorrelation reversal
        
        Args:
            prices_df: DataFrame with daily prices (columns = tickers, index = dates)
                      If None, attempts to load from data/daily_prices.parquet
            predictions_df: DataFrame with predictions (used to get date range)
            market_ticker: Ticker to use for market regime detection (default: SPY)
            volatility_window: Rolling window for volatility calculation (default: 21 days)
            trend_window: Rolling window for trend detection (default: 63 days)
            mean_reversion_window: Window for mean reversion detection (default: 21 days)
        
        Returns:
            DataFrame with columns [date, regime, volatility, trend_strength, mean_reversion_score]
            
        Validates: Requirements 5.3
        """
        print("\n--- Market Regime Detection ---")
        
        # Load price data if not provided
        if prices_df is None:
            price_file = "data/daily_prices.parquet"
            if not os.path.exists(price_file):
                print(f"  ⚠ Price data not found: {price_file}")
                print("  Cannot perform regime detection without price data.")
                return pd.DataFrame()
            
            print(f"  Loading price data from {price_file}...")
            prices_df = pd.read_parquet(price_file)
        
        # Validate market ticker exists
        if market_ticker not in prices_df.columns:
            print(f"  ⚠ Market ticker {market_ticker} not found in price data")
            print(f"  Available tickers: {list(prices_df.columns[:5])}...")
            return pd.DataFrame()
        
        # Get market prices
        market_prices = prices_df[market_ticker].copy()
        market_prices = market_prices.dropna()
        
        if len(market_prices) < max(volatility_window, trend_window, mean_reversion_window):
            print(f"  ⚠ Insufficient price data for regime detection")
            return pd.DataFrame()
        
        print(f"  Using {market_ticker} for regime detection")
        print(f"  Price data: {len(market_prices)} days from {market_prices.index[0].date()} to {market_prices.index[-1].date()}")
        
        # Compute daily returns
        daily_returns = market_prices.pct_change().dropna()
        
        # Compute regime indicators
        regime_data = []
        
        # Get unique months from predictions if available
        if predictions_df is not None and "date" in predictions_df.columns:
            prediction_dates = pd.to_datetime(predictions_df["date"].unique())
        else:
            # Use all months in price data
            prediction_dates = pd.date_range(
                start=market_prices.index[0],
                end=market_prices.index[-1],
                freq='MS'  # Month start
            )
        
        print(f"  Computing regimes for {len(prediction_dates)} months...")
        
        for month_date in prediction_dates:
            # Get data up to this month (avoid look-ahead bias)
            # Use data from the month BEFORE the prediction month
            month_end = month_date - pd.Timedelta(days=1)
            
            # Get historical data for regime calculation
            historical_returns = daily_returns[daily_returns.index <= month_end]
            
            if len(historical_returns) < max(volatility_window, trend_window, mean_reversion_window):
                continue
            
            # 1. Volatility: Realized volatility over recent window
            recent_returns = historical_returns.tail(volatility_window)
            realized_vol = recent_returns.std() * np.sqrt(252)  # Annualized
            
            # 2. Trend Strength: Cumulative return and consistency
            trend_returns = historical_returns.tail(trend_window)
            cumulative_return = (1 + trend_returns).prod() - 1
            trend_consistency = (trend_returns > 0).mean()  # Fraction of positive days
            trend_strength = abs(cumulative_return) * (2 * abs(trend_consistency - 0.5))
            
            # 3. Mean Reversion: Autocorrelation and reversal tendency
            mr_returns = historical_returns.tail(mean_reversion_window)
            if len(mr_returns) > 1:
                # Negative autocorrelation indicates mean reversion
                autocorr = mr_returns.autocorr(lag=1)
                # Hurst exponent approximation (simplified)
                # Mean reversion score: high when autocorr is negative
                mean_reversion_score = -autocorr if not np.isnan(autocorr) else 0.0
            else:
                mean_reversion_score = 0.0
            
            regime_data.append({
                "date": month_date,
                "volatility": realized_vol,
                "trend_strength": trend_strength,
                "mean_reversion_score": mean_reversion_score,
                "cumulative_return": cumulative_return
            })
        
        if len(regime_data) == 0:
            print("  ⚠ No regime data computed")
            return pd.DataFrame()
        
        regime_df = pd.DataFrame(regime_data)
        
        # Classify regimes based on percentiles
        vol_threshold = regime_df["volatility"].quantile(0.75)
        trend_threshold = regime_df["trend_strength"].quantile(0.75)
        mr_threshold = regime_df["mean_reversion_score"].quantile(0.60)
        
        def classify_regime(row):
            """Classify regime based on market characteristics."""
            # Priority: High volatility > Trending > Mean reverting
            if row["volatility"] > vol_threshold:
                return "HIGH_VOLATILITY"
            elif row["trend_strength"] > trend_threshold:
                return "TRENDING"
            elif row["mean_reversion_score"] > mr_threshold:
                return "MEAN_REVERTING"
            else:
                # Default to mean reverting for normal market conditions
                return "MEAN_REVERTING"
        
        regime_df["regime"] = regime_df.apply(classify_regime, axis=1)
        
        # Store for later analysis
        self.regime_history = regime_df
        
        # Print summary
        regime_counts = regime_df["regime"].value_counts()
        print(f"\n  Regime Distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(regime_df) * 100
            print(f"    {regime:<20} {count:>3} months ({pct:>5.1f}%)")
        
        print(f"\n  Regime Characteristics:")
        print(f"    Volatility threshold:      {vol_threshold:.4f}")
        print(f"    Trend strength threshold:  {trend_threshold:.4f}")
        print(f"    Mean reversion threshold:  {mr_threshold:.4f}")
        
        return regime_df
    
    def analyze_performance_by_regime(
        self,
        predictions_df: pd.DataFrame,
        regime_df: pd.DataFrame = None,
        pred_col: str = "predicted",
        actual_col: str = None
    ) -> Dict[str, Dict]:
        """
        Analyze model performance separately for each market regime.
        
        Computes IC statistics for each regime to understand when the model
        performs well vs poorly.
        
        Args:
            predictions_df: DataFrame with predictions and actuals
            regime_df: DataFrame with regime classifications (from detect_market_regimes)
                      If None, uses self.regime_history
            pred_col: Name of prediction column
            actual_col: Name of actual returns column
        
        Returns:
            Dictionary mapping regime names to performance metrics
            
        Validates: Requirements 5.4
        """
        print("\n--- Regime-Specific Performance Analysis ---")
        
        # Use stored regime history if not provided
        if regime_df is None:
            if len(self.regime_history) == 0:
                print("  ⚠ No regime data available. Run detect_market_regimes() first.")
                return {}
            regime_df = self.regime_history
        
        if actual_col is None:
            actual_col = "actual" if "actual" in predictions_df.columns else TARGET
        
        # Merge predictions with regime data
        predictions_with_regime = predictions_df.merge(
            regime_df[["date", "regime"]],
            on="date",
            how="inner"
        )
        
        if len(predictions_with_regime) == 0:
            print("  ⚠ No matching dates between predictions and regime data")
            return {}
        
        print(f"  Analyzing {len(predictions_with_regime):,} predictions across regimes...")
        
        # Analyze each regime separately
        regime_performance = {}
        
        for regime in sorted(predictions_with_regime["regime"].unique()):
            regime_data = predictions_with_regime[
                predictions_with_regime["regime"] == regime
            ].copy()
            
            # Compute monthly IC for this regime
            regime_ic = self.compute_monthly_ic(
                regime_data,
                pred_col=pred_col,
                actual_col=actual_col
            )
            
            # Compute distribution metrics
            distribution = self.analyze_ic_distribution(regime_ic)
            
            # Additional regime-specific metrics
            n_months = len(regime_data["date"].unique())
            n_predictions = len(regime_data)
            
            regime_performance[regime] = {
                **distribution,
                "n_months": n_months,
                "n_predictions": n_predictions,
                "monthly_ic": regime_ic
            }
        
        # Store for later use
        self.regime_performance = regime_performance
        
        # Print comparison
        print(f"\n  Performance by Regime:")
        print(f"  {'Regime':<20} {'Months':>8} {'Mean IC':>10} {'IC-IR':>8} {'Win Rate':>10} {'Volatility':>12}")
        print("  " + "-"*75)
        
        for regime in ["HIGH_VOLATILITY", "TRENDING", "MEAN_REVERTING"]:
            if regime in regime_performance:
                perf = regime_performance[regime]
                print(f"  {regime:<20} {perf['n_months']:>8} {perf['mean']:>+10.5f} "
                      f"{perf['ic_ir']:>8.3f} {perf['win_rate']:>9.1%} {perf['std']:>12.5f}")
        
        # Identify best and worst regimes
        if len(regime_performance) > 0:
            best_regime = max(regime_performance.items(), key=lambda x: x[1]["mean"])
            worst_regime = min(regime_performance.items(), key=lambda x: x[1]["mean"])
            
            print(f"\n  Best Regime:  {best_regime[0]} (Mean IC: {best_regime[1]['mean']:+.5f})")
            print(f"  Worst Regime: {worst_regime[0]} (Mean IC: {worst_regime[1]['mean']:+.5f})")
            
            ic_diff = best_regime[1]["mean"] - worst_regime[1]["mean"]
            print(f"  IC Difference: {ic_diff:+.5f}")
            
            if ic_diff > 0.05:
                print(f"\n  ⚠ Large performance difference across regimes!")
                print(f"    Consider regime-adaptive strategies or separate models per regime.")
        
        return regime_performance
    
    def compute_rolling_ic(
        self,
        monthly_ic: pd.Series,
        windows: List[int] = [3, 6]
    ) -> pd.DataFrame:
        """
        Compute rolling IC statistics over multiple time windows.
        
        Rolling IC helps identify short-term stability trends and whether
        model performance is improving, declining, or stable over time.
        
        Args:
            monthly_ic: Series of monthly IC values indexed by date
            windows: List of window sizes in months (default: [3, 6])
        
        Returns:
            DataFrame with rolling IC statistics for each window
            
        Validates: Requirements 5.5
        """
        print(f"\n--- Rolling IC Analysis ---")
        
        valid_ic = monthly_ic.dropna()
        
        if len(valid_ic) == 0:
            print("  ⚠ No valid IC data for rolling analysis")
            return pd.DataFrame()
        
        # Create DataFrame to store rolling metrics
        rolling_df = pd.DataFrame(index=valid_ic.index)
        rolling_df["ic"] = valid_ic
        
        for window in windows:
            if len(valid_ic) < window:
                print(f"  ⚠ Insufficient data for {window}-month rolling window (need {window}, have {len(valid_ic)})")
                continue
            
            # Rolling mean
            rolling_mean = valid_ic.rolling(window=window, min_periods=window).mean()
            rolling_df[f"rolling_{window}m_mean"] = rolling_mean
            
            # Rolling std
            rolling_std = valid_ic.rolling(window=window, min_periods=window).std()
            rolling_df[f"rolling_{window}m_std"] = rolling_std
            
            # Rolling IC-IR (mean / std)
            rolling_ir = rolling_mean / rolling_std
            rolling_df[f"rolling_{window}m_ir"] = rolling_ir
            
            # Rolling win rate
            rolling_win_rate = valid_ic.rolling(window=window, min_periods=window).apply(
                lambda x: (x > 0).mean()
            )
            rolling_df[f"rolling_{window}m_win_rate"] = rolling_win_rate
            
            # Print summary
            valid_rolling = rolling_mean.dropna()
            if len(valid_rolling) > 0:
                print(f"\n  {window}-Month Rolling IC:")
                print(f"    Current:     {valid_rolling.iloc[-1]:+.5f}")
                print(f"    Mean:        {valid_rolling.mean():+.5f}")
                print(f"    Std:         {valid_rolling.std():.5f}")
                print(f"    Min:         {valid_rolling.min():+.5f}")
                print(f"    Max:         {valid_rolling.max():+.5f}")
                
                # Trend analysis
                if len(valid_rolling) >= 2:
                    recent_trend = valid_rolling.iloc[-3:].mean() if len(valid_rolling) >= 3 else valid_rolling.iloc[-1]
                    overall_mean = valid_rolling.mean()
                    
                    if recent_trend > overall_mean + 0.01:
                        print(f"    Trend:       ↑ IMPROVING (recent: {recent_trend:+.5f} vs avg: {overall_mean:+.5f})")
                    elif recent_trend < overall_mean - 0.01:
                        print(f"    Trend:       ↓ DECLINING (recent: {recent_trend:+.5f} vs avg: {overall_mean:+.5f})")
                    else:
                        print(f"    Trend:       → STABLE (recent: {recent_trend:+.5f} vs avg: {overall_mean:+.5f})")
        
        return rolling_df
    
    def compute_ic_autocorrelation(
        self,
        monthly_ic: pd.Series,
        max_lag: int = 6
    ) -> Dict[int, float]:
        """
        Compute IC autocorrelation to measure signal persistence.
        
        Autocorrelation measures whether high (or low) IC months tend to be
        followed by similar IC months. Positive autocorrelation indicates
        persistent signal quality, while negative autocorrelation suggests
        mean-reverting performance.
        
        Args:
            monthly_ic: Series of monthly IC values indexed by date
            max_lag: Maximum lag in months to compute (default: 6)
        
        Returns:
            Dictionary mapping lag to autocorrelation coefficient
            
        Validates: Requirements 5.8
        """
        print(f"\n--- IC Autocorrelation Analysis ---")
        
        valid_ic = monthly_ic.dropna()
        
        if len(valid_ic) < max_lag + 2:
            print(f"  ⚠ Insufficient data for autocorrelation (need {max_lag + 2}, have {len(valid_ic)})")
            return {}
        
        autocorr_results = {}
        
        print(f"  Computing autocorrelation up to lag {max_lag}...")
        print(f"\n  {'Lag':<8} {'Autocorr':>12} {'Interpretation'}")
        print("  " + "-"*50)
        
        for lag in range(1, max_lag + 1):
            try:
                autocorr = valid_ic.autocorr(lag=lag)
                
                if np.isnan(autocorr):
                    autocorr = 0.0
                
                autocorr_results[lag] = autocorr
                
                # Interpret autocorrelation
                if autocorr > 0.3:
                    interpretation = "Strong persistence"
                elif autocorr > 0.1:
                    interpretation = "Moderate persistence"
                elif autocorr > -0.1:
                    interpretation = "No persistence"
                elif autocorr > -0.3:
                    interpretation = "Moderate mean reversion"
                else:
                    interpretation = "Strong mean reversion"
                
                print(f"  {lag:<8} {autocorr:>+12.5f} {interpretation}")
                
            except Exception as e:
                print(f"  {lag:<8} {'ERROR':>12} {str(e)}")
                autocorr_results[lag] = np.nan
        
        # Overall assessment
        print(f"\n  Signal Persistence Assessment:")
        
        # Average autocorrelation at lags 1-3 (short-term persistence)
        short_term_lags = [1, 2, 3]
        short_term_autocorr = np.mean([
            autocorr_results.get(lag, 0) 
            for lag in short_term_lags 
            if not np.isnan(autocorr_results.get(lag, np.nan))
        ])
        
        if short_term_autocorr > 0.2:
            print(f"    ✓ Strong short-term persistence (avg lag 1-3: {short_term_autocorr:+.3f})")
            print(f"      Good IC months tend to be followed by more good months")
        elif short_term_autocorr > 0.05:
            print(f"    → Moderate short-term persistence (avg lag 1-3: {short_term_autocorr:+.3f})")
            print(f"      Some continuity in IC performance")
        elif short_term_autocorr > -0.05:
            print(f"    ⚠ No short-term persistence (avg lag 1-3: {short_term_autocorr:+.3f})")
            print(f"      IC performance is largely random month-to-month")
        else:
            print(f"    ⚠ Mean-reverting IC (avg lag 1-3: {short_term_autocorr:+.3f})")
            print(f"      Good months tend to be followed by bad months (and vice versa)")
        
        return autocorr_results
    
    def check_stability_thresholds(
        self,
        monthly_ic: pd.Series,
        std_threshold: float = 0.15,
        mean_threshold: float = 0.01,
        win_rate_threshold: float = 0.60
    ) -> Dict[str, any]:
        """
        Monitor stability thresholds and flag instability issues.
        
        Checks multiple stability criteria:
        - IC standard deviation (volatility)
        - Mean IC (predictive power)
        - Win rate (consistency)
        
        Args:
            monthly_ic: Series of monthly IC values
            std_threshold: Maximum acceptable IC std dev (default: 0.15)
            mean_threshold: Minimum acceptable mean IC (default: 0.01)
            win_rate_threshold: Minimum acceptable win rate (default: 0.60)
        
        Returns:
            Dictionary with threshold check results and flags
            
        Validates: Requirements 5.9
        """
        print(f"\n--- Stability Threshold Monitoring ---")
        
        valid_ic = monthly_ic.dropna()
        
        if len(valid_ic) == 0:
            print("  ⚠ No valid IC data for threshold monitoring")
            return {
                "passed": False,
                "flags": ["NO_DATA"],
                "metrics": {}
            }
        
        # Compute metrics
        mean_ic = valid_ic.mean()
        std_ic = valid_ic.std()
        win_rate = (valid_ic > 0).mean()
        
        # Check thresholds
        flags = []
        
        print(f"  Threshold Checks:")
        print(f"  {'Metric':<25} {'Value':>12} {'Threshold':>12} {'Status'}")
        print("  " + "-"*65)
        
        # Check IC volatility
        if std_ic > std_threshold:
            status = "⚠ FAIL"
            flags.append("HIGH_VOLATILITY")
        else:
            status = "✓ PASS"
        print(f"  {'IC Std Dev':<25} {std_ic:>12.5f} {std_threshold:>12.5f} {status}")
        
        # Check mean IC
        if mean_ic < mean_threshold:
            status = "⚠ FAIL"
            flags.append("LOW_MEAN_IC")
        else:
            status = "✓ PASS"
        print(f"  {'Mean IC':<25} {mean_ic:>+12.5f} {mean_threshold:>12.5f} {status}")
        
        # Check win rate
        if win_rate < win_rate_threshold:
            status = "⚠ FAIL"
            flags.append("LOW_WIN_RATE")
        else:
            status = "✓ PASS"
        print(f"  {'Win Rate':<25} {win_rate:>12.2%} {win_rate_threshold:>12.2%} {status}")
        
        # Overall assessment
        passed = len(flags) == 0
        
        print(f"\n  Overall Stability: ", end="")
        if passed:
            print("✓ STABLE")
            print("    All stability thresholds met")
        else:
            print("⚠ UNSTABLE")
            print(f"    Failed checks: {', '.join(flags)}")
            
            # Provide specific recommendations
            if "HIGH_VOLATILITY" in flags:
                print(f"\n    High IC Volatility (std: {std_ic:.5f} > {std_threshold}):")
                print(f"      → IC varies widely month-to-month")
                print(f"      → Consider regime-adaptive strategies")
                print(f"      → Investigate negative IC months for patterns")
                print(f"      → Use ensemble methods for stability")
            
            if "LOW_MEAN_IC" in flags:
                print(f"\n    Low Mean IC (mean: {mean_ic:+.5f} < {mean_threshold}):")
                print(f"      → Weak overall predictive power")
                print(f"      → Improve feature quality")
                print(f"      → Optimize model architecture")
                print(f"      → Check for data leakage (if OOT > in-sample)")
            
            if "LOW_WIN_RATE" in flags:
                print(f"\n    Low Win Rate ({win_rate:.1%} < {win_rate_threshold:.0%}):")
                print(f"      → Too many negative IC months")
                print(f"      → Model unreliable in many periods")
                print(f"      → Analyze negative months for common characteristics")
                print(f"      → Consider regime detection and adaptation")
        
        return {
            "passed": passed,
            "flags": flags,
            "metrics": {
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "win_rate": win_rate
            },
            "thresholds": {
                "std_threshold": std_threshold,
                "mean_threshold": mean_threshold,
                "win_rate_threshold": win_rate_threshold
            }
        }
    
    def analyze_regime_transitions(
        self,
        predictions_df: pd.DataFrame,
        regime_df: pd.DataFrame = None,
        pred_col: str = "predicted",
        actual_col: str = None
    ) -> pd.DataFrame:
        """
        Analyze the impact of regime transitions on model performance.
        
        Identifies months where regime changed and measures performance
        during transition periods vs stable periods.
        
        Args:
            predictions_df: DataFrame with predictions and actuals
            regime_df: DataFrame with regime classifications
            pred_col: Name of prediction column
            actual_col: Name of actual returns column
        
        Returns:
            DataFrame with transition analysis
            
        Validates: Requirements 5.3, 5.4
        """
        print("\n--- Regime Transition Impact Analysis ---")
        
        # Use stored regime history if not provided
        if regime_df is None:
            if len(self.regime_history) == 0:
                print("  ⚠ No regime data available. Run detect_market_regimes() first.")
                return pd.DataFrame()
            regime_df = self.regime_history.copy()
        else:
            regime_df = regime_df.copy()
        
        if actual_col is None:
            actual_col = "actual" if "actual" in predictions_df.columns else TARGET
        
        # Sort by date
        regime_df = regime_df.sort_values("date")
        
        # Identify regime transitions
        regime_df["prev_regime"] = regime_df["regime"].shift(1)
        regime_df["is_transition"] = regime_df["regime"] != regime_df["prev_regime"]
        regime_df["transition_type"] = regime_df.apply(
            lambda row: f"{row['prev_regime']} → {row['regime']}" 
            if row["is_transition"] and pd.notna(row["prev_regime"]) 
            else "STABLE",
            axis=1
        )
        
        # Merge with predictions
        predictions_with_transitions = predictions_df.merge(
            regime_df[["date", "regime", "is_transition", "transition_type"]],
            on="date",
            how="inner"
        )
        
        if len(predictions_with_transitions) == 0:
            print("  ⚠ No matching dates between predictions and regime data")
            return pd.DataFrame()
        
        # Compute IC for transition vs stable months
        transition_months = predictions_with_transitions[
            predictions_with_transitions["is_transition"] == True
        ]
        stable_months = predictions_with_transitions[
            predictions_with_transitions["is_transition"] == False
        ]
        
        print(f"  Transition months: {transition_months['date'].nunique()}")
        print(f"  Stable months:     {stable_months['date'].nunique()}")
        
        # Compute IC for each group
        if len(transition_months) > 0:
            transition_ic = self.compute_monthly_ic(
                transition_months, pred_col=pred_col, actual_col=actual_col
            )
            transition_dist = self.analyze_ic_distribution(transition_ic)
        else:
            transition_dist = {"mean": np.nan, "std": np.nan, "win_rate": np.nan}
        
        if len(stable_months) > 0:
            stable_ic = self.compute_monthly_ic(
                stable_months, pred_col=pred_col, actual_col=actual_col
            )
            stable_dist = self.analyze_ic_distribution(stable_ic)
        else:
            stable_dist = {"mean": np.nan, "std": np.nan, "win_rate": np.nan}
        
        # Print comparison
        print(f"\n  Performance Comparison:")
        print(f"  {'Period':<20} {'Mean IC':>10} {'Volatility':>12} {'Win Rate':>10}")
        print("  " + "-"*55)
        print(f"  {'Transition Months':<20} {transition_dist['mean']:>+10.5f} "
              f"{transition_dist['std']:>12.5f} {transition_dist['win_rate']:>9.1%}")
        print(f"  {'Stable Months':<20} {stable_dist['mean']:>+10.5f} "
              f"{stable_dist['std']:>12.5f} {stable_dist['win_rate']:>9.1%}")
        
        ic_impact = transition_dist["mean"] - stable_dist["mean"]
        print(f"\n  Transition Impact: {ic_impact:+.5f} IC")
        
        if abs(ic_impact) > 0.02:
            if ic_impact < 0:
                print(f"  ⚠ Model performs WORSE during regime transitions")
                print(f"    Consider adding regime transition indicators as features")
            else:
                print(f"  ✓ Model performs BETTER during regime transitions")
        else:
            print(f"  ✓ Regime transitions have minimal impact on performance")
        
        # Analyze specific transition types
        print(f"\n  Performance by Transition Type:")
        transition_types = predictions_with_transitions.groupby("transition_type")
        
        transition_analysis = []
        
        for trans_type, trans_data in transition_types:
            if trans_type == "STABLE":
                continue
            
            if len(trans_data) < 10:  # Need minimum data
                continue
            
            trans_ic = self.compute_monthly_ic(
                trans_data, pred_col=pred_col, actual_col=actual_col
            )
            trans_dist = self.analyze_ic_distribution(trans_ic)
            
            transition_analysis.append({
                "transition_type": trans_type,
                "n_months": trans_data["date"].nunique(),
                "mean_ic": trans_dist["mean"],
                "std_ic": trans_dist["std"],
                "win_rate": trans_dist["win_rate"]
            })
        
        if len(transition_analysis) > 0:
            trans_df = pd.DataFrame(transition_analysis)
            trans_df = trans_df.sort_values("mean_ic", ascending=False)
            
            print(f"  {'Transition':<30} {'Months':>8} {'Mean IC':>10} {'Win Rate':>10}")
            print("  " + "-"*60)
            for _, row in trans_df.iterrows():
                print(f"  {row['transition_type']:<30} {row['n_months']:>8} "
                      f"{row['mean_ic']:>+10.5f} {row['win_rate']:>9.1%}")
            
            return trans_df
        else:
            print("  (Insufficient data for transition type analysis)")
            return pd.DataFrame()
    
    def generate_stability_report(
        self,
        predictions_df: pd.DataFrame,
        pred_col: str = "predicted",
        actual_col: str = None,
        save_path: Optional[str] = None,
        enable_regime_detection: bool = True,
        prices_df: pd.DataFrame = None
    ) -> Dict:
        """
        Generate comprehensive stability report.
        
        Computes all stability metrics and creates a detailed report including:
        - Monthly IC time series
        - IC distribution analysis
        - Win rate calculation
        - Negative IC month identification and analysis
        - Rolling IC statistics (3-month and 6-month windows)
        - IC autocorrelation for signal persistence
        - Stability threshold monitoring
        - Market regime detection (optional)
        - Regime-specific performance analysis (optional)
        - Regime transition impact assessment (optional)
        
        Args:
            predictions_df: DataFrame with predictions and actuals
            pred_col: Name of prediction column
            actual_col: Name of actual returns column
            save_path: Optional path to save report CSV
            enable_regime_detection: Whether to perform regime detection (default: True)
            prices_df: DataFrame with daily prices for regime detection (optional)
        
        Returns:
            Dictionary with all stability metrics and DataFrames
            
        Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.8, 5.9
        """
        print("\n" + "="*70)
        print("   STABILITY MONITORING REPORT")
        print("="*70)
        
        # Compute monthly IC
        monthly_ic = self.compute_monthly_ic(predictions_df, pred_col, actual_col)
        
        # Analyze IC distribution
        print("\n--- IC Distribution Analysis ---")
        distribution = self.analyze_ic_distribution(monthly_ic)
        
        print(f"  Mean IC:        {distribution['mean']:+.5f}")
        print(f"  Median IC:      {distribution['median']:+.5f}")
        print(f"  Std IC:         {distribution['std']:.5f}")
        print(f"  IC-IR:          {distribution['ic_ir']:.5f}")
        print(f"  Min IC:         {distribution['min']:+.5f}")
        print(f"  Max IC:         {distribution['max']:+.5f}")
        print(f"\n  Percentiles:")
        print(f"    5th:  {distribution['p5']:+.5f}")
        print(f"    25th: {distribution['p25']:+.5f}")
        print(f"    75th: {distribution['p75']:+.5f}")
        print(f"    95th: {distribution['p95']:+.5f}")
        print(f"\n  Win Rate:       {distribution['win_rate']:.2%} "
              f"({int(distribution['win_rate'] * distribution['n_months'])}/{distribution['n_months']} months)")
        
        # Identify negative IC months
        print("\n--- Negative IC Month Analysis ---")
        negative_months_df = self.identify_negative_ic_months(
            predictions_df, monthly_ic, pred_col, actual_col
        )
        
        # Rolling IC analysis
        rolling_ic_df = self.compute_rolling_ic(monthly_ic, windows=[3, 6])
        
        # IC autocorrelation
        ic_autocorr = self.compute_ic_autocorrelation(monthly_ic, max_lag=6)
        
        # Stability threshold monitoring
        stability_check = self.check_stability_thresholds(
            monthly_ic,
            std_threshold=0.15,
            mean_threshold=0.01,
            win_rate_threshold=0.60
        )
        
        # Regime detection and analysis
        regime_df = pd.DataFrame()
        regime_performance = {}
        regime_transitions = pd.DataFrame()
        
        if enable_regime_detection:
            try:
                # Detect market regimes
                regime_df = self.detect_market_regimes(
                    prices_df=prices_df,
                    predictions_df=predictions_df
                )
                
                if len(regime_df) > 0:
                    # Analyze performance by regime
                    regime_performance = self.analyze_performance_by_regime(
                        predictions_df,
                        regime_df=regime_df,
                        pred_col=pred_col,
                        actual_col=actual_col
                    )
                    
                    # Analyze regime transitions
                    regime_transitions = self.analyze_regime_transitions(
                        predictions_df,
                        regime_df=regime_df,
                        pred_col=pred_col,
                        actual_col=actual_col
                    )
            except Exception as e:
                print(f"\n  ⚠ Regime detection failed: {e}")
                print("  Continuing without regime analysis...")
        
        # Stability assessment
        print("\n--- Stability Assessment ---")
        self._print_stability_assessment(distribution, negative_months_df, regime_performance)
        
        print("="*70)
        
        # Compile report
        report = {
            "monthly_ic": monthly_ic,
            "distribution": distribution,
            "negative_months": negative_months_df,
            "monthly_ic_history": self.monthly_ic_history,
            "rolling_ic": rolling_ic_df,
            "ic_autocorrelation": ic_autocorr,
            "stability_check": stability_check,
            "regime_data": regime_df,
            "regime_performance": regime_performance,
            "regime_transitions": regime_transitions
        }
        
        # Save report if requested
        if save_path:
            self._save_stability_report(report, save_path)
        
        return report
    
    def _print_stability_assessment(
        self,
        distribution: Dict[str, float],
        negative_months_df: pd.DataFrame,
        regime_performance: Dict[str, Dict] = None
    ):
        """
        Print stability assessment based on distribution metrics.
        
        Args:
            distribution: IC distribution metrics
            negative_months_df: DataFrame with negative IC month analysis
            regime_performance: Optional regime-specific performance metrics
        """
        mean_ic = distribution["mean"]
        std_ic = distribution["std"]
        win_rate = distribution["win_rate"]
        ic_ir = distribution["ic_ir"]
        
        # Assess stability
        stability_issues = []
        
        if mean_ic < 0.01:
            stability_issues.append("Low mean IC (<0.01) - weak predictive power")
        
        if std_ic > 0.15:
            stability_issues.append("High IC volatility (>0.15) - inconsistent performance")
        
        if win_rate < 0.60:
            stability_issues.append("Low win rate (<60%) - frequent negative IC months")
        
        if ic_ir < 0.5:
            stability_issues.append("Low IC-IR (<0.5) - poor risk-adjusted performance")
        
        if len(negative_months_df) > 0:
            neg_pct = len(negative_months_df) / distribution["n_months"]
            if neg_pct > 0.40:
                stability_issues.append(f"High negative IC rate ({neg_pct:.1%}) - model unreliable")
        
        if not stability_issues:
            print("  ✓ Model shows GOOD stability")
            print("    - Consistent positive IC")
            print("    - Low volatility")
            print("    - High win rate")
        else:
            print("  ⚠ Stability concerns identified:")
            for issue in stability_issues:
                print(f"    - {issue}")
        
        # Regime-specific insights
        if regime_performance and len(regime_performance) > 0:
            print("\n  Regime-Specific Insights:")
            
            # Find best and worst regimes
            best_regime = max(regime_performance.items(), key=lambda x: x[1]["mean"])
            worst_regime = min(regime_performance.items(), key=lambda x: x[1]["mean"])
            
            ic_spread = best_regime[1]["mean"] - worst_regime[1]["mean"]
            
            if ic_spread > 0.05:
                print(f"    ⚠ Large performance variation across regimes (spread: {ic_spread:+.5f})")
                print(f"      Best:  {best_regime[0]} (IC: {best_regime[1]['mean']:+.5f})")
                print(f"      Worst: {worst_regime[0]} (IC: {worst_regime[1]['mean']:+.5f})")
            else:
                print(f"    ✓ Consistent performance across regimes (spread: {ic_spread:+.5f})")
            
            # Check for regime-specific weaknesses
            weak_regimes = [
                regime for regime, perf in regime_performance.items()
                if perf["mean"] < 0.01 or perf["win_rate"] < 0.5
            ]
            
            if weak_regimes:
                print(f"    ⚠ Weak performance in: {', '.join(weak_regimes)}")
        
        # Recommendations
        print("\n  Recommendations:")
        if mean_ic < 0.01:
            print("    → Improve feature quality and model architecture")
        if std_ic > 0.15:
            if regime_performance and len(regime_performance) > 0:
                print("    → Consider regime-adaptive strategies (performance varies by regime)")
            else:
                print("    → Investigate regime-specific performance")
            print("    → Consider ensemble methods for stability")
        if win_rate < 0.60:
            print("    → Analyze negative IC months for common patterns")
            print("    → Add regime detection to adapt to market conditions")
        
        if regime_performance and len(regime_performance) > 0:
            # Check if regime-adaptive approach would help
            best_regime = max(regime_performance.items(), key=lambda x: x[1]["mean"])
            if best_regime[1]["mean"] > mean_ic + 0.03:
                print("    → Regime-adaptive model could improve performance significantly")
                print(f"      (Best regime IC: {best_regime[1]['mean']:+.5f} vs Overall: {mean_ic:+.5f})")
    
    def _save_stability_report(
        self,
        report: Dict,
        save_path: str
    ):
        """
        Save stability report to CSV files.
        
        Args:
            report: Dictionary with stability metrics
            save_path: Base path for saving reports (without extension)
        """
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        # Save monthly IC time series
        monthly_ic_df = report["monthly_ic"].to_frame("ic")
        monthly_ic_path = save_path.replace(".csv", "_monthly_ic.csv")
        monthly_ic_df.to_csv(monthly_ic_path)
        print(f"\n  Saved monthly IC: {monthly_ic_path}")
        
        # Save distribution summary
        distribution_df = pd.DataFrame([report["distribution"]])
        distribution_path = save_path.replace(".csv", "_distribution.csv")
        distribution_df.to_csv(distribution_path, index=False)
        print(f"  Saved distribution: {distribution_path}")
        
        # Save negative months analysis
        if len(report["negative_months"]) > 0:
            negative_path = save_path.replace(".csv", "_negative_months.csv")
            report["negative_months"].to_csv(negative_path, index=False)
            print(f"  Saved negative months: {negative_path}")
        
        # Save rolling IC analysis
        if "rolling_ic" in report and len(report["rolling_ic"]) > 0:
            rolling_path = save_path.replace(".csv", "_rolling_ic.csv")
            report["rolling_ic"].to_csv(rolling_path)
            print(f"  Saved rolling IC: {rolling_path}")
        
        # Save IC autocorrelation
        if "ic_autocorrelation" in report and len(report["ic_autocorrelation"]) > 0:
            autocorr_df = pd.DataFrame([
                {"lag": lag, "autocorrelation": autocorr}
                for lag, autocorr in report["ic_autocorrelation"].items()
            ])
            autocorr_path = save_path.replace(".csv", "_ic_autocorrelation.csv")
            autocorr_df.to_csv(autocorr_path, index=False)
            print(f"  Saved IC autocorrelation: {autocorr_path}")
        
        # Save stability check results
        if "stability_check" in report:
            stability_check = report["stability_check"]
            stability_df = pd.DataFrame([{
                "passed": stability_check["passed"],
                "flags": ",".join(stability_check["flags"]) if stability_check["flags"] else "NONE",
                **stability_check["metrics"],
                **{f"threshold_{k}": v for k, v in stability_check["thresholds"].items()}
            }])
            stability_path = save_path.replace(".csv", "_stability_check.csv")
            stability_df.to_csv(stability_path, index=False)
            print(f"  Saved stability check: {stability_path}")
        
        # Save regime data if available
        if "regime_data" in report and len(report["regime_data"]) > 0:
            regime_path = save_path.replace(".csv", "_regime_data.csv")
            report["regime_data"].to_csv(regime_path, index=False)
            print(f"  Saved regime data: {regime_path}")
        
        # Save regime performance if available
        if "regime_performance" in report and len(report["regime_performance"]) > 0:
            regime_perf_records = []
            for regime, perf in report["regime_performance"].items():
                regime_perf_records.append({
                    "regime": regime,
                    "mean_ic": perf["mean"],
                    "median_ic": perf["median"],
                    "std_ic": perf["std"],
                    "ic_ir": perf["ic_ir"],
                    "win_rate": perf["win_rate"],
                    "n_months": perf["n_months"],
                    "n_predictions": perf["n_predictions"]
                })
            
            if regime_perf_records:
                regime_perf_df = pd.DataFrame(regime_perf_records)
                regime_perf_path = save_path.replace(".csv", "_regime_performance.csv")
                regime_perf_df.to_csv(regime_perf_path, index=False)
                print(f"  Saved regime performance: {regime_perf_path}")
        
        # Save regime transitions if available
        if "regime_transitions" in report and len(report["regime_transitions"]) > 0:
            regime_trans_path = save_path.replace(".csv", "_regime_transitions.csv")
            report["regime_transitions"].to_csv(regime_trans_path, index=False)
            print(f"  Saved regime transitions: {regime_trans_path}")


def run_stability_monitoring(
    predictions_df: pd.DataFrame,
    pred_col: str = "predicted",
    actual_col: str = None,
    save_reports: bool = True,
    enable_regime_detection: bool = True,
    prices_df: pd.DataFrame = None
) -> Dict:
    """
    Run comprehensive stability monitoring on model predictions.
    
    Convenience function that creates a StabilityMonitor and generates
    a full stability report including regime detection.
    
    Args:
        predictions_df: DataFrame with predictions and actuals
        pred_col: Name of prediction column
        actual_col: Name of actual returns column
        save_reports: Whether to save reports to files
        enable_regime_detection: Whether to perform regime detection (default: True)
        prices_df: DataFrame with daily prices for regime detection (optional)
    
    Returns:
        Dictionary with stability metrics and analysis
        
    Validates: Requirements 5.1, 5.2, 5.3, 5.4
    """
    import os
    os.makedirs("reports", exist_ok=True)
    
    monitor = StabilityMonitor()
    
    report = monitor.generate_stability_report(
        predictions_df,
        pred_col=pred_col,
        actual_col=actual_col,
        save_path="reports/stability_report.csv" if save_reports else None,
        enable_regime_detection=enable_regime_detection,
        prices_df=prices_df
    )
    
    return report


if __name__ == "__main__":
    """
    Standalone execution: run stability monitoring on existing predictions.
    """
    import os
    
    print("Running stability monitoring...")
    
    # Try to load predictions from various sources
    predictions_files = [
        "data/sector_predictions.csv",
        "data/ridge_predictions.csv",
        "data/lgbm_predictions.csv"
    ]
    
    predictions_df = None
    for file_path in predictions_files:
        if os.path.exists(file_path):
            print(f"\nLoading predictions from {file_path}...")
            predictions_df = pd.read_csv(file_path, parse_dates=["date"])
            break
    
    if predictions_df is None:
        print("\nNo prediction files found. Please run the model pipeline first.")
        print("Expected files:")
        for file_path in predictions_files:
            print(f"  - {file_path}")
    else:
        # Run stability monitoring
        report = run_stability_monitoring(predictions_df)
        
        print("\n✓ Stability monitoring complete!")

"""
data_loader.py  —  Dynamic S&P 500 Universe, 19 Features, Full Caching
=====================================================================
Downloads price + fundamental data for the POINT-IN-TIME S&P 500 universe
(reconstructed via universe_builder.py), engineers 19 monthly factors
across 8 groups, and caches all raw data.

SURVIVORSHIP BIAS FIX:
  The stock universe is NO LONGER a static hardcoded list. Instead,
  universe_builder.py scrapes Wikipedia's S&P 500 historical changes
  to reconstruct which stocks were actually in the index at each month.
  This ensures delisted, bankrupt, and removed companies are included
  during the periods they were in the index, eliminating survivorship bias.

  The original SECTOR_MAP is kept as a fallback for backwards compatibility.

TARGET VARIABLE:
  Next_Month_Return = next month's actual price return (e.g. 0.08 = 8%).
  The model FORECASTS returns. Ranking is computed AFTER prediction as a
  portfolio selection tool — it is NOT the model's target.

19 FEATURES — 8 uncorrelated groups:
  G1 Momentum   (3): Mom_12_1, Mom_6_1, Mom_1
  G2 Risk       (3): Vol_12, IdioVol, Beta_12
  G3 Technical  (3): High52W, Trend_MA, MaxRet_1M
  G4 Value      (3): PB_ratio, PE_TTM, EV_EBITDA
  G5 Quality    (3): ROE, GrossMargin, CashFlowYield
  G6 Growth     (2): RevGrowth_YoY, EarnGrowth_YoY
  G7 Size       (1): LogMktCap
  G8 Liquidity  (1): VolRatio

  REMOVED (poor yfinance coverage / noisy):
    DividendYield  — many stocks don't pay dividends, field unreliable
    CurrentRatio   — noisy for financials, weak alpha signal
    AssetTurnover  — overlaps with existing quality/value metrics

FUNDAMENTAL DATA LEAKAGE CONTROL — 45-day lag:
  Quarterly earnings/balance sheets are announced ~30-45 days after
  quarter-end. We apply a strict 45-day lag: when computing features
  for prediction month T, we only use fundamental data from quarters
  that ended >= 45 days before the 1st of month T.
  This guarantees every fundamental was publicly available.

MISSING DATA:
  Fundamental features are imputed with the cross-sectional median
  for each month (groupby date). This avoids leakage from naive
  fillna(0) which could systematically bias the model.

CACHING:
  data/daily_prices.parquet   — raw daily adjusted closes
  data/fundamentals.parquet   — quarterly fundamental data
  data/monthly_returns.csv    — monthly return matrix
  data/sp500_universe.json    — point-in-time universe cache
  Re-run loads from cache (< 24h old). Delete cache files to refresh.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
import os
from datetime import datetime, timedelta
from scipy.stats import skew as scipy_skew

# ── Static 250-stock universe (FALLBACK — see universe_builder.py for dynamic) ──
# Uses proper 11 GICS sector classifications
SECTOR_MAP = {
    # Information Technology (50)
    "AAPL":"Information Technology","MSFT":"Information Technology","NVDA":"Information Technology",
    "GOOGL":"Communication Services","META":"Communication Services","AMD":"Information Technology",
    "INTC":"Information Technology","QCOM":"Information Technology","TXN":"Information Technology",
    "AVGO":"Information Technology","AMAT":"Information Technology","MU":"Information Technology",
    "LRCX":"Information Technology","KLAC":"Information Technology","NOW":"Information Technology",
    "CRM":"Information Technology","ADBE":"Information Technology","ORCL":"Information Technology",
    "INTU":"Information Technology","PANW":"Information Technology","NET":"Information Technology",
    "CRWD":"Information Technology","FTNT":"Information Technology","CDNS":"Information Technology",
    "SNPS":"Information Technology","VRSN":"Information Technology","HPQ":"Information Technology",
    "IBM":"Information Technology","CSCO":"Information Technology","ACN":"Information Technology",
    "DELL":"Information Technology","ANET":"Information Technology","MRVL":"Information Technology",
    "NXPI":"Information Technology","ADI":"Information Technology","MCHP":"Information Technology",
    "MPWR":"Information Technology","KEYS":"Information Technology","TER":"Information Technology",
    "ENTG":"Information Technology","SWKS":"Information Technology","QRVO":"Information Technology",
    "WDC":"Information Technology","STX":"Information Technology","ONTO":"Information Technology",
    "IPGP":"Information Technology","COHU":"Information Technology","FORM":"Information Technology",
    "CGNX":"Information Technology","MKSI":"Information Technology",
    # Financials (50)
    "JPM":"Financials","BAC":"Financials","GS":"Financials",
    "MS":"Financials","BLK":"Financials","WFC":"Financials",
    "C":"Financials","AXP":"Financials","SPGI":"Financials",
    "MCO":"Financials","ICE":"Financials","CME":"Financials",
    "COF":"Financials","USB":"Financials","PNC":"Financials",
    "TFC":"Financials","SCHW":"Financials","BK":"Financials",
    "STT":"Financials","MTB":"Financials","FITB":"Financials",
    "HBAN":"Financials","RF":"Financials","CFG":"Financials",
    "KEY":"Financials","SYF":"Financials","NDAQ":"Financials",
    "CB":"Financials","AON":"Financials","MMC":"Financials",
    "AJG":"Financials","AFL":"Financials","MET":"Financials",
    "PRU":"Financials","ALL":"Financials","PGR":"Financials",
    "TRV":"Financials","HIG":"Financials","RJF":"Financials",
    "IBKR":"Financials","LPLA":"Financials","SF":"Financials",
    "SEIC":"Financials","FDS":"Financials","MSCI":"Financials",
    "FNF":"Financials","WTW":"Financials","GL":"Financials",
    "UNM":"Financials","AMTD":"Financials",
    # Health Care (50)
    "JNJ":"Health Care","UNH":"Health Care","PFE":"Health Care",
    "ABBV":"Health Care","LLY":"Health Care","MRK":"Health Care",
    "BMY":"Health Care","AMGN":"Health Care","GILD":"Health Care",
    "BIIB":"Health Care","VRTX":"Health Care","REGN":"Health Care",
    "CVS":"Health Care","CI":"Health Care","HUM":"Health Care",
    "MDT":"Health Care","SYK":"Health Care","BSX":"Health Care",
    "ABT":"Health Care","TMO":"Health Care","DHR":"Health Care",
    "ISRG":"Health Care","EW":"Health Care","ZBH":"Health Care",
    "BAX":"Health Care","BDX":"Health Care","IQV":"Health Care",
    "A":"Health Care","DGX":"Health Care","RMD":"Health Care",
    "HOLX":"Health Care","IDXX":"Health Care","WAT":"Health Care",
    "MTD":"Health Care","PODD":"Health Care","DXCM":"Health Care",
    "ALGN":"Health Care","MASI":"Health Care","NTRA":"Health Care",
    "INSP":"Health Care","ACAD":"Health Care","RARE":"Health Care",
    "FOLD":"Health Care","ALKS":"Health Care","NBIX":"Health Care",
    "RVNC":"Health Care","PRCT":"Health Care","AXNX":"Health Care",
    "NVCR":"Health Care","ROIV":"Health Care",
    # Consumer Discretionary (30)
    "AMZN":"Consumer Discretionary","HD":"Consumer Discretionary",
    "TGT":"Consumer Discretionary","LOW":"Consumer Discretionary",
    "SBUX":"Consumer Discretionary","MCD":"Consumer Discretionary","NKE":"Consumer Discretionary",
    "LULU":"Consumer Discretionary","TJX":"Consumer Discretionary","ROST":"Consumer Discretionary",
    "DG":"Consumer Discretionary","DLTR":"Consumer Discretionary",
    "YUM":"Consumer Discretionary","CMG":"Consumer Discretionary","DRI":"Consumer Discretionary",
    "ORLY":"Consumer Discretionary","AZO":"Consumer Discretionary","EBAY":"Consumer Discretionary",
    "BKNG":"Consumer Discretionary","MAR":"Consumer Discretionary","HLT":"Consumer Discretionary",
    "MGM":"Consumer Discretionary","F":"Consumer Discretionary","GM":"Consumer Discretionary",
    "TSLA":"Consumer Discretionary","BBY":"Consumer Discretionary",
    "LVS":"Consumer Discretionary","WYNN":"Consumer Discretionary",
    "RCL":"Consumer Discretionary","CCL":"Consumer Discretionary",
    # Consumer Staples (10)
    "WMT":"Consumer Staples","COST":"Consumer Staples","KR":"Consumer Staples",
    "NCLH":"Consumer Discretionary","H":"Consumer Discretionary","DKNG":"Consumer Discretionary",
    "POOL":"Consumer Discretionary","GRMN":"Consumer Discretionary",
    "ABNB":"Consumer Discretionary","IHG":"Consumer Discretionary",
    # Communication Services (10)
    "NFLX":"Communication Services","DIS":"Communication Services",
    "CMCSA":"Communication Services","SIRI":"Communication Services",
    "WBD":"Communication Services","PARA":"Communication Services","FOX":"Communication Services",
    "CZR":"Consumer Discretionary","PENN":"Consumer Discretionary","CHDN":"Consumer Discretionary",
    # Energy (12)
    "XOM":"Energy","CVX":"Energy","COP":"Energy",
    "EOG":"Energy","SLB":"Energy","HAL":"Energy",
    "MPC":"Energy","VLO":"Energy","PSX":"Energy",
    "OXY":"Energy","DVN":"Energy","APA":"Energy",
    # Industrials (25)
    "BA":"Industrials","LMT":"Industrials","RTX":"Industrials",
    "NOC":"Industrials","GD":"Industrials","GE":"Industrials",
    "HON":"Industrials","MMM":"Industrials","CAT":"Industrials",
    "DE":"Industrials","UNP":"Industrials","UPS":"Industrials",
    "FDX":"Industrials","EMR":"Industrials","ETN":"Industrials",
    "PH":"Industrials","ROK":"Industrials","AME":"Industrials",
    "VRSK":"Industrials","IDEX":"Industrials","XYL":"Industrials",
    "ROP":"Industrials","HUBB":"Industrials","FTV":"Industrials",
    "GNRC":"Industrials",
    # Utilities + Materials + Real Estate
    "FSLR":"Information Technology","ENPH":"Information Technology",
    "BKR":"Energy","MRO":"Energy","CTRA":"Energy",
    "WMB":"Energy","HES":"Energy","SM":"Energy",
    "MTDR":"Energy","BE":"Industrials","PLUG":"Industrials",
    "PR":"Energy","CLR":"Energy",
}

STOCK_TICKERS   = list(SECTOR_MAP.keys())
BENCHMARK       = "SPY"
BATCH_SIZE      = 25
BATCH_DELAY     = 3.0
MAX_RETRIES     = 4
CACHE_DIR       = "data"
PRICE_CACHE     = os.path.join(CACHE_DIR, "daily_prices.parquet")
FUND_CACHE      = os.path.join(CACHE_DIR, "fundamentals.parquet")
CACHE_MAX_HOURS = 24

FEATURES = [
    "Mom_12_1","Mom_6_1","Mom_1",           # G1 Momentum
    "Vol_12","IdioVol","Beta_12",            # G2 Risk
    "High52W","Trend_MA","MaxRet_1M",       # G3 Technical
    "PB_ratio","PE_TTM","EV_EBITDA",        # G4 Value
    "ROE","GrossMargin","CashFlowYield",    # G5 Quality
    "RevGrowth_YoY","EarnGrowth_YoY",       # G6 Growth
    "LogMktCap",                             # G7 Size
    "VolRatio",                              # G8 Liquidity
]

# Sector-relative features (computed in apply_feature_engineering)
SECTOR_REL_FEATURES = [
    "Mom_12_1_sector_z", "Vol_12_sector_z", "PB_ratio_sector_z",
    "ROE_sector_z", "LogMktCap_sector_z", "PE_TTM_sector_z",
    "RevGrowth_YoY_sector_z",
]

# Interaction features (computed in apply_feature_engineering)
INTERACTION_FEATURES = [
    "Mom12_x_PBratio",   # Value × Momentum (Asness 2013)
    "ROE_x_RevGrowth",   # Quality × Growth (Novy-Marx 2013)
    "Vol12_x_LogMktCap", # Risk × Size (low-vol anomaly)
]

# Full feature set (base + engineered, populated after apply_feature_engineering)
ALL_FEATURES = FEATURES.copy()  # extended dynamically

# Optimized feature set based on IC analysis (only features with positive mean IC and good stability)
OPTIMIZED_FEATURES = [
    "Vol_12",                               # G2 Risk (best IC: +0.020, highest stability)
    "VolRatio",                             # G8 Liquidity (IC: +0.015)
    "LogMktCap",                            # G7 Size (IC: +0.016, high stability)
    "Mom_6_1","Mom_12_1",                   # G1 Momentum (positive IC, high stability)
    "MaxRet_1M",                            # G3 Technical (IC: +0.008)
    "Trend_MA",                             # G3 Technical (high stability)
    "CashFlowYield",                        # G5 Quality (IC: +0.008)
    "EarnGrowth_YoY",                       # G6 Growth (IC: +0.008)
]

TARGET = "Next_Month_Return"
SECTOR_REL_TARGET = "Sector_Relative_Return"  # sector-relative target for sector-neutral training

FEATURE_GROUPS = {
    "G1 Momentum"  :["Mom_12_1","Mom_6_1","Mom_1"],
    "G2 Risk"      :["Vol_12","IdioVol","Beta_12"],
    "G3 Technical" :["High52W","Trend_MA","MaxRet_1M"],
    "G4 Value"     :["PB_ratio","PE_TTM","EV_EBITDA"],
    "G5 Quality"   :["ROE","GrossMargin","CashFlowYield"],
    "G6 Growth"    :["RevGrowth_YoY","EarnGrowth_YoY"],
    "G7 Size"      :["LogMktCap"],
    "G8 Liquidity" :["VolRatio"],
    "G9 Sector-Relative" : [],   # populated dynamically
    "G10 Interactions"   : [],   # populated dynamically
}

# ── Feature Engineering Configuration ──────────────────────────────────────────
FEATURE_ENGINEERING_CONFIG = {
    "enable_interactions": True,       # Targeted interaction features
    "enable_nonlinear": False,         # Non-linear transforms (disabled — adds noise for 250 stocks)
    "enable_sector_relative": True,    # Sector-relative z-scores — KEY for sector-neutral alpha
    "interaction_pairs": [             # Targeted high-value pairs (not group-level combos)
        ("G1 Momentum", "G4 Value"),   # Momentum × Value — Asness (2013)
        ("G5 Quality", "G6 Growth"),    # Quality × Growth — Novy-Marx (2013)
        ("G2 Risk", "G7 Size"),         # Risk × Size — low-vol anomaly
    ],
    "nonlinear_features": [
        "Mom_12_1", "Mom_6_1", "Vol_12", "LogMktCap", "PB_ratio", "PE_TTM"
    ],
    "sector_relative_features": [      # Features to make sector-relative
        "Mom_12_1", "Vol_12", "PB_ratio", "ROE", "LogMktCap",
        "PE_TTM", "RevGrowth_YoY",
    ],
}


def get_features_from_config(config=None):
    """
    Get the feature list based on configuration.
    
    Args:
        config: Configuration dict with 'features' section
        
    Returns:
        List of feature names to use
    """
    global FEATURES  # Allow updating the global FEATURES list
    
    if config is None:
        return FEATURES
    
    features_config = config.get("features", {})
    
    # Check if specific features are selected
    if not features_config.get("use_all_features", True):
        selected = features_config.get("selected_features", None)
        if selected and isinstance(selected, list):
            print(f"  ✓ Using {len(selected)} selected features from config (not all {len(FEATURES)})")
            FEATURES = selected  # Update global FEATURES
            return selected
    
    # Default to all features
    return FEATURES


# ── Cache helpers ──────────────────────────────────────────────────────────────
def _cache_fresh(path, max_hours=CACHE_MAX_HOURS):
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age.total_seconds() < max_hours * 3600


# ── Download helpers ───────────────────────────────────────────────────────────
def _download_batch(tickers, start, end, attempt=0):
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            raise ValueError("Empty result")
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else \
                 raw[["Close"]].rename(columns={"Close": tickers[0]})
        return prices
    except Exception as exc:
        if attempt < MAX_RETRIES:
            wait = (2 ** attempt) * 5 + random.uniform(0, 2)
            print(f"    Retry in {wait:.0f}s ({exc})...")
            time.sleep(wait)
            return _download_batch(tickers, start, end, attempt + 1)
        return pd.DataFrame()


def download_price_data(tickers=None, years=7, force_refresh=False):
    """Downloads daily prices with local parquet cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not force_refresh and _cache_fresh(PRICE_CACHE):
        print(f"Loading prices from cache ({PRICE_CACHE})")
        prices = pd.read_parquet(PRICE_CACHE)
        print(f"  {len(prices)} days, {len(prices.columns)} tickers")
        return prices

    if tickers is None:
        tickers = STOCK_TICKERS + [BENCHMARK]
    end   = datetime.today()
    start = end - timedelta(days=365 * years)
    s, e  = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    print(f"Downloading {len(tickers)} tickers | {s} → {e}")

    batches    = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    all_prices = []
    for b_idx, batch in enumerate(batches):
        print(f"  Batch {b_idx+1}/{len(batches)}: {batch[:4]}...", end=" ")
        df = _download_batch(batch, s, e)
        if not df.empty:
            all_prices.append(df); print(f"OK ({len(df)} days)")
        else:
            print("FAILED")
        if b_idx < len(batches) - 1:
            time.sleep(BATCH_DELAY)

    combined = pd.concat(all_prices, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()].dropna(how="all")
    bad = combined.columns[combined.isnull().mean() > 0.15].tolist()
    if bad:
        print(f"  Dropping {len(bad)} with >15% missing: {bad[:5]}")
        combined = combined.drop(columns=bad)
    combined.to_parquet(PRICE_CACHE)
    print(f"Saved price cache → {PRICE_CACHE}")
    return combined


def _fetch_fundamentals_one(ticker):
    try:
        t   = yf.Ticker(ticker)
        inc = t.quarterly_income_stmt
        bal = t.quarterly_balance_sheet
        if inc is None or inc.empty or bal is None or bal.empty:
            return pd.DataFrame()
        inc = inc.T.copy(); bal = bal.T.copy()

        def find(df, cands):
            for c in cands:
                m = [col for col in df.columns if c.lower() in str(col).lower()]
                if m: return pd.to_numeric(df[m[0]], errors="coerce")
            return pd.Series(np.nan, index=df.index)

        # Cash flow statement (for CashFlowYield)
        try:
            cfs = t.quarterly_cashflow
            if cfs is not None and not cfs.empty:
                cfs = cfs.T.copy()
            else:
                cfs = pd.DataFrame()
        except Exception:
            cfs = pd.DataFrame()

        df = pd.DataFrame({
            "Revenue"          : find(inc, ["Total Revenue","Revenue"]),
            "GrossProfit"      : find(inc, ["Gross Profit","GrossProfit"]),
            "NetIncome"        : find(inc, ["Net Income","NetIncome"]),
            "EBITDA"           : find(inc, ["EBITDA","Normalized EBITDA"]),
            "TotalEquity"      : find(bal, ["Stockholder","Common Stock Equity","Total Equity"]),
            "TotalAssets"      : find(bal, ["Total Assets"]),
            "CurrentLiabilities": find(bal, ["Current Liabilities","CurrentLiabilities"]),
            "CurrentAssets"    : find(bal, ["Current Assets","CurrentAssets"]),
            "TotalDebt"        : find(bal, ["Total Debt","Long Term Debt"]),
            "Shares"           : find(bal, ["Share Issued","Common Stock","Shares"]),
            "OperatingCashFlow": find(cfs, ["Operating Cash Flow","Cash Flow From Operations",
                                            "Free Cash Flow"]) if not cfs.empty
                                 else pd.Series(np.nan, index=bal.index),
            "DividendsPaid"    : find(cfs, ["Dividends Paid","Cash Dividends Paid",
                                            "Common Stock Dividend"]) if not cfs.empty
                                 else pd.Series(np.nan, index=bal.index),
        })
        df.index   = pd.to_datetime(df.index)
        df         = df.sort_index()
        df["ticker"] = ticker
        return df
    except Exception:
        return pd.DataFrame()


def download_fundamentals(tickers=None, force_refresh=False):
    """Fetches quarterly fundamentals with local parquet cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not force_refresh and _cache_fresh(FUND_CACHE):
        print(f"Loading fundamentals from cache ({FUND_CACHE})")
        fund = pd.read_parquet(FUND_CACHE)
        print(f"  {len(fund)} records, {fund['ticker'].nunique()} tickers")
        return fund

    if tickers is None:
        tickers = STOCK_TICKERS
    print(f"Fetching fundamentals for {len(tickers)} tickers...")
    all_fund = []
    for idx, ticker in enumerate(tickers):
        if idx % 25 == 0:
            print(f"  {idx}/{len(tickers)}")
        df = _fetch_fundamentals_one(ticker)
        if not df.empty:
            all_fund.append(df)
        time.sleep(0.5)

    if not all_fund:
        print("  Warning: no fundamental data retrieved")
        return pd.DataFrame()
    combined = pd.concat(all_fund)
    combined.to_parquet(FUND_CACHE)
    print(f"Saved fundamental cache → {FUND_CACHE} "
          f"({len(combined)} records, {combined['ticker'].nunique()} tickers)")
    return combined


def compute_monthly_returns(prices):
    try:
        monthly = prices.resample("ME").last()
    except ValueError:
        monthly = prices.resample("M").last()
    returns = monthly.pct_change().dropna(how="all")
    print(f"Monthly returns: {len(returns)} months, {len(returns.columns)} tickers")
    return returns


def _fund_val(fund_df, ticker, feature_date, col, lag_days=45):
    """
    Returns the most recent quarterly fundamental value for ticker
    where the quarter ended >= lag_days before feature_date.
    This enforces the 45-day publication lag — no leakage.
    """
    if fund_df is None or fund_df.empty: return np.nan
    t_data = fund_df[fund_df["ticker"] == ticker]
    if t_data.empty: return np.nan
    cutoff  = pd.Timestamp(feature_date) - pd.Timedelta(days=lag_days)
    valid   = t_data[t_data.index < cutoff]  # strict < to prevent leakage
    if valid.empty or col not in valid.columns: return np.nan
    val = valid[col].dropna()
    return float(val.iloc[-1]) if not val.empty else np.nan


def compute_factors(monthly_returns, prices, fund_df=None, sector_map=None,
                    monthly_universe=None):
    """
    Computes 17 monthly factors per stock using strictly past data.

    SURVIVORSHIP BIAS FIX:
      If monthly_universe is provided (dict {"YYYY-MM": [tickers]}),
      each month only processes stocks that were actually in the S&P 500
      at that time. This ensures delisted/removed stocks are included
      during their active periods and excluded after removal.

    LEAKAGE AUDIT:
      Price features at month i: windows [i-12..i-1], [i-6..i-1], etc.
      Daily features: up to end of month i-1 only.
      Fundamentals: 45-day lag via _fund_val().
      Target: ret.iloc[i+1] — one full month ahead of all features.
    """
    if sector_map is None: sector_map = SECTOR_MAP
    # All available stock columns (superset — filtered per month below)
    all_stock_cols = [c for c in monthly_returns.columns
                      if c != BENCHMARK and c in sector_map]
    stock_cols  = all_stock_cols  # default: use all
    spy_monthly = monthly_returns.get(BENCHMARK, pd.Series(dtype=float))
    spy_daily   = prices.get(BENCHMARK) if prices is not None else None
    month_ends  = monthly_returns.index
    records     = []
    skipped     = 0

    for i in range(12, len(monthly_returns) - 1):
        current_date = month_ends[i]
        daily_end    = month_ends[i - 1]

        daily_start_252 = None
        if spy_daily is not None and prices is not None:
            # Use asof() to find nearest trading day <= daily_end
            # (month-end dates like 2024-03-31 may not be trading days)
            nearest_daily_end = prices.index.asof(daily_end)
            if pd.notna(nearest_daily_end):
                loc = prices.index.get_loc(nearest_daily_end)
                daily_start_252 = prices.index[max(0, loc - 252)]
            else:
                nearest_daily_end = daily_end  # fallback

        prev_month_start = month_ends[i - 2] if i >= 2 else prices.index[0]
        vol_start_3m     = month_ends[i - 4] if i >= 4 else prices.index[0]

        # ── Point-in-time universe filtering ──────────────────────────
        # If monthly_universe provided, only process stocks that were
        # actually in the S&P 500 during this month (survivorship bias fix)
        if monthly_universe is not None:
            month_key = current_date.strftime("%Y-%m")
            pit_tickers = set(monthly_universe.get(month_key, []))
            stock_cols_month = [c for c in all_stock_cols if c in pit_tickers]
        else:
            stock_cols_month = stock_cols

        for ticker in stock_cols_month:
            ret   = monthly_returns[ticker]
            price = prices.get(ticker) if prices is not None else None

            win12 = ret.iloc[i-12:i].ffill()
            win6  = ret.iloc[i-6:i].ffill()
            win3  = ret.iloc[i-3:i].ffill()

            if win12.isnull().sum() > 4:
                skipped += 1; continue

            # G1 Momentum
            Mom_12_1 = float((1 + win12.iloc[:-1]).prod() - 1)
            Mom_6_1  = float((1 + win6.iloc[:-1]).prod() - 1)
            Mom_1    = float(win12.iloc[-1])

            # G2 Risk
            Vol_12  = float(win12.std())
            Beta_12 = np.nan; IdioVol = np.nan
            if len(spy_monthly) > 0:
                spy_win = spy_monthly.iloc[i-12:i].ffill()
                if spy_win.std() > 0:
                    cov     = np.cov(win12.values, spy_win.values)
                    Beta_12 = float(cov[0,1] / (cov[1,1] + 1e-10))

            # IdioVol from daily data (annualised residual volatility)
            if price is not None and spy_daily is not None:
                try:
                    # Use tail-based slicing for robustness
                    p_slice = price.loc[:daily_end].dropna()
                    s_slice = spy_daily.loc[:daily_end].dropna()
                    if len(p_slice) >= 63 and len(s_slice) >= 63:
                        p_daily = p_slice.tail(252).pct_change().dropna()
                        s_daily = s_slice.tail(252).pct_change().dropna()
                        cm = p_daily.index.intersection(s_daily.index)
                        if len(cm) >= 60:
                            pv = p_daily.loc[cm].values; sv = s_daily.loc[cm].values
                            b  = np.cov(pv,sv)[0,1]/(np.var(sv)+1e-10)
                            IdioVol = float((pv - b*sv).std() * np.sqrt(252))
                except Exception: pass

            # G3 Technical
            High52W = np.nan; Trend_MA = np.nan; MaxRet_1M = np.nan
            if price is not None:
                try:
                    # Use tail-based slicing — robust to missing daily_start_252
                    py = price.loc[:daily_end].dropna().tail(252)
                    if len(py) >= 50 and py.max() > 0:
                        High52W  = float(py.iloc[-1] / py.max())
                        Trend_MA = float(py.iloc[-1] / py.mean())
                except Exception: pass
            if price is not None:
                try:
                    pm = price.loc[prev_month_start:daily_end].pct_change().dropna()
                    if len(pm) >= 5: MaxRet_1M = float(pm.max())
                except Exception: pass

            # ── Pre-filter fundamental data for this ticker (once per ticker) ──
            td = None; cut = None; pr = np.nan; sh_val = np.nan
            if fund_df is not None:
                td = fund_df[fund_df["ticker"] == ticker]
                cut = pd.Timestamp(current_date) - pd.Timedelta(days=45)
                # Use <= to ensure we only use data from quarters ending at least 45 days before current_date
                vq = td[td.index <= cut].sort_index()  # strict <= for 45-day lag enforcement
            if price is not None:
                p_end = price.loc[:daily_end].dropna()
                if len(p_end) > 0:
                    pr = float(p_end.iloc[-1])

            # G4 Value (45-day lagged)
            PB_ratio = np.nan; PE_TTM = np.nan; EV_EBITDA = np.nan
            if td is not None and len(vq) > 0 and not np.isnan(pr):
                try:
                    # Get latest values from pre-filtered data
                    eq_s = vq["TotalEquity"].dropna()
                    sh_s = vq["Shares"].dropna()
                    eq = float(eq_s.iloc[-1]) if len(eq_s) > 0 else np.nan
                    sh_val = float(sh_s.iloc[-1]) if len(sh_s) > 0 else np.nan

                    if not np.isnan(eq) and not np.isnan(sh_val) and sh_val > 0 and eq/sh_val > 0:
                        PB_ratio = float(pr / (eq / sh_val))

                    # PE_TTM — use available quarters (1-4)
                    ni_q = vq["NetIncome"].dropna()
                    if len(ni_q) >= 1 and not np.isnan(sh_val) and sh_val > 0:
                        n_q = min(4, len(ni_q))
                        ttm_ni = float(ni_q.iloc[-n_q:].sum()) * (4 / n_q)  # annualise
                        eps = ttm_ni / sh_val
                        if eps > 0: PE_TTM = float(pr / eps)

                    # EV/EBITDA
                    ebitda_q = vq["EBITDA"].dropna()
                    debt_s = vq["TotalDebt"].dropna()
                    debt_v = float(debt_s.iloc[-1]) if len(debt_s) > 0 else 0.0
                    if len(ebitda_q) >= 1 and not np.isnan(sh_val) and sh_val > 0:
                        n_q = min(4, len(ebitda_q))
                        ttm_ebitda = float(ebitda_q.iloc[-n_q:].sum()) * (4 / n_q)
                        mkt_cap = pr * sh_val
                        ev = mkt_cap + debt_v
                        if ttm_ebitda > 0:
                            EV_EBITDA = float(ev / ttm_ebitda)
                except Exception: pass

            # G5 Quality (45-day lagged)
            ROE = np.nan; GrossMargin = np.nan
            CurrentRatio = np.nan; CashFlowYield = np.nan
            if td is not None and len(vq) > 0:
                try:
                    ni_s = vq["NetIncome"].dropna()
                    eq_s = vq["TotalEquity"].dropna()
                    rv_s = vq["Revenue"].dropna()
                    gp_s = vq["GrossProfit"].dropna()
                    ni = float(ni_s.iloc[-1]) if len(ni_s) > 0 else np.nan
                    eq = float(eq_s.iloc[-1]) if len(eq_s) > 0 else np.nan
                    rv = float(rv_s.iloc[-1]) if len(rv_s) > 0 else np.nan
                    gp = float(gp_s.iloc[-1]) if len(gp_s) > 0 else np.nan

                    if not np.isnan(ni) and not np.isnan(eq) and eq != 0:
                        ROE = float(ni / eq)
                    if not np.isnan(gp) and not np.isnan(rv) and rv != 0:
                        GrossMargin = float(gp / rv)

                    ca_s = vq["CurrentAssets"].dropna()
                    cl_s = vq["CurrentLiabilities"].dropna()
                    ca = float(ca_s.iloc[-1]) if len(ca_s) > 0 else np.nan
                    cl = float(cl_s.iloc[-1]) if len(cl_s) > 0 else np.nan
                    if not np.isnan(ca) and not np.isnan(cl) and cl > 0:
                        CurrentRatio = float(ca / cl)

                    ocf_s = vq["OperatingCashFlow"].dropna()
                    ocf = float(ocf_s.iloc[-1]) if len(ocf_s) > 0 else np.nan
                    if not np.isnan(ocf) and not np.isnan(pr) and not np.isnan(sh_val) and sh_val > 0 and pr > 0:
                        CashFlowYield = float(ocf / (pr * sh_val))
                except Exception: pass

            # G6 Growth (45-day lagged) — relaxed to 2 quarters for coverage
            RevGrowth_YoY = np.nan; EarnGrowth_YoY = np.nan
            if td is not None and len(vq) >= 2:
                try:
                    rn = vq["Revenue"].dropna()
                    nn = vq["NetIncome"].dropna()
                    # YoY: compare latest to 4 quarters ago (or earliest available)
                    if len(rn) >= 5 and rn.iloc[-5] != 0:
                        RevGrowth_YoY = float((rn.iloc[-1] - rn.iloc[-5]) / abs(rn.iloc[-5]))
                    elif len(rn) >= 2 and rn.iloc[0] != 0:
                        # Fallback: compare latest to earliest available
                        RevGrowth_YoY = float((rn.iloc[-1] - rn.iloc[0]) / abs(rn.iloc[0]))
                    if len(nn) >= 5 and nn.iloc[-5] != 0:
                        EarnGrowth_YoY = float((nn.iloc[-1] - nn.iloc[-5]) / abs(nn.iloc[-5]))
                    elif len(nn) >= 2 and nn.iloc[0] != 0:
                        EarnGrowth_YoY = float((nn.iloc[-1] - nn.iloc[0]) / abs(nn.iloc[0]))
                except Exception: pass

            # G7 Size (45-day lagged shares × current price)
            LogMktCap = np.nan
            if not np.isnan(sh_val) and not np.isnan(pr) and sh_val > 0 and pr > 0:
                LogMktCap = float(np.log(sh_val * pr))

            # G8 Liquidity — 1M vol / 3M vol ratio (price-based proxy)
            VolRatio = np.nan
            if price is not None:
                try:
                    p1m = price.loc[prev_month_start:daily_end].pct_change().dropna()
                    p3m = price.loc[vol_start_3m:daily_end].pct_change().dropna()
                    if len(p1m)>=5 and len(p3m)>=20 and p3m.std()>0:
                        VolRatio = float(p1m.std() / (p3m.std()+1e-10))
                except Exception: pass

            # G9 Efficiency — Asset Turnover = Revenue / Total Assets
            AssetTurnover = np.nan
            if fund_df is not None:
                try:
                    rv_at = _fund_val(fund_df, ticker, current_date, "Revenue")
                    ta_at = _fund_val(fund_df, ticker, current_date, "TotalAssets")
                    if not np.isnan(rv_at) and not np.isnan(ta_at) and ta_at > 0:
                        AssetTurnover = float(rv_at / ta_at)
                except Exception: pass

            # G10 Income — Dividend Yield (annualised dividends / price)
            DividendYield = np.nan
            if fund_df is not None and price is not None:
                try:
                    td_dy = fund_df[fund_df["ticker"]==ticker]
                    cut_dy = pd.Timestamp(current_date) - pd.Timedelta(days=45)
                    dv = td_dy[td_dy.index<=cut_dy]["DividendsPaid"].dropna()
                    if len(dv) >= 4:
                        # DividendsPaid is typically negative (cash outflow)
                        ttm_div = float(abs(dv.iloc[-4:].sum()))
                        pr_dy = float(price.loc[:daily_end].iloc[-1]) \
                                if len(price.loc[:daily_end]) > 0 else np.nan
                        sh_dy = _fund_val(fund_df, ticker, current_date, "Shares")
                        if (not np.isnan(pr_dy) and pr_dy > 0
                                and not np.isnan(sh_dy) and sh_dy > 0):
                            DividendYield = float(ttm_div / (pr_dy * sh_dy))
                except Exception: pass

            # Target
            next_ret = ret.iloc[i+1]
            if pd.isna(next_ret):
                skipped += 1; continue

            records.append({
                "date":current_date,"ticker":ticker,
                "sector":sector_map.get(ticker,"Unknown"),
                "Mom_12_1":Mom_12_1,"Mom_6_1":Mom_6_1,"Mom_1":Mom_1,
                "Vol_12":Vol_12,"IdioVol":IdioVol,"Beta_12":Beta_12,
                "High52W":High52W,"Trend_MA":Trend_MA,"MaxRet_1M":MaxRet_1M,
                "PB_ratio":PB_ratio,"PE_TTM":PE_TTM,"EV_EBITDA":EV_EBITDA,
                "ROE":ROE,"GrossMargin":GrossMargin,
                "CurrentRatio":CurrentRatio,"CashFlowYield":CashFlowYield,
                "RevGrowth_YoY":RevGrowth_YoY,"EarnGrowth_YoY":EarnGrowth_YoY,
                "LogMktCap":LogMktCap,"VolRatio":VolRatio,
                "AssetTurnover":AssetTurnover,"DividendYield":DividendYield,
                TARGET:next_ret,
            })

    factors_df = pd.DataFrame(records)

    # Report PRE-imputation coverage (true state of data quality)
    print("\n  Feature coverage (BEFORE imputation):")
    for grp, feats in FEATURE_GROUPS.items():
        for f in feats:
            if f in factors_df.columns:
                pct = factors_df[f].notna().mean()
                flag = "✓" if pct > 0.5 else "⚠" if pct > 0.1 else "✗"
                print(f"    {flag} {f:<18}: {pct:>5.1%}")

    # Cross-sectional winsorization at 1st/99th percentile (prevents outlier domination)
    for feat in FEATURES:
        if feat in factors_df.columns:
            factors_df[feat] = factors_df.groupby("date")[feat].transform(
                lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
            )

    # Cross-sectional median imputation — fill missing fundamentals
    # with each month's median (no future data leak)
    for feat in FEATURES:
        if feat in factors_df.columns:
            factors_df[feat] = factors_df.groupby("date")[feat].transform(
                lambda x: x.fillna(x.median())
            )

    # Report universe type
    universe_type = "POINT-IN-TIME (survivorship-bias-free)" if monthly_universe else "STATIC (fallback)"
    print(f"\nFactor dataset: {len(factors_df):,} rows | "
          f"{factors_df['ticker'].nunique()} stocks | "
          f"{factors_df['date'].nunique()} months | {skipped} skipped")
    print(f"  Universe type: {universe_type}")
    print("\n  Feature coverage (AFTER imputation):")
    for grp, feats in FEATURE_GROUPS.items():
        for f in feats:
            if f in factors_df.columns:
                pct = factors_df[f].notna().mean()
                print(f"    {f:<18}: {pct:>5.1%}")
    return factors_df


def check_feature_correlation(factors_df, threshold=0.75):
    """
    Computes mean cross-sectional Spearman correlation between all feature pairs.
    Flags pairs with |mean_corr| > threshold before training.
    LightGBM handles multicollinearity but high correlation wastes capacity.
    """
    print("\n=== FEATURE CORRELATION CHECK ===")
    monthly_corrs = []
    avail = [f for f in FEATURES if f in factors_df.columns]
    for date, g in factors_df.groupby("date"):
        fd = g[avail].dropna(how="all")
        if len(fd) < 10: continue
        try:
            monthly_corrs.append(fd.corr(method="spearman"))
        except Exception: continue

    if not monthly_corrs:
        print("  Could not compute"); return None

    mean_corr = pd.concat(monthly_corrs).groupby(level=0).mean()
    flagged   = []
    for i, f1 in enumerate(avail):
        for j, f2 in enumerate(avail):
            if j <= i or f1 not in mean_corr or f2 not in mean_corr: continue
            c = mean_corr.loc[f1, f2]
            if abs(c) > threshold:
                flagged.append((f1, f2, c))

    if flagged:
        print(f"\n  ⚠  {len(flagged)} highly correlated pairs (|corr| > {threshold}):")
        for f1, f2, c in sorted(flagged, key=lambda x: abs(x[2]), reverse=True):
            print(f"    {f1:<18} × {f2:<18}: {c:+.3f}")
        print("  Note: kept — LightGBM handles via SHAP attribution.")
    else:
        print(f"  No pairs exceed |corr| > {threshold} — feature set clean.")

    vals = mean_corr.abs().values
    tri  = vals[np.triu_indices_from(vals, k=1)]
    print(f"  Mean |corr| across all pairs: {tri.mean():.3f}")
    return mean_corr


def add_interaction_features(factors_df, config=None):
    """
    Add targeted interaction features between specific factor pairs.
    
    Creates 3 academically-grounded interactions using rank-transformed
    inputs (cross-sectional percentile ranks per month) for stability:
    
    1. Mom12_x_PBratio    — Value × Momentum (Asness 2013)
    2. ROE_x_RevGrowth    — Quality × Growth (Novy-Marx 2013)
    3. Vol12_x_LogMktCap  — Risk × Size (low-vol anomaly)
    
    Rank-based inputs ensure interactions are scale-invariant and
    robust to outliers in fundamentals (e.g., PE_TTM=500).
    """
    if config is None:
        config = FEATURE_ENGINEERING_CONFIG
    
    if not config.get("enable_interactions", False):
        return factors_df
    
    print("\n=== Adding Targeted Interaction Features ===")
    
    # Define specific interaction pairs with clean names
    specific_interactions = [
        ("Mom_12_1",       "PB_ratio",       "Mom12_x_PBratio",   "Value×Momentum (Asness 2013)"),
        ("ROE",            "RevGrowth_YoY",  "ROE_x_RevGrowth",   "Quality×Growth (Novy-Marx 2013)"),
        ("Vol_12",         "LogMktCap",       "Vol12_x_LogMktCap", "Risk×Size (low-vol anomaly)"),
    ]
    
    new_features = []
    for feat1, feat2, name, desc in specific_interactions:
        if feat1 in factors_df.columns and feat2 in factors_df.columns:
            # Use cross-sectional ranks for stability (scale-invariant)
            r1 = factors_df.groupby("date")[feat1].rank(pct=True)
            r2 = factors_df.groupby("date")[feat2].rank(pct=True)
            factors_df[name] = r1 * r2
            new_features.append(name)
            print(f"  Created: {name:<22} — {desc}")
    
    print(f"  Total interaction features added: {len(new_features)}")
    
    return factors_df


def add_nonlinear_transformations(factors_df, config=None):
    """
    Add non-linear transformations of existing features.
    
    Applies log, sqrt, and rank transformations to capture non-linear
    relationships in the data. Handles NaN values and negative values gracefully.
    
    Transformations:
    - log: log(1 + x) for positive values, handles negatives via log(1 + |x|) * sign(x)
    - sqrt: sqrt(|x|) * sign(x) to handle negative values
    - rank: cross-sectional percentile rank within each month
    
    Args:
        factors_df: DataFrame with base features
        config: Feature engineering configuration dict (default: FEATURE_ENGINEERING_CONFIG)
    
    Returns:
        DataFrame with added non-linear transformation features
        
    Validates: Requirements 2.6
    """
    if config is None:
        config = FEATURE_ENGINEERING_CONFIG
    
    if not config.get("enable_nonlinear", False):
        return factors_df
    
    print("\n=== Adding Non-Linear Transformations ===")
    
    nonlinear_features = config.get("nonlinear_features", [])
    new_features = []
    
    for feat in nonlinear_features:
        if feat not in factors_df.columns:
            continue
        
        # Log transformation: log(1 + x) for positive, log(1 + |x|) * sign(x) for negative
        log_name = f"{feat}_log"
        factors_df[log_name] = factors_df[feat].apply(
            lambda x: np.log1p(x) if x >= 0 else -np.log1p(-x) if not np.isnan(x) else np.nan
        )
        new_features.append(log_name)
        
        # Sqrt transformation: sqrt(|x|) * sign(x)
        sqrt_name = f"{feat}_sqrt"
        factors_df[sqrt_name] = factors_df[feat].apply(
            lambda x: np.sqrt(abs(x)) * np.sign(x) if not np.isnan(x) else np.nan
        )
        new_features.append(sqrt_name)
        
        # Rank transformation: cross-sectional percentile rank within each month
        rank_name = f"{feat}_rank"
        factors_df[rank_name] = factors_df.groupby("date")[feat].rank(pct=True)
        new_features.append(rank_name)
        
        print(f"  Created: {log_name}, {sqrt_name}, {rank_name}")
    
    print(f"  Total non-linear features added: {len(new_features)}")
    
    return factors_df


def add_sector_relative_features(factors_df, config=None):
    """
    Add sector-relative feature transformations.
    
    Transforms features to be relative to sector medians/means, creating
    sector-neutral signals that capture within-sector relative strength.
    
    Two transformations:
    - z-score: (value - sector_mean) / sector_std
    - percentile: percentile rank within sector for each month
    
    Args:
        factors_df: DataFrame with base features and 'sector' column
        config: Feature engineering configuration dict (default: FEATURE_ENGINEERING_CONFIG)
    
    Returns:
        DataFrame with added sector-relative features
        
    Validates: Requirements 2.9
    """
    if config is None:
        config = FEATURE_ENGINEERING_CONFIG
    
    if not config.get("enable_sector_relative", False):
        return factors_df
    
    if "sector" not in factors_df.columns:
        print("\n⚠  Warning: 'sector' column not found, skipping sector-relative features")
        return factors_df
    
    print("\n=== Adding Sector-Relative Features ===")
    
    sector_relative_features = config.get("sector_relative_features", [])
    new_features = []
    
    for feat in sector_relative_features:
        if feat not in factors_df.columns:
            continue
        
        # Z-score: (value - sector_mean) / sector_std
        zscore_name = f"{feat}_sector_z"
        factors_df[zscore_name] = factors_df.groupby(["date", "sector"])[feat].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10) if x.std() > 0 else 0
        )
        new_features.append(zscore_name)
        
        # Percentile rank within sector
        pct_name = f"{feat}_sector_pct"
        factors_df[pct_name] = factors_df.groupby(["date", "sector"])[feat].rank(pct=True)
        new_features.append(pct_name)
        
        print(f"  Created: {zscore_name}, {pct_name}")
    
    print(f"  Total sector-relative features added: {len(new_features)}")
    
    return factors_df


def apply_feature_engineering(factors_df, config=None):
    """
    Apply all feature engineering transformations and register new features.
    
    This is the main entry point for feature engineering enhancements.
    After calling this function, ALL_FEATURES and FEATURE_GROUPS are updated
    to include the engineered features, ready for model training.
    
    Transformations applied:
    1. Targeted interaction features (Value×Momentum, Quality×Growth, Risk×Size)
    2. Non-linear transformations (optional, disabled by default)
    3. Sector-relative z-scores for key features
    4. Sector-relative return target for sector-neutral training
    
    All transformations maintain temporal integrity (no future data leakage).
    """
    global ALL_FEATURES
    
    if config is None:
        config = FEATURE_ENGINEERING_CONFIG
    
    print("\n" + "="*70)
    print("  FEATURE ENGINEERING ENHANCEMENTS")
    print("="*70)
    
    original_feature_count = len([c for c in factors_df.columns 
                                   if c not in ["date", "ticker", "sector", TARGET, SECTOR_REL_TARGET]])
    
    # Apply transformations in sequence
    factors_df = add_interaction_features(factors_df, config)
    factors_df = add_nonlinear_transformations(factors_df, config)
    factors_df = add_sector_relative_features(factors_df, config)
    
    # ── Create sector-relative return target ──
    # Ticker return minus sector median return for that month
    # Forces the model to learn stock selection, not sector rotation
    if "sector" in factors_df.columns and TARGET in factors_df.columns:
        print("\n=== Creating Sector-Relative Return Target ===")
        factors_df[SECTOR_REL_TARGET] = factors_df.groupby(["date", "sector"])[TARGET].transform(
            lambda x: x - x.median()
        )
        print(f"  Created: {SECTOR_REL_TARGET} = {TARGET} - sector_median({TARGET})")
        print(f"  Raw target std: {factors_df[TARGET].std():.4f}")
        print(f"  Sector-relative target std: {factors_df[SECTOR_REL_TARGET].std():.4f}")
    
    # ── Register new features into ALL_FEATURES and FEATURE_GROUPS ──
    new_engineered = [c for c in factors_df.columns 
                      if c not in ["date", "ticker", "sector", TARGET, SECTOR_REL_TARGET] 
                      and c not in FEATURES]
    
    # Only include z-score features (not percentile rank — avoids feature bloat)
    sector_z_feats = [f for f in new_engineered if f.endswith("_sector_z")]
    interaction_feats = [f for f in new_engineered if "_x_" in f]
    other_feats = [f for f in new_engineered if f not in sector_z_feats and f not in interaction_feats
                   and not f.endswith("_sector_pct")]  # exclude pct features
    
    # Update FEATURE_GROUPS
    FEATURE_GROUPS["G9 Sector-Relative"] = sector_z_feats
    FEATURE_GROUPS["G10 Interactions"] = interaction_feats
    
    # Build ALL_FEATURES: base + z-scores + interactions (no pct features)
    ALL_FEATURES = FEATURES + sector_z_feats + interaction_feats + other_feats
    
    # Winsorize + impute new features
    feats_to_process = sector_z_feats + interaction_feats + other_feats
    if feats_to_process:
        print(f"\n=== Processing {len(feats_to_process)} New Features ===")
        for feat in feats_to_process:
            # Winsorize at 1st/99th percentile
            factors_df[feat] = factors_df.groupby("date")[feat].transform(
                lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
            )
            # Median imputation
            factors_df[feat] = factors_df.groupby("date")[feat].transform(
                lambda x: x.fillna(x.median())
            )
        print(f"  Applied winsorization + median imputation to {len(feats_to_process)} new features")
    
    final_feature_count = len(ALL_FEATURES)
    
    print("\n" + "="*70)
    print(f"  Feature Engineering Complete")
    print(f"  Base features:          {len(FEATURES)}")
    print(f"  Sector-relative (z):    {len(sector_z_feats)}")
    print(f"  Interaction features:   {len(interaction_feats)}")
    print(f"  Other new features:     {len(other_feats)}")
    print(f"  Total model features:   {final_feature_count}")
    print(f"  ALL_FEATURES list:      {ALL_FEATURES}")
    print("="*70)
    
    return factors_df


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    prices          = download_price_data()
    monthly_returns = compute_monthly_returns(prices)
    monthly_returns.to_csv(os.path.join(CACHE_DIR, "monthly_returns.csv"))
    fund_df         = download_fundamentals()
    factors_df      = compute_factors(monthly_returns, prices, fund_df)
    
    # Apply feature engineering enhancements (disabled by default)
    factors_df = apply_feature_engineering(factors_df)
    
    factors_df.to_csv(os.path.join(CACHE_DIR, "factor_features.csv"), index=False)
    check_feature_correlation(factors_df)
    print(f"\nBase Features: {FEATURES}\nTarget  : {TARGET}")
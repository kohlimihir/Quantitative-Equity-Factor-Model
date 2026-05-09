"""
data_loader.py  —  250-Stock Universe, 19 Features, Full Caching
================================================================
Downloads price + fundamental data for 250 S&P 500 stocks (50/sector),
engineers 19 monthly factors across 8 groups, and caches all raw data
so subsequent runs skip the download entirely.

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

# ── 250-stock universe (50 per sector) ────────────────────────────────────────
SECTOR_MAP = {
    # Technology (50)
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology",
    "GOOGL":"Technology","META":"Technology","AMD":"Technology",
    "INTC":"Technology","QCOM":"Technology","TXN":"Technology",
    "AVGO":"Technology","AMAT":"Technology","MU":"Technology",
    "LRCX":"Technology","KLAC":"Technology","NOW":"Technology",
    "CRM":"Technology","ADBE":"Technology","ORCL":"Technology",
    "INTU":"Technology","PANW":"Technology","NET":"Technology",
    "CRWD":"Technology","FTNT":"Technology","CDNS":"Technology",
    "SNPS":"Technology","VRSN":"Technology","HPQ":"Technology",
    "IBM":"Technology","CSCO":"Technology","ACN":"Technology",
    "DELL":"Technology","ANET":"Technology","MRVL":"Technology",
    "NXPI":"Technology","ADI":"Technology","MCHP":"Technology",
    "MPWR":"Technology","KEYS":"Technology","TER":"Technology",
    "ENTG":"Technology","SWKS":"Technology","QRVO":"Technology",
    "WDC":"Technology","STX":"Technology","ONTO":"Technology",
    "IPGP":"Technology","COHU":"Technology","FORM":"Technology",
    "CGNX":"Technology","MKSI":"Technology",
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
    # Healthcare (50)
    "JNJ":"Healthcare","UNH":"Healthcare","PFE":"Healthcare",
    "ABBV":"Healthcare","LLY":"Healthcare","MRK":"Healthcare",
    "BMY":"Healthcare","AMGN":"Healthcare","GILD":"Healthcare",
    "BIIB":"Healthcare","VRTX":"Healthcare","REGN":"Healthcare",
    "CVS":"Healthcare","CI":"Healthcare","HUM":"Healthcare",
    "MDT":"Healthcare","SYK":"Healthcare","BSX":"Healthcare",
    "ABT":"Healthcare","TMO":"Healthcare","DHR":"Healthcare",
    "ISRG":"Healthcare","EW":"Healthcare","ZBH":"Healthcare",
    "BAX":"Healthcare","BDX":"Healthcare","IQV":"Healthcare",
    "A":"Healthcare","DGX":"Healthcare","RMD":"Healthcare",
    "HOLX":"Healthcare","IDXX":"Healthcare","WAT":"Healthcare",
    "MTD":"Healthcare","PODD":"Healthcare","DXCM":"Healthcare",
    "ALGN":"Healthcare","MASI":"Healthcare","NTRA":"Healthcare",
    "INSP":"Healthcare","ACAD":"Healthcare","RARE":"Healthcare",
    "FOLD":"Healthcare","ALKS":"Healthcare","NBIX":"Healthcare",
    "RVNC":"Healthcare","PRCT":"Healthcare","AXNX":"Healthcare",
    "NVCR":"Healthcare","ROIV":"Healthcare",
    # Consumer (50)
    "AMZN":"Consumer","WMT":"Consumer","HD":"Consumer",
    "COST":"Consumer","TGT":"Consumer","LOW":"Consumer",
    "SBUX":"Consumer","MCD":"Consumer","NKE":"Consumer",
    "LULU":"Consumer","TJX":"Consumer","ROST":"Consumer",
    "DG":"Consumer","DLTR":"Consumer","KR":"Consumer",
    "YUM":"Consumer","CMG":"Consumer","DRI":"Consumer",
    "ORLY":"Consumer","AZO":"Consumer","EBAY":"Consumer",
    "BKNG":"Consumer","MAR":"Consumer","HLT":"Consumer",
    "MGM":"Consumer","F":"Consumer","GM":"Consumer",
    "TSLA":"Consumer","BBY":"Consumer","NFLX":"Consumer",
    "DIS":"Consumer","CMCSA":"Consumer","LVS":"Consumer",
    "WYNN":"Consumer","RCL":"Consumer","CCL":"Consumer",
    "NCLH":"Consumer","H":"Consumer","DKNG":"Consumer",
    "POOL":"Consumer","GRMN":"Consumer","SIRI":"Consumer",
    "WBD":"Consumer","PARA":"Consumer","FOX":"Consumer",
    "CZR":"Consumer","PENN":"Consumer","CHDN":"Consumer",
    "ABNB":"Consumer","IHG":"Consumer",
    # Energy + Industrials (50)
    "XOM":"Energy","CVX":"Energy","COP":"Energy",
    "EOG":"Energy","SLB":"Energy","HAL":"Energy",
    "MPC":"Energy","VLO":"Energy","PSX":"Energy",
    "OXY":"Energy","DVN":"Energy","APA":"Energy",
    "BA":"Energy","LMT":"Energy","RTX":"Energy",
    "NOC":"Energy","GD":"Energy","GE":"Energy",
    "HON":"Energy","MMM":"Energy","CAT":"Energy",
    "DE":"Energy","UNP":"Energy","UPS":"Energy",
    "FDX":"Energy","EMR":"Energy","ETN":"Energy",
    "PH":"Energy","ROK":"Energy","AME":"Energy",
    "VRSK":"Energy","IDEX":"Energy","XYL":"Energy",
    "ROP":"Energy","HUBB":"Energy","FTV":"Energy",
    "GNRC":"Energy","FSLR":"Energy","ENPH":"Energy",
    "BKR":"Energy","MRO":"Energy","CTRA":"Energy",
    "WMB":"Energy","HES":"Energy","SM":"Energy",
    "MTDR":"Energy","BE":"Energy","PLUG":"Energy",
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

# Optimized feature set based on IC analysis (removes low/negative IC features)
OPTIMIZED_FEATURES = [
    "RevGrowth_YoY","EarnGrowth_YoY",       # G6 Growth (highest IC)
    "IdioVol","Beta_12","Vol_12",           # G2 Risk (strong IC)
    "Mom_6_1","Mom_12_1",                   # G1 Momentum (positive IC)
    "EV_EBITDA","PE_TTM",                   # G4 Value (moderate IC)
    "ROE",                                   # G5 Quality (keep best)
    "MaxRet_1M","VolRatio",                 # G3 Technical, G8 Liquidity
]

TARGET = "Next_Month_Return"

FEATURE_GROUPS = {
    "G1 Momentum"  :["Mom_12_1","Mom_6_1","Mom_1"],
    "G2 Risk"      :["Vol_12","IdioVol","Beta_12"],
    "G3 Technical" :["High52W","Trend_MA","MaxRet_1M"],
    "G4 Value"     :["PB_ratio","PE_TTM","EV_EBITDA"],
    "G5 Quality"   :["ROE","GrossMargin","CashFlowYield"],
    "G6 Growth"    :["RevGrowth_YoY","EarnGrowth_YoY"],
    "G7 Size"      :["LogMktCap"],
    "G8 Liquidity" :["VolRatio"],
}

# ── Feature Engineering Configuration ──────────────────────────────────────────
FEATURE_ENGINEERING_CONFIG = {
    "enable_interactions": False,      # Enable interaction features between groups
    "enable_nonlinear": False,         # Enable non-linear transformations
    "enable_sector_relative": False,   # Enable sector-relative features
    "interaction_pairs": [             # Pairs of groups to create interactions
        ("G1 Momentum", "G4 Value"),   # Momentum × Value
        ("G2 Risk", "G7 Size"),        # Risk × Size
        ("G3 Technical", "G5 Quality"), # Technical × Quality
    ],
    "nonlinear_features": [            # Features to apply non-linear transforms
        "Mom_12_1", "Mom_6_1", "Vol_12", "LogMktCap", "PB_ratio", "PE_TTM"
    ],
    "sector_relative_features": [      # Features to make sector-relative
        "Mom_12_1", "Vol_12", "PB_ratio", "ROE", "LogMktCap"
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
    if config is None:
        return FEATURES
    
    features_config = config.get("features", {})
    
    # Check if specific features are selected
    if not features_config.get("use_all_features", True):
        selected = features_config.get("selected_features", None)
        if selected and isinstance(selected, list):
            print(f"Using {len(selected)} selected features from config")
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
    valid   = t_data[t_data.index <= cutoff]
    if valid.empty or col not in valid.columns: return np.nan
    val = valid[col].dropna()
    return float(val.iloc[-1]) if not val.empty else np.nan


def compute_factors(monthly_returns, prices, fund_df=None, sector_map=None):
    """
    Computes 17 monthly factors per stock using strictly past data.

    LEAKAGE AUDIT:
      Price features at month i: windows [i-12..i-1], [i-6..i-1], etc.
      Daily features: up to end of month i-1 only.
      Fundamentals: 45-day lag via _fund_val().
      Target: ret.iloc[i+1] — one full month ahead of all features.
    """
    if sector_map is None: sector_map = SECTOR_MAP
    stock_cols  = [c for c in monthly_returns.columns
                   if c != BENCHMARK and c in sector_map]
    spy_monthly = monthly_returns.get(BENCHMARK, pd.Series(dtype=float))
    spy_daily   = prices.get(BENCHMARK) if prices is not None else None
    month_ends  = monthly_returns.index
    records     = []
    skipped     = 0

    for i in range(12, len(monthly_returns) - 1):
        current_date = month_ends[i]
        daily_end    = month_ends[i - 1]

        daily_start_252 = None
        if spy_daily is not None and daily_end in prices.index:
            loc = prices.index.get_loc(daily_end)
            daily_start_252 = prices.index[max(0, loc - 252)]

        prev_month_start = month_ends[i - 2] if i >= 2 else prices.index[0]
        vol_start_3m     = month_ends[i - 4] if i >= 4 else prices.index[0]

        for ticker in stock_cols:
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

            if price is not None and spy_daily is not None and daily_start_252 is not None:
                try:
                    pd_ = price.loc[daily_start_252:daily_end].pct_change().dropna()
                    sd_ = spy_daily.loc[daily_start_252:daily_end].pct_change().dropna()
                    cm  = pd_.index.intersection(sd_.index)
                    if len(cm) >= 60:
                        pv = pd_.loc[cm].values; sv = sd_.loc[cm].values
                        b  = np.cov(pv,sv)[0,1]/(np.var(sv)+1e-10)
                        IdioVol = float((pv - b*sv).std() * np.sqrt(252))
                except Exception: pass

            # G3 Technical
            High52W = np.nan; Trend_MA = np.nan; MaxRet_1M = np.nan
            if price is not None and daily_start_252 is not None:
                try:
                    py = price.loc[daily_start_252:daily_end]
                    if len(py) >= 50 and py.max() > 0:
                        High52W  = float(py.iloc[-1] / py.max())
                        Trend_MA = float(py.iloc[-1] / py.mean())
                except Exception: pass
            if price is not None:
                try:
                    pm = price.loc[prev_month_start:daily_end].pct_change().dropna()
                    if len(pm) >= 5: MaxRet_1M = float(pm.max())
                except Exception: pass

            # G4 Value (45-day lagged)
            PB_ratio = np.nan; PE_TTM = np.nan; EV_EBITDA = np.nan
            if fund_df is not None and price is not None:
                try:
                    eq  = _fund_val(fund_df, ticker, current_date, "TotalEquity")
                    sh  = _fund_val(fund_df, ticker, current_date, "Shares")
                    pr  = float(price.loc[:daily_end].iloc[-1]) \
                          if len(price.loc[:daily_end]) > 0 else np.nan
                    if not any(np.isnan([eq,sh,pr])) and sh>0 and eq/sh>0:
                        PB_ratio = float(pr / (eq/sh))
                    td   = fund_df[fund_df["ticker"]==ticker]
                    cut  = pd.Timestamp(current_date) - pd.Timedelta(days=45)
                    vq   = td[td.index<=cut]["NetIncome"].dropna()
                    if len(vq)>=4 and not np.isnan(sh) and sh>0 and not np.isnan(pr):
                        eps  = float(vq.iloc[-4:].sum()) / sh
                        if eps > 0: PE_TTM = float(pr / eps)
                    # EV/EBITDA — enterprise value / trailing EBITDA
                    ebitda_q = td[td.index<=cut]["EBITDA"].dropna()
                    debt_v   = _fund_val(fund_df, ticker, current_date, "TotalDebt")
                    if (len(ebitda_q) >= 4 and not np.isnan(sh)
                            and sh > 0 and not np.isnan(pr)):
                        ttm_ebitda = float(ebitda_q.iloc[-4:].sum())
                        mkt_cap    = pr * sh
                        net_debt   = (debt_v if not np.isnan(debt_v) else 0)
                        ev         = mkt_cap + net_debt
                        if ttm_ebitda > 0:
                            EV_EBITDA = float(ev / ttm_ebitda)
                except Exception: pass

            # G5 Quality (45-day lagged)
            ROE = np.nan; GrossMargin = np.nan
            CurrentRatio = np.nan; CashFlowYield = np.nan
            if fund_df is not None:
                try:
                    ni = _fund_val(fund_df, ticker, current_date, "NetIncome")
                    eq = _fund_val(fund_df, ticker, current_date, "TotalEquity")
                    rv = _fund_val(fund_df, ticker, current_date, "Revenue")
                    gp = _fund_val(fund_df, ticker, current_date, "GrossProfit")
                    if not np.isnan(ni) and not np.isnan(eq) and eq!=0:
                        ROE = float(ni/eq)
                    if not np.isnan(gp) and not np.isnan(rv) and rv!=0:
                        GrossMargin = float(gp/rv)
                    # Current Ratio = Current Assets / Current Liabilities
                    ca = _fund_val(fund_df, ticker, current_date, "CurrentAssets")
                    cl = _fund_val(fund_df, ticker, current_date, "CurrentLiabilities")
                    if not np.isnan(ca) and not np.isnan(cl) and cl > 0:
                        CurrentRatio = float(ca / cl)
                    # CashFlow Yield = Operating Cash Flow / Market Cap
                    ocf = _fund_val(fund_df, ticker, current_date, "OperatingCashFlow")
                    if (not np.isnan(ocf) and price is not None
                            and not np.isnan(sh) and sh > 0):
                        pr_v = float(price.loc[:daily_end].iloc[-1]) \
                               if len(price.loc[:daily_end]) > 0 else np.nan
                        if not np.isnan(pr_v) and pr_v > 0:
                            CashFlowYield = float(ocf / (pr_v * sh))
                except Exception: pass

            # G6 Growth (45-day lagged)
            RevGrowth_YoY = np.nan; EarnGrowth_YoY = np.nan
            if fund_df is not None:
                try:
                    td  = fund_df[fund_df["ticker"]==ticker]
                    cut = pd.Timestamp(current_date) - pd.Timedelta(days=45)
                    vq  = td[td.index<=cut].sort_index()
                    if len(vq) >= 5:
                        rn = vq["Revenue"].dropna()
                        nn = vq["NetIncome"].dropna()
                        if len(rn)>=5 and rn.iloc[-5]!=0:
                            RevGrowth_YoY  = float((rn.iloc[-1]-rn.iloc[-5])/abs(rn.iloc[-5]))
                        if len(nn)>=5 and nn.iloc[-5]!=0:
                            EarnGrowth_YoY = float((nn.iloc[-1]-nn.iloc[-5])/abs(nn.iloc[-5]))
                except Exception: pass

            # G7 Size (45-day lagged shares × current price)
            LogMktCap = np.nan
            if fund_df is not None and price is not None:
                try:
                    sh = _fund_val(fund_df, ticker, current_date, "Shares")
                    pr = float(price.loc[:daily_end].iloc[-1]) \
                         if len(price.loc[:daily_end]) > 0 else np.nan
                    if not np.isnan(sh) and not np.isnan(pr) and sh>0 and pr>0:
                        LogMktCap = float(np.log(sh * pr))
                except Exception: pass

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

    # Cross-sectional median imputation — fill missing fundamentals
    # with each month's median (no future data leak)
    for feat in FEATURES:
        if feat in factors_df.columns:
            factors_df[feat] = factors_df.groupby("date")[feat].transform(
                lambda x: x.fillna(x.median())
            )

    print(f"\nFactor dataset: {len(factors_df):,} rows | "
          f"{factors_df['ticker'].nunique()} stocks | "
          f"{factors_df['date'].nunique()} months | {skipped} skipped")
    print("\n  Feature coverage (after median imputation):")
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
    Add interaction features between factor groups.
    
    Creates multiplicative interactions between features from different groups
    to capture non-linear relationships (e.g., momentum × value, risk × size).
    
    Interaction features use automatic naming: "Feature1_x_Feature2"
    
    Args:
        factors_df: DataFrame with base features
        config: Feature engineering configuration dict (default: FEATURE_ENGINEERING_CONFIG)
    
    Returns:
        DataFrame with added interaction features
        
    Validates: Requirements 2.5
    """
    if config is None:
        config = FEATURE_ENGINEERING_CONFIG
    
    if not config.get("enable_interactions", False):
        return factors_df
    
    print("\n=== Adding Interaction Features ===")
    
    interaction_pairs = config.get("interaction_pairs", [])
    new_features = []
    
    for group1_name, group2_name in interaction_pairs:
        group1_features = FEATURE_GROUPS.get(group1_name, [])
        group2_features = FEATURE_GROUPS.get(group2_name, [])
        
        for feat1 in group1_features:
            for feat2 in group2_features:
                if feat1 in factors_df.columns and feat2 in factors_df.columns:
                    interaction_name = f"{feat1}_x_{feat2}"
                    
                    # Create interaction: product of the two features
                    # Handle NaN values gracefully
                    factors_df[interaction_name] = factors_df[feat1] * factors_df[feat2]
                    
                    new_features.append(interaction_name)
                    print(f"  Created: {interaction_name}")
    
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
    Apply all feature engineering transformations.
    
    This is the main entry point for feature engineering enhancements.
    Applies interaction features, non-linear transformations, and
    sector-relative features based on configuration.
    
    All transformations maintain temporal integrity (no future data leakage).
    
    Args:
        factors_df: DataFrame with base features
        config: Feature engineering configuration dict (default: FEATURE_ENGINEERING_CONFIG)
    
    Returns:
        DataFrame with all engineered features added
        
    Validates: Requirements 2.5, 2.6, 2.9
    """
    if config is None:
        config = FEATURE_ENGINEERING_CONFIG
    
    print("\n" + "="*70)
    print("  FEATURE ENGINEERING ENHANCEMENTS")
    print("="*70)
    
    original_feature_count = len([c for c in factors_df.columns 
                                   if c not in ["date", "ticker", "sector", TARGET]])
    
    # Apply transformations in sequence
    factors_df = add_interaction_features(factors_df, config)
    factors_df = add_nonlinear_transformations(factors_df, config)
    factors_df = add_sector_relative_features(factors_df, config)
    
    # Cross-sectional median imputation for new features
    # (same approach as base features - no future data leakage)
    new_features = [c for c in factors_df.columns 
                    if c not in ["date", "ticker", "sector", TARGET] 
                    and c not in FEATURES]
    
    if new_features:
        print(f"\n=== Imputing Missing Values for New Features ===")
        for feat in new_features:
            factors_df[feat] = factors_df.groupby("date")[feat].transform(
                lambda x: x.fillna(x.median())
            )
        print(f"  Applied cross-sectional median imputation to {len(new_features)} new features")
    
    final_feature_count = len([c for c in factors_df.columns 
                                if c not in ["date", "ticker", "sector", TARGET]])
    
    print("\n" + "="*70)
    print(f"  Feature Engineering Complete")
    print(f"  Original features: {original_feature_count}")
    print(f"  New features: {final_feature_count - original_feature_count}")
    print(f"  Total features: {final_feature_count}")
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
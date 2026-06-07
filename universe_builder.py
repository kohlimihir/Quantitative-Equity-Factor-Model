"""
universe_builder.py — Point-in-Time S&P 500 Universe Reconstitution
====================================================================
Eliminates survivorship bias by reconstructing the S&P 500 membership
as it existed at each historical date, using Wikipedia's public record
of all index additions and removals.

APPROACH:
  1. Scrape current S&P 500 members from Wikipedia (with sectors)
  2. Scrape the historical changes table (additions/removals with dates)
  3. Walk backwards from the current membership to reconstruct each month
  4. Cache results to data/sp500_universe.json for fast re-use

USAGE:
  from universe_builder import UniverseBuilder
  ub = UniverseBuilder()
  ub.build()
  tickers_jan_2020 = ub.get_universe_for_date("2020-01-15")
  all_tickers      = ub.get_all_historical_tickers()
  sector_map       = ub.get_sector_map_for_date("2020-01-15")

CACHING:
  Scraped data cached to data/sp500_universe.json (< 24h).
  Delete the cache file to force a fresh scrape.
"""

import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime, timedelta
from io import StringIO
import urllib.request
import warnings
warnings.filterwarnings("ignore")

# User-Agent required to avoid Wikipedia 403 blocks
_WIKI_HEADERS = {"User-Agent": "EquityFactorModel/1.0 (educational project; Python/pandas)"}

CACHE_DIR = "data"
UNIVERSE_CACHE = os.path.join(CACHE_DIR, "sp500_universe.json")
CACHE_MAX_HOURS = 24 * 7  # Cache for 1 week (Wikipedia data doesn't change often)


class UniverseBuilder:
    """
    Builds a point-in-time S&P 500 universe by scraping Wikipedia's
    historical changes table and reconstructing monthly membership.
    """

    def __init__(self, cache_path=UNIVERSE_CACHE, verbose=True):
        self.cache_path = cache_path
        self.verbose = verbose
        self.current_members = {}      # {ticker: sector}
        self.changes = []              # [{date, added, removed, added_sector}, ...]
        self.monthly_universe = {}     # {"YYYY-MM": [tickers]}
        self.sector_map_full = {}      # {ticker: sector} for ALL historical tickers
        self._built = False

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ── Cache Management ───────────────────────────────────────────────────

    def _cache_fresh(self):
        if not os.path.exists(self.cache_path):
            return False
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.cache_path))
        return age.total_seconds() < CACHE_MAX_HOURS * 3600

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        data = {
            "current_members": self.current_members,
            "changes": self.changes,
            "monthly_universe": self.monthly_universe,
            "sector_map_full": self.sector_map_full,
            "cached_at": datetime.now().isoformat(),
        }
        with open(self.cache_path, "w") as f:
            json.dump(data, f, indent=2)
        self._log(f"  ✓ Universe cache saved → {self.cache_path}")

    def _load_cache(self):
        with open(self.cache_path, "r") as f:
            data = json.load(f)
        self.current_members = data["current_members"]
        self.changes = data["changes"]
        self.monthly_universe = data["monthly_universe"]
        self.sector_map_full = data["sector_map_full"]
        self._built = True
        self._log(f"  ✓ Loaded universe from cache ({self.cache_path})")
        self._log(f"    Current members: {len(self.current_members)}")
        self._log(f"    Historical changes: {len(self.changes)}")
        self._log(f"    Monthly universes: {len(self.monthly_universe)}")
        self._log(f"    Total unique tickers: {len(self.sector_map_full)}")

    # ── Wikipedia Scraping ─────────────────────────────────────────────────

    def _scrape_current_members(self):
        """Scrape current S&P 500 constituents from Wikipedia."""
        self._log("  Scraping current S&P 500 members from Wikipedia...")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        try:
            req = urllib.request.Request(url, headers=_WIKI_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8")
            tables = pd.read_html(StringIO(html))
        except Exception as e:
            self._log(f"  ⚠ Failed to scrape Wikipedia: {e}")
            self._log("  Using fallback hardcoded universe...")
            return self._fallback_current_members()

        # First table = current constituents
        if len(tables) < 1:
            self._log("  ⚠ No tables found on Wikipedia page")
            return self._fallback_current_members()

        df = tables[0]

        # Map columns — Wikipedia may change column names slightly
        ticker_col = None
        sector_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if "symbol" in col_str or "ticker" in col_str:
                ticker_col = col
            if "gics sector" in col_str or "sector" in col_str:
                sector_col = col

        if ticker_col is None:
            # Try first column
            ticker_col = df.columns[0]
        if sector_col is None:
            # Try column index 2 or 3
            sector_col = df.columns[2] if len(df.columns) > 2 else df.columns[1]

        members = {}
        for _, row in df.iterrows():
            ticker = str(row[ticker_col]).strip().replace(".", "-")
            sector = str(row[sector_col]).strip() if pd.notna(row[sector_col]) else "Unknown"
            # Simplify GICS sectors to match existing model categories
            sector = self._simplify_sector(sector)
            members[ticker] = sector

        self._log(f"  ✓ Found {len(members)} current S&P 500 members")
        return members

    def _scrape_historical_changes(self):
        """Scrape S&P 500 historical changes (additions/removals) from Wikipedia."""
        self._log("  Scraping S&P 500 historical changes from Wikipedia...")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        try:
            req = urllib.request.Request(url, headers=_WIKI_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8")
            tables = pd.read_html(StringIO(html))
        except Exception as e:
            self._log(f"  ⚠ Failed to scrape changes: {e}")
            return []

        # Second table = historical changes
        if len(tables) < 2:
            self._log("  ⚠ Changes table not found")
            return []

        df = tables[1]

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(str(c) for c in col).strip() for col in df.columns]

        # Identify columns by content
        date_col = None
        added_col = None
        removed_col = None

        for col in df.columns:
            col_lower = str(col).lower()
            if "date" in col_lower:
                date_col = col
            elif "added" in col_lower and "ticker" in col_lower:
                added_col = col
            elif "added" in col_lower and added_col is None:
                added_col = col
            elif "removed" in col_lower and "ticker" in col_lower:
                removed_col = col
            elif "removed" in col_lower and removed_col is None:
                removed_col = col

        if date_col is None:
            date_col = df.columns[0]
        if added_col is None:
            added_col = df.columns[1]
        if removed_col is None:
            # Look for the column after added
            added_idx = list(df.columns).index(added_col)
            # Usually removed ticker is 2-3 columns after added
            for i in range(added_idx + 1, min(added_idx + 4, len(df.columns))):
                col_lower = str(df.columns[i]).lower()
                if "removed" in col_lower or "ticker" in col_lower:
                    removed_col = df.columns[i]
                    break
            if removed_col is None and added_idx + 2 < len(df.columns):
                removed_col = df.columns[added_idx + 2]

        changes = []
        for _, row in df.iterrows():
            try:
                date_str = str(row[date_col]).strip()
                if pd.isna(date_str) or date_str == "nan":
                    continue

                # Parse date — Wikipedia uses various formats
                change_date = self._parse_date(date_str)
                if change_date is None:
                    continue

                added = str(row[added_col]).strip().replace(".", "-") if pd.notna(row[added_col]) else ""
                removed = str(row[removed_col]).strip().replace(".", "-") if removed_col and pd.notna(row[removed_col]) else ""

                # Skip if both empty or nan
                if (not added or added == "nan") and (not removed or removed == "nan"):
                    continue

                changes.append({
                    "date": change_date.strftime("%Y-%m-%d"),
                    "added": added if added != "nan" else "",
                    "removed": removed if removed != "nan" else "",
                })
            except Exception:
                continue

        # Sort by date descending (most recent first)
        changes.sort(key=lambda x: x["date"], reverse=True)
        self._log(f"  ✓ Found {len(changes)} historical changes")

        # Print date range
        if changes:
            self._log(f"    Date range: {changes[-1]['date']} → {changes[0]['date']}")

        return changes

    def _parse_date(self, date_str):
        """Parse various date formats from Wikipedia."""
        formats = [
            "%B %d, %Y",     # "June 20, 2024"
            "%b %d, %Y",     # "Jun 20, 2024"
            "%Y-%m-%d",      # "2024-06-20"
            "%m/%d/%Y",      # "06/20/2024"
            "%d %B %Y",      # "20 June 2024"
            "%d %b %Y",      # "20 Jun 2024"
        ]
        # Clean up
        date_str = re.sub(r"\s+", " ", date_str.strip())
        # Remove footnote references like [1], [2]
        date_str = re.sub(r"\[.*?\]", "", date_str).strip()

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    def _simplify_sector(self, gics_sector):
        """
        Map GICS sectors to proper 11 GICS sector names.
        Uses the official GICS taxonomy for institutional-grade analysis.
        """
        mapping = {
            "Information Technology": "Information Technology",
            "Communication Services": "Communication Services",
            "Financials": "Financials",
            "Health Care": "Health Care",
            "Consumer Discretionary": "Consumer Discretionary",
            "Consumer Staples": "Consumer Staples",
            "Energy": "Energy",
            "Industrials": "Industrials",
            "Materials": "Materials",
            "Utilities": "Utilities",
            "Real Estate": "Real Estate",
        }
        return mapping.get(gics_sector, "Industrials")  # Default to Industrials

    def _fallback_current_members(self):
        """Fallback to hardcoded list if Wikipedia scrape fails."""
        from data_loader import SECTOR_MAP
        self._log(f"  Using fallback: {len(SECTOR_MAP)} stocks from hardcoded SECTOR_MAP")
        return dict(SECTOR_MAP)

    # ── Universe Reconstruction ────────────────────────────────────────────

    def _build_monthly_universes(self, start_year=2018):
        """
        Reconstruct point-in-time S&P 500 membership for each month.

        Algorithm:
          - Start with current members
          - Walk backwards through changes:
            - If a ticker was "added" on date D, remove it from months before D
            - If a ticker was "removed" on date D, add it back for months before D
          - Build a monthly universe for each month from start_year to now
        """
        self._log("  Building monthly universe maps...")

        # Start with current membership
        current_set = set(self.current_members.keys())

        # Parse changes into a structured format
        parsed_changes = []
        for ch in self.changes:
            try:
                dt = datetime.strptime(ch["date"], "%Y-%m-%d")
                parsed_changes.append({
                    "date": dt,
                    "added": ch["added"],
                    "removed": ch["removed"],
                })
            except Exception:
                continue

        # Sort changes by date (oldest first for forward reconstruction)
        parsed_changes.sort(key=lambda x: x["date"])

        # Generate all months from start_year to now
        start_date = datetime(start_year, 1, 1)
        end_date = datetime.now()
        months = []
        current = start_date
        while current <= end_date:
            months.append(current.strftime("%Y-%m"))
            # Move to next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)

        # For each month, determine membership by walking changes backwards
        # Start with current members and undo changes that happened after that month
        self.monthly_universe = {}
        self.sector_map_full = dict(self.current_members)  # Start with current sectors

        for month_str in reversed(months):
            month_start = datetime.strptime(month_str + "-01", "%Y-%m-%d")
            month_end = month_start + timedelta(days=32)
            month_end = month_end.replace(day=1)  # First of next month

            # Start from current set and remove/add based on changes after this month
            universe_at_month = set(current_set)

            for ch in parsed_changes:
                if ch["date"] > month_end:
                    # This change happened after our target month
                    # Undo it: if ticker was added after this month, remove it
                    if ch["added"] and ch["added"] in universe_at_month:
                        universe_at_month.discard(ch["added"])
                    # If ticker was removed after this month, add it back
                    if ch["removed"]:
                        universe_at_month.add(ch["removed"])
                        # Track sector for removed tickers
                        if ch["removed"] not in self.sector_map_full:
                            self.sector_map_full[ch["removed"]] = "Unknown"

            self.monthly_universe[month_str] = sorted(list(universe_at_month))

        # Try to fill in sectors for removed tickers via yfinance
        self._fill_missing_sectors()

        n_months = len(self.monthly_universe)
        # Compute all_tickers inline to avoid recursion (self._built is still False)
        all_tickers_set = set()
        for tickers in self.monthly_universe.values():
            all_tickers_set.update(tickers)
        sizes = [len(v) for v in self.monthly_universe.values()]
        self._log(f"  ✓ Built {n_months} monthly universes")
        self._log(f"    Total unique tickers: {len(all_tickers_set)}")
        self._log(f"    Universe size range: {min(sizes)} - {max(sizes)} stocks/month")

        # Print sample months
        sample_months = sorted(self.monthly_universe.keys())
        for m in [sample_months[0], sample_months[len(sample_months)//2], sample_months[-1]]:
            self._log(f"    {m}: {len(self.monthly_universe[m])} stocks")

    def _fill_missing_sectors(self):
        """Fill in sectors for historical tickers that lack sector info."""
        unknown = [t for t, s in self.sector_map_full.items() if s == "Unknown"]
        if not unknown:
            return

        self._log(f"  Looking up sectors for {len(unknown)} historical tickers...")

        try:
            import yfinance as yf
            batch_size = 50
            found = 0
            for i in range(0, len(unknown), batch_size):
                batch = unknown[i:i+batch_size]
                for ticker in batch:
                    try:
                        info = yf.Ticker(ticker).info
                        sector = info.get("sector", None)
                        if sector:
                            self.sector_map_full[ticker] = self._simplify_sector(sector)
                            found += 1
                    except Exception:
                        pass
            self._log(f"  ✓ Found sectors for {found}/{len(unknown)} tickers via yfinance")
        except Exception:
            self._log("  ⚠ yfinance lookup failed, using 'Unknown' for missing sectors")

        # For remaining unknowns, assign a reasonable default based on ticker
        still_unknown = [t for t, s in self.sector_map_full.items() if s == "Unknown"]
        if still_unknown:
            self._log(f"  Assigning default sectors to {len(still_unknown)} remaining tickers")
            for ticker in still_unknown:
                self.sector_map_full[ticker] = "Energy"  # Default catch-all

    # ── Public API ─────────────────────────────────────────────────────────

    def build(self, start_year=2018):
        """
        Build the point-in-time universe. Uses cache if available.

        Args:
            start_year: Start year for historical reconstruction (default: 2018)
        """
        self._log("\n=== Building Point-in-Time S&P 500 Universe ===")

        if self._cache_fresh():
            self._load_cache()
            return

        self.current_members = self._scrape_current_members()
        self.changes = self._scrape_historical_changes()
        self._build_monthly_universes(start_year=start_year)
        self._save_cache()
        self._built = True

    def get_universe_for_date(self, date):
        """
        Get the list of S&P 500 tickers as of a specific date.

        Args:
            date: str or datetime — the target date

        Returns:
            List of ticker strings that were in the S&P 500 on that date
        """
        if not self._built:
            self.build()

        if isinstance(date, str):
            date = pd.Timestamp(date)
        elif isinstance(date, datetime):
            date = pd.Timestamp(date)

        month_key = date.strftime("%Y-%m")

        if month_key in self.monthly_universe:
            return self.monthly_universe[month_key]

        # If exact month not found, find nearest
        available = sorted(self.monthly_universe.keys())
        if month_key < available[0]:
            return self.monthly_universe[available[0]]
        if month_key > available[-1]:
            return self.monthly_universe[available[-1]]

        # Find closest month
        for m in reversed(available):
            if m <= month_key:
                return self.monthly_universe[m]

        return self.monthly_universe[available[-1]]

    def get_all_historical_tickers(self):
        """
        Get the union of all tickers that were ever in the S&P 500
        during the backtest window. Use this to download price data.

        Returns:
            List of all unique ticker strings
        """
        if not self._built:
            self.build()

        all_tickers = set()
        for tickers in self.monthly_universe.values():
            all_tickers.update(tickers)
        return sorted(list(all_tickers))

    def get_sector_map_for_date(self, date):
        """
        Get the ticker→sector mapping for a specific date.
        Only includes tickers that were in the index on that date.

        Args:
            date: str or datetime

        Returns:
            Dict {ticker: sector}
        """
        if not self._built:
            self.build()

        tickers = self.get_universe_for_date(date)
        return {t: self.sector_map_full.get(t, "Unknown") for t in tickers}

    def get_full_sector_map(self):
        """
        Get sector mapping for ALL historical tickers (not filtered by date).
        Useful for compute_factors which processes all tickers at once.

        Returns:
            Dict {ticker: sector} for every ticker ever in the index
        """
        if not self._built:
            self.build()
        return dict(self.sector_map_full)

    def get_universe_summary(self):
        """Print a summary of the historical universe."""
        if not self._built:
            self.build()

        print("\n" + "="*70)
        print("  POINT-IN-TIME S&P 500 UNIVERSE SUMMARY")
        print("="*70)
        print(f"  Current S&P 500 members: {len(self.current_members)}")
        print(f"  Historical changes tracked: {len(self.changes)}")
        print(f"  Total unique tickers (ever in index): {len(self.get_all_historical_tickers())}")
        print(f"  Monthly universes built: {len(self.monthly_universe)}")

        # Sample sizes
        months = sorted(self.monthly_universe.keys())
        print(f"\n  Monthly universe sizes:")
        for m in months[::12]:  # Every 12 months
            n = len(self.monthly_universe[m])
            print(f"    {m}: {n} stocks")
        if months[-1] not in months[::12]:
            print(f"    {months[-1]}: {len(self.monthly_universe[months[-1]])} stocks")

        # Sector distribution for latest month
        latest = months[-1]
        sector_counts = {}
        for t in self.monthly_universe[latest]:
            s = self.sector_map_full.get(t, "Unknown")
            sector_counts[s] = sector_counts.get(s, 0) + 1
        print(f"\n  Sector distribution ({latest}):")
        for s, c in sorted(sector_counts.items(), key=lambda x: -x[1]):
            print(f"    {s:<14}: {c} stocks")

        print("="*70)


# ── Standalone execution ───────────────────────────────────────────────────
if __name__ == "__main__":
    ub = UniverseBuilder()
    ub.build(start_year=2018)
    ub.get_universe_summary()

    # Test specific dates
    print("\n  Sample lookups:")
    for date in ["2019-01-01", "2020-06-01", "2022-03-01", "2024-01-01"]:
        tickers = ub.get_universe_for_date(date)
        print(f"    {date}: {len(tickers)} stocks")

    # Check known delistings
    print("\n  Survivorship bias checks:")
    checks = {
        "SIVB": ("2023-03-15", "Silicon Valley Bank — should be present before Mar 2023"),
        "FRC": ("2023-05-01", "First Republic — should be present before May 2023"),
    }
    for ticker, (date, desc) in checks.items():
        tickers = ub.get_universe_for_date(date)
        status = "✓ PRESENT" if ticker in tickers else "✗ MISSING"
        print(f"    {ticker} ({date}): {status} — {desc}")

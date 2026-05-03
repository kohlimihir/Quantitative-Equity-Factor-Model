"""
oot_validation.py  —  Stage 4: Out-of-Time Validation
======================================================
Held-out period: Jan 2025 onwards. Never seen during any development step.
Uses same EWM smoothing + rebalancing threshold + holding-period bonus
as sector_neutralisation.py.
Portfolio: top-3 per sector = 15 stocks/month.

LEAKAGE AUDIT:
  dev_df: strict < OOT_START — boundary month not in development
  oot_df: strict >= OOT_START — no overlap with dev
  model.fit(X_dev, y_dev): OOT features/labels completely hidden
  Features: raw values — no cross-sectional transformation
  EWM: uses only prev_ranks carried from dev period, never future OOT months
  Holding bonus: tenure tracked from past months only, never future
  Threshold: uses current + previous scores only
  Missing data: cross-sectional median imputation (no future leak)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

from data_loader import FEATURES, TARGET
from sector_neutralisation import (
    add_sector_features, EWM_ALPHA, REBAL_THRESHOLD, TOP_PER_SECTOR,
    HOLD_BONUS_PER_MONTH, HOLD_BONUS_CAP,
)

OOT_START = "2025-01-01"
RF        = 0.045 / 12

LGB_PARAMS = {
    "objective":"regression","metric":"rmse","n_estimators":300,
    "learning_rate":0.03,"num_leaves":31,"min_child_samples":80,
    "subsample":0.7,"colsample_bytree":0.7,"reg_alpha":0.2,
    "reg_lambda":0.2,"random_state":42,"verbose":-1,"n_jobs":-1,
}


def run_oot_validation(factors_df_sector=None, z_features=None,
                        sector_results_full=None, sector_port_full=None,
                        save_report=True):
    import os
    os.makedirs("reports",exist_ok=True); os.makedirs("outputs",exist_ok=True)

    if factors_df_sector is None:
        raw = pd.read_csv("data/factor_features.csv", parse_dates=["date"])
        factors_df_sector, z_features = add_sector_features(raw)

    # Strict split — OOT data completely hidden during training
    dev_df = factors_df_sector[factors_df_sector["date"] <  OOT_START].copy()
    oot_df = factors_df_sector[factors_df_sector["date"] >= OOT_START].copy()
    n_dev  = dev_df["date"].nunique()
    n_oot  = oot_df["date"].nunique()

    print(f"  Dev : {dev_df['date'].min().date()} → "
          f"{dev_df['date'].max().date()} ({n_dev} months, "
          f"{dev_df['ticker'].nunique()} stocks)")
    print(f"  OOT : {oot_df['date'].min().date()} → "
          f"{oot_df['date'].max().date()} ({n_oot} months)")
    print(f"  OOT%: {n_oot/(n_dev+n_oot)*100:.1f}% of total")

    if len(oot_df) == 0:
        print("  No OOT data available.")
        return {}

    # Train frozen model on full dev set — never sees OOT
    print("  Training final model on full dev period...")
    # Cross-sectional median imputation (no future leak)
    X_dev_df = dev_df[z_features].copy()
    for col in z_features:
        med = X_dev_df[col].median()
        X_dev_df[col] = X_dev_df[col].fillna(med)
    X_dev_df = X_dev_df.fillna(0)
    X_dev = X_dev_df.values
    y_dev = dev_df[TARGET].values
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_dev, y_dev)

    # Predict OOT month by month with EWM smoothing
    oot_results   = []
    prev_ranks    = {}
    prev_held     = {}
    hold_tenure   = {}    # {ticker: consecutive months held}
    prev_held_all = set()  # all tickers in portfolio last month

    for date, g in oot_df.groupby("date"):
        # Median imputation per OOT month (no future leak)
        X_test_df = g[z_features].copy()
        for col in z_features:
            med = X_test_df[col].median()
            X_test_df[col] = X_test_df[col].fillna(med)
        X_test_df = X_test_df.fillna(0)
        X_test = X_test_df.values
        scores = model.predict(X_test)
        temp   = g.copy()
        temp["raw_score"] = scores
        temp["raw_rank"]  = temp.groupby("sector")["raw_score"].rank(pct=True)

        smoothed = []
        for _, row in temp.iterrows():
            curr = row["raw_rank"]
            prev = prev_ranks.get(row["ticker"], curr)
            s    = EWM_ALPHA * curr + (1 - EWM_ALPHA) * prev
            # Holding-period bonus (same as sector_neutralisation)
            ticker = row["ticker"]
            if ticker in prev_held_all:
                tenure = min(hold_tenure.get(ticker, 0) + 1, HOLD_BONUS_CAP)
                s += tenure * HOLD_BONUS_PER_MONTH
                hold_tenure[ticker] = tenure
            else:
                hold_tenure[ticker] = 0
            smoothed.append(s)
            prev_ranks[row["ticker"]] = s
        temp["smoothed_rank"] = smoothed

        # Track selections for next month's holding bonus
        month_selected = set()
        for sector_sel, sg_sel in temp.groupby("sector"):
            top_sel = sg_sel.nlargest(TOP_PER_SECTOR, "smoothed_rank")["ticker"].values
            month_selected.update(top_sel)
        prev_held_all = month_selected

        for _, row in temp.iterrows():
            oot_results.append({
                "date"     : date,
                "ticker"   : row["ticker"],
                "sector"   : row["sector"],
                "actual"   : row[TARGET],
                "predicted": row["smoothed_rank"],
            })

    oot_df_res = pd.DataFrame(oot_results)

    # IC metrics
    oot_ic      = (
        oot_df_res.groupby("date")
        .apply(lambda g: g["actual"].corr(g["predicted"], method="spearman"))
    )
    oot_mean_ic = oot_ic.mean()
    oot_ic_ir   = oot_mean_ic / oot_ic.std() if oot_ic.std() > 0 else 0
    oot_pos     = (oot_ic > 0).sum()

    # OOT portfolio with threshold rebalancing
    oot_port_rows = []
    for date, g in oot_df_res.groupby("date"):
        month_ret = []
        for sector, sg in g.groupby("sector"):
            sg_s  = sg.sort_values("predicted", ascending=False)
            naive = set(sg_s.head(TOP_PER_SECTOR)["ticker"].values)
            held  = prev_held.get(sector, naive)
            final = set()
            for h in held:
                h_row = sg_s[sg_s["ticker"] == h]
                if h_row.empty: continue
                h_rank = h_row["predicted"].values[0]
                chall  = sg_s[~sg_s["ticker"].isin(held)]
                if chall.empty or \
                   chall["predicted"].max() - h_rank <= REBAL_THRESHOLD:
                    final.add(h)
            remaining = TOP_PER_SECTOR - len(final)
            if remaining > 0:
                others = sg_s[~sg_s["ticker"].isin(final)]
                final.update(others.head(remaining)["ticker"].values)
            if len(final) < TOP_PER_SECTOR:
                final = naive
            sel = g[g["ticker"].isin(final)]
            month_ret.extend(sel["actual"].values)
            prev_held[sector] = final
        oot_port_rows.append({
            "date"            : date,
            "portfolio_return": np.mean(month_ret) if month_ret else 0,
        })

    oot_port   = pd.DataFrame(oot_port_rows).set_index("date")["portfolio_return"]
    oot_sharpe = ((oot_port-RF).mean()*12) / (oot_port.std()*np.sqrt(12))
    oot_cum    = (1+oot_port).prod() - 1
    oot_mdd    = ((1+oot_port).cumprod()/(1+oot_port).cumprod().cummax()-1).min()
    oot_wr     = (oot_port > 0).mean()

    # In-sample comparison
    is_ic_s = None
    if sector_results_full is not None:
        is_df   = sector_results_full[sector_results_full["date"] < OOT_START]
        is_ic_s = is_df.groupby("date").apply(
            lambda g: g["actual"].corr(g["predicted"], method="spearman"))
    else:
        try:
            tmp     = pd.read_csv("data/sector_predictions.csv", parse_dates=["date"])
            is_df   = tmp[tmp["date"] < OOT_START]
            is_ic_s = is_df.groupby("date").apply(
                lambda g: g["actual"].corr(g["predicted"], method="spearman"))
        except FileNotFoundError:
            pass

    print("\n" + "="*64)
    print("   OUT-OF-TIME VALIDATION RESULTS")
    print("="*64)

    def vf(d, t):
        return "SAME OR BETTER" if d>=0 else (
            "small drop (normal)" if abs(d)<=t else "NOTABLE DROP")

    if is_ic_s is not None and len(is_ic_s) > 0:
        is_mean   = is_ic_s.mean()
        is_icir   = is_mean / is_ic_s.std() if is_ic_s.std() > 0 else 0
        is_sharpe = float("nan")
        if sector_port_full is not None:
            sp  = (sector_port_full if isinstance(sector_port_full, pd.Series)
                   else sector_port_full["portfolio_return"])
            isp = sp[sp.index < OOT_START]
            if len(isp) > 0:
                is_sharpe = ((isp-RF).mean()*12) / (isp.std()*np.sqrt(12))

        d_ic = oot_mean_ic - is_mean
        d_ir = oot_ic_ir   - is_icir
        d_sh = oot_sharpe  - is_sharpe

        print(f"\n  {'Metric':<24}{'In-Sample':>12}{'OOT':>12}"
              f"{'Diff':>10}  Verdict")
        print("  " + "-"*70)
        print(f"  {'Mean IC':<24}{is_mean:>12.5f}{oot_mean_ic:>12.5f}"
              f"{d_ic:>+9.5f}  {vf(d_ic,0.02)}")
        print(f"  {'IC-IR':<24}{is_icir:>12.5f}{oot_ic_ir:>12.5f}"
              f"{d_ir:>+9.5f}  {vf(d_ir,0.10)}")
        print(f"  {'Sharpe':<24}{is_sharpe:>12.3f}{oot_sharpe:>12.3f}"
              f"{d_sh:>+9.3f}  {vf(d_sh,0.20)}")
        print(f"  {'OOT Max DD':<24}{'—':>12}{oot_mdd:>11.2%}")
        print(f"  {'OOT Win Rate':<24}{'—':>12}{oot_wr:>11.2%}")
        print(f"  {'Positive IC':<24}{'—':>12}{oot_pos}/{len(oot_ic)}")

        drops   = sum([d_ic<-0.02, d_ir<-0.10, d_sh<-0.20])
        verdict = ("STRONG — generalises well, no overfitting." if drops==0
                   else "ACCEPTABLE — minor drop, normal in finance." if drops==1
                   else "CAUTION — review features/complexity.")
        print(f"\n  Verdict: {verdict}")
    else:
        print(f"  OOT Mean IC: {oot_mean_ic:.5f} | IC-IR: {oot_ic_ir:.5f}")
        print(f"  OOT Sharpe : {oot_sharpe:.3f} | Return: {oot_cum:.2%}")

    print("="*64)
    print("\n  OOT monthly IC:")
    for date, ic in oot_ic.items():
        f = "GOOD" if ic>0.05 else ("ok" if ic>0 else "WEAK")
        print(f"    {date.strftime('%Y-%m')}: IC={ic:+.4f}  [{f}]")

    if save_report:
        oot_df_res.to_csv("reports/oot_predictions.csv", index=False)
        oot_port.to_frame("portfolio_return").to_csv("reports/oot_portfolio.csv")
        pd.DataFrame({
            "OOT_Mean_IC":[oot_mean_ic],"OOT_IC_IR":[oot_ic_ir],
            "OOT_Sharpe" :[oot_sharpe], "OOT_Cum"  :[oot_cum],
            "OOT_MDD"    :[oot_mdd],    "OOT_WinRate":[oot_wr],
            "OOT_Months" :[n_oot],
        }).to_csv("reports/oot_validation_report.csv", index=False)
        _plot_oot(oot_port, oot_ic, sector_port_full, sector_results_full)
        print("  Saved: reports/oot_*.csv, outputs/oot_validation.png")

    return {"OOT_Mean_IC":oot_mean_ic,"OOT_IC_IR":oot_ic_ir,
            "OOT_Sharpe":oot_sharpe,"OOT_MDD":oot_mdd,
            "OOT_Months":n_oot}


def _plot_oot(oot_port, oot_ic, sector_port_full=None,
              sector_results_full=None,
              save_path="outputs/oot_validation.png"):
    NAVY="#1f4e79"; TEAL="#0f6e56"; ORANGE="#c55a11"; GREY="#a6a6a6"
    oot_start = pd.Timestamp(OOT_START)
    fig, axes = plt.subplots(2,1,figsize=(13,8),
                              gridspec_kw={"height_ratios":[3,2]})
    fig.patch.set_facecolor("white")

    ax = axes[0]; ax.set_facecolor("#f8f9fa")
    if sector_port_full is not None:
        sp  = (sector_port_full if isinstance(sector_port_full, pd.Series)
               else sector_port_full["portfolio_return"])
        is_d = sp[sp.index < OOT_START]
        if len(is_d) > 0:
            is_cum = (1+is_d).cumprod()-1
            oot_ch = (1+oot_port).cumprod()*(1+is_cum.iloc[-1])-1
            ax.plot(is_cum.index, is_cum*100, color=NAVY, lw=2.2,
                    label="In-sample")
            ax.plot(oot_ch.index, oot_ch*100, color=TEAL, lw=2.5,
                    label="OOT (unseen)")
    else:
        ax.plot((1+oot_port).cumprod().index,
                ((1+oot_port).cumprod()-1)*100, color=TEAL, lw=2.5)
    ax.axvspan(oot_start, oot_port.index[-1], alpha=0.07, color=TEAL,
               label="OOT period")
    ax.axvline(oot_start, color=ORANGE, lw=1.5, linestyle="--",
               label="OOT start")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("OOT Validation — Cumulative Return (15-stock portfolio, 19 features)",
                 fontweight="bold", fontsize=13)
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left")
    ax.axhline(0, color=GREY, lw=0.8, linestyle=":")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax2 = axes[1]; ax2.set_facecolor("#f8f9fa")
    if sector_results_full is not None:
        full_ic = sector_results_full.groupby("date").apply(
            lambda g: g["actual"].corr(g["predicted"], method="spearman"))
        is_ic_d = full_ic[full_ic.index < OOT_START]
        ax2.bar(is_ic_d.index, is_ic_d.values, color=GREY,
                alpha=0.5, width=20, label="In-sample IC")
    ax2.bar(oot_ic.index, oot_ic.values, color=TEAL, alpha=0.85,
            width=20, label="OOT IC")
    ax2.axvspan(oot_start, oot_ic.index[-1], alpha=0.06, color=TEAL)
    ax2.axvline(oot_start, color=ORANGE, lw=1.5, linestyle="--")
    ax2.axhline(0,    color="black", lw=0.8, linestyle=":")
    ax2.axhline(0.05, color=NAVY,   lw=1.2, linestyle="--",
                label="IC=0.05 threshold")
    ax2.set_title("Monthly IC — In-Sample vs OOT", fontweight="bold", fontsize=11)
    ax2.set_ylabel("IC (Spearman)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("OOT validation standalone...")
    run_oot_validation()
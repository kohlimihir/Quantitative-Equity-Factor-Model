"""
transaction_costs.py  —  Stage 5: Transaction Cost Modelling
============================================================
With EWM smoothing + rebalancing threshold, turnover drops from ~72%
to ~30-40%/month, significantly improving net-of-cost performance.

Cost assumptions (round-trip per trade):
  Optimistic  (5 bps): institutional algo, dark pools
  Realistic  (10 bps): retail/semi-institutional broker
  Pessimistic(20 bps): wide spreads, high market impact
  1 bps = 0.01% = 0.0001
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

from sector_neutralisation import TOP_PER_SECTOR, REBAL_THRESHOLD

RF = 0.045 / 12

COST_SCENARIOS = {
    "Optimistic (5 bps)" : 0.0005,
    "Realistic (10 bps)" : 0.0010,
    "Pessimistic (20 bps)": 0.0020,
}


def compute_turnover(sector_predictions_path, top_per_sector=TOP_PER_SECTOR):
    """
    Replicates the threshold-based portfolio construction to measure
    actual monthly turnover after EWM smoothing + threshold rebalancing.
    """
    df = pd.read_csv(sector_predictions_path, parse_dates=["date"])
    rank_col  = "predicted"
    prev_held = {}
    monthly_holdings = {}

    for date, g in df.groupby("date"):
        month_sel = set()
        for sector, sg in g.groupby("sector"):
            sg_s  = sg.sort_values(rank_col, ascending=False)
            naive = set(sg_s.head(top_per_sector)["ticker"].values)
            held  = prev_held.get(sector, naive)
            final = set()
            for h in held:
                h_row = sg_s[sg_s["ticker"] == h]
                if h_row.empty: continue
                h_rank = h_row[rank_col].values[0]
                chall  = sg_s[~sg_s["ticker"].isin(held)]
                if chall.empty or \
                   chall[rank_col].max() - h_rank <= REBAL_THRESHOLD:
                    final.add(h)
            remaining = top_per_sector - len(final)
            if remaining > 0:
                others = sg_s[~sg_s["ticker"].isin(final)]
                final.update(others.head(remaining)["ticker"].values)
            if len(final) < top_per_sector:
                final = naive
            month_sel.update(final)
            prev_held[sector] = final
        monthly_holdings[date] = month_sel

    dates   = sorted(monthly_holdings.keys())
    records = []
    for i in range(1, len(dates)):
        prev = monthly_holdings[dates[i-1]]
        curr = monthly_holdings[dates[i]]
        sold   = prev - curr
        turnov = len(sold) / len(curr) if len(curr) > 0 else 0
        records.append({
            "date"         : dates[i],
            "stocks_sold"  : sorted(sold),
            "stocks_bought": sorted(curr - prev),
            "n_changed"    : len(sold),
            "n_stocks"     : len(curr),
            "turnover"     : turnov,
        })

    t_df = pd.DataFrame(records)
    print("=== TURNOVER ANALYSIS ===")
    print(f"  Avg monthly   : {t_df['turnover'].mean():.1%}")
    print(f"  Max monthly   : {t_df['turnover'].max():.1%}")
    print(f"  Min monthly   : {t_df['turnover'].min():.1%}")
    print(f"  Annualised    : {t_df['turnover'].mean()*12:.1%}")
    print("\n  Highest turnover months:")
    for _, row in t_df.nlargest(3, "turnover").iterrows():
        print(f"    {row['date'].strftime('%Y-%m')}: "
              f"sold {row['stocks_sold'][:4]} → "
              f"bought {row['stocks_bought'][:4]}")
    return t_df


def apply_transaction_costs(portfolio_path, turnover_df, cost_bps):
    port = pd.read_csv(portfolio_path, parse_dates=["date"]).set_index("date")
    col  = "portfolio_return" if "portfolio_return" in port.columns \
           else port.columns[0]
    gross        = port[col]
    turnover_s   = turnover_df.set_index("date")["turnover"].reindex(
                    gross.index).fillna(0)
    monthly_cost = turnover_s * cost_bps
    return gross, gross - monthly_cost, monthly_cost


def _metrics(ret):
    cum  = (1 + ret).prod() - 1
    yrs  = len(ret) / 12
    ann  = (1 + cum)**(1/yrs) - 1 if yrs > 0 else 0
    vol  = ret.std() * np.sqrt(12)
    shr  = ((ret - RF).mean() * 12) / vol if vol > 0 else 0
    down = ret[ret < 0].std() * np.sqrt(12)
    srt  = (ann - 0.045) / down if down > 0 else 0
    cum_c = (1 + ret).cumprod()
    mdd  = ((cum_c - cum_c.cummax()) / cum_c.cummax()).min()
    return dict(cum=cum, ann=ann, vol=vol, sharpe=shr,
                sortino=srt, mdd=mdd, wr=(ret > 0).mean())


def print_cost_analysis(portfolio_path, turnover_df):
    gross, _, _ = apply_transaction_costs(portfolio_path, turnover_df, 0)
    gm = _metrics(gross)

    print("\n" + "="*64)
    print("   TRANSACTION COST ANALYSIS")
    print("="*64)
    print(f"\n  Gross baseline:")
    print(f"    Sharpe  : {gm['sharpe']:.3f}")
    print(f"    Sortino : {gm['sortino']:.3f}")
    print(f"    Return  : {gm['cum']:.2%}")
    print(f"    Max DD  : {gm['mdd']:.2%}")

    print(f"\n  {'Scenario':<24} {'Sharpe':>8} {'Return':>10} "
          f"{'Max DD':>10} {'Drag':>8}")
    print("  " + "-"*62)
    for name, cost in COST_SCENARIOS.items():
        _, net, _ = apply_transaction_costs(portfolio_path, turnover_df, cost)
        nm = _metrics(net)
        print(f"  {name:<24} {nm['sharpe']:>8.3f} "
              f"{nm['cum']:>10.2%} {nm['mdd']:>10.2%} "
              f"{nm['sharpe']-gm['sharpe']:>+7.3f}")

    print("\n  Break-even cost (Sharpe thresholds):")
    for target, label in [(0.0,"break-even"),(0.5,"deployable"),(1.0,"strong")]:
        lo, hi = 0.0, 0.02
        for _ in range(60):
            mid = (lo + hi) / 2
            _, net, _ = apply_transaction_costs(portfolio_path, turnover_df, mid)
            s = _metrics(net)["sharpe"]
            if s > target: lo = mid
            else:          hi = mid
        print(f"    Sharpe {target:.1f} ({label:<11}): "
              f"survives up to {mid*10000:.1f} bps")
    print("="*64)


def plot_cost_scenarios(portfolio_path, turnover_df,
                         save_path="outputs/transaction_cost_analysis.png"):
    gross, _, _ = apply_transaction_costs(portfolio_path, turnover_df, 0)
    gross_cum   = (1 + gross).cumprod() - 1
    fig, axes   = plt.subplots(2, 1, figsize=(12, 8),
                               gridspec_kw={"height_ratios":[3,1]})
    fig.patch.set_facecolor("white")

    ax = axes[0]; ax.set_facecolor("#f8f9fa")
    ax.plot(gross_cum.index, gross_cum*100, color="#1f4e79",
            lw=2.5, label="Gross (no costs)")
    colours = ["#2d8a4e","#e8a838","#c0392b"]
    for (name, cost), colour in zip(COST_SCENARIOS.items(), colours):
        _, net, _ = apply_transaction_costs(portfolio_path, turnover_df, cost)
        net_cum   = (1 + net).cumprod() - 1
        nm_sharpe = _metrics(net)["sharpe"]
        short     = name.split("(")[1].rstrip(")")
        ax.plot(net_cum.index, net_cum*100, color=colour, lw=1.8,
                linestyle="--",
                label=f"Net {short}  (Sharpe {nm_sharpe:.2f})")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Cumulative Return: Gross vs Net of Transaction Costs "
                 "(19-feature, EWM smoothed)", fontweight="bold", fontsize=13)
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left", fontsize=10)
    ax.axhline(0, color="#a6a6a6", lw=0.8, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]; ax2.set_facecolor("#f8f9fa")
    ax2.bar(turnover_df["date"], turnover_df["turnover"]*100,
            color="#1f4e79", alpha=0.6, width=20)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_title("Monthly Portfolio Turnover (EWM + threshold rebalancing)",
                  fontweight="bold", fontsize=11)
    ax2.set_ylabel("Turnover (%)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_sharpe_sensitivity(portfolio_path, turnover_df,
                              save_path="outputs/sharpe_vs_costs.png"):
    cost_range = np.linspace(0, 0.0030, 80)
    sharpes    = [_metrics(apply_transaction_costs(
                    portfolio_path, turnover_df, c)[1])["sharpe"]
                  for c in cost_range]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor("#f8f9fa")
    ax.plot(cost_range*10000, sharpes, color="#1f4e79", lw=2.5)
    ax.fill_between(cost_range*10000, sharpes, 0,
                    where=[s > 0 for s in sharpes],
                    alpha=0.10, color="#1f4e79")
    ax.axhline(0,   color="#c0392b", lw=1.0, linestyle="--",
               label="Break-even (0)")
    ax.axhline(0.5, color="#e8a838", lw=1.0, linestyle="--",
               label="Deployable (0.5)")
    ax.axhline(1.0, color="#2d8a4e", lw=1.5, linestyle="--",
               label="Strong (1.0)")
    for name, cost in COST_SCENARIOS.items():
        _, net, _ = apply_transaction_costs(portfolio_path, turnover_df, cost)
        s = _metrics(net)["sharpe"]
        bps = cost * 10000
        ax.axvline(bps, color="#888", lw=0.8, linestyle=":", alpha=0.6)
        ax.annotate(f"{bps:.0f}bps\n{s:.2f}",
                    xy=(bps, s), xytext=(bps+0.3, s+0.05),
                    fontsize=9, color="#444")
    ax.set_xlabel("Round-trip cost (basis points)")
    ax.set_ylabel("Net Sharpe Ratio")
    ax.set_title("Sharpe Sensitivity to Transaction Costs",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    os.makedirs("reports",exist_ok=True); os.makedirs("outputs",exist_ok=True)
    SECTOR_PRED = "data/sector_predictions.csv"
    PORTFOLIO   = "reports/sector_portfolio.csv"
    turnover_df = compute_turnover(SECTOR_PRED)
    print_cost_analysis(PORTFOLIO, turnover_df)
    plot_cost_scenarios(PORTFOLIO, turnover_df)
    plot_sharpe_sensitivity(PORTFOLIO, turnover_df)
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
    print("Saved: reports/cost_adjusted_returns.csv")
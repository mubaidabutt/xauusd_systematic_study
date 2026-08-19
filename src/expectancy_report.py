#!/usr/bin/env python3
"""
XAUUSD -- Expectancy report for H1 and H4 strategies
=====================================================

Answers one question: what is the win rate, payoff ratio, and expectancy of
each strategy variant, on H1 and on H4?

Why expectancy and not win rate
-------------------------------
Win rate alone cannot tell you whether a strategy makes money. At 40% wins you
need a payoff ratio above 1.5 just to break even; at 1:1 you lose 20R per 100
trades. The number that decides everything is

    expectancy_R = win_rate * payoff - loss_rate

measured in units of risk (R), where 1R is the distance to your stop. It is
directly comparable across strategies, timeframes, and account sizes, which
raw dollar profit is not.

Every variant is reported at BOTH zero cost and real Exness cost
($0.30 spread + $0.10 slippage), because the gap between those two columns is
what decides whether a strategy is tradeable or merely backtestable.

Usage
-----
    python expectancy_report.py
    python expectancy_report.py --timeframe H4
    python expectancy_report.py --min-expectancy 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from engine import Config, run_backtest
from strategies import FAMILIES

SEP = "=" * 92
SUB = "-" * 92


def locate(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for base in (Path("data"), Path(".")):
        for pat in ("*_h1_mt5.parquet", "*h1*.parquet"):
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0]
    sys.exit("H1 parquet not found. Pass --file path/to/XAUUSDm_h1_mt5.parquet")


def to_h4(h1: pd.DataFrame) -> pd.DataFrame:
    """Resample H1 -> H4 on UTC boundaries, matching what MT5 builds."""
    d = (h1.set_index("datetime")
           .resample("4h", origin="start_day")
           .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
           .dropna()
           .reset_index())
    return d


def expectancy_profile(trades: pd.DataFrame, cfg: Config) -> dict:
    """
    Full expectancy breakdown, in R units and in dollars.

    R is reconstructed per trade from the actual risk taken: the stop distance
    times position size. Expressing results in R makes them comparable across
    variants that size differently.
    """
    if trades.empty:
        return {}

    pnl = trades["net_pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    n = len(pnl)

    win_rate = len(wins) / n
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else np.inf

    # 1R = intended risk per trade = equity * risk_per_trade, held constant
    # by the sizing rule, so dollar P&L divides cleanly into R units.
    r_unit = cfg.initial_equity * cfg.risk_per_trade
    exp_dollar = float(pnl.mean())
    exp_r = exp_dollar / r_unit if r_unit else 0.0

    gross_win = float(wins.sum())
    gross_loss = float(abs(losses.sum()))

    return {
        "n_trades": n,
        "win_rate": win_rate * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff": payoff,
        "expectancy_dollar": exp_dollar,
        "expectancy_r": exp_r,
        "profit_factor": (gross_win / gross_loss) if gross_loss else np.inf,
        "net_profit": float(pnl.sum()),
        "breakeven_payoff": (1 - win_rate) / win_rate if win_rate > 0 else np.inf,
        "edge_over_breakeven": payoff - ((1 - win_rate) / win_rate) if win_rate > 0 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--timeframe", choices=["H1", "H4", "both"], default="both")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2024-01-01")
    ap.add_argument("--min-expectancy", type=float, default=0.0,
                    help="only list variants above this expectancy in R")
    ap.add_argument("--outdir", default=Path("./data"), type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    path = locate(args.file)
    print(f"{SEP}\nEXPECTANCY REPORT\n{SEP}")
    print(f"  Data: {path}")
    h1 = pd.read_parquet(path)
    h1["datetime"] = pd.to_datetime(h1["datetime"])
    h1 = h1[["datetime", "open", "high", "low", "close"]].sort_values("datetime")
    h1 = h1[(h1.datetime >= pd.Timestamp(args.start)) &
            (h1.datetime < pd.Timestamp(args.end))].reset_index(drop=True)

    frames = {}
    if args.timeframe in ("H1", "both"):
        frames["H1"] = h1
    if args.timeframe in ("H4", "both"):
        frames["H4"] = to_h4(h1)
    for tf, df in frames.items():
        print(f"  {tf}: {len(df):,} bars   {df.datetime.min().date()} -> {df.datetime.max().date()}")

    free = Config(spread_per_oz=0.0, slippage_per_oz=0.0)
    real = Config(spread_per_oz=0.30, slippage_per_oz=0.10)

    print(f"\n  Expectancy is in R units (1R = risk per trade = "
          f"${free.initial_equity * free.risk_per_trade:.0f}).")
    print(f"  'BE payoff' is the payoff ratio needed to break even at that win rate.")
    print(f"  'edge' = actual payoff minus break-even payoff. Positive = profitable.\n")

    rows = []
    for tf, bars in frames.items():
        for family, (fn, grid) in FAMILIES.items():
            for vi, params in enumerate(grid()):
                try:
                    sig = fn(bars, **params)
                    pf = expectancy_profile(run_backtest(bars, sig, free).to_frame(), free)
                    pr = expectancy_profile(run_backtest(bars, sig, real).to_frame(), real)
                except Exception:
                    continue
                if not pf or pf["n_trades"] < 30:
                    continue
                rows.append({
                    "timeframe": tf, "family": family, "variant": vi,
                    "n_trades": pf["n_trades"],
                    "win_rate": pf["win_rate"],
                    "payoff_free": pf["payoff"],
                    "be_payoff": pf["breakeven_payoff"],
                    "edge_free": pf["edge_over_breakeven"],
                    "exp_r_free": pf["expectancy_r"],
                    "exp_r_real": pr.get("expectancy_r", 0.0),
                    "exp_usd_real": pr.get("expectancy_dollar", 0.0),
                    "pf_free": pf["profit_factor"],
                    "pf_real": pr.get("profit_factor", 0.0),
                })

    if not rows:
        sys.exit("No variants produced enough trades.")

    df = pd.DataFrame(rows).sort_values("exp_r_real", ascending=False)

    print(f"{SUB}\nALL VARIANTS, RANKED BY EXPECTANCY AT REAL COST\n{SUB}")
    print(f"  {'tf':<4}{'family':<22}{'#':>3}{'trades':>7}{'win%':>7}{'payoff':>8}"
          f"{'BE pay':>8}{'edge':>7}{'expR free':>11}{'expR real':>11}{'PF real':>9}")
    print("  " + "-" * 88)
    for _, r in df.iterrows():
        if r.exp_r_free < args.min_expectancy:
            continue
        flag = "  <<<" if r.exp_r_real > 0 else ""
        print(f"  {r.timeframe:<4}{r.family[:21]:<22}{int(r.variant):>3}{int(r.n_trades):>7}"
              f"{r.win_rate:>6.1f}%{r.payoff_free:>8.2f}{r.be_payoff:>8.2f}"
              f"{r.edge_free:>+7.2f}{r.exp_r_free:>+11.4f}{r.exp_r_real:>+11.4f}"
              f"{r.pf_real:>9.3f}{flag}")

    print(f"\n{SUB}\nSUMMARY\n{SUB}")
    pos_free = int((df.exp_r_free > 0).sum())
    pos_real = int((df.exp_r_real > 0).sum())
    print(f"  Variants tested                        : {len(df)}")
    print(f"  Positive expectancy at ZERO cost       : {pos_free}")
    print(f"  Positive expectancy at REAL cost       : {pos_real}")

    if pos_real:
        best = df.iloc[0]
        print(f"\n  Best at real cost: {best.timeframe} {best.family} #{int(best.variant)}")
        print(f"    {int(best.n_trades)} trades, {best.win_rate:.1f}% win rate, "
              f"payoff {best.payoff_free:.2f}")
        print(f"    Expectancy {best.exp_r_real:+.4f}R per trade "
              f"(${best.exp_usd_real:+.2f} on a $10k account)")
        per_yr = best.n_trades / 6.0
        print(f"    ~{per_yr:.0f} trades/year -> {best.exp_r_real*per_yr:+.2f}R/year "
              f"= {best.exp_r_real*per_yr*100/10:.1f}% annual return at 1% risk")
        print(f"\n    NOTE: this is IN-SAMPLE on the window it was selected from.")
        print(f"    Expect 30-50% degradation out of sample.")
    else:
        print(f"\n  NO variant has positive expectancy at real Exness cost.")
        print(f"  The zero-cost column shows what the signal is worth before")
        print(f"  the broker takes its share; the real column is what your")
        print(f"  account would actually do.")

    out = args.outdir / "expectancy_report.csv"
    df.to_csv(out, index=False)
    print(f"\n  Full table -> {out}\n{SEP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
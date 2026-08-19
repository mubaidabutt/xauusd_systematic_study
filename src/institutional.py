#!/usr/bin/env python3
"""
XAUUSD -- Institutional time-series momentum with volatility targeting
=======================================================================

What this tests, and why it is different
----------------------------------------
Every previous study tested RETAIL mechanics: a discrete entry signal, an ATR
stop, a trailing exit. Institutional systematic desks -- CTAs, managed futures
-- generally do not trade that way. The canonical construction is:

    position_t = sign(return over lookback) * (target_vol / realized_vol)

Always in the market. No stops. No entry timing. Long or short depending on
the sign of past returns, sized inversely to recent volatility so that risk
contribution stays constant whether gold is calm at $1,200 or wild at $5,000.

This is the Moskowitz-Ooi-Pedersen time-series momentum construction, and it
is the single most documented systematic strategy in the literature. It is
mechanically different from everything tested so far, so it deserves a test.

Why a separate backtester
-------------------------
The main engine models discrete trades with stops. This strategy has neither.
A vectorised approach is both correct and far faster here:

    pnl_t = position_{t-1} * return_t

The shift is the whole lookahead defence: position must be decided on data
through t-1 and applied to the return realised at t.

Costs are ZERO by decision. Turnover is reported instead, since for an
always-in strategy turnover is what would eventually determine cost.

Usage
-----
    python institutional.py
    python institutional.py --target-vol 0.15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEP = "=" * 88
SUB = "-" * 88

RESEARCH_END = "2024-01-01"      # 2024+ stays sealed

# bars per year, per timeframe -- used to annualise and to convert
# economically meaningful horizons (months) into bar counts
BARS_PER_YEAR = {"H1": 6000, "H4": 1500, "D1": 252, "W1": 52}


def load_frames(path: Path, start: str, end: str) -> dict[str, pd.DataFrame]:
    h1 = pd.read_parquet(path)
    h1["datetime"] = pd.to_datetime(h1["datetime"])
    h1 = h1[["datetime", "open", "high", "low", "close"]].sort_values("datetime")
    h1 = h1[(h1.datetime >= pd.Timestamp(start)) & (h1.datetime < pd.Timestamp(end))]
    h1 = h1.reset_index(drop=True)

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    out = {"H1": h1}
    for label, rule in (("H4", "4h"), ("D1", "1D"), ("W1", "1W")):
        r = (h1.set_index("datetime").resample(rule, origin="start_day")
               .agg(agg).dropna().reset_index())
        out[label] = r
    return out


def tsmom(bars: pd.DataFrame, lookback: int, vol_window: int,
          target_vol: float, bars_per_year: int,
          max_leverage: float = 3.0) -> dict:
    """
    Time-series momentum with volatility targeting.

    Both the momentum signal and the volatility estimate use data through t-1
    only; the position is then applied to the return realised at t. Nothing
    here can see its own outcome.
    """
    px = bars["close"].astype(float)
    ret = px.pct_change()

    # signal: sign of the cumulative return over the lookback, known at t-1
    momentum = px / px.shift(lookback) - 1.0
    signal = np.sign(momentum).shift(1)

    # realised vol, annualised, known at t-1
    realized = ret.rolling(vol_window, min_periods=vol_window).std().shift(1)
    realized_ann = realized * np.sqrt(bars_per_year)

    scale = (target_vol / realized_ann).clip(upper=max_leverage)
    position = (signal * scale).fillna(0.0)

    strat_ret = (position * ret).fillna(0.0)
    n = len(strat_ret)
    if n < 100 or strat_ret.std() == 0:
        return {}

    ann_factor = np.sqrt(bars_per_year)
    sharpe = float(strat_ret.mean() / strat_ret.std() * ann_factor)

    equity = (1 + strat_ret).cumprod()
    dd = float(((equity - equity.cummax()) / equity.cummax()).min() * 100)
    years = n / bars_per_year
    total = float(equity.iloc[-1] - 1)
    cagr = ((1 + total) ** (1 / years) - 1) * 100 if total > -1 and years > 0 else -100.0

    # Sharpe standard error ~ sqrt((1 + S^2/2)/N_years_of_obs)
    se = float(np.sqrt((1 + sharpe**2 / 2) / years)) if years > 0 else np.inf
    turnover = float(position.diff().abs().mean() * bars_per_year)
    long_share = float((position > 0).mean() * 100)

    return {
        "sharpe": round(sharpe, 3),
        "sharpe_ci_lo": round(sharpe - 1.96 * se, 3),
        "sharpe_ci_hi": round(sharpe + 1.96 * se, 3),
        "significant": bool(sharpe - 1.96 * se > 0),
        "cagr_pct": round(cagr, 2),
        "max_dd_pct": round(dd, 1),
        "realized_vol_pct": round(float(strat_ret.std() * ann_factor * 100), 1),
        "pct_time_long": round(long_share, 1),
        "turnover_per_year": round(turnover, 1),
        "years": round(years, 2),
        "_returns": strat_ret,
    }


def buy_hold(bars: pd.DataFrame, bars_per_year: int) -> dict:
    ret = bars["close"].astype(float).pct_change().dropna()
    if len(ret) < 50 or ret.std() == 0:
        return {}
    ann = np.sqrt(bars_per_year)
    sharpe = float(ret.mean() / ret.std() * ann)
    eq = (1 + ret).cumprod()
    return {
        "sharpe": round(sharpe, 3),
        "cagr_pct": round(float((eq.iloc[-1] ** (bars_per_year / len(ret)) - 1) * 100), 2),
        "max_dd_pct": round(float(((eq - eq.cummax()) / eq.cummax()).min() * 100), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--target-vol", type=float, default=0.15,
                    help="annualised volatility target, e.g. 0.15 = 15%%")
    ap.add_argument("--outdir", default=Path("./data"), type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    path = Path(args.file) if args.file else None
    if path is None:
        for base in (Path("data"), Path(".")):
            hits = sorted(base.glob("*_h1_mt5.parquet"))
            if hits:
                path = hits[0]
                break
    if path is None or not path.exists():
        sys.exit("H1 parquet not found. Pass --file path/to/XAUUSDm_h1_mt5.parquet")

    print(f"{SEP}\nINSTITUTIONAL TIME-SERIES MOMENTUM  (vol-targeted, always in market)\n{SEP}")
    print(f"  Source      : {path}")
    frames = load_frames(path, args.start, RESEARCH_END)
    for tf, df in frames.items():
        print(f"  {tf:<3}: {len(df):>7,} bars")
    print(f"  SEALED      : {RESEARCH_END} onward -- not touched")
    print(f"  Target vol  : {args.target_vol*100:.0f}% annualised, max 3x leverage")
    print(f"  Cost        : zero by decision (turnover reported instead)")

    print(f"\n{SUB}\nBUY AND HOLD REFERENCE\n{SUB}")
    for tf, df in frames.items():
        bh = buy_hold(df, BARS_PER_YEAR[tf])
        if bh:
            print(f"  {tf:<3}  Sharpe {bh['sharpe']:+.3f}   CAGR {bh['cagr_pct']:+6.2f}%   "
                  f"maxDD {bh['max_dd_pct']:.1f}%")

    print(f"\n{SUB}\nRESULTS  (lookback in months, converted per timeframe)\n{SUB}")
    print(f"  {'tf':<4}{'lookback':>10}{'sharpe':>9}{'95% CI':>18}{'sig':>5}"
          f"{'CAGR':>8}{'maxDD':>8}{'vol':>7}{'%long':>7}{'turnover':>10}")

    rows = []
    for tf, df in frames.items():
        bpy = BARS_PER_YEAR[tf]
        for months in (1, 3, 6, 12):
            lb = max(int(bpy * months / 12), 5)
            vw = max(int(bpy / 12), 10)          # ~1 month vol window
            if lb >= len(df) // 3:
                continue
            r = tsmom(df, lb, vw, args.target_vol, bpy)
            if not r:
                continue
            print(f"  {tf:<4}{months:>8}mo{r['sharpe']:>+9.3f}"
                  f"  [{r['sharpe_ci_lo']:+.2f},{r['sharpe_ci_hi']:+.2f}]"
                  f"{'  YES' if r['significant'] else '   no':>5}"
                  f"{r['cagr_pct']:>7.1f}%{r['max_dd_pct']:>7.1f}%"
                  f"{r['realized_vol_pct']:>6.1f}%{r['pct_time_long']:>6.0f}%"
                  f"{r['turnover_per_year']:>10.1f}")
            rows.append({"timeframe": tf, "lookback_months": months,
                         **{k: v for k, v in r.items() if not k.startswith("_")}})

    if not rows:
        sys.exit("No configurations produced results.")

    df_all = pd.DataFrame(rows)

    print(f"\n{SUB}\nSUMMARY\n{SUB}")
    n_sig = int(df_all.significant.sum())
    print(f"  Configurations tested          : {len(df_all)}")
    print(f"  Statistically significant      : {n_sig}")
    print(f"  Median Sharpe                  : {df_all.sharpe.median():+.3f}")
    print(f"  Best                           : {df_all.sharpe.max():+.3f}")

    print(f"\n  By timeframe (median Sharpe):")
    for tf, g in df_all.groupby("timeframe", sort=False):
        print(f"    {tf:<4} {g.sharpe.median():+.3f}   "
              f"(best {g.sharpe.max():+.3f}, {int(g.significant.sum())}/{len(g)} significant)")

    print(f"\n  By lookback (median Sharpe across timeframes):")
    for lb, g in df_all.groupby("lookback_months"):
        print(f"    {lb:>2} months  {g.sharpe.median():+.3f}")

    if n_sig:
        best = df_all.loc[df_all.sharpe.idxmax()]
        print(f"\n  BEST: {best.timeframe} {int(best.lookback_months)}-month lookback")
        print(f"    Sharpe {best.sharpe:+.3f} [{best.sharpe_ci_lo:+.2f}, {best.sharpe_ci_hi:+.2f}]")
        print(f"    CAGR {best.cagr_pct:+.1f}%, max drawdown {best.max_dd_pct:.1f}%, "
              f"{best.pct_time_long:.0f}% of time long")
        print(f"    Turnover {best.turnover_per_year:.1f}x/year -- this is what")
        print(f"    determines cost sensitivity for an always-in strategy.")
        print(f"\n    IN-SAMPLE. The sealed 2024-2026 window is the real test.")
    else:
        print(f"\n  NOTHING is statistically significant. Point estimates without")
        print(f"  intervals excluding zero are not evidence of an edge.")

    out = args.outdir / "institutional.csv"
    df_all.to_csv(out, index=False)
    print(f"\n  -> {out}\n{SEP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
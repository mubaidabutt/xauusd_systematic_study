#!/usr/bin/env python3
"""
XAUUSD Research Project -- Step 7: Daily null benchmarks
=========================================================

The H1 thresholds do not transfer. They were derived from runs of ~800 trades
on hourly bars; a daily strategy trades 10-30 times a year. Fewer trades means
a wider null distribution, which means the "best of N by luck" bar sits
somewhere different. It has to be re-derived before any daily strategy exists.

New here: a REGIME BREAKDOWN. The 22-year daily series contains a parabolic
bull, a four-year bear, a low-vol range, and a melt-up. Buy-and-hold Sharpe
over the whole span hides all of that, and a single number would let a
long-biased strategy hide in it too.

Costs are ZERO by decision (see Engine.py). Research runs cost-free; survivors
get one cost check before deployment.

Usage
-----
    python daily_null.py
    python daily_null.py --end 2024-01-01     (respects the sealed window)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from engine import Config, metrics, run_backtest
from indicators import atr

SEP = "=" * 76
SUB = "-" * 76

# Named regimes, chosen from gold's actual history -- NOT fitted to results.
REGIMES = [
    ("2004-2007 early bull",  "2004-06-11", "2008-01-01"),
    ("2008-2011 parabolic",   "2008-01-01", "2011-09-01"),
    ("2011-2015 BEAR",        "2011-09-01", "2016-01-01"),
    ("2016-2018 range",       "2016-01-01", "2018-01-01"),
    ("2018-2023 dev window",  "2018-01-01", "2024-01-01"),
]


def load_daily(explicit: str | None) -> pd.DataFrame:
    if explicit:
        p = Path(explicit)
    else:
        p = Path("data/xauusd_daily.parquet")
        if not p.exists():
            hits = sorted(Path(".").rglob("xauusd_daily.parquet"))
            if not hits:
                sys.exit("xauusd_daily.parquet not found. Run build_daily first.")
            p = hits[0]
    print(f"  Using: {p}")
    df = pd.read_parquet(p)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def buy_and_hold(bars: pd.DataFrame) -> dict:
    px = bars.set_index("datetime")["close"]
    rets = px.pct_change().dropna()
    if len(rets) < 30 or rets.std() == 0:
        return {}
    sharpe = rets.mean() / rets.std() * np.sqrt(252)
    curve = (1 + rets).cumprod()
    dd = (curve - curve.cummax()) / curve.cummax()
    years = (px.index[-1] - px.index[0]).days / 365.25
    total = float(px.iloc[-1] / px.iloc[0] - 1)
    return {
        "sharpe": round(float(sharpe), 3),
        "total_return_pct": round(total * 100, 1),
        "cagr_pct": round(((1 + total) ** (1 / years) - 1) * 100, 2),
        "max_drawdown_pct": round(float(-dd.min() * 100), 1),
        "years": round(years, 2),
    }


def random_signals(bars: pd.DataFrame, a: pd.Series, seed: int, rate: float,
                   hold: int, stop_mult: float, bias: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(bars)
    fire = rng.random(n) < rate
    if bias == "long":
        d = np.where(fire, 1, 0)
    elif bias == "short":
        d = np.where(fire, -1, 0)
    else:
        d = np.where(fire, rng.choice([-1, 1], n), 0)
    stop = (a * stop_mult).to_numpy()
    valid = np.isfinite(stop) & (stop > 0)
    return pd.DataFrame({
        "direction": np.where(valid, d, 0).astype(int),
        "stop_distance": np.where(valid, stop, 0.0),
        "trail_distance": np.zeros(n),
        "max_hold": np.full(n, hold, dtype=int),
    })


def run_null(bars, a, cfg, trials, rate, hold, stop_mult, bias):
    out = []
    for seed in range(trials):
        sig = random_signals(bars, a, seed, rate, hold, stop_mult, bias)
        m = metrics(run_backtest(bars, sig, cfg))
        if m["n_trades"] > 10:
            out.append(m["sharpe"])
    return np.array(out)


def selection_threshold(null: np.ndarray, variants: int, draws: int = 20_000) -> dict:
    if null.size < 20:
        return {}
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    rng = np.random.default_rng(0)
    maxima = rng.normal(mu, sd, size=(draws, variants)).max(axis=1)
    return {
        "null_mean": round(mu, 4), "null_std": round(sd, 4),
        "expected_best_by_luck": round(float(np.median(maxima)), 3),
        "p95_best_by_luck": round(float(np.quantile(maxima, 0.95)), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--start", default="2004-06-11")
    ap.add_argument("--end", default="2024-01-01",
                    help="EXCLUSIVE. 2024+ stays sealed.")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--variants", type=int, default=40)
    ap.add_argument("--rate", type=float, default=0.06,
                    help="daily signal rate ~ 15 trades/yr")
    ap.add_argument("--hold", type=int, default=20, help="bars (days)")
    ap.add_argument("--stop-mult", type=float, default=2.5)
    ap.add_argument("--outdir", default=Path("./data"), type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"{SEP}\nDAILY NULL BENCHMARKS\n{SEP}")
    full = load_daily(args.file)
    bars = full[(full.datetime >= pd.Timestamp(args.start)) &
                (full.datetime < pd.Timestamp(args.end))].reset_index(drop=True)
    print(f"  Research window: {bars.datetime.min().date()} -> {bars.datetime.max().date()}"
          f"  ({len(bars):,} days)")
    print(f"  SEALED         : {args.end} -> {full.datetime.max().date()}")

    cfg = Config()
    print(f"  Cost           : ${cfg.spread_per_oz + cfg.slippage_per_oz:.2f}/oz "
          f"(zero by decision)")

    # ---- 1. buy and hold, whole window and by regime ---------------------
    print(f"\n{SUB}\n1. BUY AND HOLD\n{SUB}")
    bh = buy_and_hold(bars)
    print(f"  Whole window : Sharpe {bh['sharpe']:+.3f}   "
          f"CAGR {bh['cagr_pct']:+.2f}%   maxDD {bh['max_drawdown_pct']:.1f}%")

    print(f"\n  By regime -- a single number would hide all of this:")
    print(f"    {'regime':<24}{'sharpe':>9}{'return':>10}{'maxDD':>9}")
    regimes = {}
    for name, s, e in REGIMES:
        sub = full[(full.datetime >= pd.Timestamp(s)) &
                   (full.datetime < pd.Timestamp(e))].reset_index(drop=True)
        if len(sub) < 60:
            continue
        r = buy_and_hold(sub)
        if not r:
            continue
        regimes[name] = r
        print(f"    {name:<24}{r['sharpe']:>+9.2f}{r['total_return_pct']:>9.1f}%"
              f"{r['max_drawdown_pct']:>8.1f}%")

    # ---- 2. noise floor --------------------------------------------------
    print(f"\n{SUB}\n2. RANDOM ENTRIES ({args.trials} trials, "
          f"~{args.rate*252:.0f} signals/yr, {args.hold}-day hold)\n{SUB}")
    a = atr(bars, 14)
    nulls, summary = {}, {}
    for bias in ("both", "long", "short"):
        arr = run_null(bars, a, cfg, args.trials, args.rate,
                       args.hold, args.stop_mult, bias)
        nulls[bias] = arr
        if arr.size:
            summary[bias] = {
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std()), 4),
                "p95": round(float(np.quantile(arr, 0.95)), 3),
                "max": round(float(arr.max()), 3), "n": int(arr.size),
            }
            label = f"{bias}-only" if bias != "both" else "both dirs"
            print(f"  {label:<12} mean {arr.mean():+.3f}   std {arr.std():.3f}   "
                  f"p95 {np.quantile(arr,0.95):+.3f}   max {arr.max():+.3f}")

    drift = summary.get("long", {}).get("mean", 0) - summary.get("both", {}).get("mean", 0)
    print(f"\n  Free Sharpe from simply being long: {drift:+.3f}")

    # ---- 3. thresholds ---------------------------------------------------
    print(f"\n{SUB}\n3. SELECTION THRESHOLD (best of {args.variants} variants)\n{SUB}")
    thr = {}
    for bias in ("both", "long", "short"):
        t = selection_threshold(nulls[bias], args.variants)
        if t:
            thr[bias] = t
            label = f"{bias}-only" if bias != "both" else "both dirs"
            print(f"  {label:<12} typical best {t['expected_best_by_luck']:+.3f}   "
                  f"p95 {t['p95_best_by_luck']:+.3f}")

    print(f"\n{SUB}\nTHE DAILY BAR\n{SUB}")
    bh_s = bh["sharpe"]
    luck_long = thr.get("long", {}).get("p95_best_by_luck", 0)
    luck_sym = thr.get("both", {}).get("p95_best_by_luck", 0)
    bar_long, bar_sym = max(bh_s, luck_long), max(bh_s, luck_sym)
    print(f"  Buy and hold                  : {bh_s:+.3f}")
    print(f"  Best-of-{args.variants} luck, long-biased : {luck_long:+.3f}")
    print(f"  Best-of-{args.variants} luck, symmetric   : {luck_sym:+.3f}")
    print(f"\n  >> Mostly-LONG strategy must clear   {bar_long:+.3f}")
    print(f"  >> Direction-neutral must clear      {bar_sym:+.3f}")
    print("\n  Compare to the H1 study: +0.94 / +0.39.")
    print("  Fewer trades widens the null, so these are NOT the same bars.")

    out = args.outdir / "daily_null.json"
    out.write_text(json.dumps({
        "window": f"{args.start}..{args.end}",
        "buy_and_hold": bh, "regimes": regimes,
        "random_entries": summary, "thresholds": thr,
        "bar_long": round(bar_long, 3), "bar_neutral": round(bar_sym, 3),
        "params": vars(args) | {"outdir": str(args.outdir)},
    }, indent=2, default=str))
    print(f"\nWrote -> {out}\n{SEP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
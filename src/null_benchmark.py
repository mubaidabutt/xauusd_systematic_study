#!/usr/bin/env python3
"""
XAUUSD Research Project -- Step 3: Null Benchmarks
===================================================

The question this answers
-------------------------
"Sharpe 1.2" means nothing on its own. Meaningful compared to WHAT?

Two reference points have to exist before any strategy result can be read:

  1. BUY AND HOLD. Gold roughly tripled across this sample. If simply owning
     gold produces a respectable Sharpe, then a long-biased strategy clearing
     that bar has demonstrated nothing -- it has rediscovered beta with extra
     steps and more ways to fail.

  2. THE NOISE FLOOR. Random entries, run hundreds of times, produce a
     DISTRIBUTION of Sharpe ratios. The upper tail of that distribution is
     what luck alone achieves. And when you test 160 variants, you are not
     drawing once from it -- you are taking the maximum of 160 draws, which
     sits far higher than any single draw.

The number this script exists to produce is the last one: the Sharpe a
strategy must clear before it is distinguishable from the best of 160 coin
flips. Your previous round had no such number, which is why 14/160 passing
could not be interpreted.

Usage
-----
    python null_benchmark.py
    python null_benchmark.py --trials 400 --variants 160
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


def locate_parquet(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"Not found: {p.resolve()}")
        return p
    for base in (Path("data"), Path(".")):
        for pat in ("*_h1_mt5.parquet", "*h1*.parquet", "*.parquet"):
            hits = sorted(base.glob(pat))
            if hits:
                print(f"  Using: {hits[0]}")
                return hits[0]
    sys.exit("No parquet found. Pass --file path/to/XAUUSDm_h1_mt5.parquet")


# ==========================================================================
# BENCHMARK 1 -- BUY AND HOLD
# ==========================================================================

def buy_and_hold(bars: pd.DataFrame) -> dict:
    """
    Sharpe of daily gold returns. Unlevered, so no sizing assumptions -- and
    Sharpe is scale-invariant, which makes this directly comparable to any
    strategy result regardless of position sizing.
    """
    px = bars.set_index("datetime")["close"]
    daily = px.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()

    sharpe = rets.mean() / rets.std() * np.sqrt(252)
    curve = (1 + rets).cumprod()
    dd = (curve - curve.cummax()) / curve.cummax()
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    total = float(px.iloc[-1] / px.iloc[0] - 1)

    return {
        "sharpe": round(float(sharpe), 4),
        "total_return_pct": round(total * 100, 2),
        "cagr_pct": round(((1 + total) ** (1 / years) - 1) * 100, 2),
        "max_drawdown_pct": round(float(-dd.min() * 100), 2),
        "years": round(years, 2),
    }


def per_year_gold(bars: pd.DataFrame) -> pd.DataFrame:
    px = bars.set_index("datetime")["close"]
    daily = px.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    rows = []
    for year, grp in rets.groupby(rets.index.year):
        if len(grp) < 60:
            continue
        rows.append({
            "year": int(year),
            "return_pct": round(float((1 + grp).prod() - 1) * 100, 1),
            "sharpe": round(float(grp.mean() / grp.std() * np.sqrt(252)), 2),
            "ann_vol_pct": round(float(grp.std() * np.sqrt(252) * 100), 1),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# BENCHMARK 2 -- RANDOM ENTRIES
# ==========================================================================

def random_signals(bars: pd.DataFrame, atr_series: pd.Series, seed: int,
                   rate: float, hold: int, stop_mult: float,
                   bias: str = "both") -> pd.DataFrame:
    """
    Random entries with realistic mechanics -- ATR stops, time exit, and a
    trade rate tuned to resemble the strategies we will actually test.

    `bias` isolates directional exposure. Long-only random entries in a
    tripling market are the control that reveals how much of an apparent edge
    is simply being long.
    """
    rng = np.random.default_rng(seed)
    n = len(bars)
    fire = rng.random(n) < rate
    if bias == "long":
        d = np.where(fire, 1, 0)
    elif bias == "short":
        d = np.where(fire, -1, 0)
    else:
        d = np.where(fire, rng.choice([-1, 1], n), 0)

    stop = (atr_series * stop_mult).to_numpy()
    valid = np.isfinite(stop) & (stop > 0)
    d = np.where(valid, d, 0)

    return pd.DataFrame({
        "direction": d.astype(int),
        "stop_distance": np.where(valid, stop, 0.0),
        "trail_distance": np.zeros(n),
        "max_hold": np.full(n, hold, dtype=int),
    })


def run_null(bars: pd.DataFrame, atr_series: pd.Series, cfg: Config,
             trials: int, rate: float, hold: int, stop_mult: float,
             bias: str) -> np.ndarray:
    out = []
    for seed in range(trials):
        sig = random_signals(bars, atr_series, seed, rate, hold, stop_mult, bias)
        m = metrics(run_backtest(bars, sig, cfg))
        if m["n_trades"] > 20:
            out.append(m["sharpe"])
    return np.array(out)


def describe(name: str, arr: np.ndarray) -> dict:
    if arr.size == 0:
        print(f"  {name:<14} no valid trials")
        return {}
    q = {f"p{int(p*100)}": round(float(np.quantile(arr, p)), 3)
         for p in (0.05, 0.25, 0.50, 0.75, 0.95, 0.99)}
    print(f"  {name:<14} mean {arr.mean():+.3f}   median {q['p50']:+.3f}   "
          f"p95 {q['p95']:+.3f}   p99 {q['p99']:+.3f}   max {arr.max():+.3f}")
    return {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4),
            "n": int(arr.size), **q, "max": round(float(arr.max()), 4)}


# ==========================================================================
# BENCHMARK 3 -- MULTIPLE-TESTING THRESHOLD
# ==========================================================================

def selection_threshold(null: np.ndarray, variants: int,
                        draws: int = 20_000) -> dict:
    """
    The number that makes 14/160 interpretable.

    Testing 160 variants and keeping the best is not one draw from the null --
    it is the MAXIMUM of 160 draws. That maximum has its own distribution,
    centred far above any single draw.

    Fitting a normal to the null and sampling the max parametrically, rather
    than bootstrapping the observed Sharpes directly. Resampling 160 values
    from a few hundred observations returns the sample maximum nearly every
    time, so every percentile collapses onto the same number and the answer
    looks falsely precise.
    """
    if null.size < 20:
        return {}
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    rng = np.random.default_rng(0)
    maxima = rng.normal(mu, sd, size=(draws, variants)).max(axis=1)
    return {
        "variants_tested": variants,
        "null_mean": round(mu, 4),
        "null_std": round(sd, 4),
        "expected_best_by_luck": round(float(np.median(maxima)), 3),
        "p95_best_by_luck": round(float(np.quantile(maxima, 0.95)), 3),
        "p99_best_by_luck": round(float(np.quantile(maxima, 0.99)), 3),
    }


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--start", default=None,
                    help="ISO date, e.g. 2018-01-01. Benchmark and backtest "
                         "MUST cover the same window to be comparable.")
    ap.add_argument("--end", default=None, help="ISO date, exclusive")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--variants", type=int, default=160)
    ap.add_argument("--rate", type=float, default=0.021)
    ap.add_argument("--hold", type=int, default=24)
    ap.add_argument("--stop-mult", type=float, default=2.5)
    ap.add_argument("--outdir", default=Path("./data"), type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"{SEP}\nLOADING\n{SEP}")
    bars = pd.read_parquet(locate_parquet(args.file))
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    bars = bars.sort_values("datetime").reset_index(drop=True)

    if args.start:
        bars = bars[bars.datetime >= pd.Timestamp(args.start)]
    if args.end:
        bars = bars[bars.datetime < pd.Timestamp(args.end)]
    bars = bars.reset_index(drop=True)
    if len(bars) < 1000:
        sys.exit(f"Only {len(bars)} bars in window -- too few to benchmark.")
    print(f"  {len(bars):,} bars   {bars.datetime.min()} -> {bars.datetime.max()}")

    # A year holding far fewer bars than a full one (~5,900) is a cache
    # fragment, not history. It will not break the daily-close benchmark, but
    # it cannot support a backtest -- and mixing the two silently compares a
    # strategy against a benchmark measured over a different period.
    counts = bars.datetime.dt.year.value_counts().sort_index()
    final_year = int(counts.index.max())
    # The last year is short because it is still in progress, not because it
    # is a fragment. Flagging it would cry wolf on every run.
    sparse = [int(y) for y, n in counts.items() if n < 4500 and y != final_year]
    if sparse:
        print(f"\n  >> WARNING: sparse years in window: {sparse}")
        for y in sparse:
            print(f"       {y}: {counts[y]:,} bars (a full year is ~5,900)")
        print("     These are cache fragments. Re-run with --start to exclude")
        print("     them, and use the SAME window when backtesting.")

    cfg = Config()
    a = atr(bars, 14)

    # ---- 1. buy and hold -------------------------------------------------
    print(f"\n{SUB}\n1. BUY AND HOLD  (the beta you must beat)\n{SUB}")
    bh = buy_and_hold(bars)
    print(f"  Sharpe           : {bh['sharpe']:+.3f}")
    print(f"  Total return     : {bh['total_return_pct']:+,.1f}%  over {bh['years']} years")
    print(f"  CAGR             : {bh['cagr_pct']:+.2f}%")
    print(f"  Max drawdown     : {bh['max_drawdown_pct']:.2f}%")

    print(f"\n  Gold by year:")
    print(f"    {'year':<6}{'return':>10}{'sharpe':>9}{'ann vol':>10}")
    for _, r in per_year_gold(bars).iterrows():
        print(f"    {int(r.year):<6}{r.return_pct:>9.1f}%{r.sharpe:>9.2f}{r.ann_vol_pct:>9.1f}%")

    # ---- 2. noise floor --------------------------------------------------
    print(f"\n{SUB}\n2. RANDOM ENTRIES  ({args.trials} trials each)\n{SUB}")
    print(f"  ATR({14})x{args.stop_mult} stop, {args.hold}-bar max hold, "
          f"cost ${cfg.spread_per_oz + cfg.slippage_per_oz:.2f}/oz\n")
    nulls, summary = {}, {}
    for bias in ("both", "long", "short"):
        arr = run_null(bars, a, cfg, args.trials, args.rate,
                       args.hold, args.stop_mult, bias)
        nulls[bias] = arr
        summary[bias] = describe(f"{bias}-only" if bias != "both" else "both dirs", arr)

    # How much Sharpe does the cost model alone destroy? Same random entries,
    # costs switched off. The gap is the honest answer to "does spread matter",
    # measured rather than argued.
    free = run_null(bars, a, Config(spread_per_oz=0.0, slippage_per_oz=0.0),
                    args.trials, args.rate, args.hold, args.stop_mult, "both")
    summary["both_zero_cost"] = describe("both, no cost", free)
    if free.size and nulls["both"].size:
        drag = float(free.mean() - nulls["both"].mean())
        summary["cost_drag_sharpe"] = round(drag, 4)
        print(f"\n  >> COST DRAG: {drag:.3f} Sharpe units at "
              f"${cfg.spread_per_oz + cfg.slippage_per_oz:.2f}/oz.")
        print(f"     Every strategy pays this before it earns anything.")

    # ---- 3. selection threshold -----------------------------------------
    print(f"\n{SUB}\n3. SELECTION THRESHOLD  (best of {args.variants} variants)\n{SUB}")
    print("  The null must MATCH the strategy's directional exposure. A mostly-long")
    print("  strategy competes against random longs, not against a symmetric coin")
    print("  flip -- comparing it to the symmetric null credits it with gold's drift.\n")
    thr = {}
    for bias in ("both", "long", "short"):
        t = selection_threshold(nulls[bias], args.variants)
        if not t:
            continue
        thr[bias] = t
        label = f"{bias}-only" if bias != "both" else "both dirs"
        print(f"  {label:<11} typical best {t['expected_best_by_luck']:+.3f}   "
              f"p95 {t['p95_best_by_luck']:+.3f}   p99 {t['p99_best_by_luck']:+.3f}")

    # ---- verdict ---------------------------------------------------------
    print(f"\n{SUB}\nTHE BAR TO CLEAR\n{SUB}")
    bh_sharpe = bh["sharpe"]
    luck_long = thr.get("long", {}).get("p95_best_by_luck", 0.0)
    luck_both = thr.get("both", {}).get("p95_best_by_luck", 0.0)
    drift = summary.get("long", {}).get("mean", 0.0) - summary.get("both", {}).get("mean", 0.0)

    print(f"  Buy and hold                     : {bh_sharpe:+.3f}")
    print(f"  Best-of-{args.variants}, symmetric null    : {luck_both:+.3f}")
    print(f"  Best-of-{args.variants}, LONG-biased null  : {luck_long:+.3f}")
    print(f"  Free Sharpe from being long      : {drift:+.3f}")

    bar_long = max(bh_sharpe, luck_long)
    bar_sym = max(bh_sharpe, luck_both)
    print(f"\n  >> A mostly-LONG strategy must clear   Sharpe {bar_long:+.3f}")
    print(f"  >> A direction-neutral strategy clears Sharpe {bar_sym:+.3f}")
    print("\n  Record these in the trial registry BEFORE testing anything.")
    print("  A threshold chosen after seeing results is not a threshold.")

    out = args.outdir / "null_benchmark.json"
    out.write_text(json.dumps({
        "buy_and_hold": bh, "random_entries": summary,
        "selection_threshold": thr,
        "params": {"trials": args.trials, "variants": args.variants,
                   "rate": args.rate, "hold": args.hold,
                   "stop_mult": args.stop_mult},
    }, indent=2))
    print(f"\nWrote -> {out}\n{SEP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
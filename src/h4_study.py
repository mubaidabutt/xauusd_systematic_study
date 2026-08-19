#!/usr/bin/env python3
"""
XAUUSD -- H4 Volatility Regime Study
=====================================

Why this study exists
---------------------
The expectancy report found something the Sharpe-based studies missed. On H4,
20 of 28 variants had positive expectancy at real cost, median +0.058R. On H1,
6 of 28, median -0.052R. The whole distribution shifts with timeframe.

And the strongest family on H4 was H2_volatility_regime -- the same family that
failed decisively on H1. The mechanism (volatility compression resolving into
expansion) was plausible all along; H1 was simply the wrong timeframe to
express it, because the edge per trade was smaller than the cost per trade.

So this study tests that family properly on H4: its own null, its own
thresholds, expanded parameter grid, confidence intervals on every estimate.

Costs are ZERO by decision. Both columns are reported anyway, because the H4
finding was strongest AT real cost and hiding that would be dishonest.

The single most important output
--------------------------------
Not expectancy. The CONFIDENCE INTERVAL on expectancy.

The top H4 variant showed +0.582R over 58 trades -- but with a per-trade
standard deviation of 2.74R, its 95% interval was [-0.12, +1.29]. That
includes zero. A 26% win rate with a 5x payoff produces enormous variance, so
expectancy converges slowly and a high point estimate on few trades means very
little. Every result below carries its interval.

Usage
-----
    python h4_study.py
    python h4_study.py --trials 400
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from engine import Config, metrics, run_backtest
from indicators import atr

SEP = "=" * 94
SUB = "-" * 94

RESEARCH_END = "2024-01-01"   # 2024+ stays sealed

REGIMES = [
    ("2018-2019", "2018-01-01", "2020-01-01"),
    ("2020-2021", "2020-01-01", "2022-01-01"),
    ("2022-2023", "2022-01-01", "2024-01-01"),
]


# ==========================================================================
# DATA
# ==========================================================================

def load_h4(explicit: str | None, start: str, end: str) -> pd.DataFrame:
    p = Path(explicit) if explicit else None
    if p is None:
        for base in (Path("data"), Path(".")):
            hits = sorted(base.glob("*_h1_mt5.parquet"))
            if hits:
                p = hits[0]
                break
    if p is None or not p.exists():
        sys.exit("H1 parquet not found. Pass --file path/to/XAUUSDm_h1_mt5.parquet")

    print(f"  Source: {p}")
    h1 = pd.read_parquet(p)
    h1["datetime"] = pd.to_datetime(h1["datetime"])
    h1 = h1[["datetime", "open", "high", "low", "close"]].sort_values("datetime")

    # UTC 4h boundaries -- identical to what the Exness terminal builds
    h4 = (h1.set_index("datetime")
            .resample("4h", origin="start_day")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna().reset_index())
    h4 = h4[(h4.datetime >= pd.Timestamp(start)) & (h4.datetime < pd.Timestamp(end))]
    return h4.reset_index(drop=True)


# ==========================================================================
# STRATEGY -- volatility compression breakout, expanded grid
# ==========================================================================

def vol_regime(bars: pd.DataFrame, *, lookback: int = 100, percentile: int = 20,
               breakout_lookback: int = 20, atr_period: int = 14,
               stop_mult: float = 2.0, trail_mult: float = 2.0,
               max_hold: int = 48) -> pd.DataFrame:
    """
    ATR sits in the bottom `percentile` of its own trailing window -> the
    market is compressed. Trade whichever way it breaks out of the recent
    range. Direction comes from price, not from a view.

    Both conditions are EVENTS. `close > high` stays true for many bars after
    a break, so the prior bar must have been inside the range for the signal
    to fire. That distinction is what broke the original H1 study.
    """
    n = len(bars)
    sig = pd.DataFrame({
        "direction": np.zeros(n, dtype=int),
        "stop_distance": np.zeros(n),
        "trail_distance": np.zeros(n),
        "max_hold": np.full(n, max_hold, dtype=int),
        "exit_signal": np.zeros(n, dtype=bool),
    })

    a = atr(bars, atr_period)
    rank = a.rolling(lookback, min_periods=lookback).rank(pct=True) * 100
    compressed = (rank <= percentile).to_numpy()

    close = bars["close"]
    hi = bars["high"].rolling(breakout_lookback, min_periods=breakout_lookback).max().shift(1)
    lo = bars["low"].rolling(breakout_lookback, min_periods=breakout_lookback).min().shift(1)

    above = (close > hi)
    below = (close < lo)
    fresh_up = (above & ~above.shift(1).astype("boolean").fillna(False)).to_numpy()
    fresh_dn = (below & ~below.shift(1).astype("boolean").fillna(False)).to_numpy()

    av = a.to_numpy()
    ok = np.isfinite(av) & (av > 0) & np.isfinite(hi.to_numpy()) & np.isfinite(lo.to_numpy())
    valid = ok & compressed

    sig.loc[valid & fresh_up, "direction"] = 1
    sig.loc[valid & fresh_dn, "direction"] = -1
    sig["stop_distance"] = np.where(ok, av * stop_mult, 0.0)
    sig["trail_distance"] = np.where(ok, av * trail_mult, 0.0)
    return sig


def grid() -> list[dict]:
    """24 variants. Coarse steps so neighbours are genuinely different."""
    out = []
    for lb in (50, 100):
        for pct in (15, 30):
            for stop in (1.5, 2.5):
                for trail in (2.0, 3.0):
                    out.append({"lookback": lb, "percentile": pct, "stop_mult": stop,
                                "trail_mult": trail, "breakout_lookback": 20,
                                "max_hold": 48})
    for bl in (10, 30):                       # breakout-horizon controls
        for stop in (1.5, 2.5):
            out.append({"lookback": 100, "percentile": 20, "stop_mult": stop,
                        "trail_mult": 2.0, "breakout_lookback": bl, "max_hold": 48})
    for pct in (50, 100):                     # NO-COMPRESSION CONTROLS
        out.append({"lookback": 100, "percentile": pct, "stop_mult": 2.0,
                    "trail_mult": 2.0, "breakout_lookback": 20, "max_hold": 48})
    return out


N_VARIANTS = len(grid())
# Indices of the pct=100 control (compression filter disabled entirely).
CONTROL_IDX = [i for i, p in enumerate(grid()) if p["percentile"] >= 50]


# ==========================================================================
# STATISTICS
# ==========================================================================

def expectancy_ci(trades: pd.DataFrame, cfg: Config) -> dict:
    """
    Expectancy in R, with a 95% confidence interval.

    The interval is the point of this function. A 26% win rate with a 5x
    payoff has a per-trade standard deviation near 2.7R, so 58 trades give an
    interval roughly +/- 0.7R wide. Reporting +0.58R without that context
    invites a decision the data cannot support.
    """
    if trades.empty:
        return {}
    r_unit = cfg.initial_equity * cfg.risk_per_trade
    r = (trades["net_pnl"] / r_unit).to_numpy()
    n = len(r)
    mean, sd = float(r.mean()), float(r.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n else 0.0
    wins = r[r > 0]
    losses = r[r <= 0]
    return {
        "n": n,
        "expectancy_r": mean,
        "sd_r": sd,
        "se_r": se,
        "ci_lo": mean - 1.96 * se,
        "ci_hi": mean + 1.96 * se,
        "significant": bool(mean - 1.96 * se > 0),
        "win_rate": 100.0 * len(wins) / n,
        "payoff": (wins.mean() / abs(losses.mean())) if len(losses) and len(wins) else np.inf,
    }


def null_distribution(bars, a, cfg, trials, rate, hold, stop_mult) -> np.ndarray:
    """Random symmetric entries at the strategy's own trade frequency."""
    out = []
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        n = len(bars)
        fire = rng.random(n) < rate
        d = np.where(fire, rng.choice([-1, 1], n), 0)
        stop = (a * stop_mult).to_numpy()
        ok = np.isfinite(stop) & (stop > 0)
        sig = pd.DataFrame({
            "direction": np.where(ok, d, 0).astype(int),
            "stop_distance": np.where(ok, stop, 0.0),
            "trail_distance": np.zeros(n),
            "max_hold": np.full(n, hold, dtype=int),
        })
        m = metrics(run_backtest(bars, sig, cfg))
        if m["n_trades"] > 10:
            out.append(m["sharpe"])
    return np.array(out)


def best_of_n(null: np.ndarray, k: int, draws: int = 20_000) -> dict:
    if null.size < 20:
        return {}
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    rng = np.random.default_rng(0)
    mx = rng.normal(mu, sd, size=(draws, k)).max(axis=1)
    return {"null_mean": round(mu, 4), "null_sd": round(sd, 4),
            "typical_best": round(float(np.median(mx)), 3),
            "p95_best": round(float(np.quantile(mx, 0.95)), 3)}


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--outdir", default=Path("./data"), type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"{SEP}\nH4 VOLATILITY REGIME STUDY\n{SEP}")
    bars = load_h4(args.file, args.start, RESEARCH_END)
    print(f"  H4 bars : {len(bars):,}   {bars.datetime.min()} -> {bars.datetime.max()}")
    print(f"  SEALED  : {RESEARCH_END} onward -- not touched")

    free = Config()                                   # zero cost, by decision
    real = Config(spread_per_oz=0.30, slippage_per_oz=0.10)
    print(f"  Cost    : $0.00/oz primary (zero by decision); "
          f"$0.40/oz reported alongside")
    print(f"  Variants: {N_VARIANTS}  (including {len(CONTROL_IDX)} no-compression controls)")

    # ---- 1. NULL, computed BEFORE looking at any strategy result ---------
    print(f"\n{SUB}\n1. NULL THRESHOLD (random entries, computed first)\n{SUB}")
    a = atr(bars, 14)
    probe = vol_regime(bars, **grid()[0])
    rate = float((probe["direction"] != 0).mean())
    print(f"  Matching the strategy's own signal rate: {rate*100:.2f}% of bars")
    null = null_distribution(bars, a, free, args.trials, rate, 48, 2.0)
    thr = best_of_n(null, N_VARIANTS)
    if not thr:
        sys.exit("Null distribution too small.")
    bar = thr["p95_best"]
    print(f"  Null Sharpe        : mean {thr['null_mean']:+.3f}  sd {thr['null_sd']:.3f}")
    print(f"  Best-of-{N_VARIANTS} by luck : typical {thr['typical_best']:+.3f}   "
          f"p95 {thr['p95_best']:+.3f}")
    print(f"\n  >> BAR: Sharpe must exceed {bar:+.3f}")

    # ---- 2. VARIANTS ------------------------------------------------------
    print(f"\n{SUB}\n2. VARIANTS  (expR = expectancy in R, with 95% CI)\n{SUB}")
    print(f"  {'#':<3}{'trades':>7}{'win%':>7}{'payoff':>8}{'sharpe':>8}"
          f"{'expR free':>11}{'95% CI':>20}{'expR real':>11}{'sig':>5}  note")

    rows = []
    for vi, params in enumerate(grid()):
        sig = vol_regime(bars, **params)
        rf = run_backtest(bars, sig, free)
        rr = run_backtest(bars, sig, real)
        mf = metrics(rf)
        ef = expectancy_ci(rf.to_frame(), free)
        er = expectancy_ci(rr.to_frame(), real)
        if not ef or ef["n"] < 20:
            print(f"  {vi:<3}{ef.get('n', 0):>7}   too few trades")
            continue

        is_ctrl = vi in CONTROL_IDX
        note = "CONTROL (no compression filter)" if is_ctrl else ""
        if mf["sharpe"] > bar and ef["significant"]:
            note = ("PASSES BAR + significant " + note).strip()

        print(f"  {vi:<3}{ef['n']:>7}{ef['win_rate']:>6.1f}%{ef['payoff']:>8.2f}"
              f"{mf['sharpe']:>+8.3f}{ef['expectancy_r']:>+11.4f}"
              f"  [{ef['ci_lo']:+.3f},{ef['ci_hi']:+.3f}]"
              f"{er.get('expectancy_r', 0):>+11.4f}"
              f"{'  YES' if ef['significant'] else '   no':>5}  {note}")

        rows.append({"variant": vi, "params": json.dumps(params), "control": is_ctrl,
                     "n_trades": ef["n"], "win_rate": round(ef["win_rate"], 2),
                     "payoff": round(float(ef["payoff"]), 3),
                     "sharpe": mf["sharpe"], "max_dd": mf["max_drawdown_pct"],
                     "exp_r_free": round(ef["expectancy_r"], 4),
                     "ci_lo": round(ef["ci_lo"], 4), "ci_hi": round(ef["ci_hi"], 4),
                     "significant": ef["significant"],
                     "exp_r_real": round(er.get("expectancy_r", 0), 4),
                     "passes_bar": bool(mf["sharpe"] > bar)})

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No variants produced enough trades.")

    # ---- 3. CONTROL COMPARISON -------------------------------------------
    print(f"\n{SUB}\n3. CONTROL -- does the compression filter actually do anything?\n{SUB}")
    real_v = df[~df.control]
    ctrl_v = df[df.control]
    if len(ctrl_v):
        gap = float(real_v.exp_r_free.median() - ctrl_v.exp_r_free.median())
        print(f"  Compression-filtered : median expR {real_v.exp_r_free.median():+.4f}  "
              f"(n={len(real_v)})")
        print(f"  No filter (control)  : median expR {ctrl_v.exp_r_free.median():+.4f}  "
              f"(n={len(ctrl_v)})")
        print(f"  Difference           : {gap:+.4f}R")
        print(f"  >> {'Compression filter ADDS value.' if gap > 0.02 else 'Filter adds NOTHING -- the edge, if any, is the breakout alone.'}")

    # ---- 4. REGIME --------------------------------------------------------
    print(f"\n{SUB}\n4. REGIME BREAKDOWN (top 3 by expectancy)\n{SUB}")
    for _, r in df[~df.control].nlargest(3, "exp_r_free").iterrows():
        params = json.loads(r.params)
        line = [f"  variant {int(r.variant):<3}"]
        for name, s, e in REGIMES:
            sub = bars[(bars.datetime >= pd.Timestamp(s)) &
                       (bars.datetime < pd.Timestamp(e))].reset_index(drop=True)
            if len(sub) < 100:
                continue
            e2 = expectancy_ci(run_backtest(sub, vol_regime(sub, **params), free).to_frame(), free)
            line.append(f"{name}: {e2.get('expectancy_r', 0):+.3f}R (n={e2.get('n',0)})")
        print("   ".join(line))

    # ---- 5. VERDICT -------------------------------------------------------
    print(f"\n{SUB}\n5. VERDICT\n{SUB}")
    passed = df[(~df.control) & df.passes_bar & df.significant]
    print(f"  Variants clearing the Sharpe bar AND statistically significant: "
          f"{len(passed)}/{len(real_v)}")
    median_exp = float(real_v.exp_r_free.median())
    print(f"  Family median expectancy (zero cost) : {median_exp:+.4f}R")
    print(f"  Family median expectancy (real cost) : {real_v.exp_r_real.median():+.4f}R")
    print(f"  Variants with CI excluding zero      : {int(real_v.significant.sum())}/{len(real_v)}")

    if len(passed):
        print(f"\n  CANDIDATES:")
        for _, r in passed.iterrows():
            print(f"    variant {int(r.variant)}: {int(r.n_trades)} trades, "
                  f"{r.win_rate:.1f}% win, payoff {r.payoff:.2f}, "
                  f"expR {r.exp_r_free:+.3f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}]")
        print(f"\n  Next: sealed-window test on 2024-2026. ONE evaluation.")
    else:
        print(f"\n  No variant is both above the bar and statistically distinguishable")
        print(f"  from zero. High point estimates on few trades are not evidence.")

    out = args.outdir / "h4_study.csv"
    df.to_csv(out, index=False)
    (args.outdir / "h4_study_meta.json").write_text(json.dumps(
        {"bar": bar, "null": thr, "n_variants": N_VARIANTS,
         "window": f"{args.start}..{RESEARCH_END}",
         "timestamp": datetime.now().isoformat(timespec="seconds")}, indent=2))
    print(f"\n  -> {out}\n{SEP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
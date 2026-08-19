#!/usr/bin/env python3
"""
XAUUSD Research Project -- Step 9: Daily study runner
======================================================

Runs the 40 pre-registered daily variants on 2004-2023, applies the tightened
stages from PREREGISTRATION_DAILY.md, and reports the two tests the H1 study
could not run: regime robustness and direction balance.

Tightened from the H1 runner, deliberately and before seeing results:
  * Stage 0 caps any single trade at 25% of profit (H1 allowed 100%, and
    variants passed showing 974%).
  * Stage 2 requires the MEDIAN variant to clear the actual bar (H1 asked only
    for median > 0, and flagged a family "coherent" at +0.060 with 0/16
    passing).

Costs are zero by decision. Usage:

    python run_daily_study.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from engine import Config, metrics, run_backtest
from strategies_daily import FAMILIES_DAILY, pct_signals_long

SEP = "=" * 80
SUB = "-" * 80

# LOCKED in PREREGISTRATION_DAILY.md section 4.
BAR_LONG = 1.050
BAR_NEUTRAL = 0.705
LONG_CUTOFF = 60.0
MAX_TRADE_CONCENTRATION = 0.25
MIN_TRADES = 40

REGIMES = [
    ("2004-2007 early bull", "2004-06-11", "2008-01-01"),
    ("2008-2011 parabolic",  "2008-01-01", "2011-09-01"),
    ("2011-2015 BEAR",       "2011-09-01", "2016-01-01"),
    ("2016-2018 range",      "2016-01-01", "2018-01-01"),
    ("2018-2023 dev",        "2018-01-01", "2024-01-01"),
]

REGISTRY_FIELDS = [
    "timestamp", "family", "variant", "params", "window", "n_trades",
    "sharpe", "pct_long", "threshold", "exposure_class", "passed",
    "max_drawdown_pct", "cagr_pct", "profit_factor", "net_profit",
    "long_pnl", "short_pnl", "regimes_positive", "stage0", "stage0_reason",
]


def load_daily(explicit: str | None) -> pd.DataFrame:
    p = Path(explicit) if explicit else Path("data/xauusd_daily.parquet")
    if not p.exists():
        hits = sorted(Path(".").rglob("xauusd_daily.parquet"))
        if not hits:
            sys.exit("xauusd_daily.parquet not found. Run the daily builder first.")
        p = hits[0]
    print(f"  Using: {p}")
    df = pd.read_parquet(p)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def threshold_for(pct_long: float) -> tuple[float, str]:
    if pct_long > LONG_CUTOFF:
        return BAR_LONG, "long-biased"
    if pct_long < 100 - LONG_CUTOFF:
        return BAR_NEUTRAL, "short-biased"
    return BAR_NEUTRAL, "neutral"


def stage0(m: dict, trades: pd.DataFrame) -> tuple[bool, str]:
    if m["n_trades"] < MIN_TRADES:
        return False, f"only {m['n_trades']} trades"
    if m["net_profit"] > 0 and not trades.empty:
        top = trades["net_pnl"].max()
        share = top / m["net_profit"]
        if share > MAX_TRADE_CONCENTRATION:
            return False, f"top trade = {share*100:.0f}% of profit"
    return True, ""


def direction_split(trades: pd.DataFrame) -> tuple[float, float]:
    if trades.empty:
        return 0.0, 0.0
    return (float(trades[trades.direction == 1]["net_pnl"].sum()),
            float(trades[trades.direction == -1]["net_pnl"].sum()))


def regime_check(bars, fn, params, cfg) -> tuple[int, list]:
    """Positive Sharpe in how many of the 5 named regimes?"""
    out = []
    for name, s, e in REGIMES:
        sub = bars[(bars.datetime >= pd.Timestamp(s)) &
                   (bars.datetime < pd.Timestamp(e))].reset_index(drop=True)
        if len(sub) < 120:
            continue
        try:
            m = metrics(run_backtest(sub, fn(sub, **params), cfg))
            out.append((name, m["sharpe"], m["n_trades"]))
        except Exception:
            out.append((name, float("nan"), 0))
    return sum(1 for _, s, _ in out if s > 0), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--start", default="2004-06-11")
    ap.add_argument("--end", default="2024-01-01")
    ap.add_argument("--outdir", default=Path("./data"), type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"{SEP}\nDAILY STUDY RUNNER\n{SEP}")
    full = load_daily(args.file)
    bars = full[(full.datetime >= pd.Timestamp(args.start)) &
                (full.datetime < pd.Timestamp(args.end))].reset_index(drop=True)
    cfg = Config()
    print(f"  Research : {bars.datetime.min().date()} -> {bars.datetime.max().date()} "
          f"({len(bars):,} days)")
    print(f"  SEALED   : {args.end} -> {full.datetime.max().date()}  (not touched)")
    print(f"  Cost     : ${cfg.spread_per_oz + cfg.slippage_per_oz:.2f}/oz (zero by decision)")
    print(f"  Bars     : long-biased {BAR_LONG:+.3f}   neutral {BAR_NEUTRAL:+.3f}")

    stamp = datetime.now().isoformat(timespec="seconds")
    window = f"{args.start}..{args.end}"
    rows, fam_results = [], {}

    for family, (fn, grid) in FAMILIES_DAILY.items():
        variants = grid()
        print(f"\n{SUB}\n{family}  ({len(variants)} variants)\n{SUB}")
        print(f"  {'#':<3}{'trades':>7}{'sharpe':>8}{'%long':>7}{'bar':>7}"
              f"{'maxDD':>7}{'CAGR':>7}{'long$':>9}{'short$':>9}{'reg':>5}  verdict")

        results = []
        for vi, params in enumerate(variants):
            try:
                sig = fn(bars, **params)
                res = run_backtest(bars, sig, cfg)
                m = metrics(res)
                tr = res.to_frame()
            except Exception as exc:
                print(f"  {vi:<3} EXCEPTION {type(exc).__name__}: {exc}")
                continue

            pl = pct_signals_long(sig)
            thr, cls = threshold_for(pl)
            ok0, why0 = stage0(m, tr)
            lp, sp = direction_split(tr)
            nreg, _ = regime_check(bars, fn, params, cfg)
            passed = bool(ok0 and m["sharpe"] > thr)

            verdict = ("PASS" if passed else why0 if not ok0 else "below bar")
            print(f"  {vi:<3}{m['n_trades']:>7}{m['sharpe']:>+8.3f}{pl:>6.0f}%"
                  f"{thr:>7.2f}{m['max_drawdown_pct']:>6.1f}%{m['cagr_pct']:>6.1f}%"
                  f"{lp:>9,.0f}{sp:>9,.0f}{nreg:>4}/5  {verdict}")

            results.append({**m, "pct_long": pl, "threshold": thr,
                            "passed": passed, "long_pnl": lp, "short_pnl": sp,
                            "regimes_positive": nreg})
            rows.append({
                "timestamp": stamp, "family": family, "variant": vi,
                "params": json.dumps(params), "window": window,
                "n_trades": m["n_trades"], "sharpe": m["sharpe"],
                "pct_long": round(pl, 1), "threshold": thr,
                "exposure_class": cls, "passed": passed,
                "max_drawdown_pct": m["max_drawdown_pct"], "cagr_pct": m["cagr_pct"],
                "profit_factor": m["profit_factor"], "net_profit": m["net_profit"],
                "long_pnl": round(lp, 2), "short_pnl": round(sp, 2),
                "regimes_positive": nreg, "stage0": ok0, "stage0_reason": why0,
            })
        fam_results[family] = results

    # ---- STAGE 2: family coherence, tightened -----------------------------
    print(f"\n{SEP}\nSTAGE 2 -- FAMILY COHERENCE (median must clear the BAR)\n{SEP}")
    print("  Tightened from the H1 study, which asked only for median > 0 and")
    print("  flagged a family 'coherent' at +0.060 with 0/16 variants passing.\n")
    print(f"  {'family':<24}{'n':>3}{'median':>9}{'bar':>7}{'best':>9}"
          f"{'worst':>9}{'pass':>7}  verdict")

    summary, survivors = {}, []
    for family, results in fam_results.items():
        if not results:
            continue
        sh = np.array([r["sharpe"] for r in results])
        bar = float(np.median([r["threshold"] for r in results]))
        med = float(np.median(sh))
        npass = sum(1 for r in results if r["passed"])
        coherent = med > bar
        if coherent:
            survivors.append(family)
        summary[family] = {
            "n": len(sh), "median_sharpe": round(med, 3), "bar": bar,
            "best": round(float(sh.max()), 3), "worst": round(float(sh.min()), 3),
            "n_passed": npass, "stage2_coherent": coherent,
            "median_regimes_positive": float(np.median([r["regimes_positive"] for r in results])),
            "long_pnl_total": round(sum(r["long_pnl"] for r in results), 2),
            "short_pnl_total": round(sum(r["short_pnl"] for r in results), 2),
        }
        print(f"  {family:<24}{len(sh):>3}{med:>+9.3f}{bar:>7.2f}{sh.max():>+9.3f}"
              f"{sh.min():>+9.3f}{npass:>4}/{len(sh)}  "
              f"{'COHERENT' if coherent else 'below bar -- discard'}")

    # ---- STAGE 5: direction balance --------------------------------------
    print(f"\n{SUB}\nSTAGE 5 -- DIRECTION BALANCE\n{SUB}")
    print("  Both sides must be profitable. A long side carrying a losing short")
    print("  side is gold's drift wearing a costume, not a trend mechanism.\n")
    for family, s in summary.items():
        lp, sp = s["long_pnl_total"], s["short_pnl_total"]
        ok = lp > 0 and sp > 0
        print(f"  {family:<24} long {lp:>+12,.0f}   short {sp:>+12,.0f}   "
              f"{'BALANCED' if ok else 'ONE-SIDED -- D1 mechanism not demonstrated'}")
        s["direction_balanced"] = bool(ok)

    # ---- registry ---------------------------------------------------------
    reg = args.outdir / "trial_registry.csv"
    exists = reg.exists()
    with reg.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REGISTRY_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)
    total = sum(1 for _ in reg.open(encoding="utf-8")) - 1
    print(f"\n{SUB}\n  Logged {len(rows)} trials -> {reg}   (registry total: {total})")

    (args.outdir / "daily_study_summary.json").write_text(
        json.dumps({"window": window, "bars": {"long": BAR_LONG, "neutral": BAR_NEUTRAL},
                    "families": summary}, indent=2))

    print(f"\n{SEP}")
    real = [f for f in survivors
            if summary[f]["direction_balanced"] and summary[f]["median_regimes_positive"] >= 3]
    if real:
        print(f"  SURVIVORS (coherent + balanced + regime-robust): {real}")
        print("  Next: parameter surface, then walk-forward.")
    elif survivors:
        print(f"  Coherent but failed balance/regime: {survivors}")
        print("  Not tradeable as-is. Investigate before proceeding.")
    else:
        print("  NO family cleared Stage 2.")
        print("  Per the stopping rule: trade nothing, revise hypotheses,")
        print("  do NOT lower the bar.")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
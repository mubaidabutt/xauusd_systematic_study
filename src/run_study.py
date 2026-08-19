#!/usr/bin/env python3
"""
XAUUSD Research Project -- Step 5: Study Runner
================================================

Runs every pre-registered variant on the DEVELOPMENT window only, logs each
trial to the registry, and applies stages 0-3 from the pre-registration.

What this deliberately does NOT do
----------------------------------
It does not touch 2024-2026. That window is sealed and gets exactly one
evaluation, later, for strategies that have already earned it.

It does not pick a winner. It reports FAMILY-LEVEL evidence, because a single
variant clearing a threshold is weak -- it might be the best of 48 lottery
tickets. A family whose MEDIAN variant is profitable is strong evidence, since
that cannot be produced by finding one magic parameter.

Every run appends to trial_registry.csv. The registry count is what the
multiple-testing correction consumes, so an unlogged trial silently invalidates
every threshold in the pre-registration.

Usage
-----
    python run_study.py
    python run_study.py --dev-end 2024-01-01 --cost-mult 2
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
from strategies import FAMILIES, pct_time_long

SEP = "=" * 78
SUB = "-" * 78

# Locked in PREREGISTRATION.md section 4. Not to be edited after seeing results.
THRESHOLD_LONG = 0.94      # mostly-long: must beat buy & hold
THRESHOLD_NEUTRAL = 0.39   # direction-neutral: must beat best-of-N luck
LONG_EXPOSURE_CUTOFF = 60.0


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


def threshold_for(pct_long: float) -> tuple[float, str]:
    """Exposure decides the bar -- not the strategy author's intent."""
    if pct_long > LONG_EXPOSURE_CUTOFF:
        return THRESHOLD_LONG, "long-biased"
    if pct_long < 100 - LONG_EXPOSURE_CUTOFF:
        return THRESHOLD_NEUTRAL, "short-biased"
    return THRESHOLD_NEUTRAL, "neutral"


# ==========================================================================
# STAGE CHECKS
# ==========================================================================

def stage0_sanity(m: dict, trades: pd.DataFrame) -> tuple[bool, str]:
    """
    Minimum credibility. A strategy whose profit rests on one trade has not
    demonstrated an edge, it has demonstrated one lucky trade.
    """
    if m["n_trades"] < 100:
        return False, f"only {m['n_trades']} trades"
    if m["net_profit"] > 0 and not trades.empty:
        top = trades["net_pnl"].max()
        if top > 0.10 * m["net_profit"]:
            return False, f"top trade = {top / m['net_profit'] * 100:.0f}% of profit"
    return True, ""


def run_variant(bars: pd.DataFrame, fn, params: dict, cfg: Config) -> dict:
    sig = fn(bars, **params)
    res = run_backtest(bars, sig, cfg)
    m = metrics(res)
    m["pct_long"] = round(pct_time_long(sig), 1)
    return m, res.to_frame()


# ==========================================================================
# REGISTRY
# ==========================================================================

REGISTRY_FIELDS = [
    "timestamp", "family", "variant", "params", "window", "cost_mult",
    "n_trades", "sharpe", "pct_long", "threshold", "exposure_class",
    "passed", "max_drawdown_pct", "cagr_pct", "profit_factor", "net_profit",
    "stage0", "stage0_reason",
]


def log_trials(rows: list[dict], path: Path) -> None:
    """Append-only. Every trial ever run, including the failures."""
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REGISTRY_FIELDS)
        if not exists:
            w.writeheader()
        w.writerows(rows)


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--dev-start", default="2018-01-01")
    ap.add_argument("--dev-end", default="2024-01-01",
                    help="EXCLUSIVE. Everything after this is sealed.")
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--outdir", default=Path("./data"), type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"{SEP}\nSTUDY RUNNER -- development window only\n{SEP}")
    bars = pd.read_parquet(locate_parquet(args.file))
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    bars = bars.sort_values("datetime").reset_index(drop=True)

    full_end = bars.datetime.max()
    bars = bars[(bars.datetime >= pd.Timestamp(args.dev_start)) &
                (bars.datetime < pd.Timestamp(args.dev_end))].reset_index(drop=True)
    if len(bars) < 5000:
        sys.exit(f"Only {len(bars)} bars in dev window.")

    print(f"  Dev window : {bars.datetime.min()} -> {bars.datetime.max()}  "
          f"({len(bars):,} bars)")
    print(f"  SEALED     : {args.dev_end} -> {full_end}  (not touched)")

    base = Config()
    cfg = Config(spread_per_oz=base.spread_per_oz * args.cost_mult,
                 slippage_per_oz=base.slippage_per_oz * args.cost_mult)
    print(f"  Cost       : ${cfg.spread_per_oz + cfg.slippage_per_oz:.2f}/oz "
          f"({args.cost_mult:g}x)")

    stamp = datetime.now().isoformat(timespec="seconds")
    window = f"{args.dev_start}..{args.dev_end}"
    rows, family_results = [], {}

    for family, (fn, grid) in FAMILIES.items():
        variants = grid()
        print(f"\n{SUB}\n{family}  ({len(variants)} variants)\n{SUB}")
        print(f"  {'#':<3} {'trades':>7} {'sharpe':>8} {'%long':>7} "
              f"{'bar':>6} {'maxDD':>7} {'CAGR':>7}  verdict")

        results = []
        for vi, params in enumerate(variants):
            try:
                m, trades = run_variant(bars, fn, params, cfg)
            except Exception as exc:
                print(f"  {vi:<3} EXCEPTION: {type(exc).__name__}: {exc}")
                continue

            thr, cls = threshold_for(m["pct_long"])
            s0, s0_reason = stage0_sanity(m, trades)
            passed = bool(s0 and m["sharpe"] > thr)

            verdict = ("PASS" if passed
                       else f"stage0: {s0_reason}" if not s0
                       else "below bar")
            print(f"  {vi:<3} {m['n_trades']:>7} {m['sharpe']:>+8.3f} "
                  f"{m['pct_long']:>6.0f}% {thr:>6.2f} "
                  f"{m['max_drawdown_pct']:>6.1f}% {m['cagr_pct']:>6.1f}%  {verdict}")

            results.append({**m, "passed": passed, "threshold": thr})
            rows.append({
                "timestamp": stamp, "family": family, "variant": vi,
                "params": json.dumps(params), "window": window,
                "cost_mult": args.cost_mult, "n_trades": m["n_trades"],
                "sharpe": m["sharpe"], "pct_long": m["pct_long"],
                "threshold": thr, "exposure_class": cls, "passed": passed,
                "max_drawdown_pct": m["max_drawdown_pct"],
                "cagr_pct": m["cagr_pct"], "profit_factor": m["profit_factor"],
                "net_profit": m["net_profit"],
                "stage0": s0, "stage0_reason": s0_reason,
            })
        family_results[family] = results

    # ---- STAGE 3: family coherence --------------------------------------
    print(f"\n{SEP}\nSTAGE 3 -- FAMILY COHERENCE (the number that matters)\n{SEP}")
    print("  One variant clearing the bar is weak evidence -- it may be the best")
    print("  of 48 lottery tickets. A family whose MEDIAN variant is profitable")
    print("  cannot be produced by finding one magic parameter.\n")
    print(f"  {'family':<24} {'n':>3} {'median':>8} {'mean':>8} {'best':>8} "
          f"{'worst':>8} {'passed':>7}  verdict")

    summary = {}
    for family, results in family_results.items():
        if not results:
            continue
        sh = np.array([r["sharpe"] for r in results])
        npass = sum(1 for r in results if r["passed"])
        median = float(np.median(sh))
        coherent = median > 0
        summary[family] = {
            "n_variants": len(sh), "median_sharpe": round(median, 3),
            "mean_sharpe": round(float(sh.mean()), 3),
            "best_sharpe": round(float(sh.max()), 3),
            "worst_sharpe": round(float(sh.min()), 3),
            "n_passed": npass, "stage3_coherent": coherent,
        }
        print(f"  {family:<24} {len(sh):>3} {median:>+8.3f} {sh.mean():>+8.3f} "
              f"{sh.max():>+8.3f} {sh.min():>+8.3f} {npass:>4}/{len(sh)}  "
              f"{'COHERENT' if coherent else 'incoherent -- discard'}")

    # ---- H3 control ------------------------------------------------------
    sess = family_results.get("H3_session_breakout", [])
    if sess:
        variants = FAMILIES["H3_session_breakout"][1]()
        real = [r["sharpe"] for r, p in zip(sess, variants)
                if p.get("session_hour") in (7, 13)]
        ctrl = [r["sharpe"] for r, p in zip(sess, variants)
                if p.get("session_hour") not in (7, 13)]
        if real and ctrl:
            print(f"\n{SUB}\nH3 CONTROL -- do session hours actually matter?\n{SUB}")
            print(f"  London/NY hours (07,13) : median Sharpe {np.median(real):+.3f}"
                  f"  (n={len(real)})")
            print(f"  Arbitrary control hours : median Sharpe {np.median(ctrl):+.3f}"
                  f"  (n={len(ctrl)})")
            gap = float(np.median(real) - np.median(ctrl))
            print(f"  Difference              : {gap:+.3f}")
            print(f"  >> {'Session effect present.' if gap > 0.2 else 'NO session effect -- H3 falsified.'}")
            summary["H3_control_gap"] = round(gap, 3)

    reg = args.outdir / "trial_registry.csv"
    log_trials(rows, reg)
    total = sum(1 for _ in reg.open(encoding="utf-8")) - 1
    print(f"\n{SUB}\n  Logged {len(rows)} trials -> {reg}")
    print(f"  Registry now holds {total} trials in total.")
    print("  That count feeds the multiple-testing correction. Keep it honest.")

    (args.outdir / "study_summary.json").write_text(json.dumps({
        "window": window, "cost_mult": args.cost_mult,
        "thresholds": {"long": THRESHOLD_LONG, "neutral": THRESHOLD_NEUTRAL},
        "families": summary,
    }, indent=2))

    survivors = [f for f, s in summary.items()
                 if isinstance(s, dict) and s.get("stage3_coherent")]
    print(f"\n{SEP}")
    if survivors:
        print(f"  Families surviving Stage 3: {survivors}")
        print("  Next: cost stress at 2x, then parameter-surface inspection.")
    else:
        print("  NO family survived Stage 3.")
        print("  Per the pre-registration stopping rule, the correct action is to")
        print("  trade nothing and revise the hypotheses -- not to lower the bar.")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
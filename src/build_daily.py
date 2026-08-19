#!/usr/bin/env python3
"""
XAUUSD Research Project -- Step 6: Build the daily dataset
===========================================================

Why daily, and why now
----------------------
Measured on the H1 study: cost drag was 0.14 Sharpe for variants trading <300
times, and 0.80 Sharpe for variants trading >700 times. The strategies with the
strongest raw signal were exactly the ones that paid it all away. Daily bars
held for weeks capture $100-300 moves against a $0.40 cost -- the drag becomes
negligible rather than decisive.

The bigger prize is history. We discarded the old CSV because it runs on an
EET/EEST clock, so its H4 candles were 2-3 hours misaligned with the ones
Exness builds. That was fatal on H1/H4. On DAILY bars held for weeks, a
boundary shift changes the exact daily close slightly but does not change
whether gold trended up for six weeks. So the CSV becomes usable again, and it
carries 2004-2018 -- including the 2011-2015 bear market (-45% over four years)
that the MT5 data does not reach.

The 2018-2023 window we have been testing contains no sustained bear market at
all. That is very likely why a trend filter kept coming out 50% long instead of
the predicted 60%+: we were testing a trend hypothesis on data with weak
trends, in one direction only.

What this does
--------------
    1. Loads the EET CSV, shifts it to UTC, resamples to daily
    2. Loads the MT5 parquet (already UTC), resamples to daily
    3. Compares the two over their overlap -- the honesty check
    4. Writes a merged series: CSV for the early years, MT5 from 2018 on

Usage
-----
    python build_daily.py --csv "XAUUSD Data/XAUUSD_1H.csv"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEP = "=" * 76
SUB = "-" * 76

# MT5 is broker-native and authoritative wherever it exists.
MT5_PRIORITY_FROM = "2018-01-01"


def load_csv_eet(path: Path) -> pd.DataFrame:
    """
    Read the archived H1 CSV and convert its clock to UTC.

    The audit established this feed runs on EET/EEST -- UTC+2 in winter, UTC+3
    in summer, following EU DST. Rather than hardcode an offset, we localise
    with the real timezone rules so both seasons are handled correctly.
    """
    print(f"  Reading {path}")
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}

    date_col = cols.get("date") or cols.get("datetime") or cols.get("time")
    if date_col is None:
        sys.exit(f"No date column found. Columns: {list(df.columns)}")

    dt = pd.to_datetime(df[date_col], format="mixed", dayfirst=False, errors="coerce")
    out = pd.DataFrame({"datetime_eet": dt})
    for f in ("open", "high", "low", "close"):
        src = cols.get(f)
        if src is None:
            sys.exit(f"Missing '{f}' column. Columns: {list(df.columns)}")
        out[f] = pd.to_numeric(df[src], errors="coerce")

    out = out.dropna().sort_values("datetime_eet").reset_index(drop=True)
    print(f"  {len(out):,} H1 bars, {out.datetime_eet.min()} -> {out.datetime_eet.max()} (EET)")

    # ambiguous='NaT' / nonexistent='NaT' drop the duplicated autumn hour and
    # the missing spring hour rather than guessing at them. A handful of bars
    # per year, and guessing would corrupt the clock we are trying to fix.
    local = out["datetime_eet"].dt.tz_localize(
        "Europe/Bucharest", ambiguous="NaT", nonexistent="NaT")
    out["datetime"] = local.dt.tz_convert("UTC").dt.tz_localize(None)
    dropped = out["datetime"].isna().sum()
    if dropped:
        print(f"  Dropped {dropped} bars at DST transitions (ambiguous/nonexistent)")
    return out.dropna(subset=["datetime"]).drop(columns=["datetime_eet"])


def load_mt5(path: Path) -> pd.DataFrame:
    print(f"  Reading {path}")
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[["datetime", "open", "high", "low", "close"]].sort_values("datetime")
    print(f"  {len(df):,} H1 bars, {df.datetime.min()} -> {df.datetime.max()} (UTC)")
    return df.reset_index(drop=True)


def to_daily(h1: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Aggregate H1 -> daily on UTC calendar days.

    Gold's trading day runs roughly 22:00 UTC to 21:00 UTC, so a UTC calendar
    day splits it. That is deliberate and harmless here: we are measuring
    multi-week trends, and a consistent daily boundary matters far more than
    matching the broker's exact session edge. What would NOT be acceptable is
    an inconsistent boundary, which is what the EET feed gave us on H4.
    """
    d = (h1.set_index("datetime")
           .resample("1D")
           .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
           .dropna())
    d = d[d.index.dayofweek < 5]  # drop weekend stubs
    print(f"  {label}: {len(d):,} daily bars  {d.index.min().date()} -> {d.index.max().date()}")
    return d


def compare_overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """
    The honesty check.

    If the converted CSV and the broker feed agree closely where they overlap,
    the CSV's earlier years can be trusted as an extension. If they diverge,
    the extension is not safe and we work with MT5 alone.
    """
    idx = a.index.intersection(b.index)
    if len(idx) < 100:
        print("  Insufficient overlap to validate.")
        return {"overlap_days": int(len(idx)), "usable": False}

    ca, cb = a.loc[idx, "close"], b.loc[idx, "close"]
    diff_pct = ((ca - cb) / cb * 100).abs()
    ra = ca.pct_change().dropna()
    rb = cb.pct_change().dropna()
    common = ra.index.intersection(rb.index)
    corr = float(np.corrcoef(ra.loc[common], rb.loc[common])[0, 1])

    out = {
        "overlap_days": int(len(idx)),
        "median_abs_diff_pct": round(float(diff_pct.median()), 4),
        "p95_abs_diff_pct": round(float(diff_pct.quantile(0.95)), 4),
        "max_abs_diff_pct": round(float(diff_pct.max()), 4),
        "daily_return_correlation": round(corr, 5),
    }
    print(f"\n  Overlap            : {out['overlap_days']:,} days")
    print(f"  Median |close diff|: {out['median_abs_diff_pct']:.3f}%")
    print(f"  p95 |close diff|   : {out['p95_abs_diff_pct']:.3f}%")
    print(f"  Return correlation : {out['daily_return_correlation']:.4f}")

    # Different brokers, different feeds -- exact equality is not expected.
    # High return correlation is what matters for trend research.
    usable = corr > 0.95 and out["median_abs_diff_pct"] < 0.5
    out["usable"] = bool(usable)
    print(f"\n  >> {'CSV extension is TRUSTWORTHY.' if usable else 'CSV DIVERGES -- do not extend.'}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="archived H1 CSV (EET clock)")
    ap.add_argument("--mt5", default=None, help="MT5 parquet (UTC)")
    ap.add_argument("--outdir", default=Path("./data"), type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    mt5_path = Path(args.mt5) if args.mt5 else None
    if mt5_path is None:
        for base in (Path("data"), Path(".")):
            hits = sorted(base.glob("*_h1_mt5.parquet"))
            if hits:
                mt5_path = hits[0]
                break
    if mt5_path is None or not mt5_path.exists():
        sys.exit("MT5 parquet not found. Pass --mt5 path/to/XAUUSDm_h1_mt5.parquet")

    print(f"{SEP}\nLOADING MT5 (authoritative, UTC)\n{SEP}")
    mt5_daily = to_daily(load_mt5(mt5_path), "MT5 daily")

    if not args.csv:
        print("\n  No --csv given. Writing MT5-only daily series.")
        mt5_daily.reset_index().to_parquet(args.outdir / "xauusd_daily.parquet", index=False)
        print(f"  -> {args.outdir / 'xauusd_daily.parquet'}")
        return 0

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path.resolve()}")

    print(f"\n{SEP}\nLOADING ARCHIVED CSV (EET -> UTC)\n{SEP}")
    csv_daily = to_daily(load_csv_eet(csv_path), "CSV daily")

    print(f"\n{SUB}\nOVERLAP VALIDATION\n{SUB}")
    report = compare_overlap(csv_daily, mt5_daily)

    print(f"\n{SUB}\nMERGING\n{SUB}")
    if report.get("usable"):
        cutoff = pd.Timestamp(MT5_PRIORITY_FROM)
        early = csv_daily[csv_daily.index < cutoff]
        late = mt5_daily[mt5_daily.index >= cutoff]
        merged = pd.concat([early, late]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        print(f"  CSV  {early.index.min().date()} -> {early.index.max().date()}  ({len(early):,} days)")
        print(f"  MT5  {late.index.min().date()} -> {late.index.max().date()}  ({len(late):,} days)")
        source = "merged"
    else:
        merged = mt5_daily
        print("  Using MT5 only -- CSV failed validation.")
        source = "mt5_only"

    years = (merged.index.max() - merged.index.min()).days / 365.25
    print(f"\n  MERGED: {len(merged):,} daily bars over {years:.1f} years")
    print(f"          {merged.index.min().date()} -> {merged.index.max().date()}")

    print(f"\n  Coverage by year:")
    counts = merged.index.year.value_counts().sort_index()
    for y, n in counts.items():
        flag = "  <-- thin" if n < 200 else ""
        print(f"    {y}  {n:>4} days{flag}")

    out = args.outdir / "xauusd_daily.parquet"
    merged.reset_index().rename(columns={"index": "datetime"}).to_parquet(out, index=False)
    (args.outdir / "daily_build_report.json").write_text(
        json.dumps({"source": source, "overlap": report,
                    "n_days": int(len(merged)), "years": round(years, 2)}, indent=2))
    print(f"\n  -> {out}\n{SEP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
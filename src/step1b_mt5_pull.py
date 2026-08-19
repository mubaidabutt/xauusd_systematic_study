#!/usr/bin/env python3
"""
XAUUSD Research Project -- Step 1b: Pull broker-native history from Exness MT5
==============================================================================

Why this exists
---------------
The CSV we audited was on an EET/EEST clock (UTC+2 winter / UTC+3 summer). Its
H4 candles are 2-3 hours offset from the ones Exness actually builds, so any
higher-timeframe signal derived from it describes candles the live EA will
never see. Pulling straight from the broker eliminates that class of bug
entirely: the bars in the backtest become the bars in production.

It also hands us something the CSV never had -- a per-bar `spread` column
recorded by the terminal. That converts the cost model from an assumption into
a measurement, which matters enormously on a Standard account where the whole
cost is baked into the spread.

What it does
------------
    1. Connects to a running MT5 terminal
    2. Resolves the real symbol name (Exness uses suffixes: XAUUSDm, XAUUSDz...)
    3. Probes how far back H1 history actually goes, year by year
    4. Dumps the full symbol specification -- contract size, swaps, stops level,
       lot limits, digits -- everything Step 2's cost model needs
    5. Pulls all available history in yearly chunks (dodges the max-bars cap)
    6. Validates it: break hours must be [21, 22] for a genuine UTC+0 feed
    7. Writes parquet + a JSON spec sheet

IMPORTANT: the MetaTrader5 package is Windows-only. On macOS/Linux you need a
Windows VM or Wine. The terminal must be installed, logged in, and running.

Usage
-----
    pip install MetaTrader5 pandas numpy pyarrow
    python step1b_mt5_pull.py
    python step1b_mt5_pull.py --symbol XAUUSDm --outdir ./data

If history looks shallow, open an H1 XAUUSD chart in the terminal and hold
Home / scroll left to force the terminal to download deeper history from the
server, then re-run. MT5 fetches lazily and will not backfill on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit(
        "MetaTrader5 package not found.\n"
        "  pip install MetaTrader5\n"
        "Note: this package only runs on Windows. On macOS/Linux use a VM or Wine."
    )

SEP = "=" * 78
SUB = "-" * 78
PROBE_START_YEAR = 2000


# ==========================================================================
# CONNECTION & SYMBOL RESOLUTION
# ==========================================================================

def connect(terminal_path: str | None) -> None:
    ok = mt5.initialize(path=terminal_path) if terminal_path else mt5.initialize()
    if not ok:
        sys.exit(f"MT5 initialize() failed: {mt5.last_error()}\n"
                 "Is the terminal installed, logged in, and running?")
    info, acct = mt5.terminal_info(), mt5.account_info()
    print(f"  Terminal : {info.name} (build {info.build})")
    print(f"  Connected: {info.connected}   Trade allowed: {info.trade_allowed}")
    if acct:
        print(f"  Account  : {acct.login} @ {acct.server}  [{acct.company}]")
        print(f"  Currency : {acct.currency}   Leverage: 1:{acct.leverage}")
        if "exness" not in (acct.company or "").lower():
            print("  >> WARNING: this is not an Exness account. Server timezone may")
            print("     differ from UTC+0 and the validation below will flag it.")


def resolve_symbol(preferred: str | None) -> str:
    """
    Find the actual gold symbol. Exness appends account-type suffixes, so the
    plain name often does not exist on your particular server.
    """
    if preferred:
        if mt5.symbol_info(preferred) is None:
            sys.exit(f"Symbol {preferred!r} not found on this server.")
        mt5.symbol_select(preferred, True)
        return preferred

    candidates = [s.name for s in mt5.symbols_get() or []
                  if s.name.upper().startswith("XAUUSD")]
    if not candidates:
        sys.exit("No XAUUSD* symbol found. Check Market Watch / your account type.")

    # Prefer the exact name, then the shortest suffix.
    candidates.sort(key=lambda n: (n.upper() != "XAUUSD", len(n)))
    chosen = candidates[0]
    if len(candidates) > 1:
        print(f"  Gold symbols available: {candidates}")
    print(f"  Using symbol: {chosen}")
    mt5.symbol_select(chosen, True)
    return chosen


def symbol_specification(symbol: str) -> dict:
    """
    Everything Step 2 needs to model cost and sizing correctly. Reading these
    from the broker beats hardcoding them -- contract size and stops level in
    particular differ between account types.
    """
    s = mt5.symbol_info(symbol)
    if s is None:
        sys.exit(f"symbol_info({symbol}) returned None")
    tick = mt5.symbol_info_tick(symbol)

    spec = {
        "symbol": s.name,
        "description": s.description,
        "digits": s.digits,
        "point": s.point,
        "tick_size": s.trade_tick_size,
        "tick_value": s.trade_tick_value,
        "contract_size": s.trade_contract_size,
        "volume_min": s.volume_min,
        "volume_max": s.volume_max,
        "volume_step": s.volume_step,
        "stops_level_points": s.trade_stops_level,
        "freeze_level_points": s.trade_freeze_level,
        "swap_long": s.swap_long,
        "swap_short": s.swap_short,
        "swap_mode": s.swap_mode,
        "swap_rollover3days": s.swap_rollover3days,
        "current_spread_points": s.spread,
        "spread_float": s.spread_float,
        "currency_profit": s.currency_profit,
        "currency_margin": s.currency_margin,
    }
    if tick:
        spec["current_bid"] = tick.bid
        spec["current_ask"] = tick.ask
        spec["current_spread_price"] = round(tick.ask - tick.bid, s.digits)

    print(f"\n{SUB}\nSYMBOL SPECIFICATION  (feeds the Step 2 cost model)\n{SUB}")
    print(f"  Contract size        : {spec['contract_size']} oz per lot")
    print(f"  Digits / point       : {spec['digits']} / {spec['point']}")
    print(f"  Min / step / max lot : {spec['volume_min']} / {spec['volume_step']} / {spec['volume_max']}")
    print(f"  Min stop distance    : {spec['stops_level_points']} points "
          f"(= ${spec['stops_level_points'] * spec['point']:.2f})")
    print(f"  Swap long / short    : {spec['swap_long']} / {spec['swap_short']}  (mode {spec['swap_mode']})")
    print(f"  Triple swap day      : {spec['swap_rollover3days']}  (0=Mon .. 3=Wed)")
    if tick:
        print(f"  Live spread right now: ${spec['current_spread_price']} "
              f"({spec['current_spread_points']} points)")
    return spec


# ==========================================================================
# HISTORY
# ==========================================================================

def probe_depth(symbol: str, timeframe: int) -> int | None:
    """
    Find the earliest year with real H1 data by walking forward from 2000.

    This is the number that decides whether the archived CSV is redundant or
    load-bearing. MT5 downloads history lazily, so a thin result here usually
    means the terminal has not cached it rather than that the server lacks it.
    """
    print(f"\n{SUB}\nPROBING HISTORY DEPTH\n{SUB}")
    this_year = datetime.now().year
    for year in range(PROBE_START_YEAR, this_year + 1):
        rates = mt5.copy_rates_range(symbol, timeframe,
                                     datetime(year, 1, 1), datetime(year, 12, 31, 23, 59))
        n = 0 if rates is None else len(rates)
        if n > 100:
            print(f"  Earliest year with usable H1 data: {year}  ({n:,} bars)")
            return year
        if n:
            print(f"  {year}: {n} bars (too thin, skipping)")
    print("  No usable history found at any year.")
    return None


def fetch_history(symbol: str, timeframe: int, start_year: int) -> pd.DataFrame:
    """Pull year by year -- a single whole-range call hits the terminal's bar cap."""
    print(f"\n{SUB}\nFETCHING HISTORY\n{SUB}")
    frames = []
    for year in range(start_year, datetime.now().year + 1):
        rates = mt5.copy_rates_range(symbol, timeframe,
                                     datetime(year, 1, 1),
                                     min(datetime(year, 12, 31, 23, 59), datetime.now()))
        if rates is None or not len(rates):
            print(f"  {year}: no data")
            continue
        frames.append(pd.DataFrame(rates))
        print(f"  {year}: {len(rates):>6,} bars")

    if not frames:
        sys.exit("No data retrieved.")

    df = pd.concat(frames, ignore_index=True)
    # MT5 stamps bars in SERVER time as a unix epoch. Exness servers are UTC+0,
    # so this is genuinely UTC -- which the validation below confirms rather
    # than assumes.
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = (df.drop(columns=["time"])
            .drop_duplicates(subset="datetime")
            .sort_values("datetime")
            .reset_index(drop=True))
    df = df.rename(columns={"tick_volume": "tick_volume", "real_volume": "real_volume"})
    return df


# ==========================================================================
# VALIDATION
# ==========================================================================

def validate(df: pd.DataFrame, spec: dict) -> dict:
    """
    Re-run the checks that mattered in Step 1a, plus the UTC+0 confirmation.

    The break-hour test is the important one. Exness rollover follows US DST --
    21:00 UTC in summer, 22:00 UTC in winter -- and MT5 labels bars by open
    time, so a true UTC+0 open-labelled feed is missing hour 21 in summer and
    hour 22 in winter. Seeing exactly [21, 22] proves the clock is right.
    """
    print(f"\n{SUB}\nVALIDATION\n{SUB}")
    dt = df["datetime"]
    out: dict = {}

    monthly = {}
    for m in range(1, 13):
        sub = dt[dt.dt.month == m]
        if len(sub) > 200:
            monthly[m] = int(sub.dt.hour.value_counts().reindex(range(24), fill_value=0).idxmin())
    breaks = sorted(set(monthly.values()))
    out["break_hours"] = breaks
    out["is_utc"] = breaks == [21, 22]

    print(f"  Break hours observed : {breaks}   (UTC+0 expects [21, 22])")
    if out["is_utc"]:
        print("  [  OK  ] Confirmed UTC+0. H4 candles will match the live terminal exactly.")
    else:
        print("  [ FAIL ] NOT UTC+0. Do not proceed -- resolve before building anything.")

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    bad = int(((h < l) | (h < o) | (h < c) | (l > o) | (l > c)).sum())
    out["ohlc_violations"] = bad
    out["duplicate_timestamps"] = int(dt.duplicated().sum())
    out["n_sunday_bars"] = int((dt.dt.dayofweek == 6).sum())
    print(f"  OHLC violations      : {bad}")
    print(f"  Duplicate timestamps : {out['duplicate_timestamps']}")

    gaps = dt.diff().dt.total_seconds() / 3600
    holes = df[gaps > 72]
    out["n_holes_over_72h"] = int(len(holes))
    print(f"  Holes > 72h          : {len(holes)}")
    for i in holes.index[:10]:
        print(f"      {dt[i-1]} -> {dt[i]}   {gaps[i]:>6.0f}h")

    # Empirical spread -- the whole reason a Standard account needs measuring
    # rather than guessing.
    if "spread" in df.columns:
        pt = spec["point"]
        sp = df["spread"] * pt
        out["spread_usd"] = {
            "median": round(float(sp.median()), 3),
            "mean": round(float(sp.mean()), 3),
            "p75": round(float(sp.quantile(0.75)), 3),
            "p95": round(float(sp.quantile(0.95)), 3),
            "p99": round(float(sp.quantile(0.99)), 3),
            "max": round(float(sp.max()), 3),
        }
        print(f"\n  RECORDED SPREAD (USD/oz) -- this replaces guesswork in Step 2:")
        for k, v in out["spread_usd"].items():
            print(f"      {k:>6}: ${v}")
        by_hour = sp.groupby(dt.dt.hour).median().round(3)
        worst = by_hour.nlargest(3)
        print(f"      Widest hours (UTC): " +
              ", ".join(f"{h:02d}:00 ${v}" for h, v in worst.items()))
        out["spread_by_hour_median"] = {int(k): float(v) for k, v in by_hour.items()}

    out["first_bar"] = str(dt.min())
    out["last_bar"] = str(dt.max())
    out["n_bars"] = int(len(df))
    return out


# ==========================================================================
# MAIN
# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Pull XAUUSD H1 history from Exness MT5")
    ap.add_argument("--symbol", default=None, help="override symbol (e.g. XAUUSDm)")
    ap.add_argument("--terminal", default=None, help="path to terminal64.exe")
    ap.add_argument("--outdir", default=Path("./data"), type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"{SEP}\nCONNECTING TO MT5\n{SEP}")
    connect(args.terminal)

    try:
        symbol = resolve_symbol(args.symbol)
        spec = symbol_specification(symbol)

        start = probe_depth(symbol, mt5.TIMEFRAME_H1)
        if start is None:
            return 1

        df = fetch_history(symbol, mt5.TIMEFRAME_H1, start)
        report = validate(df, spec)
    finally:
        mt5.shutdown()

    out_parquet = args.outdir / f"{symbol}_h1_mt5.parquet"
    df.to_parquet(out_parquet, index=False)
    (args.outdir / f"{symbol}_spec.json").write_text(
        json.dumps({"specification": spec, "validation": report}, indent=2, default=str))

    print(f"\n{SEP}")
    print(f"Wrote {len(df):,} bars  ({report['first_bar']} -> {report['last_bar']})")
    print(f"  -> {out_parquet}")
    print(f"  -> {args.outdir / f'{symbol}_spec.json'}")
    years = (pd.Timestamp(report["last_bar"]) - pd.Timestamp(report["first_bar"])).days / 365.25
    print(f"\nHistory depth: {years:.1f} years.")
    if years < 8:
        print(">> Shallow. Try forcing a deeper download (open the H1 chart, hold Home),")
        print("   then re-run. If it stays shallow, the archived CSV is load-bearing")
        print("   and we WILL need the EET->UTC converter for pre-broker history.")
    else:
        print(">> Deep enough to stand alone. The CSV becomes a cross-check only.")
    print(SEP)
    return 0
if __name__ == "__main__":
    sys.exit(main())



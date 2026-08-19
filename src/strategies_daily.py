#!/usr/bin/env python3
"""
XAUUSD Research Project -- Step 8: Daily strategy families
===========================================================

Two families, per PREREGISTRATION_DAILY.md.

Both are built SYMMETRIC by design. The neutral bar is +0.705 while the
long-biased bar is +1.050, and the underlying mechanism is identical in either
direction. Taking shorts is what makes the hypothesis testable: the short-only
null p95 is +0.045, so a profitable short side is doing something random
shorts essentially never do.

Same contract as the H1 families: a signal on row i uses bars 0..i only, and
the engine applies the execution delay itself.

Entries are EVENTS, not states. The H1 study failed partly because
`RSI < 50` -- true a third of the time -- was used as an entry, which kept the
strategy permanently in the market and made entry timing arbitrary. Every
condition below fires on a transition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import atr, ema, sma


def _empty(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "direction": np.zeros(n, dtype=int),
        "stop_distance": np.zeros(n),
        "trail_distance": np.zeros(n),
        "max_hold": np.zeros(n, dtype=int),
        "exit_signal": np.zeros(n, dtype=bool),
    })


def _cross(series: pd.Series, other: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(crossed_above, crossed_below) -- transitions only, never states."""
    a, b = series.to_numpy(), other.to_numpy()
    ap, bp = np.roll(a, 1), np.roll(b, 1)
    ap[0] = bp[0] = np.nan
    up = (a > b) & (ap <= bp)
    dn = (a < b) & (ap >= bp)
    return np.nan_to_num(up, nan=False).astype(bool), np.nan_to_num(dn, nan=False).astype(bool)


def pct_signals_long(sig: pd.DataFrame) -> float:
    d = sig["direction"].to_numpy()
    a = d[d != 0]
    return float((a == 1).mean() * 100) if a.size else 0.0


# ==========================================================================
# D1 -- BIDIRECTIONAL TREND PERSISTENCE
# ==========================================================================

def trend_persistence_daily(bars: pd.DataFrame, *, fast: int = 50,
                            slow: int = 200, atr_period: int = 14,
                            stop_mult: float = 3.0, trail_mult: float = 4.0,
                            max_hold: int = 0) -> pd.DataFrame:
    """
    HYPOTHESIS D1: macro flows make multi-month trends persist, symmetrically.

    Entry on the moving-average cross -- a discrete regime-change event, maybe
    3-6 times a year. Stops and trails scale with ATR so risk is constant in
    volatility units across a sample where gold ranges from $380 to $5,500.

    A trailing stop rather than a fixed target: trend persistence says the move
    runs further than expected, so the exit should be open-ended. Capping it
    with a fixed target would test a different hypothesis.

    Expected exposure: NEUTRAL (both directions) -> bar +0.705.
    """
    n = len(bars)
    sig = _empty(n)
    f = sma(bars["close"], fast)
    s = sma(bars["close"], slow)
    a = atr(bars, atr_period)

    up, dn = _cross(f, s)
    av = a.to_numpy()
    valid = np.isfinite(av) & (av > 0) & np.isfinite(f.to_numpy()) & np.isfinite(s.to_numpy())

    sig.loc[valid & up, "direction"] = 1
    sig.loc[valid & dn, "direction"] = -1
    sig["stop_distance"] = np.where(valid, av * stop_mult, 0.0)
    sig["trail_distance"] = np.where(valid, av * trail_mult, 0.0)
    sig["max_hold"] = max_hold
    return sig


# ==========================================================================
# D2 -- LONG-HORIZON BREAKOUT
# ==========================================================================

def horizon_breakout(bars: pd.DataFrame, *, lookback: int = 60,
                     atr_period: int = 14, stop_mult: float = 3.0,
                     trail_mult: float = 4.0, max_hold: int = 0,
                     confirm: int = 1) -> pd.DataFrame:
    """
    HYPOTHESIS D2: multi-month range breaks mark genuine regime transitions.

    lookback is in TRADING DAYS: 60 ~ 3 months, 120 ~ 6 months.

    The fresh-break requirement matters. Once price clears a 60-day high it
    stays cleared for many bars, so `close > high` is a state, not an event.
    Requiring the prior bar to have been inside the range isolates the break
    itself -- the same distinction that broke the H1 study.
    """
    n = len(bars)
    sig = _empty(n)
    close = bars["close"]
    hi = bars["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    lo = bars["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    a = atr(bars, atr_period)
    av = a.to_numpy()

    above = close > hi
    below = close < lo
    # `confirm` consecutive closes beyond the level, with the bar before the
    # run still inside -- filters one-day pokes without waiting so long the
    # move is over.
    if confirm > 1:
        run_up = above.rolling(confirm, min_periods=confirm).sum() == confirm
        run_dn = below.rolling(confirm, min_periods=confirm).sum() == confirm
        fresh_up = run_up & ~above.shift(confirm).fillna(False)
        fresh_dn = run_dn & ~below.shift(confirm).fillna(False)
    else:
        fresh_up = above & ~above.shift(1).fillna(False)
        fresh_dn = below & ~below.shift(1).fillna(False)

    valid = np.isfinite(av) & (av > 0) & np.isfinite(hi.to_numpy()) & np.isfinite(lo.to_numpy())
    sig.loc[valid & fresh_up.fillna(False).to_numpy(), "direction"] = 1
    sig.loc[valid & fresh_dn.fillna(False).to_numpy(), "direction"] = -1
    sig["stop_distance"] = np.where(valid, av * stop_mult, 0.0)
    sig["trail_distance"] = np.where(valid, av * trail_mult, 0.0)
    sig["max_hold"] = max_hold
    return sig


# ==========================================================================
# GRIDS -- 20 per family, horizons in months not arbitrary integers
# ==========================================================================

def grid_trend_daily() -> list[dict]:
    out = []
    for fast, slow in ((20, 100), (50, 200), (20, 200), (50, 100)):   # ~1mo/5mo etc
        for stop in (2.0, 3.0):
            for trail in (3.0, 5.0):
                out.append({"fast": fast, "slow": slow, "stop_mult": stop,
                            "trail_mult": trail, "max_hold": 0})
    for stop in (2.0, 3.0):                                          # no-trail control
        for slow in (100, 200):
            out.append({"fast": 50, "slow": slow, "stop_mult": stop,
                        "trail_mult": 0.0, "max_hold": 60})
    return out[:20]


def grid_breakout_daily() -> list[dict]:
    out = []
    for lb in (60, 120):                     # ~3mo, ~6mo
        for stop in (2.0, 3.0):
            for trail in (3.0, 5.0):
                for confirm in (1, 2):
                    out.append({"lookback": lb, "stop_mult": stop,
                                "trail_mult": trail, "confirm": confirm,
                                "max_hold": 0})
    for lb in (40, 250):                     # ~2mo, ~1yr horizon controls
        for stop in (2.0, 3.0):
            out.append({"lookback": lb, "stop_mult": stop, "trail_mult": 4.0,
                        "confirm": 1, "max_hold": 0})
    return out[:20]


FAMILIES_DAILY = {
    "D1_trend_persistence": (trend_persistence_daily, grid_trend_daily),
    "D2_horizon_breakout": (horizon_breakout, grid_breakout_daily),
}
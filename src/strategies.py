#!/usr/bin/env python3
"""
XAUUSD Research Project -- Step 4: Strategy families
=====================================================

Three families, one per pre-registered hypothesis. Each is a function that
turns bars into a signals DataFrame the engine can execute.

CONTRACT
--------
Every signal on row i must be computable from bars 0..i ONLY. The engine
applies the one-bar execution delay itself, so strategies here may use bar i's
close -- but never bar i+1's anything.

The single most common way to break this: computing an indicator, then calling
.bfill(). That fills leading NaNs with LATER values, which is lookahead
wearing a disguise. Leave the NaNs; emit direction 0 where the lookback is not
yet satisfied.

H4 AGGREGATION
--------------
Higher-timeframe values are built by resampling to 4h, computing on CLOSED
candles, then shifting one candle forward before mapping back to H1. Without
that shift, every H1 bar inside a forming H4 candle sees that candle's final
value -- a lookahead bug worth several Sharpe points and completely invisible
in the equity curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import atr, ema, rsi, sma


# ==========================================================================
# HELPERS
# ==========================================================================

def _empty_signals(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "direction": np.zeros(n, dtype=int),
        "stop_distance": np.zeros(n),
        "trail_distance": np.zeros(n),
        "max_hold": np.zeros(n, dtype=int),
        "exit_signal": np.zeros(n, dtype=bool),
    })


def h4_series(bars: pd.DataFrame, fn, *args) -> pd.Series:
    """
    Compute an indicator on H4 candles and map it back to H1, lag-safe.

    The shift(1) is the critical line. An H4 candle covering 08:00-11:59 only
    becomes known at 12:00. Without the shift, the H1 bar at 08:00 would see
    the completed candle's close -- four hours of future information.
    """
    idx = pd.to_datetime(bars["datetime"])
    h4 = (bars.set_index(idx)[["open", "high", "low", "close"]]
              .resample("4h", origin="start_day")
              .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
              .dropna())
    value = fn(h4, *args).shift(1)
    return value.reindex(idx, method="ffill").to_numpy()


def pct_time_long(signals: pd.DataFrame) -> float:
    """
    Directional exposure -- decides which pre-registered threshold applies.

    Measured on emitted signals, not on realised holding time, so it is known
    before the backtest runs and cannot be tuned after seeing results.
    """
    d = signals["direction"].to_numpy()
    active = d[d != 0]
    if active.size == 0:
        return 0.0
    return float((active == 1).mean() * 100)


# ==========================================================================
# H1 -- VOLATILITY-SCALED TREND PERSISTENCE
# ==========================================================================

def trend_persistence(bars: pd.DataFrame, *, fast: int = 50, slow: int = 200,
                      atr_period: int = 14, stop_mult: float = 2.5,
                      trail_mult: float = 0.0, max_hold: int = 0,
                      pullback_rsi: float = 50.0) -> pd.DataFrame:
    """
    HYPOTHESIS H1: macro flows make intermediate trends persist.

    Trend defined on H4 (slow-moving macro, not H1 noise), entry timed on an
    H1 pullback so we are not buying an extended move. Stops scale with ATR so
    risk is constant in volatility units rather than dollars -- essential when
    the sample spans gold at $1,200 and gold at $5,500.

    Expected exposure: MOSTLY LONG -> must clear Sharpe +0.94.
    """
    n = len(bars)
    sig = _empty_signals(n)

    h4_fast = h4_series(bars, lambda d, p: sma(d["close"], p), fast)
    h4_slow = h4_series(bars, lambda d, p: sma(d["close"], p), slow)
    r = rsi(bars["close"], 14).to_numpy()
    a = atr(bars, atr_period).to_numpy()

    up = h4_fast > h4_slow
    down = h4_fast < h4_slow
    valid = np.isfinite(h4_fast) & np.isfinite(h4_slow) & np.isfinite(r) & np.isfinite(a) & (a > 0)

    # EVENT, not state. `r < pullback_rsi` is true roughly a third of the time,
    # so it does not mark a pullback -- it marks "we are in the lower half of
    # the RSI range", which describes most bars. Using it as an entry keeps the
    # strategy permanently in the market and makes entry timing arbitrary.
    #
    # The RECOVERY cross is what the hypothesis actually describes: price
    # pulled back against the trend and has now turned back in its direction.
    r_prev = np.roll(r, 1)
    r_prev[0] = np.nan
    long_entry = valid & up & (r > pullback_rsi) & (r_prev <= pullback_rsi)
    short_entry = valid & down & (r < 100 - pullback_rsi) & (r_prev >= 100 - pullback_rsi)

    sig.loc[long_entry, "direction"] = 1
    sig.loc[short_entry, "direction"] = -1
    sig["stop_distance"] = np.where(valid, a * stop_mult, 0.0)
    sig["trail_distance"] = np.where(valid, a * trail_mult, 0.0)
    sig["max_hold"] = max_hold
    return sig


# ==========================================================================
# H2 -- VOLATILITY REGIME REVERSION
# ==========================================================================

def volatility_regime(bars: pd.DataFrame, *, lookback: int = 100,
                      percentile: int = 20, atr_period: int = 14,
                      stop_mult: float = 2.0, trail_mult: float = 2.0,
                      max_hold: int = 48, breakout_lookback: int = 20) -> pd.DataFrame:
    """
    HYPOTHESIS H2: volatility clusters and mean-reverts, independent of direction.

    When ATR sits in the bottom percentile of its own recent history, the
    market is compressed. Compression resolves into expansion. We take
    whichever way it breaks -- the signal is about volatility STATE, and
    direction is decided by price.

    That symmetry is the point. This family should not inherit gold's drift,
    so it competes against the +0.39 bar rather than +0.94.
    """
    n = len(bars)
    sig = _empty_signals(n)

    a = atr(bars, atr_period)
    # Rank of current ATR within its own trailing window. rolling().rank() uses
    # only past values, so no lookahead.
    rank = a.rolling(lookback, min_periods=lookback).rank(pct=True) * 100
    compressed = (rank <= percentile).to_numpy()

    hi = bars["high"].rolling(breakout_lookback, min_periods=breakout_lookback).max().shift(1)
    lo = bars["low"].rolling(breakout_lookback, min_periods=breakout_lookback).min().shift(1)
    close = bars["close"]

    av = a.to_numpy()
    valid = np.isfinite(av) & (av > 0) & compressed & np.isfinite(hi) & np.isfinite(lo)

    # Same state-vs-event trap as H1. Once price clears the 20-bar high it
    # stays cleared for many bars, so `close > hi` describes a condition, not
    # a breakout. Requiring the PREVIOUS bar to have been inside the range
    # isolates the moment of the break.
    prev_close = close.shift(1)
    fresh_up = (close > hi) & (prev_close <= hi.shift(1))
    fresh_dn = (close < lo) & (prev_close >= lo.shift(1))

    sig.loc[valid & fresh_up.fillna(False), "direction"] = 1
    sig.loc[valid & fresh_dn.fillna(False), "direction"] = -1
    sig["stop_distance"] = np.where(np.isfinite(av) & (av > 0), av * stop_mult, 0.0)
    sig["trail_distance"] = np.where(np.isfinite(av) & (av > 0), av * trail_mult, 0.0)
    sig["max_hold"] = max_hold
    return sig


# ==========================================================================
# H3 -- SESSION-BOUNDARY LIQUIDITY
# ==========================================================================

def session_breakout(bars: pd.DataFrame, *, session_hour: int = 7,
                     range_hours: int = 8, atr_period: int = 14,
                     stop_mult: float = 2.0, trail_mult: float = 0.0,
                     max_hold: int = 12) -> pd.DataFrame:
    """
    HYPOTHESIS H3: liquidity step-changes at session boundaries.

    At `session_hour` UTC, look at the range built over the preceding
    `range_hours`. Trade the break of that range. London opens 07:00 UTC,
    New York 13:00 UTC.

    The control is built in: run this at all 24 hours. If 07:00 and 13:00 do
    not stand out from arbitrary hours, the hypothesis is false and any single
    hour that looks good is noise.
    """
    n = len(bars)
    sig = _empty_signals(n)
    hour = pd.to_datetime(bars["datetime"]).dt.hour.to_numpy()

    prior_hi = bars["high"].rolling(range_hours, min_periods=range_hours).max().shift(1)
    prior_lo = bars["low"].rolling(range_hours, min_periods=range_hours).min().shift(1)
    a = atr(bars, atr_period)
    av = a.to_numpy()

    at_open = hour == session_hour
    valid = at_open & np.isfinite(av) & (av > 0) & np.isfinite(prior_hi) & np.isfinite(prior_lo)
    close = bars["close"]

    sig.loc[valid & (close > prior_hi), "direction"] = 1
    sig.loc[valid & (close < prior_lo), "direction"] = -1
    sig["stop_distance"] = np.where(np.isfinite(av) & (av > 0), av * stop_mult, 0.0)
    sig["trail_distance"] = np.where(np.isfinite(av) & (av > 0), av * trail_mult, 0.0)
    sig["max_hold"] = max_hold
    return sig


# ==========================================================================
# PARAMETER GRIDS -- 20 variants per family, per the pre-registration
# ==========================================================================

def grid_trend() -> list[dict]:
    out = []
    for slow in (100, 200):                      # ~2wk, ~1mo on H4
        for stop in (1.5, 2.5, 3.5):
            for pull in (45, 55):
                out.append({"fast": 50, "slow": slow, "stop_mult": stop,
                            "pullback_rsi": pull, "trail_mult": 0.0, "max_hold": 0})
    for stop in (2.5, 3.5):                      # trailing variants
        for trail in (2.0, 3.0):
            out.append({"fast": 50, "slow": 200, "stop_mult": stop,
                        "pullback_rsi": 50, "trail_mult": trail, "max_hold": 0})
    return out[:20]


def grid_volatility() -> list[dict]:
    out = []
    for lookback in (50, 100):
        for pct in (10, 20):
            for stop in (1.5, 2.5):
                for trail in (0.0, 2.0):
                    out.append({"lookback": lookback, "percentile": pct,
                                "stop_mult": stop, "trail_mult": trail,
                                "breakout_lookback": 20, "max_hold": 48})
    for bl in (10, 40):
        for stop in (1.5, 2.5):
            out.append({"lookback": 100, "percentile": 20, "stop_mult": stop,
                        "trail_mult": 2.0, "breakout_lookback": bl, "max_hold": 48})
    return out[:20]


def grid_session() -> list[dict]:
    """Includes non-session hours deliberately -- they are the control."""
    out = []
    for hour in (7, 13):                          # London, NY
        for rng in (8, 12):
            for stop in (1.5, 2.5):
                out.append({"session_hour": hour, "range_hours": rng,
                            "stop_mult": stop, "max_hold": 12})
    for hour in (2, 10, 17, 20):                  # control hours
        out.append({"session_hour": hour, "range_hours": 8,
                    "stop_mult": 2.0, "max_hold": 12})
    return out[:20]


FAMILIES = {
    "H1_trend_persistence": (trend_persistence, grid_trend),
    "H2_volatility_regime": (volatility_regime, grid_volatility),
    "H3_session_breakout": (session_breakout, grid_session),
}
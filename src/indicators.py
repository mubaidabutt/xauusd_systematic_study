#!/usr/bin/env python3
"""
XAUUSD Research Project -- Indicator library
=============================================

Every indicator here obeys one rule: the value at bar i uses ONLY bars 0..i.
No centring, no forward-filling from the future, no rolling(center=True).

Each returns a Series aligned to the input index, with leading NaNs where the
lookback is not yet satisfied. Callers must handle those NaNs rather than
back-filling them -- bfill() on an indicator is a lookahead bug wearing a
convincing disguise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(bars: pd.DataFrame) -> pd.Series:
    """max(H-L, |H-C_prev|, |L-C_prev|) -- captures gaps, unlike H-L alone."""
    h, l = bars["high"], bars["low"]
    prev_close = bars["close"].shift(1)
    return pd.concat([
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder's ATR -- an EMA with alpha = 1/period, not a simple mean.

    This is what MT5's iATR computes. A simple rolling mean gives visibly
    different values, and if the backtest and the live EA disagree on ATR,
    they disagree on every stop distance and every position size.
    """
    return true_range(bars).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, matching MT5's iRSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
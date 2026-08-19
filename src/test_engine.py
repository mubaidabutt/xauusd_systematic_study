#!/usr/bin/env python3
"""
XAUUSD Research Project -- Step 2b: Adversarial Engine Tests
=============================================================

These tests are not here to confirm the engine works. They are here to CATCH
IT CHEATING. Each one targets a specific way a backtest engine can silently
manufacture profit that does not exist in reality.

The nine failure modes covered:

  1. LOOKAHEAD        -- results change when future data changes
  2. EXECUTION DELAY  -- signals filling on the signal bar instead of the next
  3. ZERO EDGE        -- random entries producing positive expectancy
  4. COST MONOTONIC   -- costs not actually reducing profit
  5. ACCOUNTING       -- equity not equalling the sum of trade P&L
  6. GAP FILL         -- gaps filling at the stop price instead of the open
  7. INTRABAR ORDER   -- trailing stop saving a trade the stop already killed
  8. SIZING           -- positions ignoring the broker's lot step / minimum
  9. DETERMINISM      -- same input producing different output

Run:  python test_engine.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from engine import Config, Result, metrics, position_size_oz, run_backtest

PASS, FAIL = "  PASS  ", "  FAIL  "
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name:<44} {detail}")


def make_bars(n: int, seed: int = 0, start: float = 2000.0,
              vol: float = 0.002, drift: float = 0.0) -> pd.DataFrame:
    """Deterministic synthetic H1 bars on a random walk."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    close = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    op = np.concatenate([[start], close[:-1]])
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    return pd.DataFrame({"datetime": idx, "open": op, "high": hi,
                         "low": lo, "close": close})


def flat_signals(bars: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "direction": np.zeros(len(bars), dtype=int),
        "stop_distance": np.zeros(len(bars)),
        "trail_distance": np.zeros(len(bars)),
        "max_hold": np.zeros(len(bars), dtype=int),
    })


def random_signals(bars: pd.DataFrame, seed: int, rate: float = 0.02,
                   hold: int = 24) -> pd.DataFrame:
    """
    Random entries with a TIME-BASED exit.

    The max_hold matters for fairness. With only a stop and no exit, a random
    walk is recurrent -- every position stops out eventually with probability
    1, so even a perfect engine would show a loss. A symmetric time exit makes
    zero the correct expected answer.
    """
    rng = np.random.default_rng(seed)
    n = len(bars)
    fire = rng.random(n) < rate
    d = np.where(fire, rng.choice([-1, 1], n), 0)
    atr = bars["close"].rolling(14).std().bfill().to_numpy()
    return pd.DataFrame({
        "direction": d.astype(int),
        "stop_distance": np.maximum(atr * 4.0, 1.0),
        "trail_distance": np.zeros(n),
        "max_hold": np.full(n, hold, dtype=int),
    })


# ==========================================================================
# 1. LOOKAHEAD -- truncation invariance
# ==========================================================================
def test_no_lookahead() -> None:
    """
    Trades that closed before bar k must be IDENTICAL whether the engine saw
    bars[0:k] or the full series. If appending future data changes a past
    trade, the engine is reading forward.
    """
    bars = make_bars(1500, seed=1)
    sig = random_signals(bars, seed=1)
    cfg = Config()

    full = run_backtest(bars, sig, cfg).to_frame()
    k = 1000
    trunc = run_backtest(bars.iloc[:k].reset_index(drop=True),
                         sig.iloc[:k].reset_index(drop=True), cfg).to_frame()

    if trunc.empty or full.empty:
        check("1. No lookahead (truncation invariant)", False, "no trades generated")
        return

    cutoff = bars["datetime"].iloc[k - 1]
    a = trunc[trunc.exit_time < cutoff].reset_index(drop=True)
    b = full[full.exit_time < cutoff].reset_index(drop=True)
    same = len(a) == len(b) and np.allclose(
        a["net_pnl"].to_numpy(), b["net_pnl"].to_numpy(), atol=1e-9)
    check("1. No lookahead (truncation invariant)", same,
          f"{len(a)} closed trades compared")


# ==========================================================================
# 2. EXECUTION DELAY
# ==========================================================================
def test_execution_delay() -> None:
    """A signal on bar i must fill at bar i+1's OPEN. Not bar i's anything."""
    bars = make_bars(50, seed=2)
    sig = flat_signals(bars)
    sig.loc[10, "direction"] = 1
    sig.loc[10, "stop_distance"] = 50.0

    res = run_backtest(bars, sig, Config(spread_per_oz=0, slippage_per_oz=0))
    if not res.trades:
        check("2. Executes on NEXT bar open", False, "no trade")
        return
    t = res.trades[0]
    ok = (t.entry_index == 11) and np.isclose(t.entry_price, bars["open"].iloc[11])
    check("2. Executes on NEXT bar open", ok,
          f"filled bar {t.entry_index} @ {t.entry_price:.2f} "
          f"(bar 11 open = {bars['open'].iloc[11]:.2f})")


# ==========================================================================
# 3. ZERO EDGE
# ==========================================================================
def test_zero_edge() -> None:
    """
    Random entries on a driftless random walk, with costs switched off, must
    average out to roughly nothing. A consistently positive result means the
    engine is handing out free money somewhere.
    """
    cfg = Config(spread_per_oz=0.0, slippage_per_oz=0.0)
    sharpes = []
    for seed in range(12):
        bars = make_bars(3000, seed=100 + seed, drift=0.0)
        res = run_backtest(bars, random_signals(bars, seed=200 + seed), cfg)
        sharpes.append(metrics(res)["sharpe"])
    mean = float(np.mean(sharpes))
    ok = abs(mean) < 0.45
    check("3. Zero edge on random entries", ok,
          f"mean Sharpe {mean:+.3f} across 12 seeds")


# ==========================================================================
# 4. COST MONOTONICITY
# ==========================================================================
def test_cost_monotonic() -> None:
    """More cost must mean less profit. Always. No exceptions."""
    bars = make_bars(3000, seed=4, drift=0.00005)
    sig = random_signals(bars, seed=4)
    profits = []
    for spread in (0.0, 0.30, 1.00, 3.00):
        res = run_backtest(bars, sig, Config(spread_per_oz=spread, slippage_per_oz=0))
        profits.append(metrics(res)["net_profit"])
    ok = all(profits[i] > profits[i + 1] for i in range(len(profits) - 1))
    check("4. Cost strictly reduces profit", ok,
          " > ".join(f"{p:,.0f}" for p in profits))


# ==========================================================================
# 5. ACCOUNTING IDENTITY
# ==========================================================================
def test_accounting() -> None:
    """final_equity - initial_equity must equal the sum of net trade P&L."""
    bars = make_bars(3000, seed=5)
    res = run_backtest(bars, random_signals(bars, seed=5), Config())
    df = res.to_frame()
    m = metrics(res)
    expected = Config().initial_equity + df["net_pnl"].sum()
    ok = np.isclose(expected, m["final_equity"], rtol=1e-6)
    check("5. Accounting identity holds", ok,
          f"sum(P&L)+start = {expected:,.2f}  vs  equity {m['final_equity']:,.2f}")


# ==========================================================================
# 6. GAP FILL
# ==========================================================================
def test_gap_fill() -> None:
    """
    A bar that gaps THROUGH the stop must fill at the open, not the stop.
    Assuming the stop price would fabricate money that a real gap destroys.
    """
    idx = pd.date_range("2020-01-01", periods=6, freq="h")
    bars = pd.DataFrame({
        "datetime": idx,
        "open":  [100, 100, 100,  90,  90,  90],   # bar 3 gaps down to 90
        "high":  [101, 101, 101,  91,  91,  91],
        "low":   [ 99,  99,  99,  88,  88,  88],
        "close": [100, 100, 100,  89,  89,  89],
    })
    sig = flat_signals(bars)
    sig.loc[0, ["direction", "stop_distance"]] = [1, 5.0]   # entry bar1, stop 95

    res = run_backtest(bars, sig, Config(spread_per_oz=0, slippage_per_oz=0,
                                         risk_per_trade=0.01))
    if not res.trades:
        check("6. Gap fills at open, not stop", False, "no trade")
        return
    t = res.trades[0]
    ok = np.isclose(t.exit_price, 90.0)
    check("6. Gap fills at open, not stop", ok,
          f"exit @ {t.exit_price:.2f} (open=90, stop=95)")


# ==========================================================================
# 7. INTRABAR ORDERING
# ==========================================================================
def test_intrabar_pessimism() -> None:
    """
    Bar 3 dips to 94 and then rallies to close at 128.

    Correct engine : trail sat at 97 from bar 2. The dip triggers it, filling
                     at 97 for a loss.
    Buggy engine   : advances the trail on bar 3's close to 125 BEFORE the stop
                     check, so the fill lands at the open, 100 -- breakeven.

    97 versus 100 is the whole test. Letting the trail advance first is the
    single most flattering bug an engine can have.
    """
    idx = pd.date_range("2020-01-01", periods=6, freq="h")
    bars = pd.DataFrame({
        "datetime": idx,
        "open":  [100, 100, 100, 100, 100, 100],
        "high":  [101, 101, 101, 130, 101, 101],   # bar 3 rallies hard...
        "low":   [ 99,  99,  99,  94,  99,  99],   # ...after dipping to 94
        "close": [100, 100, 100, 128, 100, 100],
    })
    sig = flat_signals(bars)
    sig.loc[0, ["direction", "stop_distance", "trail_distance"]] = [1, 5.0, 3.0]

    res = run_backtest(bars, sig, Config(spread_per_oz=0, slippage_per_oz=0))
    if not res.trades:
        check("7. Stop beats trail within a bar", False, "no trade")
        return
    t = res.trades[0]
    ok = (t.exit_reason == "stop" and np.isclose(t.exit_price, 97.0)
          and t.net_pnl < 0)
    check("7. Stop beats trail within a bar", ok,
          f"exit @ {t.exit_price:.2f} (correct=97, buggy=100), pnl {t.net_pnl:.2f}")


# ==========================================================================
# 8. SIZING
# ==========================================================================
def test_sizing() -> None:
    """Every position must be a legal broker size: multiple of step, >= min."""
    cfg = Config()
    step_oz = cfg.lot_step * cfg.contract_size
    min_oz = cfg.min_lot * cfg.contract_size

    legal = True
    for eq in (500, 1_000, 10_000, 137_429):
        for sd in (0.7, 3.3, 12.5, 47.9):
            size = position_size_oz(eq, sd, cfg)
            if size == 0:
                continue
            if size < min_oz - 1e-9 or abs(size / step_oz - round(size / step_oz)) > 1e-6:
                legal = False

    # never risk more than intended
    over = position_size_oz(10_000, 10.0, cfg) * 10.0 > 10_000 * cfg.risk_per_trade + 1e-6
    check("8. Sizing respects lot step & minimum", legal and not over,
          f"step={step_oz:g}oz min={min_oz:g}oz, no over-risk")


# ==========================================================================
# 9. DETERMINISM
# ==========================================================================
def test_determinism() -> None:
    bars = make_bars(2000, seed=9)
    sig = random_signals(bars, seed=9)
    a = metrics(run_backtest(bars, sig, Config()))
    b = metrics(run_backtest(bars, sig, Config()))
    check("9. Deterministic across runs", a == b, "identical metrics")


# ==========================================================================
def main() -> int:
    print("=" * 78)
    print("ADVERSARIAL ENGINE TESTS -- trying to catch the engine cheating")
    print("=" * 78)
    for fn in (test_no_lookahead, test_execution_delay, test_zero_edge,
               test_cost_monotonic, test_accounting, test_gap_fill,
               test_intrabar_pessimism, test_sizing, test_determinism):
        try:
            fn()
        except Exception as exc:
            check(fn.__name__, False, f"EXCEPTION: {type(exc).__name__}: {exc}")

    print("=" * 78)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} passed")
    if passed < len(results):
        print("\nDO NOT TRUST ANY BACKTEST RESULT UNTIL THESE PASS.")
    print("=" * 78)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
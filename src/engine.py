#!/usr/bin/env python3
"""
XAUUSD Research Project -- Step 2: Backtest Engine
===================================================

Design principle
----------------
Every ambiguous situation resolves AGAINST the strategy. A backtest that
flatters you is worse than no backtest, because it costs real money to
discover the flattery. Where reality is unknowable from H1 bars, we assume
the worse branch.

Concretely:
  * Signals are computed on a completed bar and executed at the NEXT bar's
    open. The engine enforces the shift itself so a strategy author cannot
    forget it.
  * Stops are checked BEFORE trailing stops advance. If a bar contains both
    the stop and a favourable trail update, the stop wins.
  * A bar that gaps through a stop fills at the OPEN, not at the stop price.
    Gaps do not politely fill at your requested level.
  * Cost is charged on entry and exit, always adverse.
  * Position size respects the broker's real lot step and minimum.

Everything here is verified by test_engine.py, which tries to catch the engine
cheating. Do not trust this file on its own -- run the tests.

Conventions
-----------
Bars are BID prices (what MT5 gives you). Buying happens at ask = bid + spread,
selling at bid. So one spread is crossed per round trip regardless of side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ==========================================================================
# CONFIGURATION
# ==========================================================================

@dataclass
class Config:
    """Broker and risk parameters. Defaults match Exness XAUUSDm, swap-free."""

    # --- cost model -------------------------------------------------------
    # ZERO BY DEFAULT for the daily study, by explicit decision.
    #
    # Rationale, measured rather than assumed: on the H1 study, cost drag was
    # 0.14 Sharpe for variants trading <300 times and 0.80 Sharpe for variants
    # trading >700 times. Daily strategies holding for weeks capture $100-300
    # moves against a ~$0.40 cost, so the drag is negligible by construction.
    #
    # Research runs at zero cost. Survivors get ONE cost check before any live
    # deployment -- set spread_per_oz=0.30, slippage_per_oz=0.10 for that.
    spread_per_oz: float = 0.0
    slippage_per_oz: float = 0.0
    swap_long_per_lot: float = 0.0   # swap-free account -> 0. Flip to stress.
    swap_short_per_lot: float = 0.0

    # --- instrument spec (from MT5 symbol_info) ---------------------------
    contract_size: float = 100.0     # oz per lot
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 200.0

    # --- risk -------------------------------------------------------------
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.01     # fraction of equity risked to the stop
    max_leverage: float = 100.0      # sanity cap on notional/equity

    def cost_per_oz(self) -> float:
        """Adverse price adjustment applied at entry and again at exit."""
        return self.spread_per_oz + self.slippage_per_oz


@dataclass
class Trade:
    direction: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    size_oz: float = 0.0
    stop: float = 0.0
    trail_distance: float = 0.0
    max_hold: int = 0
    entry_index: int = 0
    nights_held: int = 0
    gross_pnl: float = 0.0
    cost: float = 0.0
    swap_cost: float = 0.0
    exit_reason: str = ""

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.cost - self.swap_cost


@dataclass
class Result:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    config: Config | None = None

    def to_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "direction": t.direction,
            "entry_time": t.entry_time, "entry_price": t.entry_price,
            "exit_time": t.exit_time, "exit_price": t.exit_price,
            "size_oz": t.size_oz, "nights": t.nights_held,
            "gross_pnl": t.gross_pnl, "cost": t.cost, "swap": t.swap_cost,
            "net_pnl": t.net_pnl, "exit_reason": t.exit_reason,
        } for t in self.trades])


# ==========================================================================
# SIZING
# ==========================================================================

def position_size_oz(equity: float, stop_distance: float, cfg: Config) -> float:
    """
    Fixed-fractional sizing, rounded to the broker's real lot step.

    Rounding DOWN matters: rounding up would silently risk more than intended,
    and on a 0.01 lot step that error is largest exactly when the account is
    small -- which is when it hurts most.
    """
    if stop_distance <= 0:
        return 0.0
    target_oz = (equity * cfg.risk_per_trade) / stop_distance
    lots = target_oz / cfg.contract_size
    lots = np.floor(lots / cfg.lot_step) * cfg.lot_step
    lots = min(lots, cfg.max_lot)
    if lots < cfg.min_lot:
        return 0.0
    return round(lots * cfg.contract_size, 8)


# ==========================================================================
# ENGINE
# ==========================================================================

def run_backtest(bars: pd.DataFrame, signals: pd.DataFrame, cfg: Config) -> Result:
    """
    bars    : datetime, open, high, low, close   (bid prices, H1)
    signals : direction (-1/0/1), stop_distance, trail_distance
              Indexed identically to `bars`. A signal on row i was computed
              from information available at bar i's CLOSE, and the engine
              executes it at bar i+1's OPEN.
    """
    required = {"datetime", "open", "high", "low", "close"}
    if not required.issubset(bars.columns):
        raise ValueError(f"bars missing columns: {required - set(bars.columns)}")
    if len(bars) != len(signals):
        raise ValueError("bars and signals must be the same length")

    dt = pd.to_datetime(bars["datetime"]).to_numpy()
    op = bars["open"].to_numpy(dtype=float)
    hi = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    cl = bars["close"].to_numpy(dtype=float)

    direction = signals["direction"].to_numpy(dtype=int)
    stop_dist = signals["stop_distance"].to_numpy(dtype=float)
    trail_dist = (signals["trail_distance"].to_numpy(dtype=float)
                  if "trail_distance" in signals.columns
                  else np.zeros(len(bars)))
    exit_sig = (signals["exit_signal"].to_numpy(dtype=bool)
                if "exit_signal" in signals.columns
                else np.zeros(len(bars), dtype=bool))
    max_hold = (signals["max_hold"].to_numpy(dtype=int)
                if "max_hold" in signals.columns
                else np.zeros(len(bars), dtype=int))

    # Calendar day index, for counting rollovers crossed.
    day = pd.to_datetime(bars["datetime"]).dt.normalize().to_numpy()

    n = len(bars)
    equity = cfg.initial_equity
    equity_curve = np.empty(n, dtype=float)
    pos: Trade | None = None
    trades: list[Trade] = []
    unit_cost = cfg.cost_per_oz()

    def close_position(p: Trade, price: float, idx: int, reason: str) -> float:
        """Realise a position at `price`, charging exit cost and swap."""
        p.exit_time, p.exit_price, p.exit_reason = dt[idx], price, reason
        p.gross_pnl = (price - p.entry_price) * p.direction * p.size_oz
        # cost charged on both legs
        p.cost = 2.0 * unit_cost * p.size_oz
        nights = int(np.busday_count(
            np.datetime64(day[p.entry_index], "D"), np.datetime64(day[idx], "D")))
        p.nights_held = max(nights, 0)
        rate = cfg.swap_long_per_lot if p.direction == 1 else cfg.swap_short_per_lot
        p.swap_cost = abs(rate) * (p.size_oz / cfg.contract_size) * p.nights_held
        trades.append(p)
        return p.net_pnl

    for i in range(n):
        was_flat = pos is None

        # ---- 1. ENTRY (at this bar's open) --------------------------------
        # Uses the signal from bar i-1. Only if we were already flat when the
        # bar began -- otherwise we would be acting on knowledge of an exit
        # that happens inside this same bar.
        if was_flat and i > 0 and direction[i - 1] != 0 and stop_dist[i - 1] > 0:
            d = int(direction[i - 1])
            sd = float(stop_dist[i - 1])
            size = position_size_oz(equity, sd, cfg)
            notional = size * op[i]
            if size > 0 and notional <= equity * cfg.max_leverage:
                # Entry price stays RAW. Cost is booked once, explicitly, in
                # close_position -- baking it into the price too would charge
                # it twice, which is the kind of quiet error that makes an
                # engine look conservative while being wrong.
                entry = op[i]
                pos = Trade(
                    direction=d, entry_time=dt[i], entry_price=entry,
                    size_oz=size, stop=entry - d * sd,
                    trail_distance=float(trail_dist[i - 1]),
                    max_hold=int(max_hold[i - 1]), entry_index=i,
                )

        # ---- 2. OPEN-PRICE EXITS ------------------------------------------
        # An exit decided on the previous close fills at THIS bar's open,
        # which chronologically precedes anything that happens inside the bar.
        # So these are evaluated before the stop check, not after.
        #
        # direction == 0 means "no new entry", NOT "close the position".
        # Closing requires an explicit exit_signal, an opposite-direction
        # signal, or the max-hold limit.
        if pos is not None and pos.entry_index < i:
            prev_dir = direction[i - 1]
            reversal = prev_dir != 0 and prev_dir != pos.direction
            timed_out = pos.max_hold > 0 and (i - pos.entry_index) >= pos.max_hold
            if exit_sig[i - 1] or reversal or timed_out:
                reason = ("max_hold" if timed_out
                          else "reversal" if reversal else "exit_signal")
                equity += close_position(pos, op[i], i, reason)
                pos = None

        # ---- 3. STOP CHECK (before trailing advances) ---------------------
        # Pessimistic ordering: if a bar both hits the stop and would have
        # moved the trail favourably, the stop wins.
        if pos is not None:
            hit = (lo[i] <= pos.stop) if pos.direction == 1 else (hi[i] >= pos.stop)
            if hit:
                if pos.entry_index < i:
                    # Gapped through? Fill at the open, not at the stop.
                    fill = (min(op[i], pos.stop) if pos.direction == 1
                            else max(op[i], pos.stop))
                else:
                    fill = pos.stop
                # cost is booked in close_position, not folded into the fill
                equity += close_position(pos, fill, i, "stop")
                pos = None

        # ---- 4. TRAILING STOP (on completed bar only) ---------------------
        if pos is not None and pos.trail_distance > 0:
            c = cl[i]
            if pos.direction == 1:
                pos.stop = max(pos.stop, c - pos.trail_distance)
            else:
                pos.stop = min(pos.stop, c + pos.trail_distance)

        equity_curve[i] = equity + (
            0.0 if pos is None
            else (cl[i] - pos.entry_price) * pos.direction * pos.size_oz)

    # Force-close anything still open at the end of the sample.
    if pos is not None:
        equity += close_position(pos, cl[-1], n - 1, "end_of_data")
        equity_curve[-1] = equity

    return Result(
        trades=trades,
        equity_curve=pd.Series(equity_curve, index=pd.to_datetime(bars["datetime"])),
        config=cfg,
    )




# ==========================================================================
# METRICS
# ==========================================================================

def metrics(result: Result) -> dict:
    """
    Sharpe is computed on DAILY equity returns, annualised by sqrt(252).

    Not on per-trade returns. Per-trade Sharpe annualised by trade count makes
    high-frequency strategies look artificially good and is not comparable
    across strategies with different trade rates -- which is exactly the
    comparison we need to make.
    """
    df = result.to_frame()
    eq = result.equity_curve
    cfg = result.config

    if df.empty or eq.empty:
        return {"n_trades": 0, "sharpe": 0.0, "max_drawdown_pct": 0.0,
                "net_profit": 0.0, "profit_factor": 0.0, "win_rate_pct": 0.0,
                "expectancy": 0.0, "cagr_pct": 0.0, "total_cost": 0.0}

    daily = eq.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

    running_max = eq.cummax()
    dd = (eq - running_max) / running_max
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    final = float(eq.iloc[-1])
    cagr = ((final / cfg.initial_equity) ** (1 / years) - 1) * 100 if final > 0 else -100.0

    wins = df[df.net_pnl > 0]["net_pnl"]
    losses = df[df.net_pnl <= 0]["net_pnl"]
    gross_win, gross_loss = wins.sum(), abs(losses.sum())

    return {
        "n_trades": int(len(df)),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown_pct": round(float(-dd.min() * 100), 2),
        "net_profit": round(float(df.net_pnl.sum()), 2),
        "total_return_pct": round((final / cfg.initial_equity - 1) * 100, 2),
        "cagr_pct": round(float(cagr), 2),
        "profit_factor": round(float(gross_win / gross_loss), 4) if gross_loss > 0 else np.inf,
        "win_rate_pct": round(100.0 * len(wins) / len(df), 2),
        "expectancy": round(float(df.net_pnl.mean()), 4),
        "total_cost": round(float(df.cost.sum() + df.swap.sum()), 2),
        "final_equity": round(final, 2),
    }
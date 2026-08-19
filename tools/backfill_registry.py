#!/usr/bin/env python3
"""Backfill H4 and momentum trials into the unified registry."""
import pandas as pd, shutil, json
from pathlib import Path

REG = Path("trial_registry.csv")
shutil.copy(REG, "trial_registry.pre_backfill.bak")
reg = pd.read_csv(REG)

# ---- Study V: four-hourly volatility regime (22 configs) ----------------
h4 = pd.read_csv("h4_study.csv")
h4_rows = pd.DataFrame({
    "schema_version": 2,
    "study": "V-fourhourly-volatility",
    "timestamp": "2026-08-02T19:00:00",
    "family": h4.control.map({True: "H2_volatility_control",
                              False: "H2_volatility_regime"}),
    "variant": h4.variant,
    "params": h4.params,
    "window": "2018-01-01..2024-01-01",
    "cost_mult": 0.0,
    "timeframe": "H4",
    "n_trades": h4.n_trades,
    "sharpe": h4.sharpe,
    "threshold": 0.989,
    "exposure_class": "neutral",
    "passed": h4.passes_bar,
    "max_drawdown_pct": h4.max_dd,
    "stage0": h4.n_trades >= 40,
    "stage0_reason": "",
})

# ---- Study VI: institutional time-series momentum (16 configs) ----------
inst = pd.read_csv("institutional.csv")
BH = {"H1": 0.607, "H4": 0.578, "D1": 0.531, "W1": 0.578}
inst_rows = pd.DataFrame({
    "schema_version": 2,
    "study": "VI-momentum",
    "timestamp": "2026-08-02T20:00:00",
    "family": "TSMOM_vol_targeted",
    "variant": range(len(inst)),
    "params": inst.apply(lambda r: json.dumps(
        {"lookback_months": int(r.lookback_months), "target_vol": 0.15,
         "max_leverage": 3.0}), axis=1),
    "window": "2018-01-01..2024-01-01",
    "cost_mult": 0.0,
    "timeframe": inst.timeframe,
    "n_trades": "",                       # always-in strategy: no discrete trades
    "sharpe": inst.sharpe,
    "pct_long": inst.pct_time_long,
    "threshold": inst.timeframe.map(BH),  # benchmark is passive holding
    "exposure_class": "neutral",
    "passed": inst.significant,
    "max_drawdown_pct": inst.max_dd_pct,
    "cagr_pct": inst.cagr_pct,
    "stage0": True,
    "stage0_reason": "n/a: continuous exposure",
})

out = pd.concat([reg, h4_rows, inst_rows], ignore_index=True)[reg.columns.tolist()]
out.to_csv(REG, index=False)
print(f"{len(reg)} + {len(h4_rows)} + {len(inst_rows)} = {len(out)} trials")
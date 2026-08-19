
"""One-time repair: merge the two incompatible registry schemas into one."""
import csv, shutil
from pathlib import Path

SRC = Path("trial_registry.csv")
shutil.copy(SRC, SRC.with_suffix(".csv.bak"))

HOURLY = ["timestamp","family","variant","params","window","cost_mult","n_trades",
          "sharpe","pct_long","threshold","exposure_class","passed",
          "max_drawdown_pct","cagr_pct","profit_factor","net_profit",
          "stage0","stage0_reason"]

DAILY  = ["timestamp","family","variant","params","window","n_trades","sharpe",
          "pct_long","threshold","exposure_class","passed","max_drawdown_pct",
          "cagr_pct","profit_factor","net_profit","long_pnl","short_pnl",
          "regimes_positive","stage0","stage0_reason"]

UNIFIED = ["schema_version","study","timestamp","family","variant","params","window",
           "cost_mult","timeframe","n_trades","sharpe","pct_long","threshold",
           "exposure_class","passed","max_drawdown_pct","cagr_pct","profit_factor",
           "net_profit","long_pnl","short_pnl","regimes_positive",
           "stage0","stage0_reason"]

def label(ts, cost):
    if ts.startswith("2026-08-01"):            return "I-hourly-defective"
    if cost == "1.0":                          return "II-hourly-corrected-realistic"
    if cost == "0.0":                          return "III-hourly-corrected-zerocost"
    return "IV-daily"

out = []
for row in list(csv.reader(SRC.open(newline="", encoding="utf-8")))[1:]:
    if len(row) == 18:
        r = dict(zip(HOURLY, row)); r["timeframe"] = "H1"
    elif len(row) == 20:
        r = dict(zip(DAILY, row)); r["timeframe"] = "D1"; r["cost_mult"] = "0.0"
    else:
        raise SystemExit(f"Unexpected row width {len(row)}: {row[:3]}")
    r["schema_version"] = "2"
    r["study"] = label(r["timestamp"], r["cost_mult"])
    out.append({k: r.get(k, "") for k in UNIFIED})

with SRC.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=UNIFIED)
    w.writeheader(); w.writerows(out)

print(f"Repaired {len(out)} rows -> {SRC}")
#!/usr/bin/env python3
"""Verify strategies contain no lookahead: signals must be truncation-invariant."""
import numpy as np, pandas as pd, sys
from strategies import FAMILIES

def main():
    bars = pd.read_parquet("syn.parquet")
    k = 30000
    ok_all = True
    print("=" * 70)
    print("STRATEGY LOOKAHEAD TEST -- signals[0:k] must not depend on bars[k:]")
    print("=" * 70)
    for name, (fn, grid) in FAMILIES.items():
        for vi, params in enumerate(grid()):
            full = fn(bars, **params).iloc[:k]
            trunc = fn(bars.iloc[:k].reset_index(drop=True), **params)
            same = (
                np.array_equal(full["direction"].to_numpy(), trunc["direction"].to_numpy())
                and np.allclose(full["stop_distance"].to_numpy(),
                                trunc["stop_distance"].to_numpy(),
                                atol=1e-9, equal_nan=True))
            if not same:
                d = (full["direction"].to_numpy() != trunc["direction"].to_numpy()).sum()
                print(f"  [FAIL] {name} variant {vi}: {d} differing signals")
                ok_all = False
                break
        else:
            print(f"  [PASS] {name}: all {len(grid())} variants truncation-invariant")
    print("=" * 70)
    return 0 if ok_all else 1

if __name__ == "__main__":
    sys.exit(main())
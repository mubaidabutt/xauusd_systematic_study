# XAUUSD Daily Study — Pre-Registration

**Committed:** 2026-08-02
**Status:** LOCKED before any daily strategy was written or run.

Supersedes the H1 pre-registration for daily work. The H1 study stands as a
completed, failed result: 144 trials, 0 passes. It is not reinterpreted here.

---

## 1. Why the study moved to daily bars

The H1 study produced one clear, quantified finding — not about gold, but
about frequency:

| Trade count (H1 family) | Cost drag |
|---|---|
| Under 300 trades | 0.141 Sharpe |
| Over 700 trades | **0.803 Sharpe** |

The variants with the strongest raw signal (+0.598, +0.609 at zero cost) were
the *worst* after costs (−0.387, −0.358). The apparent edge lived precisely at
frequencies where it could not survive contact with a broker.

Daily bars held for weeks capture $100–300 moves against a ~$0.40 cost. The
drag becomes structurally negligible rather than decisive.

**Second reason, equally important:** the archived CSV became usable again.
It was rejected for H1/H4 work because its EET clock misaligned intraday
candles. On daily bars that misalignment is immaterial, and validation
confirmed it empirically — **return correlation 0.9994, median close
difference 0.003%** against the broker feed over 3,088 overlapping days.

That recovered 2004–2018, including a regime the entire H1 study never saw.

## 2. Data

| Item | Value |
|---|---|
| Series | `xauusd_daily.parquet`, 5,711 bars, 2004-06-11 → 2026-07-31 |
| Early years | Archived CSV, EET→UTC converted (2004–2017) |
| Recent years | Exness MT5, broker-native UTC (2018–2026) |
| Research window | **2004-06-11 → 2023-12-31** (5,040 days) |
| **Sealed** | **2024-01-01 → 2026-07-31** — one evaluation, later |

## 3. Cost model — ZERO, by explicit decision

```
spread    $0.00
slippage  $0.00
swap      $0.00   (Islamic swap-free account)
```

Research runs cost-free. This is a deliberate choice, and its consequence is
symmetric: removing costs raises strategy results **and** raises the random
benchmarks they are measured against. The comparison stays fair; the bar
simply moves. It is recorded here so nobody later mistakes a zero-cost result
for a net-of-cost one.

**Survivors get exactly one cost check before any live deployment**, at
$0.30 spread + $0.10 slippage. A daily strategy holding for weeks should barely
notice it. If one does, that is disqualifying information.

---

## 4. Thresholds — LOCKED

From 300 random-entry trials per direction on the real 2004–2023 daily series,
~15 signals/year, 20-day hold, zero cost.

| Null | mean | σ | p95 | Best-of-40 (p95) |
|---|---|---|---|---|
| Symmetric | +0.026 | 0.225 | +0.386 | **+0.705** |
| Long-only | +0.526 | 0.173 | +0.831 | **+1.050** |
| Short-only | −0.465 | 0.169 | −0.179 | +0.045 |

Buy and hold, 2004–2023: **+0.576** (CAGR 8.98%, max drawdown 44.8%).

### The bar

| Strategy type | Must clear |
|---|---|
| **Mostly long** (>60% of signals long) | **Sharpe +1.050** |
| **Direction-neutral** (40–60%) | **Sharpe +0.705** |
| **Mostly short** (<40%) | **Sharpe +0.705** |

Both bars are HIGHER than the H1 study's +0.94 / +0.39. Fewer trades widens
the null, and zero cost lifts the random benchmark too. This is the correct
direction — a harder bar on cleaner data.

**Free Sharpe from simply being long: +0.500.** Across 22 years *including a
four-year bear market*, random long entries beat random symmetric entries by
half a Sharpe point. Gold's drift is persistent and substantial, and any
long-biased strategy is handed it before demonstrating anything.

## 5. Regime structure — what any strategy is actually up against

| Regime | B&H Sharpe | Return | Max DD |
|---|---|---|---|
| 2004–2007 early bull | +1.32 | +116.6% | 22.0% |
| 2008–2011 parabolic | +1.00 | +113.5% | 28.9% |
| **2011–2015 bear** | **−0.64** | **−41.9%** | **44.5%** |
| 2016–2018 range | +0.81 | +21.4% | 17.4% |
| 2018–2023 (H1 dev window) | +0.59 | +56.3% | 21.4% |

The 2011–2015 bear is the regime the entire H1 study never contained, and it
is very likely why the H1 trend filter kept reporting 50% long against a
predicted 60%+: there was no sustained downtrend to detect.

**Stage 6 requires positive performance in ≥3 of these 5 regimes.** A strategy
carried by one regime has not demonstrated an edge; it has demonstrated
exposure to that regime.

---

## 6. Hypotheses

### D1 — Bidirectional trend persistence

**Mechanism.** Gold trends are driven by slow macro forces — real interest
rates, central-bank reserve policy, currency debasement — that persist over
months. Positioning adjusts gradually, so price continues in the direction of
an established multi-month trend more often than a random walk implies. The
mechanism is **symmetric**: it applies to 2011–2015 falling as much as to
2008–2011 rising.

**Prediction.** Entering in the direction of a multi-month trend, holding for
weeks, produces positive expectancy in both directions.

**Falsified if:** performance is positive only in rising regimes, or the short
side loses money across the full sample.

**Design requirement — this is the key design decision of the study.** The
strategy must be built to trade both directions, *not* long-biased. Rationale:
the long-biased bar is +1.05 while the neutral bar is +0.705, and the
underlying mechanism is identical either way. Taking shorts is not a
concession — it is what makes the hypothesis testable, since the short-only
null p95 is only +0.045 and a profitable short side would be doing something
random shorts essentially never do.

### D2 — Long-horizon breakout

**Mechanism.** Multi-month range breaks mark genuine regime transitions —
positioning shifts, new participants enter. Unlike intraday breakouts, which
the H1 study found are dominated by noise and cost, monthly-horizon breaks
reflect real repricing.

**Prediction.** Breaks of 3–6 month extremes, held for weeks with volatility
stops, produce positive expectancy symmetrically.

**Falsified if:** breakout entries do not beat entries taken at random times
with identical exits.

### D3 — Not tested: mean reversion

Deliberately excluded. Gold's daily series shows strong regime persistence and
the H1 volatility-reversion family (H2) failed decisively — median −0.781 with
costs, −0.156 without, 0/20 passing. Nothing has changed at the daily horizon
that would revive it. Recorded here so its absence is a decision, not an
oversight.

---

## 7. Variant budget — LOCKED

**Maximum 40 variants: 20 per family.**

The threshold table above is computed for exactly 40. Testing more invalidates
it. Parameters are chosen at economically meaningful horizons — months, not
arbitrary integers — and no parameter is tested at more than 4 values.

**Registry stands at 144 trials from the H1 study.** Daily trials are 145–184.
The count is cumulative and permanent.

## 8. Evaluation sequence

| Stage | Test | Pass condition |
|---|---|---|
| 0 | Sanity | ≥40 trades; no single trade > 25% of profit |
| 1 | Threshold | Clears its exposure-matched bar |
| 2 | Family coherence | **Median variant clears the bar** — not merely > 0 |
| 3 | Parameter surface | Neighbouring parameters behave similarly |
| 4 | Regime robustness | Positive in ≥3 of 5 regimes |
| 5 | Direction balance | Long and short sides *both* profitable |
| 6 | Walk-forward | OOS/IS efficiency > 0.5 |
| 7 | **Sealed 2024–2026** | Single evaluation |
| 8 | Cost check | Survives $0.40/oz |
| 9 | Demo forward test | ≥3 months live-fill comparison |

**Stage 0 and Stage 2 are tightened from the H1 study, deliberately.**

The H1 Stage 0 allowed a single trade to contribute up to 100% of profit, and
variants passed showing "top trade = 974% of profit" — meaning every other
trade combined lost money. The daily limit is 25%.

The H1 Stage 2 asked only for median > 0, and flagged a family with median
+0.060 as "coherent" when 0/16 variants cleared the actual bar. The daily
criterion requires the **median variant to clear the threshold itself.** That
is a substantially harder test and it is set now, before any results exist.

**Trade-count note:** at ~15 trades/year over 19 years, a variant produces
roughly 250–300 trades. Adequate, but each individual trade carries more
weight than on H1. Stage 0's 25% concentration limit is the guard against a
result resting on one or two exceptional moves.

## 9. Stopping rule

Unchanged and reaffirmed: **if no family clears Stage 2, the correct outcome
is to trade nothing.**

The H1 study ended this way, correctly. Two failed studies is evidence, not
misfortune. Gold may not offer a retail-accessible edge, and finding that out
for free is the cheapest possible result.

**What will not happen:** lowering a threshold, extending the window, adding
variants beyond 40, or unsealing 2024–2026 early because results were
disappointing.

---

## Amendments

*(append-only; date and justify each)*

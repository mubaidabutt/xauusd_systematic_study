# XAUUSD Strategy Study — Pre-Registration

**Committed:** 2026-08-01
**Status:** LOCKED before any strategy backtest was run.

Nothing in this document may be revised after seeing results. If a threshold
turns out to be inconvenient, that is information about the strategy, not a
reason to move the threshold. Amendments get appended with dates and reasons,
never edited in place.

---

## 1. Why this document exists

The previous round tested 160 variants, 14 passed a Sharpe > 1 filter, and the
result could not be interpreted. Not because the number was small — because
there was no way to distinguish these three explanations:

1. A real edge was found.
2. 160 lottery tickets were bought and 14 happened to win.
3. Gold rose 200% and long-biased strategies inherited the drift.

Explanations 2 and 3 are now quantified (Section 4). Committing thresholds in
advance is what converts "it passed" from a description into evidence.

---

## 2. Data and environment

| Item | Value |
|---|---|
| Instrument | XAUUSDm, Exness-MT5Real35, account 255330834 |
| Source | MT5 terminal, broker-native, UTC+0 confirmed (break hours `[21,22]`) |
| Bars | 56,645 H1, 2014-01-14 → 2026-07-30 |
| **Usable window** | **2018-01-01 → 2026-07-30** (2014–2016 are ~300-bar cache fragments) |
| Contract | 100 oz/lot, min 0.01, step 0.01 |
| Swap | Zero — Islamic swap-free account |

**Swap-free is a broker policy, not a market fact.** Every strategy is also run
at the standard rate (−48.83/lot/night long, 0 short) to establish whether it
depends on that policy. A strategy that only works at zero swap is a bet on
terms of service.

## 3. Cost model

```
spread   $0.30/oz     (conservative vs $0.24 observed live)
slippage $0.10/oz     (invisible in history — assumed, not measured)
```

Charged adversely on both legs. The MT5 bar `spread` column was found to be a
backfill constant (median = p95 in every year, rollover cheaper than midday)
and is **not** used.

Measured cost drag on random entries: **≈0.36 Sharpe units.** Every strategy
pays this before earning anything.

**Every result is reported at 1×, 2× and 3× cost.** A strategy that dies at 2×
is too fragile to trade, because slippage is the one input we have no
historical measurement of.

---

## 4. Thresholds — LOCKED

Derived from 300 random-entry trials per direction on the real data.

| Null | mean | σ | Best-of-160 (typical) | Best-of-160 (p95) |
|---|---|---|---|---|
| Symmetric | −0.455 | 0.249 | +0.20 | **+0.39** |
| Long-only | −0.019 | 0.223 | +0.57 | **+0.74** |

Buy and hold: **+0.611** over the full window, **≈+0.94** over 2018+ only.

### The bar

| Strategy type | Must clear |
|---|---|
| **Mostly long** (>60% of exposure long) | **Sharpe +0.94** |
| **Direction-neutral** (40–60%) | **Sharpe +0.39** |
| **Mostly short** | **Sharpe +0.39** |

The asymmetry is the whole point. Random long entries earn **+0.44 Sharpe more
than random symmetric entries** for no skill — that is gold's drift, and a
long-biased strategy is handed it for free. So it must clear a bar more than
twice as high.

**Consequence:** every strategy reports `pct_time_long` alongside Sharpe. That
number selects which threshold applies. This is not optional and is decided by
the strategy's behaviour, not by its author's intent.

---

## 5. Hypotheses

Each states a mechanism, a directional prediction, and what would falsify it.
A family with no mechanism is curve-fitting with extra steps.

### H1 — Volatility-scaled trend persistence

**Mechanism.** Gold trends are driven by slow-moving macro flows — real rates,
central-bank buying, currency debasement — that persist over weeks. Positioning
adjusts gradually, so price continues in the direction of the intermediate
trend more often than a random walk implies.

**Prediction.** Entering in the direction of an established multi-week trend,
with ATR-scaled stops, produces positive expectancy with low win rate and high
payoff ratio.

**Falsified if:** win rate × payoff ≈ break-even after costs, or all the edge
sits in 2024–2025.

**Expected exposure:** mostly long → must clear **+0.94**. This is the family
most at risk of being disguised beta, and it dominated the previous round
(8/14 survivors).

### H2 — Volatility regime reversion

**Mechanism.** Volatility clusters and mean-reverts. Compressed ranges resolve
into expansion; extreme expansion decays. This is one of the most robust
empirical regularities in finance and, crucially, is **direction-agnostic**.

**Prediction.** Conditioning entries on volatility state — rather than price
direction — produces edge with roughly symmetric long/short exposure.

**Falsified if:** the volatility signal adds nothing over the same entry taken
unconditionally.

**Expected exposure:** neutral → must clear **+0.39**. Structurally the most
promising family precisely *because* it doesn't inherit drift.

### H3 — Session-boundary liquidity effects

**Mechanism.** Gold trades nearly 24h but liquidity is wildly uneven. The
London open (07:00 UTC) and NY open (13:00 UTC) bring order-flow step-changes;
the Asian session is thin and range-bound. Overnight ranges get tested and
either break or reject at these boundaries.

**Prediction.** Entry conditions tied to session boundaries outperform the same
logic applied at arbitrary hours.

**Falsified if:** the session filter's edge is within noise of random hours.
**This family has a built-in control** — run the identical logic at every hour
and check whether 07:00/13:00 genuinely stand out.

**Expected exposure:** neutral → **+0.39**.

### H4 — Not tested: news-driven strategies

Deliberately excluded. Gold's largest moves cluster around NFP, CPI and FOMC,
but we have no historical spread or slippage data for those windows, and they
are exactly when execution is worst. Backtesting them with a flat cost model
would produce results we could not trust. Revisit after forward-test data
exists.

---

## 6. Variant budget — LOCKED

**48 variants total: 16 trend / 20 volatility / 12 session.**

Down from 160, deliberately. The selection threshold scales with how many
things you try — testing 160 raises the long-biased luck bar to +0.74, while 60
lowers it to about +0.70. More importantly, a smaller grid forces parameter
choices to be justified rather than swept.

Parameters are chosen on **economic reasoning**, not by gridding every value:

- Lookbacks at meaningful horizons (day, week, month), not every integer
- Stops at 1.5 / 2.5 / 3.5 ATR — coarse enough that neighbours are genuinely different
- No parameter tested at more than 4 values

Grids are unequal because parameters were chosen per hypothesis rather than
padded to a round number. The session family is smallest because a third of its
grid is control hours, not candidates.

**Every backtest run is logged in `trial_registry.csv`, including failures and
abandoned ideas.** The registry count is what the multiple-testing correction
uses. An unlogged trial makes every threshold in this document wrong.

---

## 7. Evaluation sequence

A strategy must pass each stage to reach the next. No stage is skipped, and a
failure at any stage ends that strategy's candidacy.

| Stage | Test | Pass condition |
|---|---|---|
| 0 | Sanity | ≥100 trades; no single trade > 10% of profit |
| 1 | In-sample, 2018–2023 | Clears its exposure-matched threshold |
| 2 | Cost stress | Survives 2× cost |
| 3 | Family coherence | Median Sharpe across the family's 20 variants is positive |
| 4 | Parameter surface | Neighbouring parameters give similar results — plateau, not spike |
| 5 | Walk-forward | Efficiency (OOS/IS) > 0.5 |
| 6 | Regime split | Positive in ≥3 of 4 sub-periods; not carried by 2025 alone |
| 7 | **Held-out 2024–2026** | Single evaluation. See below. |
| 8 | Monte Carlo | Bootstrapped p95 drawdown used for sizing, not observed |
| 9 | Demo forward test | ≥3 months; realised fills compared to modelled |

**Stage 3 is the one that matters most.** A single variant clearing the bar is
weak evidence. A family whose *median* variant is profitable is strong
evidence, because it means the effect doesn't depend on finding the magic
parameter.

### The held-out period

**2024-01-01 → 2026-07-30 is sealed.** No looking, no exploratory plots, no
"just checking." It is examined exactly once, for strategies that have passed
stages 0–6.

This window matters more than its length suggests: gold rose 27.2% in 2024 and
64.6% in 2025 (Sharpe 2.46), then reversed with 2026 realised vol at 30.1% —
roughly double every prior year. It contains both the strongest trend and the
sharpest regime break in the entire dataset.

**Expected out-of-sample degradation is 30–50%.** A strategy showing in-sample
1.2 that delivers 0.7 held-out has performed *well*. Anything that matches its
in-sample number exactly should be treated with suspicion, not celebration.

---

## 8. Stopping rule

If no strategy family clears its threshold at Stage 3, **the correct outcome is
to trade nothing.** 

This needs saying in advance, because after weeks of work the temptation to
relax a threshold is enormous and always feels justified in the moment. Gold on
H1 through a single broker may simply not contain an exploitable edge at
retail cost levels. That is a legitimate finding and far cheaper than
discovering it with real money.

---

## Amendments

*(append-only; date and justify each)*

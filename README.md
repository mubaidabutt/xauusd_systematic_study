# Timeframe Selection in Short-Horizon Systematic Trading

Replication materials for a pre-registered study of **222 strategy configurations**
tested on spot gold (XAU/USD), including every configuration that failed.

The central claim of the accompanying paper is that undisclosed multiple testing
invalidates most retail trading research. Publishing every trial rather than only
the survivors is what makes that claim verifiable rather than merely asserted.

---

## Headline results

| Finding | Statistic |
|---|---|
| Configurations logged | 222 |
| Passed Stage 0 sanity criteria | 152 |
| Cleared their pre-registered threshold | **0** |
| Expectancy estimates with 95% CI excluding zero | **0** |
| Cost drag, under 300 vs. over 700 trades | 0.14 vs. 0.80 Sharpe units |
| Free Sharpe from long-only exposure | approximately 0.50 |
| Timeframe effect, family-matched paired test | +0.158R, *t*(36) = 4.35, *p* < .001 |

The only significant result concerns bar interval, not strategy.

---

## Repository structure

```
├── src/                    analysis toolchain (15 modules)
├── results/                trial registry and result tables
│   └── archive/            pre-repair registry snapshots
├── logs/                   verbatim execution output, one file per study
├── preregistration/        both protocols, committed before testing
├── paper/                  manuscript and figure generation
└── figures/                all seven figures
```

**Price data are not included.** Broker feeds are licensed and cannot be
redistributed. `src/step1b_mt5_pull.py` retrieves an equivalent series directly
from a MetaTrader 5 terminal.

### On the registry snapshots

`results/archive/` holds two earlier states of the trial registry. The registry was
originally written by two runners using incompatible column schemas, which made the
file unparseable; it was repaired into a unified schema, and the four-hourly and
momentum studies were then backfilled from their own result files. Both prior states
are retained so the correction is auditable rather than invisible.

---

## Reproducing the analysis

```bash
pip install -r requirements.txt
```

Python 3.11 or later. Developed on 3.13 with pandas 3.0.2 and NumPy 2.4.4.

### 1. Validate the engine before trusting any result

```bash
python src/test_engine.py
```

Expected: `9/9 passed`. These tests attempt to catch the engine fabricating profit
through lookahead bias, favourable intrabar ordering, optimistic gap fills, and six
other mechanisms. **If any test fails, no downstream result is meaningful.**

```bash
python src/test_strategies.py
```

Verifies that signal generation is truncation invariant.

### 2. Retrieve and validate data

```bash
python src/step1b_mt5_pull.py          # requires a running MT5 terminal, Windows only
python src/build_daily.py --csv "path/to/archived_hourly.csv"
```

`step1b_mt5_pull.py` verifies the feed is genuinely UTC+0 by checking that the daily
maintenance break falls on hours {21, 22}, the signature of an open-labelled UTC
series under US daylight-saving rollover.

### 3. Establish thresholds before testing anything

```bash
python src/null_benchmark.py --start 2018-01-01
python src/daily_null.py
```

This is the step most commonly skipped in practitioner research. A Sharpe ratio has
no meaning without knowing what random entries achieve on the same series, and what
the *best of N* random entries achieves.

### 4. Run the studies

```bash
python src/run_study.py                # hourly, 48 configurations
python src/run_daily_study.py          # daily, 40 configurations
python src/h4_study.py                 # four-hourly, 22 configurations
python src/institutional.py            # momentum, 16 configurations
python src/expectancy_report.py        # expectancy across timeframes
python src/timeframe_analysis.py       # the timeframe effect
```

Each runner appends to `results/trial_registry.csv`. The registry is append-only by
design: the multiple-comparison correction consumes the trial count, so an unlogged
trial silently invalidates every threshold.

### 5. Rebuild the figures

```bash
python paper/make_figures.py
```

Figures are computed from `results/`; no value is entered by hand.

---

## Methodological notes

**The sealed period.** Data from 2024-01-01 onward was withheld from all analysis and
remains unused. No configuration survived to a stage warranting its evaluation. It is
available for future validation work.

**Zero-cost analyses.** Primary analyses for the daily, four-hourly and momentum
studies were run without transaction costs at the researcher's election, with
realistic-cost results reported alongside. The effect is symmetric: removing costs
raises both strategy results and the random benchmarks against which they are
compared.

**A defect found by pre-registration.** The initial trend implementation specified
entry as a persistent state rather than a discrete event, producing near-continuous
market exposure. It was detected because the protocol predicted predominantly long
exposure and observed exposure was 51%, a failed prediction about the
*implementation* that is visible without reference to any performance metric. All 48
affected trials remain in the registry.

**Version control.** The project was not under version control during the research
period, so no independent timestamp exists for the pre-registration documents. Their
stated commitment dates are self-asserted. The execution timestamps recorded in the
trial registry are consistent with those dates and with the stated sequence.

---

## Disclaimer

This repository documents academic research and is **not investment advice**. The
central finding is that none of the strategies examined demonstrated a tradeable
edge. The code is provided for replication and methodological reuse, not for
deployment. Trading leveraged instruments carries substantial risk of loss.

## License

MIT. See [LICENSE](LICENSE).

## Contact

Muhammad Ubaida Butt · Suleman Dawood School of Business, Lahore University of
Management Sciences · mubaidabutt@gmail.com
#!/usr/bin/env python3
"""Generate figures for the XAUUSD paper from recorded study results.

All values are computed from files in results/ or drawn from recorded study
output. No figure carries a title: explanatory material belongs in the
caption, per journal convention. Panels are labelled (a)/(b) only.

Run from the project root:
    python paper/make_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
RESULTS = Path("results")

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight",
})
C = {"pos": "#1a5490", "neg": "#a02020", "neu": "#666666", "hl": "#d4820a"}

PANEL = dict(fontsize=9, loc="left")


# --- Figure 1: evaluation funnel (computed from the registry) -------------
reg = pd.read_csv(RESULTS / "trial_registry.csv")
n_total, n_stage0 = len(reg), int(reg.stage0.sum())
n_bar, n_sig = int(reg.passed.sum()), 0

fig, ax = plt.subplots(figsize=(6.5, 3.4))
stages = ["Trials\nrun", "Passed\nStage 0", "Cleared\nthreshold", "Statistically\nsignificant"]
vals = [n_total, n_stage0, n_bar, n_sig]
bars = ax.bar(stages, vals, color=[C["neu"], C["neu"], C["neg"], C["neg"]], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + max(vals)*0.02, str(v),
            ha="center", fontweight="bold")
ax.set_ylabel("Number of strategy configurations")
ax.set_ylim(0, max(vals) * 1.15)
fig.savefig(OUT / "fig1_funnel.png")
plt.close(fig)
print(f"Figure 1: {n_total} -> {n_stage0} -> {n_bar} -> {n_sig}")


# --- Figure 2: regime structure of gold -----------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
regimes = ["2004–07\nearly bull", "2008–11\nparabolic", "2011–15\nbear",
           "2016–18\nrange", "2018–23\ndev"]
sharpe = [1.32, 1.00, -0.64, 0.81, 0.59]
rets = [116.6, 113.5, -41.9, 21.4, 56.3]
cols = [C["pos"] if s > 0 else C["neg"] for s in sharpe]

a1.bar(range(5), sharpe, color=cols, width=0.65)
a1.axhline(0, color="black", lw=0.8)
a1.set_xticks(range(5)); a1.set_xticklabels(regimes, fontsize=7)
a1.set_ylabel("Buy-and-hold Sharpe ratio")
a1.set_title("(a)", **PANEL)

a2.bar(range(5), rets, color=cols, width=0.65)
a2.axhline(0, color="black", lw=0.8)
a2.set_xticks(range(5)); a2.set_xticklabels(regimes, fontsize=7)
a2.set_ylabel("Total return (%)")
a2.set_title("(b)", **PANEL)

fig.savefig(OUT / "fig2_regimes.png")
plt.close(fig)


# --- Figure 3: cost drag vs trade frequency -------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
trades = [399, 362, 290, 240, 223, 203, 284, 238, 202, 160, 159, 130, 1046, 761, 1029, 705]
free = [0.424, 0.341, 0.237, 0.439, 0.266, 0.295, 0.182, 0.368, 0.067, 0.293,
        0.032, 0.219, 0.598, 0.378, 0.609, 0.449]
real = [0.129, 0.072, 0.047, 0.275, 0.119, 0.170, -0.015, 0.211, -0.075, 0.193,
        -0.075, 0.142, -0.387, -0.271, -0.358, -0.164]
drag = np.array(free) - np.array(real)

a1.scatter(trades, drag, c=C["hl"], s=28, edgecolor="black", linewidth=0.4, zorder=3)
z = np.polyfit(trades, drag, 1)
xs = np.linspace(min(trades), max(trades), 50)
a1.plot(xs, np.polyval(z, xs), "--", color=C["neu"], lw=1)
a1.set_xlabel("Number of trades (6 years)")
a1.set_ylabel("Cost drag (Sharpe units)")
a1.set_title("(a)", **PANEL)

idx = np.argsort(trades)
sc = a2.scatter(np.array(free)[idx], np.array(real)[idx],
                c=np.array(trades)[idx], cmap="YlOrRd", s=34,
                edgecolor="black", linewidth=0.4, zorder=3)
lim = [-0.5, 0.7]
a2.plot(lim, lim, "--", color=C["neu"], lw=1, label="No cost effect")
a2.axhline(0, color="black", lw=0.8); a2.axvline(0, color="black", lw=0.8)
a2.set_xlim(lim); a2.set_ylim(lim)
a2.set_xlabel("Sharpe ratio, zero cost")
a2.set_ylabel("Sharpe ratio, realistic cost")
a2.legend(fontsize=7, loc="upper left")
a2.set_title("(b)", **PANEL)
cb = fig.colorbar(sc, ax=a2, pad=0.02)
cb.set_label("Trades", fontsize=7)
cb.ax.tick_params(labelsize=6)

fig.savefig(OUT / "fig3_cost.png")
plt.close(fig)


# --- Figure 4: null distributions and selection thresholds ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
x = np.linspace(-1.6, 1.6, 400)
for mu, sd, lab, c in [(-0.455, 0.249, "Symmetric entries", C["neu"]),
                       (-0.019, 0.223, "Long-only entries", C["pos"])]:
    a1.plot(x, np.exp(-((x-mu)**2)/(2*sd**2))/(sd*np.sqrt(2*np.pi)),
            color=c, lw=1.5, label=lab)
a1.axvline(0.94, color=C["neg"], ls="--", lw=1.2, label="Buy and hold (+0.94)")
a1.set_xlabel("Sharpe ratio (hourly null, 300 trials each)")
a1.set_ylabel("Density")
a1.legend(fontsize=7)
a1.set_title("(a)", **PANEL)

labels = ["Single\nrandom trial", "Best of 40", "Best of 160", "Buy and\nhold"]
vals = [0.526, 0.894, 1.05, 0.576]
a2.bar(range(4), vals, color=[C["neu"], C["hl"], C["hl"], C["neg"]], width=0.6)
for i, v in enumerate(vals):
    a2.text(i, v + 0.02, f"{v:+.2f}", ha="center", fontsize=8, fontweight="bold")
a2.set_xticks(range(4)); a2.set_xticklabels(labels, fontsize=7)
a2.set_ylabel("Sharpe ratio (daily, long-only null)")
a2.set_ylim(0, max(vals) * 1.18)
a2.set_title("(b)", **PANEL)

fig.savefig(OUT / "fig4_null.png")
plt.close(fig)


# --- Figure 5: four-hourly confidence intervals ---------------------------
h4 = pd.read_csv(RESULTS / "h4_study.csv")
h4 = h4[~h4.control].reset_index(drop=True)      # 20 filtered configurations
h4 = h4.sort_values("exp_r_free").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(6.8, 4.2))
y = np.arange(len(h4))
ax.errorbar(h4.exp_r_free, y,
            xerr=[h4.exp_r_free - h4.ci_lo, h4.ci_hi - h4.exp_r_free],
            fmt="o", color=C["pos"], ecolor=C["neu"], elinewidth=1,
            capsize=2.5, markersize=4)
ax.axvline(0, color=C["neg"], lw=1.4, ls="--")
ax.set_yticks(y)
ax.set_yticklabels([f"v{int(v)}" for v in h4.variant], fontsize=6)
ax.set_xlabel("Expectancy per trade (R units), 95% confidence interval")
ax.set_ylabel("Configuration")

fig.savefig(OUT / "fig5_ci.png")
plt.close(fig)
print(f"Figure 5: {len(h4)} configurations, "
      f"{int(((h4.ci_lo < 0) & (h4.ci_hi > 0)).sum())} intervals span zero")


# --- Figure 6: institutional momentum vs buy and hold ---------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
tfs = ["H1", "H4", "D1", "W1"]
bh = [0.607, 0.578, 0.531, 0.578]
best = [0.582, 0.622, 0.409, 0.259]
w, xp = 0.36, np.arange(4)

a1.bar(xp - w/2, bh, w, label="Buy and hold", color=C["neu"])
a1.bar(xp + w/2, best, w, label="Best momentum configuration", color=C["pos"])
a1.set_xticks(xp); a1.set_xticklabels(tfs)
a1.set_ylabel("Sharpe ratio")
a1.legend(fontsize=7)
a1.set_title("(a)", **PANEL)

lbs = [1, 3, 6, 12]
med = [0.242, 0.495, 0.300, 0.225]
a2.plot(lbs, med, "o-", color=C["hl"], lw=1.6, markersize=6, label="Observed (gold)")
a2.plot(lbs, [0.05, 0.18, 0.32, 0.45], "s--", color=C["neu"], lw=1.3,
        markersize=5, label="Literature expectation")
a2.set_xticks(lbs)
a2.set_xlabel("Momentum lookback (months)")
a2.set_ylabel("Median Sharpe across timeframes")
a2.legend(fontsize=7)
a2.set_title("(b)", **PANEL)

fig.savefig(OUT / "fig6_institutional.png")
plt.close(fig)


# --- Figure 7: benchmark comparison and the timeframe effect --------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.3))

studies = ["Study I\nhourly", "Study III\ndaily", "Study IV\nfour-hourly", "Study V\nmomentum"]
best = [0.609, 0.423, 0.582, 0.622]
bench = [0.940, 0.576, 0.578, 0.578]
xp, w = np.arange(4), 0.36

a1.bar(xp - w/2, bench, w, label="Passive benchmark", color=C["neu"])
a1.bar(xp + w/2, best, w, label="Best configuration", color=C["pos"])
a1.set_xticks(xp); a1.set_xticklabels(studies, fontsize=7)
a1.set_ylabel("Sharpe ratio")
a1.legend(fontsize=7, loc="upper right")
a1.set_title("(a)", **PANEL)

_e = pd.read_csv(RESULTS / "expectancy_report.csv")
h1e = _e[_e.timeframe == "H1"].exp_r_real.tolist()
h4e = _e[_e.timeframe == "H4"].exp_r_real.tolist()

bp = a2.boxplot([h1e, h4e], tick_labels=["Hourly", "Four-hourly"],
                patch_artist=True, widths=0.5, showmeans=True)
for patch, col in zip(bp["boxes"], [C["neg"], C["pos"]]):
    patch.set_facecolor(col); patch.set_alpha(0.45)
for med in bp["medians"]:
    med.set_color("black"); med.set_linewidth(1.4)
a2.axhline(0, color="black", lw=0.9, ls="--")
a2.set_ylabel("Expectancy per trade (R units)")
a2.set_title("(b)", **PANEL)

fig.savefig(OUT / "fig7_benchmark.png")
plt.close(fig)
print(f"Figure 7: H1 n={len(h1e)}, H4 n={len(h4e)}")

print("written:", sorted(p.name for p in OUT.glob("*.png")))
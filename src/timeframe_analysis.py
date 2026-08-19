#!/usr/bin/env python3
"""Timeframe effect: unmatched (confounded) vs family-matched (primary)."""
import numpy as np, pandas as pd
from scipy import stats

df = pd.read_csv("results/expectancy_report.csv")
df = df[df.timeframe.isin(["H1", "H4"])]

print("=" * 70, "\nGROUP COMPOSITION -- why the naive test is confounded\n", "=" * 70)
print(pd.crosstab(df.family, df.timeframe))

h1 = df[df.timeframe == "H1"].exp_r_real.values
h4 = df[df.timeframe == "H4"].exp_r_real.values
t = stats.ttest_ind(h4, h1, equal_var=False)
sp = np.sqrt(((len(h4)-1)*h4.var(ddof=1) + (len(h1)-1)*h1.var(ddof=1)) / (len(h4)+len(h1)-2))
print(f"\nUNMATCHED (report as descriptive only)")
print(f"  H1: n={len(h1)} M={h1.mean():+.4f} SD={h1.std(ddof=1):.4f}")
print(f"  H4: n={len(h4)} M={h4.mean():+.4f} SD={h4.std(ddof=1):.4f}")
print(f"  Welch t({t.df:.1f})={t.statistic:.2f} p={t.pvalue:.2g} d={(h4.mean()-h1.mean())/sp:.2f}")

# ---- family-matched pairs ----------------------------------------------
w = df.pivot_table(index=["family", "variant"], columns="timeframe",
                   values="exp_r_real").dropna()
d = (w.H4 - w.H1).values
tp = stats.ttest_rel(w.H4, w.H1)
rng = np.random.default_rng(0)
bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(20000)]

print(f"\nFAMILY-MATCHED (report as primary)")
print(f"  pairs n={len(d)}", dict(w.reset_index().family.value_counts()))
print(f"  mean diff={d.mean():+.4f}  95% CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]")
print(f"  paired t({len(d)-1})={tp.statistic:.2f} p={tp.pvalue:.4f}  dz={d.mean()/d.std(ddof=1):.2f}")
print(f"  Wilcoxon p={stats.wilcoxon(w.H4, w.H1).pvalue:.4f}")

# ---- sign-flip permutation: addresses shared-price-history dependence ---
obs = d.mean()
perm = [np.mean(d * rng.choice([-1, 1], len(d))) for _ in range(20000)]
print(f"  permutation p={np.mean(np.abs(perm) >= abs(obs)):.4f}")

print(f"\nWITHIN-FAMILY")
for fam in df.family.unique():
    a = df[(df.timeframe=="H1") & (df.family==fam)].exp_r_real.values
    b = df[(df.timeframe=="H4") & (df.family==fam)].exp_r_real.values
    if len(a) > 1 and len(b) > 1:
        tf = stats.ttest_ind(b, a, equal_var=False)
        print(f"  {fam:<24} H1 n={len(a)} M={a.mean():+.4f} | "
              f"H4 n={len(b)} M={b.mean():+.4f} | t={tf.statistic:.2f} p={tf.pvalue:.4f}")
"""Disruptionskurve: Wolkenmasse oberhalb von Tracer-Schwellen gegen die Zeit.

Aufruf:  %run ~/athena/analysis/disruption_curve.py hr       (vordefinierter Satz)
         %run ~/athena/analysis/disruption_curve.py run1 run2 (Run-Namen in ~/athena/runs)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
RUN_SETS = {
    "hr": {   # 1024 x 768, Cluster
        "g2_cart_hr":        r"$\chi \approx 2800$, $\gamma = 5/3$",
        "g2_cart_hr_rho621": r"$\chi \approx 4900$, $\gamma = 5/3$",
        "g2_cart_hr_g101":   r"$\chi \approx 2800$, $\gamma = 1.01$",
    },
    "lr": {   # 256 x 192, Mac
        "g2_cart_cc01_v2":     r"$\chi \approx 2800$, $\gamma = 5/3$",
        "g2_cart_cc01_rho621": r"$\chi \approx 4900$, $\gamma = 5/3$",
        "g2_cart_cc01_g1.01":  r"$\chi \approx 2800$, $\gamma = 1.01$",
    },
    "s12": {  # CC01 mit Schartmann+12-Normierung, bis 2060
        "g2_cart_hr_s12": r"CC01 (Schartmann+12), $r_{\rm in} = 1.9 \times 10^{15}$ cm",
    },
}
args = sys.argv[1:] or ["hr"]
if args[0] in RUN_SETS:
    SET, RUNS = args[0], RUN_SETS[args[0]]
else:
    SET, RUNS = "custom", {a: a for a in args}
THRESH = [0.9, 0.99, 0.9999]
T_PERI = 18.0     # Perizentrum 2013.5
YR     = g2_io.YR


def cloud_masses(run, n):
    """Zeit und Massen (gleiche Zellvolumina -> Summe rho*tr); None bei nan."""
    t, g = g2_io.load_snapshot(os.path.join(BASE, run), n)
    m, tr = g["rho"] * g["r0"], g["r0"]
    if not np.all(np.isfinite(m)):
        return None
    return t, [m.sum()] + [m[tr > th].sum() for th in THRESH]


results = {}
for run in RUNS:
    rows = []
    for n in g2_io.snapshot_numbers(os.path.join(BASE, run)):
        r = cloud_masses(run, n)
        if r is not None:
            rows.append([r[0] / YR] + r[1])
    df = pd.DataFrame(rows, columns=["t_yr", "M_tot"] + [f"M_tr>{th}" for th in THRESH])
    df.iloc[:, 1:] /= df["M_tot"].iloc[0]          # relativ zur Anfangsmasse
    df.to_csv(os.path.join(BASE, run, "disruption.csv"), index=False)
    results[run] = df
    print(f"{run}: {len(df)} Snapshots bis t = {df['t_yr'].iloc[-1]:.1f} yr")


def t_below(df, col, level):
    """Erster Zeitpunkt, an dem df[col] unter level faellt (linear interpoliert)."""
    f, t = df[col].to_numpy(), df["t_yr"].to_numpy()
    below = np.where(f < level)[0]
    if len(below) == 0 or below[0] == 0:
        return None
    i = below[0]
    return t[i-1] + (level - f[i-1]) * (t[i] - t[i-1]) / (f[i] - f[i-1])


print(f"\nSatz '{SET}': Zeitpunkte, an denen die Masse unter 50 % faellt")
print(f"  {'Lauf':22s} {'tr>0.9999':>10s} {'tr>0.99':>10s}   Kern nach 1. Snapshot   M_tot am Ende")
for run, df in results.items():
    a, b = t_below(df, "M_tr>0.9999", 0.5), t_below(df, "M_tr>0.99", 0.5)
    fa = f"{a:6.2f} yr" if a else "     -   "
    fb = f"{b:6.2f} yr" if b else "     -   "
    print(f"  {run:22s} {fa:>10s} {fb:>10s}   {df['M_tr>0.9999'].iloc[1]:.3f}"
          f"                  {df['M_tot'].iloc[-1]:.3f}")

fig, axes = plt.subplots(1, len(THRESH), figsize=(5 * len(THRESH), 4.2), sharey=True,
                         constrained_layout=True)
for ax, th in zip(axes, THRESH):
    for (run, label), c in zip(RUNS.items(), ["C0", "C1", "C2", "C3", "C4"]):
        df = results[run]
        ax.plot(df["t_yr"], df[f"M_tr>{th}"], color=c, label=label)
        ax.plot(df["t_yr"], df["M_tot"], color=c, ls=":", lw=1)
    ax.axvline(T_PERI, color="grey", ls="--", lw=0.8)
    ax.set_title(f"tr > {th}  ({SET})")
    ax.set_xlabel("t [yr] seit 1995.5")
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("Masse / Anfangsmasse")
axes[0].legend(loc="lower left", fontsize=8)

out = os.path.expanduser(f"~/athena/analysis/disruption_curve_{SET}.png")
fig.savefig(out, dpi=150)
print("\nGespeichert:", out)

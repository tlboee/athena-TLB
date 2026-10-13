"""Disruptionskurve: Wolkenmasse oberhalb von Tracer-Schwellen gegen die Zeit.

Aufruf (JupyterHub-Notebook):  %run ~/athena/analysis/disruption_curve.py
"""
import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE   = os.path.expanduser("~/athena/runs")
RUNS   = {
    "g2_cart_hr":        r"$\chi \approx 2800$, $\gamma = 5/3$",
    "g2_cart_hr_rho621": r"$\chi \approx 4900$, $\gamma = 5/3$",
    "g2_cart_hr_g101":   r"$\chi \approx 2800$, $\gamma = 1.01$",
}
PREFIX = "g2_cart"
THRESH = [0.9, 0.99, 0.9999]
DT_OUT = 0.5      # Jahre zwischen zwei Snapshots
NMAX   = 40
T_PERI = 18.0     # Perizentrum 2013.5


def cloud_masses(run, n):
    """Gesamtmasse und Masse oberhalb der Schwellen (gleiche Zellvolumina -> Summe rho*tr)."""
    files = sorted(glob.glob(f"{BASE}/{run}/{PREFIX}.block*.out1.{n:05d}.tab"))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f, sep=r"\s+", comment="#", header=None, usecols=[4, 9])
                   for f in files])
    rho, tr = d[4].to_numpy(), d[9].to_numpy()
    m = rho * tr
    if not np.all(np.isfinite(m)):
        return None                      # Snapshot mit nan (abgestuerzter Lauf)
    return [m.sum()] + [m[tr > t].sum() for t in THRESH]


results = {}
for run in RUNS:
    rows = []
    for n in range(NMAX + 1):
        r = cloud_masses(run, n)
        if r is not None:
            rows.append([n * DT_OUT] + r)
    df = pd.DataFrame(rows, columns=["t_yr", "M_tot"] + [f"M_tr>{t}" for t in THRESH])
    df.iloc[:, 1:] /= df["M_tot"].iloc[0]          # relativ zur Anfangsmasse
    df.to_csv(f"{BASE}/{run}/disruption.csv", index=False)
    results[run] = df
    print(f"{run}: {len(df)} Snapshots bis t = {df['t_yr'].iloc[-1]:.1f} yr")

# Zeitpunkt, an dem der kalte Kern (tr > 0.9999) unter 50 % faellt
print("\nt_50 (Kern tr > 0.9999 unter 50 % der Anfangsmasse):")
for run, df in results.items():
    f = df["M_tr>0.9999"].to_numpy()
    t = df["t_yr"].to_numpy()
    below = np.where(f < 0.5)[0]
    if len(below) and below[0] > 0:
        i = below[0]
        t50 = t[i-1] + (0.5 - f[i-1]) * (t[i] - t[i-1]) / (f[i] - f[i-1])
        print(f"  {run:20s} t_50 = {t50:5.2f} yr  ({1995.5 + t50:.1f})")
    else:
        print(f"  {run:20s} nicht unter 50 % gefallen")

fig, axes = plt.subplots(1, len(THRESH), figsize=(5 * len(THRESH), 4.2), sharey=True,
                         constrained_layout=True)
for ax, thr in zip(axes, THRESH):
    for (run, label), c in zip(RUNS.items(), ["C0", "C1", "C2"]):
        df = results[run]
        ax.plot(df["t_yr"], df[f"M_tr>{thr}"], color=c, label=label)
        ax.plot(df["t_yr"], df["M_tot"], color=c, ls=":", lw=1)
    ax.axvline(T_PERI, color="grey", ls="--", lw=0.8)
    ax.set_title(f"tr > {thr}")
    ax.set_xlabel("t [yr] seit 1995.5")
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("Masse / Anfangsmasse")
axes[0].legend(loc="lower left", fontsize=8)
axes[-1].text(T_PERI + 0.2, 1.0, "Perizentrum", fontsize=8, color="grey", va="top")

out = os.path.expanduser("~/athena/analysis/disruption_curve.png")
fig.savefig(out, dpi=150)
print("\nGespeichert:", out)

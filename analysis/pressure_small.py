"""Max' Frage (19.11.): Bleibt eine kleinere Wolke im Druckgleichgewicht?
P_core/P_hot (Median, Tracer > 0.9) fuer R, R/2, R/4, adiabatisch und mit Kuehlung.
Cache: <lauf>/pressure_eq.csv (gleiches Format wie pressure_eq.py).  Ausgabe: runs/pressure_small.png
Aufruf:  %run ~/athena/analysis/pressure_small.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/athena/analysis"))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, MU, MH, KB = 3.156e7, 0.6139, 1.6605e-24, 1.380648e-16
RHO0, T0, RR0, R_IN, T_PERI = 1.702e-21, 2.128e8, 1.0e16, 1.9e15, 18.0
RUNS = [("R", "ps_ref_big_x2_t50", "ps_ref_big_cool_x2_t50", "#c0392b"),
        ("R/2", "ps_ref_big_R0.5_x2_t50", "ps_ref_big_cool_R0.5_x2_t50", "#d9822b"),
        ("R/4", "ps_ref_big_R0.25_x4_t50", "ps_ref_big_cool_R0.25_x4_t50", "#2f6db3")]

def wmedian(x, w):
    i = np.argsort(x); c = np.cumsum(w[i])
    return float(x[i][np.searchsorted(c, 0.5 * c[-1])])

def series(run):
    d = os.path.join(BASE, run); cache = os.path.join(d, "pressure_eq.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        X, Y = np.meshgrid(s["x"], s["y"]); r = np.maximum(np.hypot(X, Y), R_IN)
        q = s["press"] / (RHO0 * T0 * (RR0 / r)**2 * KB / (MU * MH))
        rho, tr = s["rho"], np.clip(s["r0"], 0, 1)
        row = [t / YR]
        for thr in (0.9, 0.5):
            m = tr > thr
            row.append(wmedian(q[m], (rho * tr)[m]) if m.sum() > 10 else np.nan)
        m = tr > 0.9
        row.append(wmedian((s["press"] * MU * MH / (rho * KB))[m], (rho * tr)[m]) if m.sum() > 10 else np.nan)
        rows.append(row); print(f"  {run}: t = {t/YR:5.1f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t", "P_core", "P_cloud", "T_core"]); df.to_csv(cache, index=False)
    return df

fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
print(f"{'Wolke':5s} {'Kuehlung':9s} " + " ".join(f"{'P/P_hot @' + str(t):>11s}" for t in (5, 10, 15, 17)))
for lab, ra, rc, c in RUNS:
    for run, ls, cl in ((ra, "-", "adiabatic"), (rc, "--", "with cooling")):
        if not os.path.isdir(os.path.join(BASE, run)):
            print(f"{run} fehlt"); continue
        d = series(run)
        ax.semilogy(d["t"], d["P_core"], color=c, ls=ls, label=f"{lab}, {cl}")
        vals = [d["P_core"].iloc[int(np.argmin(np.abs(d["t"] - t)))] for t in (5, 10, 15, 17)]
        print(f"{lab:5s} {cl[:9]:9s} " + " ".join(f"{v:11.3f}" for v in vals))
ax.axhline(1, color="0.6", lw=0.8); ax.axvline(T_PERI, color="k", ls=":", lw=0.8)
ax.set_xlim(0, 25); ax.set_xlabel("t [yr]"); ax.set_ylabel("P_cl / P_hot(r)  (core, tracer > 0.9)")
ax.set_title("Does a smaller cloud stay in pressure equilibrium on its way to pericentre?"); ax.legend(fontsize=9, ncol=2)
fig.savefig(os.path.join(BASE, "pressure_small.png"), dpi=150)
plt.show()

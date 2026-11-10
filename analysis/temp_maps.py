"""Temperaturkarten um das Perizentrum (17/18/19 yr) fuer CC01 bei 2048, adiabatisch und mit Kuehlung.

Konturen: Tracer 0.1 / 0.5 / 0.9. Heisses Gas in tr > 0.9 -> Stossheizung; nur in 0.1 < tr < 0.9 -> Vermischung.
Ausgabe: Anteil des heissen Wolkenmaterials (T > 3 T_cl0, Gewicht rho*tr) in Zellen mit tr > 0.9.
Snapshot-Zeiten aus <lauf>/heat.csv (heat_time.py muss vorher gelaufen sein).
Aufruf (Notebook):  %run ~/athena/analysis/temp_maps.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, MU, MH, KB, AU = 3.156e7, 0.6139, 1.6605e-24, 1.380648e-16, 1.496e13
TIMES = (17.0, 18.0, 19.0)
RUNS = [("adiabatic", "ps_ref_big_x4_t50"), ("with cooling", "ps_ref_big_cool_x4_t50")]
T_CL0 = 1.03e4

def press_key(s):
    for k in ("press", "p", "P", "pres", "prs"):
        if k in s:
            return k
    raise SystemExit(f"Kein Druck-Schluessel gefunden, vorhanden: {list(s.keys())}")

def snap_at(run, t_want):
    d = os.path.join(BASE, run)
    nums = list(g2_io.snapshot_numbers(d))
    ts = pd.read_csv(os.path.join(d, "heat.csv"))["t_yr"].values
    i = int(np.argmin(np.abs(ts - t_want)))
    return g2_io.load_snapshot(d, nums[i])

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
print(f"{'Lauf':14s} {'t [yr]':>6s} {'heiss [% Wolke]':>16s} {'davon tr>0.9 [%]':>17s} {'T_max/T_cl0':>12s}")
for row, (kind, run) in enumerate(RUNS):
    for col, tw in enumerate(TIMES):
        t, s = snap_at(run, tw)
        x, y = s["x"] / AU, s["y"] / AU
        rho, tr = s["rho"], np.clip(s["r0"], 0.0, 1.0)
        T = s[press_key(s)] * MU * MH / (rho * KB)
        w = rho * tr
        hot = (T > 3 * T_CL0) & (tr > 1e-4)
        f_hot = w[hot].sum() / w.sum()
        f_core = w[hot & (tr > 0.9)].sum() / max(w[hot].sum(), 1e-300)
        print(f"{kind:14s} {t/YR:6.1f} {100*f_hot:16.1f} {100*f_core:17.1f} {T[tr > 1e-4].max()/T_CL0:12.3g}")
        # Ausschnitt um das Wolkenmaterial
        jj, ii = np.where(tr > 1e-3)
        x0, x1, y0, y1 = x[ii.min()], x[ii.max()], y[jj.min()], y[jj.max()]
        px, py = 0.15 * (x1 - x0) + 1, 0.15 * (y1 - y0) + 1
        ax = axes[row, col]
        im = ax.imshow(T, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap="inferno",
                       norm=LogNorm(1e4, 1e9), interpolation="nearest")
        ax.contour(x, y, tr, levels=[0.1, 0.5, 0.9], colors=["c", "w", "lime"], linewidths=0.7)
        ax.plot(0, 0, "w+", ms=8)
        ax.set_xlim(x0 - px, x1 + px); ax.set_ylim(y0 - py, y1 + py)
        ax.set_aspect("equal")
        ax.set_title(f"{kind}, t = {t/YR:.1f} yr")
        ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]")
fig.colorbar(im, ax=axes, label="T [K]", shrink=0.8)
fig.suptitle("CC01 2048: temperature around pericentre; tracer contours 0.1 (cyan), 0.5 (white), 0.9 (green); + = Sgr A*")
fig.savefig(os.path.join(BASE, "temp_maps.png"), dpi=150)
plt.show()

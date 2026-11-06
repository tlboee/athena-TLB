"""Kompression am Perizentrum: Geschwindigkeitsdivergenz div v und Temperatur, CC01 2048 adiabatisch, 16/17/18 yr.

div v < 0: Kompression (Stoss = schmale Linie stark negativer Divergenz), Einheit 1/yr.
Tabelle fuer fast reines Wolkengas (tr > 0.9): Massenanteil mit div v < -1/yr und gewichteter Median von T/T_cl0
im komprimierten vs. uebrigen Gas.  Snapshot-Zeiten aus heat.csv.
Aufruf (Notebook):  %run ~/athena/analysis/div_maps.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, MU, MH, KB, AU = 3.156e7, 0.62, 1.6605e-24, 1.380648e-16, 1.496e13
RUN, TIMES, T_CL0 = "ps_ref_big_x4_t50", (16.0, 17.0, 18.0), 1.03e4

def key(s, cands, what):
    for k in cands:
        if k in s:
            return k
    raise SystemExit(f"Kein Schluessel fuer {what} gefunden, vorhanden: {list(s.keys())}")

def wmed(x, w):
    i = np.argsort(x); cw = np.cumsum(w[i])
    return float(x[i][np.searchsorted(cw, 0.5 * cw[-1])]) if len(x) and cw[-1] > 0 else np.nan

d = os.path.join(BASE, RUN)
nums = list(g2_io.snapshot_numbers(d))
ts = pd.read_csv(os.path.join(d, "heat.csv"))["t_yr"].values

fig, axes = plt.subplots(2, len(TIMES), figsize=(15, 8.5), constrained_layout=True)
print(f"{'t [yr]':>6s} {'komprimiert [%]':>15s} {'T/T_cl0 kompr.':>14s} {'T/T_cl0 uebrig':>14s} {'div Median [1/yr]':>17s}")
for col, tw in enumerate(TIMES):
    t, s = g2_io.load_snapshot(d, nums[int(np.argmin(np.abs(ts - tw)))])
    x, y = s["x"], s["y"]
    vx, vy = s[key(s, ("vel1", "vx", "v1"), "v_x")], s[key(s, ("vel2", "vy", "v2"), "v_y")]
    div = (np.gradient(vx, x, axis=1) + np.gradient(vy, y, axis=0)) * YR          # [1/yr]
    rho, tr = s["rho"], np.clip(s["r0"], 0.0, 1.0)
    T = s[key(s, ("press", "p", "P", "pres", "prs"), "Druck")] * MU * MH / (rho * KB)
    core = tr > 0.9
    w = (rho * tr)[core]
    comp = div[core] < -1.0
    print(f"{t/YR:6.1f} {100*w[comp].sum()/w.sum():15.1f} {wmed(T[core][comp]/T_CL0, w[comp]):14.3g} "
          f"{wmed(T[core][~comp]/T_CL0, w[~comp]):14.3g} {wmed(div[core], w):17.3g}")
    xa, ya = x / AU, y / AU
    jj, ii = np.where(tr > 1e-3)
    x0, x1, y0, y1 = xa[ii.min()], xa[ii.max()], ya[jj.min()], ya[jj.max()]
    px, py = 0.15 * (x1 - x0) + 1, 0.15 * (y1 - y0) + 1
    ext = (xa[0], xa[-1], ya[0], ya[-1])
    for row, (img, cmap, norm, lab) in enumerate([
            (div, "RdBu_r", SymLogNorm(linthresh=0.1, vmin=-100, vmax=100), "div v [1/yr]  (blue = compression)"),
            (T, "inferno", LogNorm(1e4, 1e9), "T [K]")]):
        ax = axes[row, col]
        im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
        ax.contour(xa, ya, tr, levels=[0.1, 0.5, 0.9], colors=["c", "k" if row == 0 else "w", "lime"], linewidths=0.7)
        ax.plot(0, 0, "k+" if row == 0 else "w+", ms=8)
        ax.set_xlim(x0 - px, x1 + px); ax.set_ylim(y0 - py, y1 + py); ax.set_aspect("equal")
        ax.set_title(f"t = {t/YR:.1f} yr", fontsize=10)
        ax.set_xlabel("x [AU]"); ax.set_ylabel("y [AU]")
        if col == len(TIMES) - 1:
            fig.colorbar(im, ax=axes[row, :], label=lab, shrink=0.8)
fig.suptitle("CC01 2048 adiabatic: velocity divergence (top) and temperature (bottom); tracer 0.1 / 0.5 / 0.9 (green)")
fig.savefig(os.path.join(BASE, "div_maps.png"), dpi=150)
plt.show()

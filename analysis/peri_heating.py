"""Heizung am Perizentrum: Vermischung oder zusaetzliche Heizung (Stoss/Kompression)?  CC01 adiabatisch, 512/1024/2048.

Pro Wolkenzelle (tr > 1e-4):  q = T / T_mix,erw  mit  T_mix,erw = tr T_cl0 + (1 - tr) T_hot(r)   (reine Vermischung -> q ~ 1)
                              P / P_hot(r)                                                       (Druckgleichgewicht -> ~ 1)
Atmosphaere: rho_hot = RHO0 R0/r, T_hot = T0 R0/r (Schartmann).  Gewicht: rho * tr.  Snapshot-Zeiten aus heat.csv.
Aufruf (Notebook):  %run ~/athena/analysis/peri_heating.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, MU, MH, KB = 3.156e7, 0.6139, 1.6605e-24, 1.380648e-16
RHO0, T0, R0, R_IN = 1.702e-21, 2.128e8, 1.0e16, 1.9e15
TIMES = (18.0, 19.0, 20.0)
RUNS = [("512", "ps_ref_big"), ("1024", "ps_ref_big_x2_t50"), ("2048", "ps_ref_big_x4_t50")]

def press_key(s):
    for k in ("press", "p", "P", "pres", "prs"):
        if k in s:
            return k
    raise SystemExit(f"Kein Druck-Schluessel gefunden, vorhanden: {list(s.keys())}")

def snapshot(run, n_or_t, by_time=True):
    d = os.path.join(BASE, run)
    nums = list(g2_io.snapshot_numbers(d))
    if not by_time:
        return g2_io.load_snapshot(d, nums[0])
    ts = pd.read_csv(os.path.join(d, "heat.csv"))["t_yr"].values
    return g2_io.load_snapshot(d, nums[int(np.argmin(np.abs(ts - n_or_t)))])

def wmed(x, w):
    i = np.argsort(x); cw = np.cumsum(w[i])
    return float(x[i][np.searchsorted(cw, 0.5 * cw[-1])]) if len(x) and cw[-1] > 0 else np.nan

fig, axes = plt.subplots(len(RUNS), len(TIMES), figsize=(13, 11), sharex=True, sharey=True, constrained_layout=True)
print(f"{'Aufl.':>5s} {'t [yr]':>6s} {'heiss [%]':>9s} {'q Median':>9s} {'q>3 [%]':>8s} {'P/P_hot Med.':>12s} {'P/P_hot>3 [%]':>13s}")
xe, ye = np.linspace(-4, 0, 81), np.linspace(-5, 1.5, 131)
for row, (lab, run) in enumerate(RUNS):
    _, s0 = snapshot(run, None, by_time=False)
    T_cl0 = float(np.median((s0[press_key(s0)] * MU * MH / (s0["rho"] * KB))[s0["r0"] > 0.99]))
    for col, tw in enumerate(TIMES):
        t, s = snapshot(run, tw)
        X, Y = np.meshgrid(s["x"], s["y"])
        r = np.maximum(np.hypot(X, Y), R_IN)
        rho, tr = s["rho"], np.clip(s["r0"], 0.0, 1.0)
        P = s[press_key(s)]
        T = P * MU * MH / (rho * KB)
        T_h, rho_h = T0 * R0 / r, RHO0 * R0 / r
        P_h = rho_h * KB * T_h / (MU * MH)
        sel = tr > 1e-4
        w = (rho * tr)[sel]
        q = (T / (tr * T_cl0 + (1 - tr) * T_h))[sel]
        pr = (P / P_h)[sel]
        hot = T[sel] > 3 * T_cl0
        wh = w[hot]
        print(f"{lab:>5s} {t/YR:6.1f} {100*wh.sum()/w.sum():9.1f} {wmed(q[hot], wh):9.3g} "
              f"{100*wh[q[hot] > 3].sum()/max(wh.sum(),1e-300):8.1f} {wmed(pr[hot], wh):12.3g} "
              f"{100*wh[pr[hot] > 3].sum()/max(wh.sum(),1e-300):13.1f}")
        H, _, _ = np.histogram2d(np.log10(np.maximum(1 - tr[sel], 1e-4)), np.log10((T / T_h)[sel]),
                                 bins=(xe, ye), weights=w / w.sum())
        ax = axes[row, col]
        im = ax.pcolormesh(xe, ye, H.T, norm=LogNorm(1e-5, 1e-1), cmap="viridis")
        xx = np.linspace(-4, 0, 50)
        ax.plot(xx, np.log10(10**xx + T_cl0 / (T0 * R0 / 4e15)), "w--", lw=1)   # reine Vermischung (T_cl0 bei r_p)
        ax.set_title(f"{lab}, t = {t/YR:.1f} yr", fontsize=10)
        if row == len(RUNS) - 1: ax.set_xlabel("log₁₀ (1 − tr)  (ambient mass fraction)")
        if col == 0: ax.set_ylabel("log₁₀ T / T_hot(r)")
fig.colorbar(im, ax=axes, label="cloud mass fraction (ρ·tr)", shrink=0.6)
fig.suptitle("CC01 adiabatic: cloud gas around pericentre; dashed = pure mixing  T = tr T_cl0 + (1−tr) T_hot")
fig.savefig(os.path.join(BASE, "peri_heating.png"), dpi=150)
plt.show()

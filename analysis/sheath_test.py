"""Bugstoss-Test: Umgebungsgas im Saum um die Wolke, vorne (Flugrichtung) vs. hinten, aus den Neustarts (0.05 yr).

Saum: tr < 1e-3 innerhalb SHEATH_AU um die Wolke (tr > 0.1), per binary_dilation (Breite in Zellen pro Aufloesung).
Vorne/hinten: Vorzeichen von (x - x_cm) . v_cm, Schwerpunkt und Geschwindigkeit massengewichtet mit rho*tr.
Ausgabe: Median T/T_hot(r) und P/P_hot(r) vorne/hinten, Machzahl |v_cm| / c_s,hot(r_cm).  Cache: <lauf>/sheath.csv
Erwartung Bugstoss: vorne T/T_hot ~ 2-4, hinten ~ 1, Mach ~ 1.5.
Aufruf (Notebook):  %run ~/athena/analysis/sheath_test.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, MU, MH, KB, AU, GAM = 3.156e7, 0.6139, 1.6605e-24, 1.380648e-16, 1.496e13, 5.0 / 3.0
RHO0, T0, R0, R_IN, T_PERI, SHEATH_AU = 1.702e-21, 2.128e8, 1.0e16, 1.9e15, 18.0, 30.0
RUNS = [("1024", "ps_ref_big_x2_peri"), ("2048", "ps_ref_big_x4_peri")]
COLS = {"1024": "#21918c", "2048": "#bddf26"}
STEP = {"ps_ref_big_x4_peri": 2}   # nur jeden 2. Snapshot (0.1 yr), Lesezeit

def key(s, cands):
    for k in cands:
        if k in s:
            return k
    raise SystemExit(f"Schluessel {cands} fehlt, vorhanden: {list(s.keys())}")

def series(run):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "sheath.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for n in list(g2_io.snapshot_numbers(d))[::STEP.get(run, 1)]:
        t, s = g2_io.load_snapshot(d, n)
        x, y = s["x"], s["y"]
        X, Y = np.meshgrid(x, y)
        r = np.maximum(np.hypot(X, Y), R_IN)
        rho, tr = s["rho"], np.clip(s["r0"], 0.0, 1.0)
        P = s[key(s, ("press", "p", "P", "pres", "prs"))]
        vx, vy = s[key(s, ("vel1", "vx", "v1"))], s[key(s, ("vel2", "vy", "v2"))]
        T = P * MU * MH / (rho * KB)
        T_h = T0 * R0 / r
        P_h = (RHO0 * R0 / r) * KB * T_h / (MU * MH)
        w = rho * tr
        xc, yc = (w * X).sum() / w.sum(), (w * Y).sum() / w.sum()
        ux, uy = (w * vx).sum() / w.sum(), (w * vy).sum() / w.sum()
        ncell = max(1, int(round(SHEATH_AU * AU / (x[1] - x[0]))))
        sheath = binary_dilation(tr > 0.1, iterations=ncell) & (tr < 1e-3)
        front = sheath & ((X - xc) * ux + (Y - yc) * uy > 0)
        back = sheath & ~front
        rc = max(np.hypot(xc, yc), R_IN)
        cs = np.sqrt(GAM * KB * T0 * R0 / rc / (MU * MH))
        med = lambda a, m: float(np.median(a[m])) if m.any() else np.nan
        rows.append([t / YR, med(T / T_h, front), med(T / T_h, back), med(P / P_h, front), med(P / P_h, back),
                     np.hypot(ux, uy) / cs, rc, ncell])
        print(f"  {run}: t = {t/YR:6.2f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t", "T_front", "T_back", "P_front", "P_back", "mach", "r_cm", "ncell"])
    df.to_csv(cache, index=False)
    return df

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True, constrained_layout=True)
print(f"{'Aufl.':>5s} {'Saum [Zellen]':>13s} {'T/T_hot vorne max':>17s} {'bei t':>6s} {'hinten dort':>11s} {'P/P_hot vorne':>13s} {'Mach max':>8s}")
for lab, run in RUNS:
    if not os.path.isfile(os.path.join(BASE, run, "g2_cart.final.rst")):
        print(f"{run} fehlt oder noch nicht fertig"); continue
    df = series(run)
    i = int(np.nanargmax(df["T_front"]))
    print(f"{lab:>5s} {int(df['ncell'][0]):13d} {df['T_front'][i]:17.3g} {df['t'][i]:6.2f} {df['T_back'][i]:11.3g} "
          f"{df['P_front'][i]:13.3g} {df['mach'].max():8.3g}")
    c = COLS[lab]
    axes[0].plot(df["t"], df["T_front"], color=c, label=f"{lab} front"); axes[0].plot(df["t"], df["T_back"], color=c, ls="--", label=f"{lab} back")
    axes[1].plot(df["t"], df["P_front"], color=c, label=f"{lab} front"); axes[1].plot(df["t"], df["P_back"], color=c, ls="--", label=f"{lab} back")
    axes[2].plot(df["t"], df["mach"], color=c, label=lab)
for ax, ttl in zip(axes, ("ambient sheath: T / T_hot(r)", "ambient sheath: P / P_hot(r)", "cloud Mach number |v| / c_s,hot(r)")):
    ax.axvline(T_PERI, color="k", ls="--", lw=0.8); ax.set_title(ttl, fontsize=10); ax.set_xlabel("t [yr]")
axes[0].axhline(1, color="0.6", lw=0.8); axes[1].axhline(1, color="0.6", lw=0.8); axes[2].axhline(1, color="0.6", lw=0.8)
axes[0].legend(fontsize=8); axes[2].legend(fontsize=8)
fig.suptitle(f"CC01 adiabatic: ambient gas within {SHEATH_AU:g} AU of the cloud (tracer < 1e-3), front vs. back")
fig.savefig(os.path.join(BASE, "sheath_test.png"), dpi=150)
plt.show()

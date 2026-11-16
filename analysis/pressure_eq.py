"""Warum ist die adiabatische Wolke nicht im Druckgleichgewicht? (Max, Treffen 12.11.)

Feld 1: P_cl / P_hot(r) entlang der Bahn, massengewichteter Median ueber Tracer > 0.9 (Kern) und > 0.5 (Wolke);
        CC01 1024 adiabatisch und mit Kuehlung.
Feld 2: Schallzeit der Wolke t_sc = r_eff / c_s,cl (r_eff aus stream_width.csv, c_s aus der Median-Temperatur
        des Kerns) gegen die verbleibende Zeit bis zum Perizentrum und die Passagezeit r_p / v_p.
Cache: <lauf>/pressure_eq.csv.   Ausgabe: runs/pressure_eq.png.   Aufruf:  %run ~/athena/analysis/pressure_eq.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/athena/analysis"))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, MU, MH, KB, GAM = 3.156e7, 0.6139, 1.6605e-24, 1.380648e-16, 5.0 / 3.0
RHO0, T0, RR0, R_IN, R0 = 1.702e-21, 2.128e8, 1.0e16, 1.9e15, 1.87e15
T_PERI, T_PASS = 18.0, 0.25
RUNS = [("adiabatic", "ps_ref_big_x2_t50", "-"), ("with cooling", "ps_ref_big_cool_x2_t50", "--")]

def wmedian(x, w):
    i = np.argsort(x); c = np.cumsum(w[i])
    return float(x[i][np.searchsorted(c, 0.5 * c[-1])])

def series(run):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "pressure_eq.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        X, Y = np.meshgrid(s["x"], s["y"])
        r = np.maximum(np.hypot(X, Y), R_IN)
        P_hot = RHO0 * T0 * (RR0 / r)**2 * KB / (MU * MH)
        rho, P, tr = s["rho"], s["press"], np.clip(s["r0"], 0, 1)
        q = P / P_hot
        row = [t / YR]
        for thr in (0.9, 0.5):
            m = tr > thr
            row.append(wmedian(q[m], (rho * tr)[m]) if m.sum() > 10 else np.nan)
        m = tr > 0.9
        T_core = wmedian((P * MU * MH / (rho * KB))[m], (rho * tr)[m]) if m.sum() > 10 else np.nan
        row.append(T_core)
        rows.append(row)
        print(f"  {run}: t = {t/YR:5.1f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t", "P_core", "P_cloud", "T_core"])
    df.to_csv(cache, index=False)
    return df

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
data = {}
for lab, run, ls in RUNS:
    if not os.path.isdir(os.path.join(BASE, run)):
        print(f"{run} fehlt"); continue
    df = data[lab] = series(run)
    axes[0].semilogy(df["t"], df["P_core"], color="#c0392b", ls=ls, label=f"{lab}: core (tracer > 0.9)")
    axes[0].semilogy(df["t"], df["P_cloud"], color="#2f6db3", ls=ls, label=f"{lab}: cloud (tracer > 0.5)")
axes[0].axhline(1, color="0.6", lw=0.8); axes[0].axvline(T_PERI, color="k", ls=":", lw=0.8)
axes[0].set_xlabel("t [yr]"); axes[0].set_ylabel("P_cl / P_hot(r)  (mass-weighted median)")
axes[0].set_title("Cloud pressure relative to the local atmosphere (CC01, 1024)"); axes[0].legend(fontsize=9)

sw = os.path.join(BASE, "ps_ref_big_x2_t50", "stream_width.csv")
if os.path.exists(sw) and "adiabatic" in data:
    w, d = pd.read_csv(sw), data["adiabatic"]
    T_c = np.interp(w["t"], d["t"], d["T_core"])
    c_s = np.sqrt(GAM * KB * T_c / (MU * MH))
    t_sc = w["r_tr_R0"] * R0 / c_s / YR
    left = np.where(w["t"] < T_PERI, T_PERI - w["t"], np.nan)
    axes[1].semilogy(w["t"], t_sc, color="#6a3d9a", label="sound-crossing time of the cloud  r_eff / c_s,cl")
    axes[1].semilogy(w["t"], left, color="k", label="time left until pericentre  t_peri − t")
    axes[1].axhline(T_PASS, color="0.5", lw=1, label="pericentre passage  r_p / v_p ≈ 0.25 yr")
    axes[1].axvline(T_PERI, color="k", ls=":", lw=0.8)
    axes[1].set_xlabel("t [yr]"); axes[1].set_ylabel("time [yr]"); axes[1].set_xlim(0, 30)
    axes[1].set_title("Why the cloud cannot adjust (adiabatic, 1024)"); axes[1].legend(fontsize=9)
    print("\n t [yr]  P_core/P_hot  T_core [K]  r_eff/R0  t_sc [yr]  t_peri - t [yr]")
    for tt in (0, 5, 10, 15, 17, 18, 20, 30):
        i = int(np.argmin(np.abs(w["t"] - tt))); j = int(np.argmin(np.abs(d["t"] - tt)))
        print(f" {w['t'][i]:6.1f}  {d['P_core'][j]:12.3f}  {d['T_core'][j]:10.3g}  {w['r_tr_R0'][i]:8.3f}  {t_sc[i]:9.3g}  "
              f"{(T_PERI - w['t'][i]) if w['t'][i] < T_PERI else float('nan'):10.2f}")
fig.savefig(os.path.join(BASE, "pressure_eq.png"), dpi=150)
plt.show()

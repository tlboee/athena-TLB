"""Temperatur des Wolkenmaterials T_cl(t) und Heizzeit t_heat fuer CC01 (512/1024/2048, adiabatisch / Kuehlung).

T = P mu m_H / (rho k_B), mu = 0.6139; Gewicht w = rho * tr. Normiert auf T_cl0 = gewichteter Median im 1. Snapshot.
T_wmed: massengewichteter Median von T;  f_cold: Massenanteil mit T < 3 T_cl0.
t_heat: erster Zeitpunkt mit T_wmed > 3 T_cl0.   Cache: <lauf>/heat.csv
Aufruf (Notebook):  %run ~/athena/analysis/heat_time.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, MU, MH, KB = 3.156e7, 0.6139, 1.6605e-24, 1.380648e-16
T_END, T_PERI = 50.0, 18.0          # t_peri CC01 ~ 18 yr (wie kh_plot.py)
RUNS = {"adiabatic": [("512", "ps_ref_big"), ("1024", "ps_ref_big_x2_t50"), ("2048", "ps_ref_big_x4_t50")],
        "with cooling": [("512", "ps_ref_big_cool_t50"), ("1024", "ps_ref_big_cool_x2_t50"),
                         ("2048", "ps_ref_big_cool_x4_t50")]}
COLS = {"512": "#440154", "1024": "#21918c", "2048": "#bddf26"}

def wquant(x, w, q):
    i = np.argsort(x)
    cw = np.cumsum(w[i])
    return float(x[i][np.searchsorted(cw, q * cw[-1])]) if cw[-1] > 0 else np.nan

def press_key(s):
    for k in ("press", "p", "P", "pres", "prs"):
        if k in s:
            return k
    raise SystemExit(f"Kein Druck-Schluessel gefunden, vorhanden: {list(s.keys())}")

def series(run):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "heat.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        if df["t_yr"].max() >= T_END - 0.5:
            return df
    out, T0 = [], None
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        if t / YR > T_END + 0.5:
            break
        rho, tr = s["rho"], s["r0"]
        T = s[press_key(s)] * MU * MH / (rho * KB)
        sel = tr > 1e-4
        Ts, w = T[sel], (rho * tr)[sel]
        Tw = wquant(Ts, w, 0.5)
        if T0 is None:
            T0 = Tw                                   # T_cl0 aus dem ersten Snapshot
        out.append([t / YR, Tw, Tw / T0, float(w[Ts < 3 * T0].sum() / w.sum())])
        print(f"  {run}: t = {t/YR:5.1f} yr", end="\r")
    print()
    df = pd.DataFrame(out, columns=["t_yr", "T_wmed", "T_rel", "f_cold"])
    df.to_csv(cache, index=False)
    return df

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
print(f"{'Lauf':26s} {'T_cl0 [K]':>10s} {'t_heat [yr]':>11s} {'f_cold +10':>10s} {'f_cold +30':>10s}")
for col, (kind, lst) in enumerate(RUNS.items()):
    for lab, run in lst:
        if not os.path.isdir(os.path.join(BASE, run)):
            print(f"{run:26s} fehlt"); continue
        df = series(run)
        hot = df[df["T_rel"] > 3]
        th = hot["t_yr"].iloc[0] if len(hot) else np.nan
        fc = [np.interp(T_PERI + dt, df["t_yr"], df["f_cold"]) for dt in (10, 30)]
        print(f"{run:26s} {df['T_wmed'].iloc[0]:10.3g} {th:11.3g} {fc[0]:10.2f} {fc[1]:10.2f}")
        axes[0, col].semilogy(df["t_yr"], df["T_rel"], color=COLS[lab], label=lab)
        axes[1, col].plot(df["t_yr"], df["f_cold"], color=COLS[lab], label=lab)
    axes[0, col].axhline(3, color="gray", ls=":")
    axes[0, col].set_title(kind)
    for ax in axes[:, col]:
        ax.axvline(T_PERI, color="k", ls="--", lw=0.8)
axes[0, 0].set_ylabel("T_cl / T_cl0  (mass-weighted median)")
axes[1, 0].set_ylabel("cold fraction  (T < 3 T_cl0)")
for ax in axes[1]:
    ax.set_xlabel("t [yr]")
axes[0, 0].legend()
fig.suptitle("CC01: cloud temperature and cold fraction (dashed: pericentre)")
fig.savefig(os.path.join(BASE, "heat_time.png"), dpi=150)
plt.show()

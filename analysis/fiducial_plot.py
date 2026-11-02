"""Fiducial-Mass rho/rho_eq > chi0/3 (lokal) und kaltes Wolkenmaterial (T < 3 T_cl) ueber die Zeit,
CC01 bei 512 / 1024 / 2048, adiabatisch und mit Kuehlung.

Fiducial: Spalte dl_1/3 aus mX.csv (study_table.py), Anteil von sum(rho tr) zu Beginn.
Kalt:     Spalte m_cold aus res_cool.csv (res_cooling.py), sum(rho tr) mit T < 3e4 K.
Bei gleichem Druck sind beide gleichwertig; der Unterschied zeigt, wie weit die Wolke vom Druckgleichgewicht entfernt ist.
Aufruf (Notebook):  %run ~/athena/analysis/fiducial_plot.py
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/athena/runs")
T_END = 50.0
RUNS = {"adiabatic": [("512", "ps_ref_big"), ("1024", "ps_ref_big_x2_t50"), ("2048", "ps_ref_big_x4_t50")],
        "with cooling": [("512", "ps_ref_big_cool_t50"), ("1024", "ps_ref_big_cool_x2_t50"),
                         ("2048", "ps_ref_big_cool_x4_t50")]}
COLS = {"512": "#440154", "1024": "#21918c", "2048": "#bddf26"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
print(f"{'Lauf':26s} {'chi0/3 +10':>11s} {'chi0/3 +30':>11s} {'kalt +10':>9s} {'kalt +30':>9s}")
for ax, (kind, lst) in zip(axes, RUNS.items()):
    for lab, run in lst:
        d = os.path.join(BASE, run)
        with open(os.path.join(d, "manifest.json")) as f:
            tp = json.load(f).get("t_peri_yr", 18.0)
        mx = pd.read_csv(os.path.join(d, "mX.csv"))
        mx = mx[mx["t_yr"] <= T_END + 0.01]
        rc = pd.read_csv(os.path.join(d, "res_cool.csv"))
        rc = rc[rc["t_yr"] <= T_END + 0.01]
        ax.plot(mx["t_yr"] - tp, mx["dl_1/3"], "-", color=COLS[lab], lw=2, label=f"{lab}: ρ/ρ_eq > χ₀/3")
        ax.plot(rc["t_yr"] - tp, rc["m_cold"], ":", color=COLS[lab], lw=1.8, label=f"{lab}: T < 3 T_cl")
        vals = []
        for df, col in [(mx, "dl_1/3"), (rc, "m_cold")]:
            for dt in (10, 30):
                vals.append(float(np.interp(tp + dt, df["t_yr"], df[col])))
        print(f"{run:26s} {vals[0]:11.2f} {vals[1]:11.2f} {vals[2]:9.2f} {vals[3]:9.2f}")
    ax.axvspan(-18, 0, color="0.92", zorder=0)
    ax.text(-17, 0.06, "before pericentre: cloud cannot be\ncompressed fast enough → contrast\nfalls with the orbit",
            fontsize=7, color="0.35", va="bottom")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlim(-18, T_END - 18); ax.set_ylim(0, 1.02)
    ax.set_xlabel("t − t_peri [yr]")
    ax.set_title(f"CC01, {kind}", fontsize=10)
axes[0].set_ylabel("fraction of initial cloud mass")
axes[1].legend(fontsize=7, ncol=2, loc="upper right")
fig.suptitle("Fiducial measure ρ/ρ_eq > χ₀/3 (solid) vs. cold gas T < 3 T_cl (dotted): "
             "with cooling, dense gas in pressure equilibrium reappears after pericentre", fontsize=10)
out = os.path.join(BASE, "fiducial.png")
fig.savefig(out, dpi=150)
print("Gespeichert:", out)

"""Herkunft des kalten Gases (T < 3 T_cl0) in den CC01-Kuehllaeufen: Wolkengas (rho*tr) vs. Umgebungsgas (rho*(1-tr)).

Normiert auf die anfaengliche Wolkenmasse sum(rho*tr) im 1. Snapshot (gleichmaessiges Gitter -> Zellsumme ~ Masse).
T = P mu m_H / (rho k_B), mu = 0.62.  Cache: <lauf>/cold_origin.csv
Aufruf (Notebook):  %run ~/athena/analysis/cold_origin.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, MU, MH, KB = 3.156e7, 0.62, 1.6605e-24, 1.380648e-16
T_END, T_PERI = 50.0, 18.0
RUNS = [("512", "ps_ref_big_cool_t50"), ("1024", "ps_ref_big_cool_x2_t50"), ("2048", "ps_ref_big_cool_x4_t50")]
COLS = {"512": "#440154", "1024": "#21918c", "2048": "#bddf26"}

def press_key(s):
    for k in ("press", "p", "P", "pres", "prs"):
        if k in s:
            return k
    raise SystemExit(f"Kein Druck-Schluessel gefunden, vorhanden: {list(s.keys())}")

def series(run):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "cold_origin.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        if df["t_yr"].max() >= T_END - 0.5:
            return df
    out, T0, M0 = [], None, None
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        if t / YR > T_END + 0.5:
            break
        rho, tr = s["rho"], np.clip(s["r0"], 0.0, 1.0)
        T = s[press_key(s)] * MU * MH / (rho * KB)
        if T0 is None:
            cl = tr > 0.99
            T0, M0 = float(np.median(T[cl])), float((rho * tr).sum())
        cold = T < 3 * T0
        out.append([t / YR, (rho * tr)[cold].sum() / M0, (rho * (1 - tr))[cold].sum() / M0,
                    (rho * tr).sum() / M0])
        print(f"  {run}: t = {t/YR:5.1f} yr", end="\r")
    print()
    df = pd.DataFrame(out, columns=["t_yr", "m_cold_cloud", "m_cold_amb", "m_cloud_tot"])
    df.to_csv(cache, index=False)
    return df

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True, constrained_layout=True)
print(f"{'Lauf':26s} {'Wolke +10':>9s} {'Umg. +10':>9s} {'Wolke +30':>9s} {'Umg. +30':>9s} {'Umg. max':>9s}")
for lab, run in RUNS:
    if not os.path.isdir(os.path.join(BASE, run)):
        print(f"{run:26s} fehlt"); continue
    df = series(run)
    v = [np.interp(T_PERI + dt, df["t_yr"], df[c]) for dt in (10, 30) for c in ("m_cold_cloud", "m_cold_amb")]
    print(f"{run:26s} {v[0]:9.3f} {v[1]:9.3f} {v[2]:9.3f} {v[3]:9.3f} {df['m_cold_amb'].max():9.3f}")
    axes[0].plot(df["t_yr"], df["m_cold_cloud"], color=COLS[lab], label=f"{lab} cloud gas")
    axes[0].plot(df["t_yr"], df["m_cold_cloud"] + df["m_cold_amb"], color=COLS[lab], ls="--",
                 label=f"{lab} all gas")
    axes[1].plot(df["t_yr"], df["m_cold_amb"], color=COLS[lab], label=lab)
for ax in axes:
    ax.axvline(T_PERI, color="k", ls=":", lw=0.8)
    ax.set_xlabel("t [yr]")
axes[0].set_ylabel("cold mass / initial cloud mass")
axes[1].set_ylabel("cold ambient mass / initial cloud mass")
axes[0].set_title("cold gas (T < 3 T_cl0): cloud only (solid) vs. all (dashed)")
axes[1].set_title("cold gas of ambient origin  ρ(1 − tr)")
axes[0].legend(fontsize=8, ncol=2)
fig.savefig(os.path.join(BASE, "cold_origin.png"), dpi=150)
plt.show()

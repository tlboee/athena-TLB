"""Kontrast des Wolkenkerns: Median ueber Zellen vs. massengewichteter Median vs. 90. Perzentil,
fuer CC01 bei 512 / 1024 / 2048, adiabatisch und mit Kuehlung, verglichen mit dem idealisierten chi ~ 1/r.

Kontrast c = rho / rho_eq(r) (ungestoerte Atmosphaere); Gewicht w = rho * tr (Wolkenmaterial).
Ergebnisse je Lauf in <lauf>/core_contrast.csv (Cache).  Aufruf (Notebook):  %run ~/athena/analysis/core_contrast.py
"""
import json, math, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, KM = 3.156e7, 1.0e5
GM = 6.674e-8 * 4.31e6 * 1.989e33
X0, Y0 = -7.22e16, 2.21e16
R_START = math.hypot(X0, Y0)
RHO0, R0, R_IN = 1.702e-21, 1.0e16, 1.9e15
T_END = 50.0
RUNS = {"adiabatic": [("512", "ps_ref_big"), ("1024", "ps_ref_big_x2_t50"), ("2048", "ps_ref_big_x4_t50")],
        "with cooling": [("512", "ps_ref_big_cool_t50"), ("1024", "ps_ref_big_cool_x2_t50"),
                         ("2048", "ps_ref_big_cool_x4_t50")]}
COLS = {"512": "#440154", "1024": "#21918c", "2048": "#bddf26"}
MEAS = [("c_cellmed", "cell median (tracer > 0.5)"), ("c_wmed", "mass-weighted median"),
        ("c_w90", "mass-weighted 90th percentile (core)")]

def wquant(x, w, q):
    i = np.argsort(x)
    cw = np.cumsum(w[i])
    return float(x[i][np.searchsorted(cw, q * cw[-1])]) if cw[-1] > 0 else np.nan

def series(run):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "core_contrast.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        if df["t_yr"].max() >= T_END - 0.5:
            return df
    rows = []
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        if t / YR > T_END + 0.5:
            break
        X, Y = np.meshgrid(s["x"], s["y"])
        rho, tr = s["rho"], s["r0"]
        c = rho / (RHO0 * R0 / np.maximum(np.hypot(X, Y), R_IN))
        w = rho * tr
        sel = tr > 1e-4
        cs, ws = c[sel], w[sel]
        m9 = tr > 0.9
        rows.append([t / YR,
                     float(np.median(c[tr > 0.5])) if (tr > 0.5).any() else np.nan,
                     wquant(cs, ws, 0.5), wquant(cs, ws, 0.9),
                     float((c[m9] * w[m9]).sum() / w[m9].sum()) if m9.any() else np.nan])
        print(f"  {run}: t = {t/YR:5.1f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t_yr", "c_cellmed", "c_wmed", "c_w90", "c_mean09"])
    df.to_csv(cache, index=False)
    return df

def kepler_r(vx, vy, t_yr):
    ux, uy = vx * KM, vy * KM
    E = 0.5 * (ux * ux + uy * uy) - GM / R_START
    h = X0 * uy - Y0 * ux
    a = -GM / (2 * E)
    e = math.sqrt(1 + 2 * E * h * h / GM**2)
    n = math.sqrt(GM / a**3)
    Ea = math.acos(max(-1.0, min(1.0, (1 - R_START / a) / e)))
    M0 = -(Ea - e * math.sin(Ea)) if X0 * ux + Y0 * uy < 0 else Ea - e * math.sin(Ea)
    M = M0 + n * np.asarray(t_yr) * YR
    Ek = M + 0.85 * e * np.sign(np.sin(M))
    for _ in range(50):
        Ek = Ek - (Ek - e * np.sin(Ek) - M) / (1 - e * np.cos(Ek))
    return a * (1 - e * np.cos(Ek))

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True, sharey=True, constrained_layout=True)
print(f"{'Lauf':26s} {'t-t_p':>6s} {'Zellmed.':>9s} {'w-Med.':>8s} {'w-P90':>8s} {'tr>0.9':>8s} {'ideal':>8s}")
for row, (kind, lst) in enumerate(RUNS.items()):
    for lab, run in lst:
        with open(os.path.join(BASE, run, "manifest.json")) as f:
            man = json.load(f)
        tp = man.get("t_peri_yr", 18.0)
        df = series(run)
        df = df[df["t_yr"] <= T_END + 0.01]
        x = df["t_yr"].to_numpy() - tp
        chi0 = df["c_cellmed"].iloc[0]
        for col, (key, ti) in enumerate(MEAS):
            axes[row, col].plot(x, df[key], color=COLS[lab], label=f"{lab}")
        rid = kepler_r(man["vx_kms"], man["vy_kms"], df["t_yr"].to_numpy())
        ideal = chi0 * R_START / rid
        for dt in (-10, -5, -1, 5, 20):
            i = np.argmin(np.abs(x - dt))
            if abs(x[i] - dt) < 0.6:
                print(f"{run:26s} {x[i]:6.1f} {df['c_cellmed'].iloc[i]:9.0f} {df['c_wmed'].iloc[i]:8.0f} "
                      f"{df['c_w90'].iloc[i]:8.0f} {df['c_mean09'].iloc[i]:8.0f} {ideal[i]:8.0f}")
    tt = np.linspace(0, T_END, 2000)
    rid = kepler_r(794.59, 48.45, tt)
    for col, (key, ti) in enumerate(MEAS):
        ax = axes[row, col]
        ax.plot(tt - 18.0, 2755 * R_START / rid, "k--", lw=1.2, label="idealised χ ∝ 1/r")
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_yscale("log"); ax.set_title(f"{kind}: {ti}", fontsize=9)
        if row == 1:
            ax.set_xlabel("t − t_peri [yr]")
    axes[row, 0].set_ylabel("cloud contrast ρ/ρ_eq(r)")
axes[0, 0].legend(fontsize=7, title="resolution (cells, x)", title_fontsize=7)
fig.suptitle("CC01: cloud contrast relative to the unperturbed atmosphere – which measure follows the idealised χ ∝ 1/r?",
             fontsize=11)
out = os.path.join(BASE, "core_contrast.png")
fig.savefig(out, dpi=150)
print("Gespeichert:", out)

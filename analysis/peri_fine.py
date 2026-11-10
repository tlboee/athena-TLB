"""Zeitreihe um das Perizentrum aus den Neustarts mit Snapshots alle 0.05 yr (17-19.5 yr), CC01 adiabatisch 1024/2048.

Kern = tr > 0.9, Gewicht rho*tr, gewichtete Mediane von:
  T/T_cl0, T_ad/T_cl0 = (rho/rho_cl0)^(2/3) (rein adiabatisch), K/K0 mit K = P/rho^gamma (Stoss -> Sprung, Kompression -> konstant),
  rho/rho_cl0, div v [1/yr] (+ Anteil < -1/yr), P/P_hot(r), q = T / [tr T_cl0 + (1-tr) T_hot(r)].
T_cl0, rho_cl0, K0 aus dem 1. Snapshot des Originallaufs.  Cache: <lauf>/peri_fine.csv
Bild 1: runs/peri_fine.png (Zeitreihen), Bild 2: runs/peri_fine_maps.png (log K/K0, 2048, um den steilsten Anstieg).
Aufruf (Notebook):  %run ~/athena/analysis/peri_fine.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, MU, MH, KB, AU, GAM = 3.156e7, 0.6139, 1.6605e-24, 1.380648e-16, 1.496e13, 5.0 / 3.0
RHO0, T0, R0, R_IN, T_PERI = 1.702e-21, 2.128e8, 1.0e16, 1.9e15, 18.0
RUNS = [("1024", "ps_ref_big_x2_peri", "ps_ref_big_x2_t50"), ("2048", "ps_ref_big_x4_peri", "ps_ref_big_x4_t50")]
COLS = {"1024": "#21918c", "2048": "#bddf26"}

def key(s, cands):
    for k in cands:
        if k in s:
            return k
    raise SystemExit(f"Schluessel {cands} fehlt, vorhanden: {list(s.keys())}")

def wmed(x, w):
    i = np.argsort(x); cw = np.cumsum(w[i])
    return float(x[i][np.searchsorted(cw, 0.5 * cw[-1])]) if len(x) and cw[-1] > 0 else np.nan

def fields(s):
    rho, tr = s["rho"], np.clip(s["r0"], 0.0, 1.0)
    P = s[key(s, ("press", "p", "P", "pres", "prs"))]
    return rho, tr, P, P * MU * MH / (rho * KB)

def ref_state(orig):
    d = os.path.join(BASE, orig)
    _, s = g2_io.load_snapshot(d, list(g2_io.snapshot_numbers(d))[0])
    rho, tr, P, T = fields(s)
    cl = tr > 0.99
    return float(np.median(T[cl])), float(np.median(rho[cl])), float(np.median((P / rho**GAM)[cl]))

def series(run, orig):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "peri_fine.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    T_cl0, rho_cl0, K0 = ref_state(orig)
    rows = []
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        x, y = s["x"], s["y"]
        X, Y = np.meshgrid(x, y)
        r = np.maximum(np.hypot(X, Y), R_IN)
        rho, tr, P, T = fields(s)
        T_h = T0 * R0 / r
        P_h = (RHO0 * R0 / r) * KB * T_h / (MU * MH)
        div = (np.gradient(s[key(s, ("vel1", "vx", "v1"))], x, axis=1)
               + np.gradient(s[key(s, ("vel2", "vy", "v2"))], y, axis=0)) * YR
        c = tr > 0.9
        w = (rho * tr)[c]
        rows.append([t / YR, wmed(T[c] / T_cl0, w), wmed((rho[c] / rho_cl0)**(2 / 3), w),
                     wmed((P / rho**GAM)[c] / K0, w), wmed(rho[c] / rho_cl0, w), wmed(div[c], w),
                     float(w[div[c] < -1].sum() / w.sum()), wmed((P / P_h)[c], w),
                     wmed((T / (tr * T_cl0 + (1 - tr) * T_h))[c], w), float(w.sum())])
        print(f"  {run}: t = {t/YR:6.2f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t", "T", "T_ad", "K", "rho", "div", "f_comp", "P_Ph", "q", "m_core"])
    df.to_csv(cache, index=False)
    return df

panels = [("T", "T / T_cl0  (dashed: adiabatic (ρ/ρ_cl0)^(2/3))", True), ("K", "K / K₀  (entropy, P/ρ^γ)", True),
          ("rho", "ρ / ρ_cl0", True), ("div", "div v [1/yr]  (dotted: fraction < −1/yr, ×10)", False),
          ("P_Ph", "P / P_hot(r)", True), ("q", "q = T / T_mix,expected", True)]
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, constrained_layout=True)
data = {}
print(f"{'Aufl.':>5s} {'t(max dlogK/dt)':>15s} {'K/K0 davor':>10s} {'danach':>8s} {'T/T_cl0 davor':>13s} {'danach':>8s}")
for lab, run, orig in RUNS:
    if not os.path.isfile(os.path.join(BASE, run, "g2_cart.final.rst")):
        print(f"{run} fehlt oder noch nicht fertig"); continue
    df = series(run, orig)
    data[lab] = df
    g = np.gradient(np.log10(df["K"]), df["t"])
    i = int(np.nanargmax(g)); j0, j1 = max(i - 2, 0), min(i + 2, len(df) - 1)
    print(f"{lab:>5s} {df['t'][i]:15.2f} {df['K'][j0]:10.3g} {df['K'][j1]:8.3g} {df['T'][j0]:13.3g} {df['T'][j1]:8.3g}")
    for ax, (k, ttl, lg) in zip(axes.flat, panels):
        ax.plot(df["t"], df[k], color=COLS[lab], label=lab)
        if k == "T":
            ax.plot(df["t"], df["T_ad"], color=COLS[lab], ls="--", lw=1)
        if k == "div":
            ax.plot(df["t"], -10 * df["f_comp"], color=COLS[lab], ls=":", lw=1)
for ax, (k, ttl, lg) in zip(axes.flat, panels):
    if lg: ax.set_yscale("log")
    ax.axvline(T_PERI, color="k", ls="--", lw=0.8); ax.set_title(ttl, fontsize=10)
for ax in axes[1]:
    ax.set_xlabel("t [yr]")
axes[0, 0].legend()
fig.suptitle("CC01 adiabatic, cloud core (tracer > 0.9), snapshots every 0.05 yr; dashed vertical = pericentre")
fig.savefig(os.path.join(BASE, "peri_fine.png"), dpi=150)
plt.show()

# Entropiekarten 2048 um den steilsten Anstieg von K
if "2048" in data:
    df = data["2048"]
    g = np.gradient(np.log10(df["K"]), df["t"]); tk = df["t"][int(np.nanargmax(g))]
    d = os.path.join(BASE, "ps_ref_big_x4_peri"); nums = list(g2_io.snapshot_numbers(d))
    _, _, K0 = ref_state("ps_ref_big_x4_t50")
    fig2, axs = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    for ax, dt in zip(axs, (-0.1, 0.0, 0.1, 0.2)):
        n = nums[int(np.argmin(np.abs(df["t"].values - (tk + dt))))]
        t, s = g2_io.load_snapshot(d, n)
        rho, tr, P, T = fields(s)
        xa, ya = s["x"] / AU, s["y"] / AU
        jj, ii = np.where(tr > 1e-3)
        x0, x1, y0, y1 = xa[ii.min()], xa[ii.max()], ya[jj.min()], ya[jj.max()]
        px, py = 0.1 * (x1 - x0) + 1, 0.1 * (y1 - y0) + 1
        im = ax.imshow(np.log10(P / rho**GAM / K0), origin="lower", extent=(xa[0], xa[-1], ya[0], ya[-1]),
                       cmap="magma", vmin=0, vmax=8, interpolation="nearest")
        ax.contour(xa, ya, tr, levels=[0.1, 0.9], colors=["c", "lime"], linewidths=0.7)
        ax.plot(0, 0, "w+", ms=8)
        ax.set_xlim(x0 - px, x1 + px); ax.set_ylim(y0 - py, y1 + py); ax.set_aspect("equal")
        ax.set_title(f"t = {t/YR:.2f} yr"); ax.set_xlabel("x [AU]")
    axs[0].set_ylabel("y [AU]")
    fig2.colorbar(im, ax=axs, label="log₁₀ K / K₀  (entropy)", shrink=0.8)
    fig2.suptitle("CC01 2048 adiabatic: entropy around the steepest entropy rise; tracer 0.1 (cyan), 0.9 (green)")
    fig2.savefig(os.path.join(BASE, "peri_fine_maps.png"), dpi=150)
    plt.show()

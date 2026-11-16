"""Windkanal unter Perizentrumsbedingungen (ohne Gezeiten) gegen die Bahn.

Laeufe: wt_L1_peri_R1 (R0) und wt_L1_peri_R05 (R0/2): adiabatisch, rho_amb und T_amb bei r_p = 4e15 cm,
v = 5270 km/s (Mach ~1.5), rho_cl = rho_cl0 -> chi ~ 146.
Links: Anteil der anfaenglichen Wolkenmasse (rho*tr): unvermischt (tracer > 0.9), dicht (rho/rho_amb > sqrt(chi)),
nicht geheizt (T < 3 T_cl0).
Rechts (fairer Vergleich, unabhaengig von T_cl0): massengewichteter Median des beigemischten Umgebungsgases 1 - tr
im Wolkengas (tr > 1e-4, Gewicht rho*tr); Windkanal gegen t, Bahn (CC01 1024, jaehrlich und fein) gegen t - t_peri.
Caches: <lauf>/wt_peri2.csv, <lauf>/mixfrac.csv.   Ausgabe: runs/wt_peri.png.   Aufruf:  %run ~/athena/analysis/wt_peri.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/athena/analysis"))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, MU, MH, KB = 3.156e7, 0.6139, 1.6605e-24, 1.380648e-16
T_PERI, T_PASS, V = 18.0, 0.25, 5.27e8
RUNS = [("wind tunnel, R0", "wt_L1_peri_R1", 1.87e15, "#c0392b"), ("wind tunnel, R0/2", "wt_L1_peri_R05", 9.35e14, "#d9822b")]
ORBIT = [("orbit 1024", "ps_ref_big_x2_t50", "#21918c", "--"), ("orbit 1024, fine", "ps_ref_big_x2_peri", "#0b5d55", "-")]

def wmedian(x, w):
    i = np.argsort(x); c = np.cumsum(w[i])
    return float(x[i][np.searchsorted(c, 0.5 * c[-1])])

def series(run):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "wt_peri2.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows, M0, T0, rho_amb = [], None, None, None
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        rho, tr = s["rho"], np.clip(s["r0"], 0, 1)
        T = s["press"] * MU * MH / (rho * KB)
        w = rho * tr
        if M0 is None:
            M0, T0 = float(w.sum()), float(np.median(T[tr > 0.99]))
            rho_amb, chi = float(np.median(rho[tr < 1e-6])), float(np.median(rho[tr > 0.99])) / float(np.median(rho[tr < 1e-6]))
        cl = tr > 1e-4
        rows.append([t / YR, float(w[tr > 0.9].sum() / M0), float(w[rho / rho_amb > np.sqrt(chi)].sum() / M0),
                     float(w[cl & (T < 3 * T0)].sum() / M0), wmedian(1 - tr[cl], w[cl])])
        print(f"  {run}: t = {t/YR:6.3f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t", "unmixed", "dense", "unheated", "mix_med"]); df.to_csv(cache, index=False)
    return df

def orbit_mix(run):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "mixfrac.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        tr = np.clip(s["r0"], 0, 1); cl = tr > 1e-4
        rows.append([t / YR, wmedian(1 - tr[cl], (s["rho"] * tr)[cl])])
        print(f"  {run}: t = {t/YR:6.2f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t", "mix_med"]); df.to_csv(cache, index=False)
    return df

fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)
print(f"\n{'Lauf':20s} {'t_cc [yr]':>9s}   Anteil bei t = 0.25 / 1 / 2 yr:  unvermischt | dicht | nicht geheizt   |  Median 1-tr")
for lab, run, R, c in RUNS:
    if not os.path.isdir(os.path.join(BASE, run)):
        print(f"{run} fehlt"); continue
    df = series(run)
    tcc = np.sqrt(146.0) * R / V / YR
    for col, ls in (("unmixed", "-"), ("dense", "--"), ("unheated", ":")):
        axes[0].plot(df["t"], df[col], color=c, ls=ls, label=f"{lab}: {col}")
    axes[1].semilogy(df["t"], np.maximum(df["mix_med"], 1e-6), color=c, label=f"{lab} (no tides)")
    vals = []
    for tt in (0.25, 1.0, 2.0):
        i = int(np.argmin(np.abs(df["t"] - tt)))
        vals.append(f"{df['unmixed'][i]:.2f} | {df['dense'][i]:.2f} | {df['unheated'][i]:.2f} | {df['mix_med'][i]:.1e}")
    print(f"{lab:20s} {tcc:9.2f}   " + "  //  ".join(vals))
for lab, run, c, ls in ORBIT:
    if not os.path.isdir(os.path.join(BASE, run)):
        print(f"{run} fehlt"); continue
    o = orbit_mix(run); m = (o["t"] > T_PERI - 3) & (o["t"] < T_PERI + 8)
    axes[1].semilogy(o["t"][m] - T_PERI, np.maximum(o["mix_med"][m], 1e-6), color=c, ls=ls, label=f"{lab} (t − t_peri)")
    print(f"{lab:20s}  Median 1-tr bei t-t_peri = -1 / 0 / 0.25 / 1 / 2 yr: " + " | ".join(
        f"{o['mix_med'][int(np.argmin(np.abs(o['t'] - T_PERI - dt)))]:.1e}" for dt in (-1, 0, 0.25, 1, 2)))
for ax in axes:
    ax.axvline(T_PASS, color="0.5", lw=1)
axes[0].set_xlabel("t [yr]"); axes[0].set_ylabel("fraction of initial cloud mass")
axes[0].set_title("Wind tunnel at pericentre conditions (χ ≈ 146, Mach 1.5, adiabatic)"); axes[0].legend(fontsize=8, ncol=2)
axes[1].axvline(0, color="k", ls=":", lw=0.8)
axes[1].set_xlabel("t [yr]  (wind tunnel)  /  t − t_peri [yr]  (orbit)")
axes[1].set_ylabel("median admixed ambient fraction 1 − tr of cloud gas")
axes[1].set_title("Mixing: wind tunnel without tides vs. orbit (grey line: passage time 0.25 yr)"); axes[1].legend(fontsize=9)
axes[1].set_xlim(-3, 8)
fig.savefig(os.path.join(BASE, "wt_peri.png"), dpi=150)
plt.show()

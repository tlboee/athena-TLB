"""Windkanal (bewegte Wolke, homogenes Medium, Kuehlung): kalte Masse gegen t / t_cc fuer Lambda_fac = 1 / 0.1 / 0.01,
verglichen mit den G2-Laeufen (CC01 bei 1024, gleiche Lambda_fac, Cache cold_all.csv aus max_plots.py).
Kalt: T < 3 T_cl0 und Tracer > 1e-4; getrennt Wolkengas rho*tr und Umgebungsgas rho*(1 - tr); / anfaengliche Wolkenmasse.
Cache: <lauf>/wt_cold.csv.   Ausgabe: runs/wt_cold.png.   Aufruf (Notebook):  %run ~/athena/analysis/wt_cold.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, MU, MH, KB = 3.156e7, 0.6139, 1.6605e-24, 1.380648e-16
T_CC_WT = np.sqrt(2818) * 1.87e15 / 8.0e7 / YR      # 39.3 yr
T_CC_G2, T_PERI = 39.1, 18.0
CASES = [("6.4", "1", "ps_ref_big_cool_x2_t50", "#c0392b"), ("0.64", "0.1", "ps_ref_big_cool_L0.1_x2_t50", "#d9822b"),
         ("0.064", "0.01", "ps_ref_big_cool_L0.01_x2_t50", "#2f6db3")]

def series(run):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "wt_cold.csv")
    if os.path.exists(cache):
        c = pd.read_csv(cache)
        if c["t_yr"].max() >= 389.0:
            return c
    rows, T0, M0 = [], None, None
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        rho, tr = s["rho"], np.clip(s["r0"], 0, 1)
        T = s["press"] * MU * MH / (rho * KB)
        if T0 is None:
            T0, M0 = float(np.median(T[tr > 0.99])), float((rho * tr).sum())
        cold = (T < 3 * T0) & (tr > 1e-4)
        rows.append([t / YR, float((rho * tr)[cold].sum() / M0), float((rho * (1 - tr))[cold].sum() / M0),
                     float((rho * tr).sum() / M0)])
        print(f"  {run}: t = {t/YR:6.1f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t_yr", "m_cold_cloud", "m_cold_amb", "m_cloud_tot"])
    if df["t_yr"].max() >= 389.0:               # nur vollstaendige Laeufe cachen (final.rst entsteht auch bei Abbruch)
        df.to_csv(cache, index=False)
    return df

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True, constrained_layout=True)
print(f"{'r/r_crit':>8s} {'t/t_cc':>6s} {'kalt gesamt':>11s} {'davon Umgebung':>14s}")
for rr, fac, g2run, c in CASES:
    run = f"wt_L{fac}"
    if not os.path.isdir(os.path.join(BASE, run)):
        print(f"{run} fehlt"); continue
    df = series(run)
    tot = df["m_cold_cloud"] + df["m_cold_amb"]
    axes[0].plot(df["t_yr"] / T_CC_WT, tot, color=c, label=f"r/r_crit ≈ {rr}: all cold gas")
    axes[0].plot(df["t_yr"] / T_CC_WT, df["m_cold_cloud"], color=c, ls="--", lw=1, label=f"r/r_crit ≈ {rr}: cloud gas only")
    for tt in (1, 3, 5, 10):
        if df["t_yr"].max() / T_CC_WT >= tt:
            i = int(np.argmin(np.abs(df["t_yr"] / T_CC_WT - tt)))
            print(f"{rr:>8s} {tt:6d} {tot[i]:11.3f} {df['m_cold_amb'][i]:14.3f}")
    p = os.path.join(BASE, g2run, "cold_all.csv")
    if os.path.exists(p):
        g = pd.read_csv(p)
        axes[1].plot(g["t_yr"] / T_CC_G2, g["m_cold"], color=c, label=f"r/r_crit ≈ {rr}")
axes[0].axhline(1, color="0.6", lw=0.8); axes[1].axhline(1, color="0.6", lw=0.8)
axes[1].axvline(T_PERI / T_CC_G2, color="k", ls=":", lw=0.8)
axes[0].set_title("Wind tunnel: moving cloud in uniform hot medium (CC01 values, Mach ≈ 1)")
axes[1].set_title("G2 orbit (CC01, 1024; dotted: pericentre)")
for ax in axes:
    ax.set_xlabel("t / t_cc   (t_cc ≈ 39 yr)")
axes[0].set_ylabel("m(T < 3 T_cl0) / m_0")
axes[0].legend(fontsize=8); axes[1].legend(fontsize=9)
fig.savefig(os.path.join(BASE, "wt_cold.png"), dpi=150)
plt.show()
print("Gespeichert: runs/wt_cold.png")

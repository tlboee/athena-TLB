"""r_crit-Test: kaltes Wolkenmaterial (T < 3e4 K) fuer Lambda_fac = Dichte-Normierung f = 1, 0.1, 0.01,
CC01 bei 512 und 1024, adiabatisch als Referenz. Cache wie res_cooling.py (<lauf>/res_cool.csv).
Aufruf (Notebook):  %run ~/athena/analysis/lambda_test.py
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, KB, MH, MU = 3.156e7, 1.380649e-16, 1.6726e-24, 0.6139
T_END = 50.0
RATIO = {1.0: (0.155, 0.039), 0.1: (1.55, 0.39), 0.01: (15.5, 3.9)}   # t_cool,mix/t_cc Start / Perizentrum
RES = {"512": {"adia": "ps_ref_big", 1.0: "ps_ref_big_cool_t50", 0.1: "ps_ref_big_cool_L0.1_t50",
               0.01: "ps_ref_big_cool_L0.01_t50"},
       "1024": {"adia": "ps_ref_big_x2_t50", 1.0: "ps_ref_big_cool_x2_t50", 0.1: "ps_ref_big_cool_L0.1_x2_t50",
                0.01: "ps_ref_big_cool_L0.01_x2_t50"}}
COL = {1.0: "C0", 0.1: "C1", 0.01: "C3"}

def series(run):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "res_cool.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        if "m_cold_all" in df.columns and df["t_yr"].max() >= T_END - 0.5:
            return df
    rows = []
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        if t / YR > T_END + 0.5:
            break
        rho, tr = s["rho"], s["r0"]
        T = s["press"] * MU * MH / (rho * KB)
        m, cold = rho * tr, T < 3e4
        rows.append([t / YR, m.sum(), m[cold].sum(), rho[cold & (tr >= 1e-4)].sum(), m[tr > 0.9].sum()])
        print(f"  {run}: t = {t/YR:5.1f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t_yr", "m_dom", "m_cold", "m_cold_all", "m_unmixed"])
    m0 = df["m_dom"].iloc[0]
    for c in ("m_dom", "m_cold", "m_cold_all", "m_unmixed"):
        df[c] = df[c] / m0
    df.to_csv(cache, index=False)
    return df

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
print(f"{'Aufl.':5s} {'f':>6s} {'kalt +5':>8s} {'+10':>6s} {'+20':>6s} {'+32':>6s}")
for ax, (lab, runs) in zip(axes, RES.items()):
    for key in ("adia", 1.0, 0.1, 0.01):
        run = runs[key]
        if not os.path.exists(os.path.join(BASE, run, "g2_cart.hst")):
            print(f"  {run}: noch nicht vorhanden"); continue
        df = series(run)
        with open(os.path.join(BASE, run, "manifest.json")) as f:
            tp = json.load(f).get("t_peri_yr", 18.0)
        df = df[df["t_yr"] <= T_END + 0.01]
        x = df["t_yr"] - tp
        if key == "adia":
            ax.plot(x, df["m_cold"], "k--", lw=1.2, label="adiabatic")
        else:
            rs, rp = RATIO[key]
            ax.plot(x, df["m_cold"], "-", color=COL[key], lw=2,
                    label=f"f = {key:g}: R/r_crit = {1/rs:.2g} (start), {1/rp:.2g} (peri)")
        vals = [float(np.interp(tp + dt, df["t_yr"], df["m_cold"])) for dt in (5, 10, 20, 32)]
        print(f"{lab:5s} {str(key):>6s} " + " ".join(f"{v:6.2f}" for v in vals))
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlim(-18, 32); ax.set_ylim(0, 1.02); ax.set_xlabel("t − t_peri [yr]")
    ax.set_title(f"CC01, {lab} cells: cold cloud gas T < 3·10⁴ K", fontsize=10)
axes[0].set_ylabel("fraction of initial cloud mass")
axes[1].legend(fontsize=7, loc="upper right")
fig.suptitle("Survival test (Gronke & Oh): density normalisation f ≡ Lambda_fac moves R relative to r_crit", fontsize=11)
out = os.path.join(BASE, "lambda_test.png")
fig.savefig(out, dpi=150)
print("Gespeichert:", out)

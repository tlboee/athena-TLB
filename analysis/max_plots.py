"""Zwei Abbildungen fuer Max (Treffen 12.11.):
(1) T_cl / T_cl0 (massengewichteter Median, Wolkengas) mit und ohne Kuehlung in einem Feld, aus <lauf>/heat.csv.
(2) m(T < 3 T_cl0) / m0 gegen t / t_cc fuer CC01 bei 1024 mit Lambda_fac = 1 / 0.1 / 0.01 und adiabatisch
    (Gronke-&-Oh-Stil); kalte Masse = alles Gas (Wolke + Umgebung), normiert auf die anfaengliche Wolkenmasse.
Cache fuer (2): <lauf>/cold_all.csv.   Ausgabe: runs/max_T_cool.png, runs/max_cold_tcc.png
Aufruf (Notebook):  %run ~/athena/analysis/max_plots.py
"""
import glob, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, MU, MH, KB = 3.156e7, 0.6139, 1.6605e-24, 1.380648e-16
T_PERI, T_CC = 18.0, 39.1
COLS = {"512": "#440154", "1024": "#21918c", "2048": "#bddf26"}

# (1) Temperatur mit / ohne Kuehlung
fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
for lab, ad, co in (("512", "ps_ref_big", "ps_ref_big_cool_t50"), ("1024", "ps_ref_big_x2_t50", "ps_ref_big_cool_x2_t50"),
                    ("2048", "ps_ref_big_x4_t50", "ps_ref_big_cool_x4_t50")):
    for run, ls, kind in ((ad, "-", "adiabatic"), (co, "--", "with cooling")):
        p = os.path.join(BASE, run, "heat.csv")
        if not os.path.exists(p):
            print(f"{run}: keine heat.csv"); continue
        d = pd.read_csv(p)
        ax.semilogy(d["t_yr"], d["T_rel"], color=COLS[lab], ls=ls, label=f"{lab}, {kind}")
ax.axvline(T_PERI, color="k", ls=":", lw=0.8); ax.axhline(3, color="0.6", lw=0.8)
ax.set_xlabel("t [yr]"); ax.set_ylabel("T_cl / T_cl0  (mass-weighted median of cloud gas)")
ax.set_title("CC01: cloud temperature with and without cooling (dotted: pericentre)")
ax.legend(fontsize=9, ncol=2)
fig.savefig(os.path.join(BASE, "max_T_cool.png"), dpi=150)

# (2) kalte Masse gegen t / t_cc
def find(pattern):
    m = sorted(glob.glob(os.path.join(BASE, pattern)))
    m = [d for d in m if not d.endswith("_fine")]
    return os.path.basename(m[0]) if m else None

def cold_series(run, ref="ps_ref_big_x2_t50"):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "cold_all.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rd = os.path.join(BASE, ref)
    _, s0 = g2_io.load_snapshot(rd, list(g2_io.snapshot_numbers(rd))[0])
    tr0 = np.clip(s0["r0"], 0, 1)
    T0 = float(np.median((s0["press"] * MU * MH / (s0["rho"] * KB))[tr0 > 0.99]))
    M0 = float((s0["rho"] * tr0).sum())
    rows = []
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        T = s["press"] * MU * MH / (s["rho"] * KB)
        cold = (T < 3 * T0) & (np.clip(s["r0"], 0, 1) > 1e-4)
        rows.append([t / YR, float(s["rho"][cold].sum() / M0)])
        print(f"  {run}: t = {t/YR:5.1f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t_yr", "m_cold"]); df.to_csv(cache, index=False)
    return df

runs = [("adiabatic", "ps_ref_big_x2_t50", "k"),
        ("r/r_crit ≈ 0.064  (Λ × 0.01)", find("ps_ref_big_cool_L0.01_x2*"), "#2f6db3"),
        ("r/r_crit ≈ 0.64  (Λ × 0.1)", find("ps_ref_big_cool_L0.1_x2*"), "#d9822b"),
        ("r/r_crit ≈ 6.4  (Λ × 1)", find("ps_ref_big_cool_x2*"), "#c0392b")]
fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
for lab, run, c in runs:
    if run is None:
        print(f"{lab}: Lauf nicht gefunden"); continue
    df = cold_series(run)
    ax.plot(df["t_yr"] / T_CC, df["m_cold"], color=c, label=f"{lab}  [{run}]")
ax.axvline(T_PERI / T_CC, color="k", ls=":", lw=0.8)
ax.set_xlabel("t / t_cc   (t_cc = 39 yr)"); ax.set_ylabel("m(T < 3 T_cl0) / m_0")
ax.set_title("CC01 at 1024: cold mass vs. time for different r/r_crit (dotted: pericentre)")
ax.legend(fontsize=9)
fig.savefig(os.path.join(BASE, "max_cold_tcc.png"), dpi=150)
plt.show()
print("Gespeichert: runs/max_T_cool.png, runs/max_cold_tcc.png")

"""Konvergenz von alpha = t_50 / t_cc fuer chi = 10, r_p = 4e15 cm bei ~7 / 13 / 27 Zellen pro Wolkenradius.

Laeufe: g_chi10_rp4 (512), g_chi10_rp4_x2_t30 (1024), g_chi10_rp4_R1_x4_t30 (2048), Werte aus runs/study_table.csv
(vorher study_table.py fuer den neuen Lauf laufen lassen).  Richardson bei Verdopplung der Aufloesung:
p = log2[(a2 - a1) / (a3 - a2)],  alpha_inf = a3 + (a3 - a2) / (2^p - 1)   (nur sinnvoll, wenn p > 0).
Aufruf:  python3 analysis/alpha_conv.py
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/athena/runs")
RUNS = [("~7", "g_chi10_rp4"), ("~13", "g_chi10_rp4_x2_t30"), ("~27", "g_chi10_rp4_R1_x4_t30")]
tab = pd.read_csv(os.path.join(BASE, "study_table.csv"))
tab = tab[(tab["Y"] == "1/2") & (tab["norm"] == "init")]
for X, name in (("0.9", "Mixing (tracer > 0.9)"), ("dq", "dichter Strom (rho/rho_eq > sqrt(chi0))"), ("da_1/3", "rho > rho_cl0/3 (SB15)")):
    a = []
    print(f"\n=== {name}")
    for N, run in RUNS:
        s = tab[(tab["run"] == run) & (tab["X"] == X)]
        if s.empty or not np.isfinite(s["t"].iloc[0]):
            print(f"  {N:>4s} Zellen/R  {run:24s} fehlt oder t_50 nicht erreicht"); a.append(np.nan); continue
        r = s.iloc[0]
        a.append(r["t"] / r["t_cc0"])
        print(f"  {N:>4s} Zellen/R  {run:24s} t_50 = {r['t']:6.2f} yr  t_cc = {r['t_cc0']:5.2f} yr  alpha = {a[-1]:5.2f}")
    if np.all(np.isfinite(a)):
        d1, d2 = a[1] - a[0], a[2] - a[1]
        print(f"  Aenderungen: {d1:+.2f} (7->13), {d2:+.2f} (13->27)")
        if d1 * d2 > 0 and abs(d2) < abs(d1):
            p = np.log2(d1 / d2)
            print(f"  Konvergenzordnung p = {p:.2f},  alpha_inf = {a[2] + d2 / (2**p - 1):.2f}  (Faktor {(a[2] + d2/(2**p - 1))/a[2]:.2f} ueber 27 Zellen/R)")
        else:
            print("  Differenzen werden nicht kleiner -> keine Konvergenz erkennbar, kein Grenzwert")

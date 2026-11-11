"""Sind die Regimegrenzen in x stabil gegen die Aufloesung, wenn alpha bei jeder Aufloesung neu kalibriert wird?

Paare: Gitterlauf (512, ~7 Zellen/R) und <lauf>_x2_t30 (1024, ~13 Zellen/R, bis 30 yr) aus runs/study_table.csv.
alpha je Aufloesung und Mass = Median t_50 / t_cc0 der chi = 10-Laeufe (alle vor dem Perizentrum zerstoert).
Regime aus y = t_50 / t_peri: vorher (< 0.85), Perizentrum (<= 1 + t_tid/t_peri + 0.3), sonst ueberlebt;
nicht erreicht bis Laufende -> 'ueberlebt' (512, 100 yr) bzw. '>30yr' (1024, Laufende 30 yr).
Aufruf:  python3 analysis/alpha_by_res.py
"""
import os, re
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/athena/runs")
tab = pd.read_csv(os.path.join(BASE, "study_table.csv"))
tab = tab[(tab["Y"] == "1/2") & (tab["norm"] == "init")]
lo = tab[tab["run"].str.match(r"^g_chi[0-9e]+_rp\d+(_big)?$")]
hi = tab[tab["run"].str.match(r"^g_chi[0-9e]+_rp\d+(_big)?_x2_t30$")]

def regime(t50, tp, ttid, t_end, short):
    if not np.isfinite(t50):
        return ">30yr" if short else "überlebt"
    y = t50 / tp
    return "vorher" if y < 0.85 else "Perizentrum" if y <= 1 + ttid / tp + 0.3 else "überlebt"

for X, name in (("0.9", "Mixing (tracer > 0.9)"), ("dq", "dichter Strom (rho/rho_eq > sqrt(chi0))")):
    L, H = lo[lo["X"] == X].set_index("run"), hi[hi["X"] == X].copy()
    H["base"] = H["run"].str.replace("_x2_t30", "", regex=False)
    H = H.set_index("base")
    a = {}
    for lab, D in (("512", L), ("1024", H)):
        c10 = D[(D["chi"].round() == 10) & np.isfinite(D["t"])]
        a[lab] = float(np.median(c10["t"] / c10["t_cc0"]))
    print(f"\n=== {name}:  alpha(512) = {a['512']:.2f}, alpha(1024) = {a['1024']:.2f}, Verhaeltnis {a['1024']/a['512']:.2f}")
    print(f"{'Lauf':20s} {'x(512)':>7s} {'Regime 512':>12s} {'x(1024)':>8s} {'Regime 1024':>12s}  {'gleich?':>7s}")
    n_same = n_cmp = 0
    for run in sorted(L.index, key=lambda r: a["512"] * L.loc[r, "t_cc0"] / L.loc[r, "t_peri"]):
        if run not in H.index:
            continue
        r1, r2 = L.loc[run], H.loc[run]
        x1, x2 = a["512"] * r1["t_cc0"] / r1["t_peri"], a["1024"] * r2["t_cc0"] / r2["t_peri"]
        g1 = regime(r1["t"], r1["t_peri"], r1["t_tid_peri"], r1["t_end"], False)
        g2 = regime(r2["t"], r2["t_peri"], r2["t_tid_peri"], r2["t_end"], True)
        same = "–" if g2 == ">30yr" else ("ja" if g1 == g2 else "NEIN")
        if same != "–":
            n_cmp += 1; n_same += same == "ja"
        print(f"{run:20s} {x1:7.2f} {g1:>12s} {x2:8.2f} {g2:>12s}  {same:>7s}")
    print(f"gleiches Regime: {n_same} von {n_cmp} vergleichbaren Paaren")

"""Ist t_50 proportional zu t_cc, oder gibt es einen Versatz?  Fit t_50 = t0 + alpha t_cc.

Laeufe aus runs/timescales.csv, die vor dem Perizentrum zerstoert werden (y < 0.85 im jeweiligen Mass) und
~7 Zellen pro Radius haben: Gitterlaeufe (R = R0 bei 512), R/2 bei x2, R/4 bei x4.  Je chi getrennt und gemeinsam.
Aufruf:  python3 analysis/offset_fit.py
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/athena/runs")
ts = pd.read_csv(os.path.join(BASE, "timescales.csv"))
seven = ts["run"].str.match(r"^g_chi[0-9e]+_rp\d+(_big)?$") | ts["run"].str.contains(r"_R0\.5_x2|_R0\.25_x4")
ts = ts[seven].copy()
for col, name in (("t50_mix", "Mixing (tracer > 0.9)"), ("t50_dq", "dichter Strom (rho/rho_eq > sqrt(chi0))")):
    d = ts[np.isfinite(ts[col]) & (ts[col] < 0.85 * ts["t_peri"])]
    print(f"\n=== {name}: {len(d)} Laeufe vor dem Perizentrum (~7 Zellen/R)")
    print(d[["run", "chi", "R_R0", "rp", "t_cc", col]].sort_values("t_cc").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    for lab, s in [("alle", d)] + [(f"chi = {c:g}", d[d["chi"].round() == c]) for c in sorted(d["chi"].round().unique())]:
        if len(s) < 3:
            continue
        a, t0 = np.polyfit(s["t_cc"], s[col], 1)
        prop = float((s["t_cc"] * s[col]).sum() / (s["t_cc"]**2).sum())
        res_lin = np.sqrt(np.mean((t0 + a * s["t_cc"] - s[col])**2))
        res_prop = np.sqrt(np.mean((prop * s["t_cc"] - s[col])**2))
        print(f"  {lab:10s} ({len(s)}): t_50 = {t0:5.2f} yr + {a:4.2f} t_cc  (Rest {res_lin:.2f} yr)   |   rein proportional: {prop:4.2f} t_cc (Rest {res_prop:.2f} yr)")

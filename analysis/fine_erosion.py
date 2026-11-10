"""tau_1/2 nach dem Perizentrum aus den Laeufen mit feiner Ausgabe (<lauf>_fine, Snapshots alle 0.1 yr), dann Neuanpassung.

Masse unvermischt (tr > 0.9) und dichter Strom (rho/rho_eq > sqrt(chi0)); chi0 und Anfangsmasse aus dem 1. Snapshot des
Originallaufs (gleiches Gitter).  Kontrolle: Massen bei t_peri gegen <lauf>/mX.csv.  Ersetzt die groben tau-Werte aus
runs/erosion.csv und passt tau ~ chi^a R^b r_p^c neu an (alle Laeufe bzw. ohne zensierte).
Ausgabe: runs/fine_erosion.csv.   Aufruf (Notebook):  %run ~/athena/analysis/fine_erosion.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, RHO0, R0, R_IN = 3.156e7, 1.702e-21, 1.0e16, 1.9e15

def masses(s, chi0):
    X, Y = np.meshgrid(s["x"], s["y"])
    rho, tr = s["rho"], np.clip(s["r0"], 0, 1)
    c = rho / (RHO0 * R0 / np.maximum(np.hypot(X, Y), R_IN))
    m = rho * tr
    return float(m[tr > 0.9].sum()), float(m[c > np.sqrt(chi0)].sum()), float(m.sum()), c, tr

E = pd.read_csv(os.path.join(BASE, "erosion.csv"))
ts = pd.read_csv(os.path.join(BASE, "timescales.csv")).set_index("run")
rows = []
for d in sorted(os.listdir(BASE)):
    if not d.endswith("_fine") or not os.path.isfile(os.path.join(BASE, d, "g2_cart.final.rst")):
        continue
    orig = d[:-5]
    if orig not in ts.index:
        print(f"{d}: {orig} nicht in timescales.csv"); continue
    tp = ts.loc[orig, "t_peri"]
    od = os.path.join(BASE, orig)
    _, s0 = g2_io.load_snapshot(od, list(g2_io.snapshot_numbers(od))[0])
    tr0, X0 = np.clip(s0["r0"], 0, 1), None
    _, _, M0, c0, _ = masses(s0, 1.0)
    chi0 = float(np.median(c0[tr0 > 0.99]))
    t, mu, mq = [], [], []
    fd = os.path.join(BASE, d)
    for n in g2_io.snapshot_numbers(fd):
        tt, s = g2_io.load_snapshot(fd, n)
        a, b, _, _, _ = masses(s, chi0)
        t.append(tt / YR); mu.append(a / M0); mq.append(b / M0)
        print(f"  {d}: t = {tt/YR:6.2f} yr", end="\r")
    print()
    t, mu, mq = map(np.array, (t, mu, mq))
    mx = pd.read_csv(os.path.join(od, "mX.csv"))
    ref_u = np.interp(tp, mx["t_yr"], mx["m_0.9"] / mx["m_tot"].iloc[0])
    ref_q = np.interp(tp, mx["t_yr"], mx["dq"] / mx["m_tot"].iloc[0])
    row = dict(run=orig, t_peri=tp, chi0=chi0)
    for lab, m, ref in (("mix", mu, ref_u), ("dq", mq, ref_q)):
        m1 = np.interp(tp, t, m)
        after = (t > tp) & (m < 0.5 * m1)
        row[f"m_peri_{lab}"], row[f"ref_{lab}"] = m1, ref
        row[f"tau_{lab}"] = (t[after][0] - tp) if after.any() else np.nan
        row[f"tau_{lab}_cens"] = not after.any()
    rows.append(row)
F = pd.DataFrame(rows)
F.to_csv(os.path.join(BASE, "fine_erosion.csv"), index=False)
pd.set_option("display.width", 200)
print("Kontrolle (Masse bei t_peri: fein gegen Original) und neue tau-Werte:")
print(F[["run", "m_peri_mix", "ref_mix", "tau_mix", "m_peri_dq", "ref_dq", "tau_dq"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

for _, r in F.iterrows():
    i = E.index[E["run"] == r["run"]]
    for lab in ("mix", "dq"):
        if not r[f"tau_{lab}_cens"]:
            E.loc[i, f"tau_{lab}"] = r[f"tau_{lab}"]
for lab in ("mix", "dq"):
    for name, ok in (("alle", E[f"tau_{lab}"].notna()),
                     ("nur fein oder > 1.5 yr", E[f"tau_{lab}"].notna() & ((E[f"tau_{lab}"] > 1.5) | E["run"].isin(F["run"])))):
        A = np.column_stack([np.log10(E["chi"][ok]), np.log10(E["R"][ok]), np.log10(E["rp"][ok]), np.ones(ok.sum())])
        y = np.log10(E[f"tau_{lab}"][ok])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        rms = np.sqrt(np.mean((A @ c - y)**2))
        print(f"[{lab}] {name}: {ok.sum()} Laeufe:  tau ~ chi^{c[0]:.2f} R^{c[1]:.2f} r_p^{c[2]:.2f}  (Streuung {10**rms:.2f}x)")

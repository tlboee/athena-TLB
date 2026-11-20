"""Max (19.11.): extrem starke Kuehlung und Verlaengerung auf 150 yr.

Pro Snapshot, jeweils / Anfangsmasse der Wolke m0 = sum(rho*tr) bei t = 0:
  cold_all   = sum(rho)    ueber T < 3 T_cl0              (inkl. kondensiertem Umgebungsgas -> > 1 heisst Wachstum)
  cold_cloud = sum(rho*tr) ueber T < 3 T_cl0              (nur Wolkengas)
  dense      = sum(rho*tr) ueber rho/rho_eq > sqrt(chi0)  (Ueberlebensmass)
  accreted   aus g2_cart.hst wie in max_plots.py
Cache: <lauf>/cold_ext.csv.   Ausgabe: runs/cool_extreme.png.   Aufruf:  %run ~/athena/analysis/cool_extreme.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/athena/analysis"))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, MU, MH, KB = 3.156e7, 0.6139, 1.6605e-24, 1.380648e-16
RHO0, RR0, R_IN, CHI0, T_PERI = 1.702e-21, 1.0e16, 1.9e15, 2755.0, 18.0
RUNS = [("R adiabatic", "ps_ref_big_x2_t50", "0.3", "--"),
        ("R, Λ×1 (r/r_crit 6.4)", "ps_ref_big_cool_x2_t150", "#c0392b", "-"),
        ("R, Λ×16 (≈100)", "ps_ref_big_cool_L16_R1_x2", "#e6ab02", "-"),
        ("R, Λ×160 (≈1000)", "ps_ref_big_cool_L160_R1_x2", "#1b9e77", "-"),
        ("R/2, Λ×1 (3.2)", "ps_ref_big_cool_R0.5_x2_t50", "#6a3d9a", ":"),
        ("R/2, Λ×100 (≈320)", "ps_ref_big_cool_L100_R0.5_x2", "#e7298a", ":")]

def series(run):
    d = os.path.join(BASE, run); cache = os.path.join(d, "cold_ext.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows, m0, T0 = [], None, None
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        rho, tr = s["rho"], np.clip(s["r0"], 0, 1)
        T = s["press"] * MU * MH / (rho * KB)
        X, Y = np.meshgrid(s["x"], s["y"])
        c = rho / (RHO0 * RR0 / np.maximum(np.hypot(X, Y), R_IN))
        if m0 is None:
            m0, T0 = float((rho * tr).sum()), float(np.median(T[tr > 0.99]))
        cold = T < 3 * T0
        rows.append([t / YR, float(rho[cold].sum() / m0), float((rho * tr)[cold].sum() / m0),
                     float((rho * tr)[c > np.sqrt(CHI0)].sum() / m0)])
        print(f"  {run}: t = {t/YR:6.1f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t", "cold_all", "cold_cloud", "dense"]); df.to_csv(cache, index=False)
    return df

def accreted(run):
    h = np.loadtxt(os.path.join(BASE, run, "g2_cart.hst"))
    t, tr, sink, buf, res = h[:, 0] / YR, h[:, 10], h[:, 11], h[:, 12], h[:, 13]
    loss, den = tr[0] - tr, sink + buf + np.abs(res)
    with np.errstate(invalid="ignore", divide="ignore"):
        return t, np.where(den > 0, loss * sink / den, 0.0) / tr[0]

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, constrained_layout=True)
titles = ["all cold gas m(T < 3 T_cl0) / m0  (> 1: growth)", "cold cloud gas only (tracer-weighted)",
          "dense gas ρ/ρ_eq > √χ₀ (survival measure)", "accreted (in sink)"]
print(f"{'Lauf':24s} " + " ".join(f"{'@'+str(t)+' yr':>22s}" for t in (19, 30, 50, 100, 150)))
print(f"{'':24s} " + " ".join(f"{'kalt/Wolke/dicht':>22s}" for _ in range(5)))
for lab, run, col, ls in RUNS:
    if not os.path.isdir(os.path.join(BASE, run)):
        print(f"{run} fehlt"); continue
    d = series(run)
    for ax, k in zip(axes.flat[:3], ("cold_all", "cold_cloud", "dense")):
        ax.plot(d["t"], d[k], color=col, ls=ls, label=lab)
    ta, acc = accreted(run); axes[1, 1].plot(ta, acc, color=col, ls=ls, label=lab)
    vals = []
    for tt in (19, 30, 50, 100, 150):
        i = int(np.argmin(np.abs(d["t"] - tt)))
        vals.append(f"{d['cold_all'][i]:6.2f}/{d['cold_cloud'][i]:5.2f}/{d['dense'][i]:5.2f}" if abs(d["t"][i] - tt) < 0.6 else f"{'-':>22s}")
    print(f"{lab:24s} " + " ".join(f"{v:>22s}" for v in vals))
for ax, ti in zip(axes.flat, titles):
    ax.set_title(ti); ax.axvline(T_PERI, color="k", ls=":", lw=0.8); ax.axhline(1, color="0.7", lw=0.8)
axes[1, 0].axhline(0.5, color="k", ls=":", lw=0.8)
for ax in axes[1]:
    ax.set_xlabel("t [yr]")
axes[0, 0].legend(fontsize=9)
fig.suptitle("CC01 orbit: extreme cooling and long-term evolution (pericentre at 18 yr, apocentre at ~87 yr)")
fig.savefig(os.path.join(BASE, "cool_extreme.png"), dpi=150)
plt.show()

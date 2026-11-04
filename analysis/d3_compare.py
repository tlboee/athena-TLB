"""3D (CC01, adiabatisch, 512x384x144) gegen 2D (ps_ref_big, gleiche Zellgroesse):
oben Massenbilanz aus der History (ganzes Volumen), unten Dichte in der Mittelebene.
Aufruf (Notebook):  %run ~/athena/analysis/d3_compare.py
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.expanduser("~/athena/vis/python"))
import athena_read

BASE = os.path.expanduser("~/athena/runs")
YR = 3.156e7
RUNS = [("d3_cc01_adiab", "3D"), ("ps_ref_big", "2D")]
TP = 18.0
SHOW = [-5, 0, 10, 30]

def budget(run):
    h = np.loadtxt(os.path.join(BASE, run, "g2_cart.hst"))
    t, tr, sink, buf, res = h[:, 0] / YR, h[:, 10], h[:, 11], h[:, 12], h[:, 13]
    loss = (tr[0] - tr) / tr[0]
    den = sink + buf + np.abs(res)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(den > 0, loss * sink / den, 0.0)
        bnd = np.where(den > 0, loss * buf / den, 0.0)
    return t, tr / tr[0], acc, bnd

def snapshot_at(run, t_yr):
    d = os.path.join(BASE, run)
    files = sorted(f for f in os.listdir(d) if ".out1." in f and f.endswith(".athdf"))
    best, bt = None, 1e9
    for f in files:
        n = int(f.split(".")[-2])
        if abs(n - t_yr) < bt:            # output1 jaehrlich: Nummer = Jahr
            best, bt = f, abs(n - t_yr)
    a = athena_read.athdf(os.path.join(d, best))
    return a["Time"] / YR, a["x1f"], a["x2f"], np.squeeze(a["rho"])

fig = plt.figure(figsize=(16, 11), constrained_layout=True)
gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 1])
ax = fig.add_subplot(gs[0, :])
print(f"{'Lauf':16s} {'t-t_p':>6s} {'Domain':>7s} {'akkr.':>6s} {'Rand':>6s}")
for (run, lab), ls in zip(RUNS, ["-", "--"]):
    t, dom, acc, bnd = budget(run)
    sel = t <= 50.01
    ax.plot(t[sel] - TP, dom[sel], ls, color="C0", label=f"{lab}: in domain")
    ax.plot(t[sel] - TP, acc[sel], ls, color="0.25", label=f"{lab}: accreted")
    ax.plot(t[sel] - TP, bnd[sel], ls, color="C3", label=f"{lab}: lost at boundary")
    for dt in (0, 10, 20, 32):
        i = np.argmin(np.abs(t - TP - dt))
        print(f"{run:16s} {t[i]-TP:6.1f} {dom[i]:7.3f} {acc[i]:6.3f} {bnd[i]:6.3f}")
ax.axvline(0, color="grey", lw=0.8); ax.set_xlim(-18, 32); ax.set_ylim(0, 1.02)
ax.set_xlabel("t − t_peri [yr]"); ax.set_ylabel("fraction of initial cloud mass")
ax.set_title("CC01 adiabatic, same cell size: mass budget in 3D (solid) and 2D (dashed)", fontsize=10)
ax.legend(fontsize=8, ncol=3)
norm = LogNorm(1e-22, 1e-18)
for row, (run, lab) in enumerate(RUNS, start=1):
    for col, dt in enumerate(SHOW):
        tt, x, y, rho = snapshot_at(run, TP + dt)
        a2 = fig.add_subplot(gs[row, col])
        im = a2.pcolormesh(x / 1e16, y / 1e16, rho, norm=norm, cmap="magma", shading="flat", rasterized=True)
        a2.set_aspect("equal"); a2.set_xlim(-7.5, 3); a2.set_ylim(-3.5, 3.5)
        a2.set_title(f"{lab}{' midplane' if lab == '3D' else ''}, t − t_peri = {tt - TP:.0f} yr", fontsize=9)
        if row == 2: a2.set_xlabel("x [10¹⁶ cm]")
        if col == 0: a2.set_ylabel("y [10¹⁶ cm]")
fig.colorbar(im, ax=fig.axes[1:], shrink=0.6, label="ρ [g cm⁻³]")
out = os.path.join(BASE, "d3_compare.png")
fig.savefig(out, dpi=130)
print("Gespeichert:", out)

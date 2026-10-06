#!/usr/bin/env python3
"""plot_g2_compare.py — Stabilitätsvergleich der drei 2D-Runs"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

r0   = 1.0e16
k_B  = 1.380649e-16
m_H  = 1.6726e-24
mu   = 0.5

RUNS = {
    "unstabil (kein NSC)":   "g2_2d",
    "NSC only (M=M_eq)":     "g2_2d_nsc",
    "NSC + relax (τ=1 yr)":  "g2_2d_relax",
    "Burkert+12 (relax only)": "g2_2d_burkert",
}
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
SNAP = 10

def read_tab(fname):
    """Spalten: i x1v j x2v rho press vel1 vel2 vel3"""
    data = np.loadtxt(fname, comments="#")
    r_raw  = data[:, 1]   # x1v
    th_raw = data[:, 3]   # x2v
    rho_r  = data[:, 4]   # rho
    P_r    = data[:, 5]   # press
    vr_r   = data[:, 6]   # vel1
    vt_r   = data[:, 7]   # vel2

    r_vals  = np.unique(r_raw)
    th_vals = np.unique(th_raw)
    nr, nt  = len(r_vals), len(th_vals)

    def avg(arr):
        return arr.reshape(nt, nr).mean(axis=0)

    return dict(r=r_vals,
                rho=avg(rho_r), P=avg(P_r),
                vr=avg(vr_r),   vt=avg(vt_r))

def snap_file(run_id, snap):
    return str(ROOT / "runs" / run_id / f"{run_id}.block0.out1.{snap:05d}.tab")

# Analytisch
def burkert_rho(r): return 9.5e-22 * (r0/r)
def burkert_T(r):   return 2.0e8   * (r0/r)
def burkert_P(r):
    rho = burkert_rho(r)
    return rho * k_B * burkert_T(r) / (mu * m_H)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("G2 Atmosphäre — Stabilitätsvergleich (t = 10 yr)", fontsize=14)
ax_rho, ax_vr, ax_T, ax_P = axes.flat

for ax in axes.flat:
    ax.set_xscale("log")
    ax.set_xlabel("r / r₀")

ax_rho.set_yscale("log"); ax_rho.set_ylabel("ρ  [g cm⁻³]");    ax_rho.set_title("Dichte")
ax_vr .set_ylabel("v_r  [km s⁻¹]");                             ax_vr .set_title("Radialgeschwindigkeit")
ax_T  .set_yscale("log"); ax_T.set_ylabel("T  [K]");            ax_T  .set_title("Temperatur")
ax_P  .set_yscale("log"); ax_P.set_ylabel("P  [dyn cm⁻²]");    ax_P  .set_title("Druck")

r_ref = np.logspace(np.log10(3e15), np.log10(3e17), 300)
x_ref = r_ref / r0
for ax, y in [(ax_rho, burkert_rho(r_ref)),
              (ax_vr,  np.zeros(300)),
              (ax_T,   burkert_T(r_ref)),
              (ax_P,   burkert_P(r_ref))]:
    ax.plot(x_ref, y, "k--", lw=1.2, label="Burkert analytisch", zorder=0)

for (label, run_id), color in zip(RUNS.items(), COLORS):
    fname = snap_file(run_id, SNAP)
    if not os.path.exists(fname):
        print(f"WARNUNG: {fname} nicht gefunden"); continue
    d = read_tab(fname)
    x = d["r"] / r0
    T = d["P"] * mu * m_H / (d["rho"] * k_B)
    kw = dict(color=color, lw=1.8, label=label)
    ax_rho.plot(x, d["rho"],       **kw)
    ax_vr .plot(x, d["vr"]/1e5,   **kw)
    ax_T  .plot(x, T,              **kw)
    ax_P  .plot(x, d["P"],         **kw)

for ax in axes.flat:
    ax.legend(fontsize=8)

fig.tight_layout()
outdir = str(ROOT / "figures")
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "g2_compare_t10.png")
fig.savefig(out, dpi=150)
print(f"Gespeichert: {out}")
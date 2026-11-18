"""Braucht Rekondensation einen dichten Rest, der das Perizentrum uebersteht?

Feld 1: Anteil dichten Gases (rho/rho_eq > sqrt(chi0), Gewicht rho*tr, / Anfangsmasse) ueber t, mit Kuehlung (durchgezogen)
        und adiabatisch (gestrichelt) fuer R, R/2, R/4.
Feld 2: Max' Diagramm x gegen r/r_crit; Farbe = kalte Masse (T < 3 T_cl0) bei 50 yr (aus cold_all.csv), Zahl = minimaler
        dichter Anteil zwischen 18 und 25 yr.  x mit alpha je Aufloesung (timescale_table.py), r/r_crit am Start.
Cache: <lauf>/seed.csv.   Ausgabe: runs/cool_seed.png.   Aufruf:  %run ~/athena/analysis/cool_seed.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/athena/analysis"))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, RHO0, RR0, R_IN, CHI0, T_PERI = 3.156e7, 1.702e-21, 1.0e16, 1.9e15, 2755.0, 18.0
#        Bezeichnung        Lauf mit Kuehlung              x     r/r_crit  adiabatisches Gegenstueck     Farbe
COOL = [("R, Λ×1",      "ps_ref_big_cool_x2_t50",       12.5, 6.4,   "ps_ref_big_x2_t50",       "#c0392b"),
        ("R, Λ×0.1",    "ps_ref_big_cool_L0.1_x2_t50",  12.5, 0.64,  None,                      "#d9822b"),
        ("R, Λ×0.01",   "ps_ref_big_cool_L0.01_x2_t50", 12.5, 0.064, None,                      "#2f6db3"),
        ("R/2, Λ×1",    "ps_ref_big_cool_R0.5_x2_t50",  4.2,  3.2,   "ps_ref_big_R0.5_x2_t50",  "#6a3d9a"),
        ("R/4, Λ×1",    "ps_ref_big_cool_R0.25_x4_t50", 2.1,  1.6,   "ps_ref_big_R0.25_x4_t50", "#b39ddb"),
        ("R/2, Λ×10",   "ps_ref_big_cool_L10_R0.5_x2_t50", 4.2, 32.0, None,                     "#e7298a"),
        ("0.75R, Λ×1",  "ps_ref_big_cool_R0.75_x2_t50", 8.1,  4.8,   "ps_ref_big_R0.75_x2_t50", "#1b9e77"),
        ("0.85R, Λ×1",  "ps_ref_big_cool_R0.85_x2_t50", 9.8,  5.4,   "ps_ref_big_R0.85_x2_t50", "#e6ab02")]

def cold_last(run):
    """Kalte Masse m(T < 3 T_cl0, tr > 1e-4)/m0 im letzten Snapshot (Ersatz, wenn cold_all.csv fehlt)."""
    d = os.path.join(BASE, run); cache = os.path.join(d, "cold_seed.csv")
    if os.path.exists(cache):
        return float(pd.read_csv(cache)["m_cold"].iloc[-1])
    ns = g2_io.snapshot_numbers(d)
    vals = []
    for n in (ns[0], ns[-1]):
        t, s = g2_io.load_snapshot(d, n)
        rho, tr = s["rho"], np.clip(s["r0"], 0, 1)
        T = s["press"] * 0.6139 * 1.6605e-24 / (rho * 1.380648e-16)
        vals.append((t / YR, rho, tr, T))
    T0 = float(np.median(vals[0][3][vals[0][2] > 0.99])); m0 = float((vals[0][1] * vals[0][2]).sum())
    t1, rho, tr, T = vals[1]
    mc = float((rho * tr)[(tr > 1e-4) & (T < 3 * T0)].sum() / m0)
    pd.DataFrame([[t1, mc]], columns=["t", "m_cold"]).to_csv(cache, index=False)
    return mc

def series(run):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "seed.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows, M0 = [], None
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        X, Y = np.meshgrid(s["x"], s["y"])
        rho, tr = s["rho"], np.clip(s["r0"], 0, 1)
        w = rho * tr
        if M0 is None:
            M0 = float(w.sum())
        c = rho / (RHO0 * RR0 / np.maximum(np.hypot(X, Y), R_IN))
        rows.append([t / YR, float(w[c > np.sqrt(CHI0)].sum() / M0)])
        print(f"  {run}: t = {t/YR:5.1f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t", "dense"]); df.to_csv(cache, index=False)
    return df

fig, axes = plt.subplots(1, 3, figsize=(25, 6.5), constrained_layout=True)
print(f"\n{'Lauf':12s} {'x':>5s} {'r/r_crit':>8s} {'dicht min (18-25 yr)':>21s} {'dicht @50':>10s} {'kalt @50':>9s} {'dicht min adiab.':>17s}")
pairs = []
for lab, run, x, rr, adia, c in COOL:
    if not os.path.isdir(os.path.join(BASE, run)):
        print(f"{run} fehlt"); continue
    d = series(run)
    axes[0].plot(d["t"], d["dense"], color=c, label=f"{lab} (cooling)")
    m = (d["t"] >= T_PERI) & (d["t"] <= 25)
    dmin, d50 = float(d["dense"][m].min()), float(d["dense"].iloc[-1])
    amin = np.nan
    if adia and os.path.isdir(os.path.join(BASE, adia)):
        a = series(adia); axes[0].plot(a["t"], a["dense"], color=c, ls="--", lw=1, label=f"{lab.split(',')[0]} (adiabatic)")
        ma = (a["t"] >= T_PERI) & (a["t"] <= 25); amin = float(a["dense"][ma].min())
    p = os.path.join(BASE, run, "cold_all.csv")
    cold50 = float(pd.read_csv(p)["m_cold"].iloc[-1]) if os.path.exists(p) else cold_last(run)
    sc = axes[1].scatter(rr, x, c=[cold50], cmap="viridis", vmin=0, vmax=0.6, s=260, edgecolors="k", zorder=3)
    axes[1].annotate(f"{lab}\ndense min {dmin:.2f}", (rr, x), xytext=(12, -4), textcoords="offset points", fontsize=10)
    print(f"{lab:12s} {x:5.1f} {rr:8.3g} {dmin:21.3f} {d50:10.3f} {cold50:9.3f} {amin:17.3f}")
    if lab.endswith("Λ×1"):
        pairs.append((x, dmin, amin))
axes[0].axvline(T_PERI, color="k", ls=":", lw=0.8)
axes[0].set_xlabel("t [yr]"); axes[0].set_ylabel("dense fraction  ρ/ρ_eq > √χ₀  (of initial cloud mass)")
axes[0].set_title("Does a dense remnant survive the pericentre?"); axes[0].legend(fontsize=8, ncol=2)
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].axvline(1, color="0.5", ls="--", lw=1); axes[1].axhline(11.5, color="k", ls=":", lw=1)
axes[1].text(1.08, 1.12, "r/r_crit = 1", color="0.4", fontsize=10)
axes[1].text(0.025, 10.6, "adiabatic survival boundary x ≈ 11.5", fontsize=10, ha="left", va="top")
axes[1].axhspan(4.2, 8.1, color="0.5", alpha=0.12, lw=0)
axes[1].text(78, 5.9, "a dense remnant survives the pericentre\nfrom somewhere in this band on", fontsize=10, ha="right", va="center", color="0.3")
axes[1].text(0.025, 1.5, "no dense remnant survives the pericentre:\ndestroyed, independent of cooling", fontsize=11, color="0.3")
axes[1].text(0.025, 22, "dense remnant survives,\nno recondensation", fontsize=11, color="0.3")
axes[1].text(1.3, 22, "remnant survives\nand recondenses", fontsize=11, color="0.3")
axes[1].set_xlim(0.02, 80); axes[1].set_ylim(1, 30)
axes[1].set_xlabel("r/r_crit = t_cc / t_cool,mix  (at start)"); axes[1].set_ylabel("x = α t_cc / t_peri")
axes[1].set_title("CC01 orbit with cooling: colour = cold mass at 50 yr")
fig.colorbar(sc, ax=axes[1], label="m(T < 3 T_cl0) / m₀ at 50 yr")
def cross(xs, ys, lvl=0.5):
    for (x1, y1), (x2, y2) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if (y1 - lvl) * (y2 - lvl) < 0:
            return float(np.exp(np.log(x1) + (lvl - y1) / (y2 - y1) * (np.log(x2) - np.log(x1))))
    return np.nan
pairs.sort()
xs, yc, ya = [q[0] for q in pairs], [q[1] for q in pairs], [q[2] for q in pairs]
ax3 = axes[2]
ax3.plot(xs, ya, "o--", color="0.4", ms=8, label="adiabatic")
ax3.plot(xs, yc, "o-", color="#c0392b", ms=8, label="with cooling (Λ×1, r/r_crit ≈ 1.6–6.4)")
ax3.axhline(0.5, color="k", ls=":", lw=1)
for y, c, lab in ((ya, "0.4", "adiabatic"), (yc, "#c0392b", "cooling")):
    xc = cross(xs, y)
    if np.isfinite(xc):
        ax3.axvline(xc, color=c, lw=1, ls="--")
        ax3.text(xc * 1.02, 0.03, f"x ≈ {xc:.1f}", color=c, fontsize=11)
        print(f"Grenze (dichter Rest = 0.5 nach dem Perizentrum), {lab}: x ≈ {xc:.2f}")
ax3.set_xscale("log"); ax3.set_xlim(1.5, 20); ax3.set_ylim(-0.02, 0.75)
ax3.set_xlabel("x = α t_cc / t_peri"); ax3.set_ylabel("dense fraction min. after pericentre (18–25 yr)")
ax3.set_title("Cooling shifts the boundary for surviving the pericentre"); ax3.legend(fontsize=10, loc="upper left")
fig.savefig(os.path.join(BASE, "cool_seed.png"), dpi=150)
plt.show()

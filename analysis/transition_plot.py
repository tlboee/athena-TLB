"""Uebergang KH -> Gezeiten: t_50 gegen chi (A) und dimensionslos (B).

Modell: Zerstoerung vor dem Perizentrum durch Cloud Crushing/KH, t ~ alpha t_cc ~ sqrt(chi);
        sonst am Perizentrum durch Gezeiten, t ~ t_peri + t_tid(r_p);  t_model = min(beide).
alpha an die chi = 10-Laeufe angepasst (Tracer > 0.9, 512).
A: t_50 gegen chi, eine Modellkurve pro r_p.
B: x = alpha t_cc / t_peri, y = t_50 / t_peri; Erwartung y = x fuer x < 1, y ~ 1 + t_tid/t_peri fuer x > 1.
Daten: runs/study_table.csv (t_50, bezogen auf m_tot(0)), nur Gitterlaeufe.
Aufruf (Notebook):  %run ~/athena/analysis/transition_plot.py
"""
import math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/athena/runs")
YR, R_C = 3.156e7, 1.87e15
V0 = math.hypot(794.59, 48.45) * 1e5
MEAS = [("0.9", "Mixing: tracer > 0.9"), ("ds", "Dense stream: ρ/ρ_eq > min(10, χ₀/3)")]

tab = pd.read_csv(os.path.join(BASE, "study_table.csv"))
g = tab[tab["run"].str.startswith("g_") & (tab["Y"] == "1/2") & (tab["norm"] == "init")].copy()
g["x2"] = g["run"].str.contains("_x2")
g["rp"] = g["rp"].map(lambda v: float(f"{v:.2g}"))
rps = sorted(g["rp"].unique())
peri = g.drop_duplicates("rp").set_index("rp")[["t_peri", "t_tid_peri"]]
cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(rps)))

ref = g[(g["X"] == "0.9") & (g["chi"] == 10) & ~g["x2"]]
alpha = float(np.median(ref["t"] / ref["t_cc0"]))
print(f"alpha = {alpha:.2f} (an chi = 10 angepasst, Tracer > 0.9, 512)")
chis = np.logspace(0.8, 4.3, 300)
tcc = lambda chi: np.sqrt(chi) * R_C / V0 / YR

# --- A: t_50 gegen chi ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(MEAS), figsize=(6.2 * len(MEAS), 4.8), sharey=True, constrained_layout=True)
for ax, (key, title) in zip(axes, MEAS):
    sub = g[g["X"] == key]
    for c, rp in zip(cols, rps):
        tp, tt = peri.loc[rp, "t_peri"], peri.loc[rp, "t_tid_peri"]
        r0 = g[g["rp"] == rp].iloc[0]
        tcc_rp = r0["t_cc0"] * np.sqrt(chis / r0["chi"])          # t_cc mit v_Start dieses r_p
        ax.plot(chis, np.minimum(alpha * tcc_rp, tp + tt), color=c, lw=1, alpha=0.7)
        s = sub[(sub["rp"] == rp) & ~sub["x2"]].sort_values("chi")
        ok = np.isfinite(s["t"]).to_numpy()
        ax.plot(s["chi"][ok], s["t"][ok], "o", color=c, ms=7, label=f"r_p = {rp:.1e} cm")
        ax.plot(s["chi"][~ok], s["t_end"][~ok], "^", color=c, mfc="none", ms=9)
        h = sub[(sub["rp"] == rp) & sub["x2"]]
        okh = np.isfinite(h["t"]).to_numpy()
        ax.plot(h["chi"][okh], h["t"][okh], "s", color=c, mfc="none", ms=7)
    ax.text(12, 4, "cloud crushing / KH\nbefore pericentre", fontsize=8, color="0.3")
    ax.text(150, 13, "at pericentre (tidal)", fontsize=8, color="0.3")
    ax.text(1500, 45, "survives\npericentre", fontsize=8, color="0.3")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(6, 2e4); ax.set_ylim(3, 150)
    ax.set_xlabel("χ"); ax.set_title(title, fontsize=10)
axes[0].set_ylabel("t_50 [yr since 1995.5]")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside right center", fontsize=8,
           title=f"lines: min({alpha:.1f} t_cc, t_peri + t_tid)\ncircles: 512, squares: 2× res.\ntriangles: not reached",
           title_fontsize=7)
outA = os.path.join(BASE, "transition_A.png")
fig.savefig(outA, dpi=150)

# --- B: dimensionslos ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(MEAS), figsize=(6.2 * len(MEAS), 4.8), sharey=True, constrained_layout=True)
xs = np.logspace(-1.5, 1.5, 100)
for ax, (key, title) in zip(axes, MEAS):
    sub = g[g["X"] == key].copy()
    sub["x"] = alpha * sub["t_cc0"] / sub["t_peri"]
    sub["y"] = sub["t"] / sub["t_peri"]
    ax.plot(xs, xs, "k--", lw=1, label="y = x (cloud crushing / KH)")
    for c, rp in zip(cols, rps):
        ax.plot([1, 15], [1 + peri.loc[rp, "t_tid_peri"] / peri.loc[rp, "t_peri"]] * 2, color=c, lw=2,
                alpha=0.5, label="1 + t_tid / t_peri (tidal)" if rp == rps[-1] else None)
    for chi, xy in [(10, (0.25, 0.3)), (100, (1.15, 0.8)), (1000, (2.5, 3.2)), (10000, (8, 6.5))]:
        ax.text(*xy, f"χ = {chi:g}", fontsize=9, color="0.3")
    ax.axvline(1, color="0.6", lw=0.8)
    for c, rp in zip(cols, rps):
        for x2, mk, mfc in [(False, "o", None), (True, "s", "none")]:
            s = sub[(sub["rp"] == rp) & (sub["x2"] == x2)]
            ok = np.isfinite(s["y"]).to_numpy()
            ax.plot(s["x"][ok], s["y"][ok], mk, color=c, mfc=mfc if mfc else c, ms=7,
                    label=f"r_p = {rp:.1e} cm" if not x2 else None)
            if not x2:
                ax.plot(s["x"][~ok], s["t_end"][~ok] / s["t_peri"][~ok], "^", color=c, mfc="none", ms=9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.08, 15); ax.set_ylim(0.2, 8)
    ax.set_xlabel(f"α t_cc / t_peri   (α = {alpha:.1f})")
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel("t_50 / t_peri")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside right center", fontsize=8,
           title="circles: 512, squares: 2× res.\ntriangles: not reached", title_fontsize=7)
outB = os.path.join(BASE, "transition_B.png")
fig.savefig(outB, dpi=150)
print("Gespeichert:", outA, outB)

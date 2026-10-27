"""Uebergang KH -> Gezeiten: t_50 gegen chi mit den Zeitskalen als Linien.

KH-/Cloud-Crushing-Bereich: t_dis ~ t_KH ~ sqrt(chi)  (steigende Gerade)
Gezeitenbereich:            t_dis ~ t_peri + t_tid(r_p) (waagerecht, unabhaengig von chi)
Zeitskalen am Start (CC01-Bahn): t_cc = sqrt(chi) R / v0, t_KH = t_cc / (2 pi) (lambda = R); ohne angepassten Vorfaktor.
Daten: runs/study_table.csv (t_50, d.h. Y = 1/2, bezogen auf m_tot(0)), nur Gitterlaeufe.
Aufruf (Notebook):  %run ~/athena/analysis/transition_plot.py
"""
import math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/athena/runs")
YR = 3.156e7
R_C = 1.87e15
V0 = math.hypot(794.59, 48.45) * 1e5
MEAS = [("0.9", "Tracer > 0.9 (Mischung)"),
        ("ds", "dichter Strom: rho/rho_eq > min(10, chi0/3)"),
        ("ek_1/3", "Entropie: K < 3^gamma K0")]

tab = pd.read_csv(os.path.join(BASE, "study_table.csv"))
g = tab[tab["run"].str.startswith("g_") & (tab["Y"] == "1/2") & (tab["norm"] == "init")].copy()
g["x2"] = g["run"].str.contains("_x2")
g["rp"] = g["rp"].map(lambda v: float(f"{v:.2g}"))     # _big-Laeufe haben minimal anderes r_p
rps = sorted(g["rp"].unique())
peri = g.drop_duplicates("rp").set_index("rp")[["t_peri", "t_tid_peri"]]

chis = np.logspace(0.8, 4.3, 100)
t_cc = np.sqrt(chis) * R_C / V0 / YR
ref = g[(g["X"] == "0.9") & (g["chi"] == 10) & ~g["x2"]]
alpha = float(np.median(ref["t"] / (np.sqrt(10.0) * R_C / V0 / YR)))   # an chi = 10 angepasst
fig, axes = plt.subplots(1, len(MEAS), figsize=(5.2 * len(MEAS), 4.6), sharey=True, constrained_layout=True)
for ax, (key, title) in zip(axes, MEAS):
    sub = g[g["X"] == key]
    for k, rp in enumerate(rps):
        c = f"C{k}"
        s = sub[(sub["rp"] == rp) & ~sub["x2"]].sort_values("chi")
        ok = np.isfinite(s["t"]).to_numpy()
        ax.plot(s["chi"][ok], s["t"][ok], "o-", color=c, label=f"r_p = {rp:.1e} cm")
        ax.plot(s["chi"][~ok], s["t_end"][~ok], "^", color=c, mfc="none", ms=9)
        h = sub[(sub["rp"] == rp) & sub["x2"]]
        okh = np.isfinite(h["t"]).to_numpy()
        ax.plot(h["chi"][okh], h["t"][okh], "s", color=c, mfc="none", ms=8)
        ax.axhline(peri.loc[rp, "t_peri"] + peri.loc[rp, "t_tid_peri"], color=c, ls=":", lw=1.2)
    ax.plot(chis, alpha * t_cc, color="grey", lw=2, alpha=0.6, label=f"{alpha:.1f} t_cc (an chi = 10 angepasst)")
    ax.plot(chis, t_cc, "k-.", lw=1, label="t_cc (Start)")
    ax.plot(chis, t_cc / (2 * np.pi), "k--", lw=1, label="t_KH (Start, lambda = R)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(6, 2e4); ax.set_ylim(0.5, 200)
    ax.set_xlabel("chi"); ax.set_title(title, fontsize=10)
axes[0].set_ylabel("t_50 [yr seit 1995.5]")
axes[0].legend(fontsize=7, loc="upper left",
               title="gepunktet: t_peri + t_tid(r_p)\nKreis 512, Quadrat doppelt, Dreieck nicht erreicht",
               title_fontsize=7)
out = os.path.join(BASE, "transition_kh_tidal.png")
fig.savefig(out, dpi=150)
print("Gespeichert:", out)

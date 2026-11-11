"""Regime-Diagramm: x = alpha t_cc / t_peri gegen r_p, Symbole nach gemessenem Regime (aus runs/timescales.csv).

Erst timescale_table.py laufen lassen.  R-Varianten (R/4 ... 4R) leicht waagerecht versetzt.
Ausgabe: runs/regime.png.   Aufruf:  python3 analysis/regime_plot.py   (oder %run im Notebook)
"""
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/athena/runs")
df = pd.read_csv(os.path.join(BASE, "timescales.csv"))
OFF = {0.25: -0.09, 0.5: -0.045, 1.0: 0.0, 2.0: 0.045, 4.0: 0.09}
STY = {"vorher": ("o", "#2f6db3", "destroyed before pericentre"),
       "Perizentrum": ("s", "#d9822b", "destroyed at pericentre"),
       "überlebt": ("^", "#1d2733", "survives pericentre")}
plt.rcParams.update({"font.size": 15})
fig, ax = plt.subplots(figsize=(16, 7))
import numpy as np
rr = np.logspace(np.log10(2.6e15), np.log10(5.2e16), 50)
ax.plot(rr, 10.5 * (rr / 4e15)**-0.7, color="#1d2733", ls=":", lw=1.5, zorder=2)
for rp0 in (4e15, 8e15, 1.6e16, 3.2e16):              # Grenze am tatsaechlichen r_p, ueber die Breite der R-Varianten
    xc = 10.5 * (rp0 / 4e15)**-0.7
    ax.plot([rp0 * 10**-0.11, rp0 * 10**0.11], [xc, xc], color="#1d2733", lw=2.5, zorder=2, solid_capstyle="butt")   # Ueberlebensgrenze (10.11.)
done = set()
for _, r in df.iterrows():
    xp = r["rp"] * 10**OFF.get(round(r["R_R0"], 2), 0.0)
    if r["run"].startswith("ps_ref_big"):                      # CC01 bei 512 / 1024 / 2048
        res = {"ps_ref_big": "512", "ps_ref_big_x2_t50": "1024", "ps_ref_big_x4_t50": "2048"}[r["run"]]
        surv = r["regime"] == "überlebt"
        ax.scatter(xp, r["x"], marker="*", s=380, zorder=4, edgecolors="#c0392b", linewidths=1.5,
                   c="#c0392b" if not surv else "white", label="CC01 (G2): 512 / 1024 / 2048" if res == "512" else None)
        ax.annotate(res, (xp, r["x"]), xytext={"512": (14, -5), "1024": (-62, -16), "2048": (-62, 4)}[res], textcoords="offset points", fontsize=12, color="#c0392b")
        if res == "512":
            ax.annotate("CC01 (G2)", (xp, r["x"]), xytext=(-150, -4), textcoords="offset points",
                        fontsize=16, color="#c0392b", weight="bold")
        continue
    m, c, lab = STY[r["regime"]]
    ax.scatter(xp, r["x"], marker=m, c=c, s=110, edgecolors="white", linewidths=0.8, zorder=3,
               label=None if r["regime"] in done else lab)
    done.add(r["regime"])
ax.axhline(1, color="#2f6db3", ls="--", lw=1.5)
ax.text(2.75e15, 1.07, "α t_cc = t_peri", color="#2f6db3", fontsize=14)
ax.text(9.5e15, 0.8, "survival boundary (dotted; bar = value at each r_p)\nx ≈ 10.5 · (r_p / 4×10¹⁵ cm)^(−0.7)", color="#1d2733", fontsize=13, va="top")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([4e15, 8e15, 1.6e16, 3.2e16]); ax.set_xticklabels(["4×10¹⁵", "8×10¹⁵", "1.6×10¹⁶", "3.2×10¹⁶"])
ax.minorticks_off(); ax.set_xlim(2.6e15, 5.2e16); ax.set_ylim(0.1, 40)
ax.set_xlabel("pericentre distance r_p [cm]   (R-variants R/4 … 4R slightly offset)")
ax.set_ylabel("x = α(N) t_cc / t_peri   (mixing measure, α per resolution)")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=14)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "regime.png"), dpi=150)
print("Gespeichert: runs/regime.png")

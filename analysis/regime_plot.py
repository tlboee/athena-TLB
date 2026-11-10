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
ax.plot(rr, 15 * (rr / 4e15)**-0.5, color="#1d2733", ls=":", lw=2, zorder=2)   # Ueberlebensgrenze (10.11.)
done = set()
for _, r in df.iterrows():
    xp = r["rp"] * 10**OFF.get(round(r["R_R0"], 2), 0.0)
    if r["run"] == "ps_ref_big":
        ax.scatter(xp, r["x"], marker="*", c="#c0392b", s=380, edgecolors="white", linewidths=0.8, zorder=4, label="CC01 (G2)")
        ax.annotate("CC01 (G2)", (xp, r["x"]), xytext=(-150, -4), textcoords="offset points",
                    fontsize=16, color="#c0392b", weight="bold")
        continue
    m, c, lab = STY[r["regime"]]
    ax.scatter(xp, r["x"], marker=m, c=c, s=110, edgecolors="white", linewidths=0.8, zorder=3,
               label=None if r["regime"] in done else lab)
    done.add(r["regime"])
ax.axhline(1, color="#2f6db3", ls="--", lw=1.5)
ax.text(2.75e15, 1.07, "α t_cc = t_peri", color="#2f6db3", fontsize=14)
ax.text(9.2e15, 5.2, "survival boundary (dotted)\nx ≈ 15 · (r_p / 4×10¹⁵ cm)^(−1/2)", color="#1d2733", fontsize=13, va="top")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([4e15, 8e15, 1.6e16, 3.2e16]); ax.set_xticklabels(["4×10¹⁵", "8×10¹⁵", "1.6×10¹⁶", "3.2×10¹⁶"])
ax.minorticks_off(); ax.set_xlim(2.6e15, 5.2e16); ax.set_ylim(0.15, 60)
ax.set_xlabel("pericentre distance r_p [cm]   (R-variants R/4 … 4R slightly offset)")
ax.set_ylabel("x = α t_cc / t_peri")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=14)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "regime.png"), dpi=150)
print("Gespeichert: runs/regime.png")

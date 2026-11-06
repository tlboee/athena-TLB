"""Test des Uebergangs ueber den Wolkenradius: t_50 / t_peri gegen alpha t_cc / t_peri.

alpha wie in Plot B: an die chi = 10-Grundlaeufe (R0, 512) angepasst, NICHT an die R-Varianten.
t_cc am Start mit dem Radius des jeweiligen Laufs (manifest: problem/R_cloud, sonst R0).
Modell: y = x fuer x < 1 (Cloud Crushing), sonst y = 1 + t_tid / t_peri (Perizentrum).
Aufruf (Notebook):  %run ~/athena/analysis/r_test.py
"""
import json, math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/athena/runs")
YR, R0 = 3.156e7, 1.87e15
MEAS = [("0.9", "Mixing: tracer > 0.9"), ("da_1/3", "Dense stream: ρ > ρ_cl0/3"),
        ("dq", "Dense stream: ρ/ρ_eq > √χ₀")]
COL = {10: "C0", 100: "C1", 1000: "C2"}

tab = pd.read_csv(os.path.join(BASE, "study_table.csv"))
t = tab[(tab["Y"] == "1/2") & (tab["norm"] == "init")].copy()

def rfac(run):
    with open(os.path.join(BASE, run, "manifest.json")) as f:
        man = json.load(f)
    R = float(man.get("params", {}).get("problem/R_cloud", R0))
    return R / R0

base = t[t["run"].str.startswith("g_") & ~t["run"].str.contains("_R") & ~t["run"].str.contains("_x")].copy()
var = t[t["run"].str.contains(r"_rp4_R")].copy()
var["f"] = [rfac(r) for r in var["run"]]
base["f"] = 1.0

fig, axes = plt.subplots(1, len(MEAS), figsize=(6.5 * len(MEAS), 5.4), sharey=True, constrained_layout=True)
print(f"{'Lauf':28s} {'Mass':>5s} {'R/R0':>5s} {'x':>6s} {'y':>6s} {'Modell':>7s}")
for ax, (key, title) in zip(axes, MEAS):
    b = base[base["X"] == key]
    ref = b[b["chi"] == 10]
    alpha = float(np.median(ref["t"] / ref["t_cc0"]))
    tt = b.drop_duplicates("rp").set_index("rp")
    # Grundlaeufe grau
    ok = np.isfinite(b["t"])
    ax.plot(alpha * b["t_cc0"][ok] / b["t_peri"][ok], b["t"][ok] / b["t_peri"][ok], "o", color="0.75", ms=6,
            label="grid (R = R₀)")
    # Modell fuer r_p = 4e15
    rp4 = b[b["rp"].round(-14) == 4.0e15].iloc[0]
    xs = np.logspace(-1.5, 1.6, 200)
    ax.plot(xs, np.where(xs < 1, xs, 1 + rp4["t_tid_peri"] / rp4["t_peri"]), "k--", lw=1,
            label="model: min(x, 1 + t_tid/t_peri)")
    # R-Varianten
    v = var[var["X"] == key]
    for chi, ss in v.groupby("chi"):
        c = COL.get(int(round(chi)), "k")
        for _, r in ss.sort_values("f").iterrows():
            x = alpha * r["t_cc0"] * r["f"] / r["t_peri"]
            reached = np.isfinite(r["t"])
            y = (r["t"] if reached else r["t_end"]) / r["t_peri"]
            ax.plot(x, y, "*" if reached else "^", color=c, ms=11, mfc=c if reached else "none")
            ax.annotate(f"{r['f']:g}R", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7, color=c)
            model = x if x < 1 else 1 + r["t_tid_peri"] / r["t_peri"]
            print(f"{r['run']:28s} {key:>5s} {r['f']:5g} {x:6.2f} {y:6.2f} {model:7.2f}")
        ax.plot([], [], "*", color=c, ms=10, label=f"χ = {chi:g}, R varied")
    ax.axvline(1, color="0.6", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.05, 40); ax.set_ylim(0.1, 8)
    ax.set_xlabel("α t_cc / t_peri  (t_cc with the run's R)")
    ax.set_title(f"{title}  (α = {alpha:.1f}, fitted to χ = 10 at R₀)", fontsize=10)
axes[0].set_ylabel("t_50 / t_peri")
axes[0].legend(fontsize=7, loc="upper left")
out = os.path.join(BASE, "r_test.png")
fig.savefig(out, dpi=150)
print("Gespeichert:", out)

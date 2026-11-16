"""Widerstand: verliert das Wolkengas Bahnenergie? (Vergleich mit Gillessen et al. 2019)

Pro Zelle e = v^2/2 - GM/r und j = x v_y - y v_x; massengewichteter Mittelwert (rho*tr) ueber drei Masken:
Kern (tracer > 0.9), Wolke (tracer > 0.5), dicht (rho/rho_eq > sqrt(chi0)).  a = -GM/(2<e>).
Ohne Wechselwirkung bleiben a/a0 und j/j0 bei 1 (Kepler).  Cache: <lauf>/drag.csv.  Ausgabe: runs/drag_test.png
Aufruf:  %run ~/athena/analysis/drag_test.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/athena/analysis"))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, GM = 3.156e7, 6.674e-8 * 4.31e6 * 1.989e33
RHO0, RR0, R_IN, CHI0, T_PERI = 1.702e-21, 1.0e16, 1.9e15, 2755.0, 18.0
RUNS = [("1024 adiabatic", "ps_ref_big_x2_t50", "#21918c", "-"),
        ("1024 with cooling", "ps_ref_big_cool_x2_t50", "#21918c", "--"),
        ("2048 adiabatic", "ps_ref_big_x4_t50", "#bddf26", "-")]
MASKS = [("core", "tracer > 0.9"), ("cloud", "tracer > 0.5"), ("dense", "ρ/ρ_eq > √χ₀")]

def vel(s, k):
    return s[[n for n in (f"vel{k}", "vx" if k == 1 else "vy", f"v{k}") if n in s][0]]

def series(run):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "drag.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        X, Y = np.meshgrid(s["x"], s["y"])
        r = np.hypot(X, Y)
        vx, vy, rho, tr = vel(s, 1), vel(s, 2), s["rho"], np.clip(s["r0"], 0, 1)
        e = 0.5 * (vx**2 + vy**2) - GM / np.maximum(r, R_IN)
        j = X * vy - Y * vx
        c = rho / (RHO0 * RR0 / np.maximum(r, R_IN))
        row = [t / YR]
        for name, m in (("core", tr > 0.9), ("cloud", tr > 0.5), ("dense", c > np.sqrt(CHI0))):
            w = (rho * tr)[m]
            if w.sum() <= 0 or m.sum() < 10:
                row += [np.nan, np.nan, np.nan]; continue
            em, jm = float((w * e[m]).sum() / w.sum()), float((w * j[m]).sum() / w.sum())
            row += [-GM / (2 * em) if em < 0 else np.nan, jm, float(w.sum())]
        rows.append(row)
        print(f"  {run}: t = {t/YR:5.1f} yr", end="\r")
    print()
    cols = ["t"] + [f"{k}_{q}" for k, _ in MASKS for q in ("a", "j", "m")]
    df = pd.DataFrame(rows, columns=cols); df.to_csv(cache, index=False)
    return df

fig, axes = plt.subplots(2, 3, figsize=(18, 8.5), sharex=True, constrained_layout=True)
data = {}
for lab, run, c, ls in RUNS:
    if not os.path.isdir(os.path.join(BASE, run)):
        print(f"{run} fehlt"); continue
    df = data[lab] = series(run)
    for i, (k, mlab) in enumerate(MASKS):
        a0, j0 = df[f"{k}_a"].iloc[0], df[f"{k}_j"].iloc[0]
        axes[0, i].plot(df["t"], df[f"{k}_a"] / a0, color=c, ls=ls, label=lab)
        axes[1, i].plot(df["t"], df[f"{k}_j"] / j0, color=c, ls=ls, label=lab)
for i, (k, mlab) in enumerate(MASKS):
    axes[0, i].set_title(f"semi-major axis a/a0  ({mlab})", fontsize=11)
    axes[1, i].set_title(f"specific angular momentum j/j0  ({mlab})", fontsize=11)
    for ax in axes[:, i]:
        ax.axhline(1, color="0.6", lw=0.8); ax.axvline(T_PERI, color="k", ls=":", lw=0.8)
    axes[1, i].set_xlabel("t [yr]")
axes[0, 0].legend(fontsize=9)
fig.suptitle("Does the cloud gas lose orbital energy? (Kepler: a/a0 = j/j0 = 1)")
fig.savefig(os.path.join(BASE, "drag_test.png"), dpi=150)
plt.show()

print(f"\na0 (t = 0, Kern) = {data[next(iter(data))]['core_a'].iloc[0]:.3e} cm  (erwartet ~6.5e16 aus P = 137.6 yr)")
print(f"{'Lauf':18s} {'Maske':6s} " + " ".join(f"{'a/a0 @' + str(tt):>9s}" for tt in (10, 17, 20, 30, 50)))
for lab, df in data.items():
    for k, _ in MASKS:
        a0 = df[f"{k}_a"].iloc[0]
        vals = []
        for tt in (10, 17, 20, 30, 50):
            i = int(np.argmin(np.abs(df["t"] - tt)))
            vals.append(f"{df[f'{k}_a'].iloc[i] / a0:9.3f}" if abs(df["t"].iloc[i] - tt) < 0.6 else f"{'-':>9s}")
        print(f"{lab:18s} {k:6s} " + " ".join(vals))

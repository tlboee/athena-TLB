"""Reine Gezeitenstreckung: Testteilchen auf Kepler-Bahnen gegen die gemessene Breite des Stroms.

Anfangswolke: Scheibe mit Radius R0 um (x_cloud, y_cloud), alle Teilchen mit (vx, vy) der Wolke (CC01-Werte).
Integration: Leapfrog mit globalem Zeitschritt dt = 2e-3 * r_min^1.5 / sqrt(GM) (r_min = kleinster Abstand eines Teilchens).
Breite: Teilchen auf das 1024er-Gitter (dx = 1.39e14 cm) verteilt, Luecken geschlossen, r_eff = 2A/P wie in stream_width.py.
Vergleich: runs/ps_ref_big_x2_t50/stream_width.csv und ps_ref_big_x2_peri/stream_width.csv (tracer > 0.5).
Ausgabe: runs/ballistic_width.png und runs/ballistic_width.csv.   Aufruf:  %run ~/athena/analysis/ballistic_width.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
try:
    from skimage.measure import find_contours
except ImportError:
    find_contours = None

BASE = os.path.expanduser("~/athena/runs")
YR, KM, GM = 3.156e7, 1e5, 6.674e-8 * 4.31e6 * 1.989e33
R0, XC, YC, VX, VY = 1.87e15, -7.22e16, 2.21e16, 794.59 * KM, 48.45 * KM
DX, N, T_END, T_PERI = 1.39e14, 200_000, 30.0, 18.0
OUT = np.unique(np.concatenate([np.arange(0, 16, 0.5), np.arange(16, 20.01, 0.05), np.arange(20.5, T_END + 0.01, 0.5)]))

def perimeter(mask, dx):
    if find_contours is not None:
        return sum(np.sum(np.hypot(*np.diff(c, axis=0).T)) for c in find_contours(mask.astype(float), 0.5)) * dx
    edges = np.sum(mask[1:, :] != mask[:-1, :]) + np.sum(mask[:, 1:] != mask[:, :-1])
    return edges * dx * np.pi / 4

def width(x, y):
    x0, y0 = x.min() - 3 * DX, y.min() - 3 * DX
    ix, iy = ((x - x0) / DX).astype(int), ((y - y0) / DX).astype(int)
    m = np.zeros((iy.max() + 4, ix.max() + 4), bool); m[iy, ix] = True
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
    return 2 * m.sum() * DX * DX / perimeter(m, DX) / R0

rng = np.random.default_rng(1)
r, phi = R0 * np.sqrt(rng.random(N)), 2 * np.pi * rng.random(N)
x, y = XC + r * np.cos(phi), YC + r * np.sin(phi)
vx, vy = np.full(N, VX), np.full(N, VY)
acc = lambda x, y: (-GM * x / np.hypot(x, y)**3, -GM * y / np.hypot(x, y)**3)
ax_, ay_ = acc(x, y)
t, rows, k = 0.0, [], 0
while k < len(OUT):
    if t >= OUT[k] * YR - 1e-6:
        rows.append([t / YR, width(x, y)]); print(f"  t = {t/YR:6.2f} yr", end="\r"); k += 1; continue
    dt = min(2e-3 * np.hypot(x, y).min()**1.5 / np.sqrt(GM), OUT[k] * YR - t)
    vx += 0.5 * dt * ax_; vy += 0.5 * dt * ay_
    x += dt * vx; y += dt * vy
    ax_, ay_ = acc(x, y)
    vx += 0.5 * dt * ax_; vy += 0.5 * dt * ay_
    t += dt
print()
bal = pd.DataFrame(rows, columns=["t", "r_eff_R0"]); bal.to_csv(os.path.join(BASE, "ballistic_width.csv"), index=False)

fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
ax.semilogy(bal["t"], bal["r_eff_R0"], color="k", lw=2, label="ballistic test particles (gravity only)")
for run, lab, c in (("ps_ref_big_x2_t50", "hydro 1024, whole orbit", "#21918c"), ("ps_ref_big_x2_peri", "hydro 1024, fine", "#0b5d55"),
                    ("ps_ref_big_x4_peri", "hydro 2048, fine", "#bddf26")):
    p = os.path.join(BASE, run, "stream_width.csv")
    if os.path.exists(p):
        d = pd.read_csv(p); ax.semilogy(d["t"], d["r_tr_R0"], color=c, label=f"{lab} (tracer > 0.5)")
ax.axvline(T_PERI, color="k", ls=":", lw=0.8); ax.axhline(1, color="0.6", lw=0.8)
ax.set_xlabel("t [yr]"); ax.set_ylabel("effective width r_eff / R0 = 2A/(P R0)"); ax.set_xlim(0, T_END)
ax.set_title("Tidal stretching alone vs. hydrodynamic simulation (CC01)"); ax.legend(fontsize=9)
fig.savefig(os.path.join(BASE, "ballistic_width.png"), dpi=150)
plt.show()
print(f"{'t [yr]':>7s} {'ballistisch':>12s} {'hydro 1024':>11s}")
h = pd.read_csv(os.path.join(BASE, "ps_ref_big_x2_t50", "stream_width.csv"))
for tt in (0, 10, 15, 17, 18, 19, 20, 25, 30):
    b = bal.iloc[int(np.argmin(np.abs(bal["t"] - tt)))]; hh = h.iloc[int(np.argmin(np.abs(h["t"] - tt)))]
    print(f"{tt:7.1f} {b['r_eff_R0']:12.3f} {hh['r_tr_R0']:11.3f}")

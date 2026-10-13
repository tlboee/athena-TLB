"""Uebersichtsplot: Dichte und Tracer zu mehreren Zeitpunkten, mit Keplerbahn.

Aufruf:  %run ~/athena/analysis/plot_g2_cart.py <run_dir> [Jahre, z.B. 0,10,15,17,18,20]
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

RUN     = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/athena/runs/g2_cart_cc01")
TIMES   = [float(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 10, 15, 17, 18, 20]
YR      = g2_io.YR
T_START = 1995.5


def kepler_orbit(t_end, n=50000):
    """Ballistische Bahn aus Burkert+12 Tabelle 1 (Leapfrog-Integration)."""
    GM  = 6.674e-8 * 4.31e6 * 1.989e33
    pos = np.array([-7.22e16, 2.21e16])
    vel = np.array([794.59e5, 48.45e5])
    dt  = t_end / n
    path = [pos.copy()]
    for _ in range(n):
        vel += 0.5 * dt * (-GM * pos / np.linalg.norm(pos)**3)
        pos += dt * vel
        vel += 0.5 * dt * (-GM * pos / np.linalg.norm(pos)**3)
        path.append(pos.copy())
    return np.array(path) / 1e16


snaps = g2_io.nearest_snapshots(RUN, TIMES)
orbit = kepler_orbit(max(max(TIMES), 1.0) * YR)

fig, axes = plt.subplots(2, len(snaps), figsize=(4.2 * len(snaps), 6.5),
                         sharex=True, sharey=True, constrained_layout=True, squeeze=False)
for col, n in enumerate(snaps):
    t, g = g2_io.load_snapshot(RUN, n)
    X, Y = np.meshgrid(g["x"] / 1e16, g["y"] / 1e16)
    p0 = axes[0, col].pcolormesh(X, Y, g["rho"], norm=LogNorm(1e-23, 1e-17), shading="auto")
    p1 = axes[1, col].pcolormesh(X, Y, g["r0"], vmin=0, vmax=1, cmap="magma", shading="auto")
    axes[0, col].set_title(f"t = {t / YR:.1f} yr  ({T_START + t / YR:.1f})")
    for ax in axes[:, col]:
        ax.plot(orbit[:, 0], orbit[:, 1], "w--", lw=0.7)   # Keplerbahn
        ax.plot(0, 0, "w+", ms=8)                           # Schwarzes Loch

for ax in axes.flat:
    ax.set_aspect("equal")
for ax in axes[-1]:
    ax.set_xlabel(r"x [$10^{16}$ cm]")
for ax in axes[:, 0]:
    ax.set_ylabel(r"y [$10^{16}$ cm]")
fig.colorbar(p0, ax=axes[0], label=r"$\rho$ [g/cm$^3$]")
fig.colorbar(p1, ax=axes[1], label="Tracer r0")

out = os.path.join(RUN, "plots")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "overview.png"), dpi=150)
print("Gespeichert:", os.path.join(out, "overview.png"))

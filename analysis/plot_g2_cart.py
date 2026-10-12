import glob, os, re, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

RUN   = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/athena/runs/g2_cart_cc01")
BASE  = "g2_cart"
YR    = 3.156e7
SNAPS = [0, 20, 30, 34, 36, 40]     # 0, 10, 15, 17, 18, 20 yr (Output alle 0.5 yr)


def load_snapshot(n):
    """Liest alle Meshblocks eines Snapshots und setzt sie zu 2D-Arrays (y, x) zusammen."""
    files = sorted(glob.glob(f"{RUN}/{BASE}.block*.out1.{n:05d}.tab"))
    with open(files[0]) as f:
        t = float(re.search(r"time=([0-9.eE+-]+)", f.readline()).group(1))
    d = np.vstack([np.loadtxt(fn) for fn in files])
    # Spalten: 0 i, 1 x, 2 j, 3 y, 4 rho, 5 press, 6 vx, 7 vy, 8 vz, 9 r0
    x, y = np.unique(d[:, 1]), np.unique(d[:, 3])
    i, j = np.searchsorted(x, d[:, 1]), np.searchsorted(y, d[:, 3])
    g = {}
    for k, name in enumerate(["rho", "press", "vx", "vy", "vz", "r0"], start=4):
        a = np.full((len(y), len(x)), np.nan)
        a[j, i] = d[:, k]
        g[name] = a
    g["X"], g["Y"] = np.meshgrid(x, y)
    return t, g


def kepler_orbit(t_end, n=20000):
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


orbit = kepler_orbit(20 * YR)

fig, axes = plt.subplots(2, len(SNAPS), figsize=(4.2 * len(SNAPS), 6.5),
                         sharex=True, sharey=True, constrained_layout=True)

for col, n in enumerate(SNAPS):
    t, g = load_snapshot(n)
    X, Y = g["X"] / 1e16, g["Y"] / 1e16
    p0 = axes[0, col].pcolormesh(X, Y, g["rho"], norm=LogNorm(1e-23, 1e-17), shading="auto")
    p1 = axes[1, col].pcolormesh(X, Y, g["r0"], vmin=0, vmax=1, cmap="magma", shading="auto")
    axes[0, col].set_title(f"t = {t / YR:.1f} yr  ({1995.5 + t / YR:.1f})")
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

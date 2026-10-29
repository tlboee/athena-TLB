"""CC01 adiabatisch vs. mit Kuehlung (gleiches Setup): Dichtebilder und Massen ueber die Zeit.
Aufruf:  python3 analysis/cool_compare.py runs/cool_test1 runs/cc01_cool_mac
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

A, C = sys.argv[1], sys.argv[2]
YR, KB, MH, MU = 3.156e7, 1.380649e-16, 1.6726e-24, 0.6139
RHO0, R0, R_IN = 1.702e-21, 1e16, 1.9e15
TIMES = [10, 18, 25, 35, 50]

def series(run):
    out = {"t": [], "cold": [], "dense": [], "cloud": []}
    snaps = {}
    for n in g2_io.snapshot_numbers(run):
        t, s = g2_io.load_snapshot(run, n)
        rho, P, tr = s["rho"], s["press"], s["r0"]
        X, Y = np.meshgrid(s["x"], s["y"])
        dA = (s["x"][1] - s["x"][0]) * (s["y"][1] - s["y"][0])
        T = P * MU * MH / (rho * KB)
        rho_eq = RHO0 * R0 / np.maximum(np.hypot(X, Y), R_IN)
        m = rho * tr * dA
        out["t"].append(t / YR)
        out["cold"].append(m[(tr > 0.5) & (T < 3e4)].sum())
        out["dense"].append(m[rho / rho_eq > 10].sum())
        out["cloud"].append(m.sum())
        snaps[t / YR] = (s["x"], s["y"], rho / rho_eq)
    return {k: np.array(v) for k, v in out.items()}, snaps

sa, na = series(A)
sc, nc = series(C)
fig = plt.figure(figsize=(17, 9), constrained_layout=True)
gs = fig.add_gridspec(3, len(TIMES))
for row, (snaps, lab) in enumerate([(na, "adiabatisch"), (nc, "mit Kuehlung")]):
    for col, tt in enumerate(TIMES):
        key = min(snaps, key=lambda k: abs(k - tt))
        x, y, con = snaps[key]
        ax = fig.add_subplot(gs[row, col])
        im = ax.pcolormesh(x / 1e16, y / 1e16, np.log10(np.maximum(con, 1)), vmin=0, vmax=4, shading="auto")
        ax.set_xlim(-14, 2); ax.set_ylim(-8, 3); ax.set_aspect("equal")
        ax.set_title(f"{lab}, t = {key:.1f} yr", fontsize=9)
fig.colorbar(im, ax=fig.axes, label="log10 rho/rho_eq", shrink=0.6)
for col, (key, lab) in enumerate([("cold", "kaltes Wolkengas (tr > 0.5, T < 3e4 K)"),
                                  ("dense", "dichter Strom (rho/rho_eq > 10)"),
                                  ("cloud", "Wolkengas gesamt (sum rho tr)")]):
    ax = fig.add_subplot(gs[2, col])
    ax.plot(sa["t"], sa[key] / sa["cloud"][0], label="adiabatisch")
    ax.plot(sc["t"], sc[key] / sc["cloud"][0], label="mit Kuehlung")
    ax.axvline(18.0, color="grey", lw=0.8); ax.set_xlabel("t [yr]"); ax.set_title(lab, fontsize=9)
    ax.legend(fontsize=8)
out = os.path.join(C, "plots", "cool_compare.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120)
for key in ("cold", "dense", "cloud"):
    i10a, i10c = np.argmin(abs(sa["t"] - 10)), np.argmin(abs(sc["t"] - 10))
    print(f"{key:6s}: t=10 yr  adiab {sa[key][i10a]/sa['cloud'][0]:.3f}  kuehl {sc[key][i10c]/sc['cloud'][0]:.3f}"
          f"   |  t=20 yr  adiab {sa[key][-1]/sa['cloud'][0]:.3f}  kuehl {sc[key][-1]/sc['cloud'][0]:.3f}")
print("Gespeichert:", out)

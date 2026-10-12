#!/usr/bin/env python3
"""
plot_g2_cloud.py  —  Visualisierung des G2-Wolken-Testlaufs
Liest Athena++ .tab Output (prim + scalar) und plottet
Dichte und Tracer als 2D-Karte in kartesischen Koordinaten.

Nutzung:
    python plot_g2_cloud.py              # t=0 (erster Snapshot)
    python plot_g2_cloud.py 00003        # Snapshot Nr. 3 (~1 Jahr)
"""

import numpy as np
import matplotlib.pyplot as plt
import glob, os, sys

# ── Konfiguration ─────────────────────────────────────────────────────────────
RUN_DIR  = os.path.expanduser("~/runs/g2_cloud")
PROB_ID  = "g2_cloud"
TIMESTEP = sys.argv[1] if len(sys.argv) > 1 else "00000"

# G2-Sollposition (zum Überprüfen)
R_G2_CM   = 7.55e16   # cm
THETA_G2  = np.radians(73.0)
X_G2      = R_G2_CM * np.sin(THETA_G2)   # in Orbitalebene
Z_G2      = R_G2_CM * np.cos(THETA_G2)
R_CLOUD   = 1.87e15   # cm (Wolkenradius)

# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def parse_header(filename):
    """Liest Spaltennamen aus dem Athena++ Tab-Header.
    Athena++ schreibt Zeilen wie:
      # [1]=x1v  [2]=x2v  [3]=rho  [4]=vel1  ...
    """
    cols = []
    with open(filename) as f:
        for line in f:
            if not line.startswith('#'):
                break
            if '[1]=' in line:
                # z.B. "# [1]=x1v  [2]=x2v  [3]=rho ..."
                parts = line.strip('# \n').split()
                for p in parts:
                    if '=' in p:
                        cols.append(p.split('=')[1])
    return cols


def read_all_blocks(pattern):
    """Alle MeshBlock-Dateien für einen Zeitschritt einlesen.
    Gibt (cols, data_array) zurück, wobei data_array alle Blöcke
    untereinander enthält."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"\nKeine Dateien gefunden: {pattern}\n"
            f"Ist der Job schon gestartet und hat Output erzeugt?\n"
            f"Dateien im Run-Dir:\n" +
            "\n".join(os.listdir(RUN_DIR)[:20])
        )
    print(f"  {len(files)} Block-Dateien gefunden für Pattern: {os.path.basename(pattern)}")

    cols = parse_header(files[0])
    if not cols:
        raise ValueError(f"Keine Spalten-Header gefunden in {files[0]}")
    print(f"  Spalten: {cols}")

    blocks = [np.loadtxt(f, comments='#') for f in files]
    return cols, np.vstack(blocks)


def to_2d_grid(cols, data, xcol, ycol, vcol):
    """Stapelt die (x, y, v)-Punkte aus allen Blöcken zu einem 2D-Grid."""
    ix = cols.index(xcol)
    iy = cols.index(ycol)
    iv = cols.index(vcol)

    x_vals = np.unique(np.round(data[:, ix], decimals=3))
    y_vals = np.unique(np.round(data[:, iy], decimals=6))
    grid   = np.full((len(y_vals), len(x_vals)), np.nan)

    for row in data:
        i = np.searchsorted(x_vals, round(row[ix], 3))
        j = np.searchsorted(y_vals, round(row[iy], 6))
        if 0 <= i < len(x_vals) and 0 <= j < len(y_vals):
            grid[j, i] = row[iv]

    return x_vals, y_vals, grid

# ── Daten einlesen ────────────────────────────────────────────────────────────
print(f"\n=== Lese Snapshot {TIMESTEP} ===")

prim_pat   = f"{RUN_DIR}/{PROB_ID}.block*.out1.{TIMESTEP}.tab"
scalar_pat = f"{RUN_DIR}/{PROB_ID}.block*.out2.{TIMESTEP}.tab"

print("\nPrimitive Variablen (prim):")
pcols, pdata = read_all_blocks(prim_pat)

print("\nPassiver Tracer (scalar):")
scols, sdata = read_all_blocks(scalar_pat)

# Koordinatenspalten: x1v = r, x2v = theta
r_col  = 'x1v'
th_col = 'x2v'

r,  theta, rho    = to_2d_grid(pcols, pdata, r_col, th_col, 'rho')
_,  _,     press  = to_2d_grid(pcols, pdata, r_col, th_col, 'press')
_,  _,     vel1   = to_2d_grid(pcols, pdata, r_col, th_col, 'vel1')
_,  _,     vel2   = to_2d_grid(pcols, pdata, r_col, th_col, 'vel2')

# Tracer-Spalte (letzte Spalte in scalar output)
tr_col = scols[-1]
_, _, tracer = to_2d_grid(scols, sdata, r_col, th_col, tr_col)

print(f"\nGrid-Größe: {len(r)} × {len(theta)}  (r × θ)")
print(f"  r:     {r.min():.2e} – {r.max():.2e} cm")
print(f"  theta: {np.degrees(theta.min()):.1f}° – {np.degrees(theta.max()):.1f}°")
print(f"  rho:   {np.nanmin(rho):.2e} – {np.nanmax(rho):.2e} g/cm³")
print(f"  tracer: max = {np.nanmax(tracer):.4f}  (sollte ~1 in Wolkenzellen sein)")

# ── Koordinaten-Transformation: sphärisch → kartesisch ───────────────────────
R_grid, TH_grid = np.meshgrid(r, theta)
X_grid = R_grid * np.sin(TH_grid)   # [cm], Abstand von Rotationsachse
Z_grid = R_grid * np.cos(TH_grid)   # [cm], entlang Rotationsachse

scale = 1e16   # cm → Einheiten von 10^16 cm

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"G2-Wolke: Snapshot {TIMESTEP}  "
             f"(t ≈ {int(TIMESTEP) * 1e7 / 3.156e7:.2f} yr)", fontsize=13)

# --- Panel 1: Dichte (log) ---
ax = axes[0]
vmin_rho = np.log10(np.nanpercentile(rho, 1))
vmax_rho = np.log10(np.nanmax(rho))
im0 = ax.pcolormesh(X_grid/scale, Z_grid/scale, np.log10(rho),
                    cmap='magma', vmin=vmin_rho, vmax=vmax_rho,
                    shading='nearest')
plt.colorbar(im0, ax=ax, label='log₁₀ ρ  [g/cm³]')
ax.set_title('Dichte ρ')

# --- Panel 2: Tracer ---
ax = axes[1]
im1 = ax.pcolormesh(X_grid/scale, Z_grid/scale, tracer,
                    cmap='Blues', vmin=0, vmax=1, shading='nearest')
plt.colorbar(im1, ax=ax, label='tr  (0=Hintergrund, 1=Wolke)')
ax.set_title('Tracer tr')

# --- Panel 3: Radialgeschwindigkeit ---
ax = axes[2]
vmax_v = np.nanpercentile(np.abs(vel1), 99)
im2 = ax.pcolormesh(X_grid/scale, Z_grid/scale, vel1/1e5,  # cm/s → km/s
                    cmap='RdBu_r', vmin=-vmax_v/1e5, vmax=vmax_v/1e5,
                    shading='nearest')
plt.colorbar(im2, ax=ax, label='v_r  [km/s]')
ax.set_title('Radialgeschwindigkeit v_r')

# G2-Sollposition und Wolkenradius einzeichnen
for ax in axes:
    ax.set_xlabel("x  [10¹⁶ cm]")
    ax.set_ylabel("z  [10¹⁶ cm]")
    ax.set_aspect('equal')
    # Sollposition G2 (Kreis = Wolkenradius)
    circle = plt.Circle((X_G2/scale, Z_G2/scale), R_CLOUD/scale,
                         color='lime', fill=False, lw=1.2, ls='--',
                         label='G2 Sollposition')
    ax.add_patch(circle)
    ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
outfile = f"{RUN_DIR}/plot_t{TIMESTEP}.png"
plt.savefig(outfile, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot gespeichert: {outfile}")
plt.show()

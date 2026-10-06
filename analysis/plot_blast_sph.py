import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "blast"
sys.path.append(str(ROOT / "vis" / "python"))
import athena_read

# Letzten Output bei t = 0.4 einlesen
r, theta, phi, data = athena_read.vtk(
    RUN / "Blast.block0.out1.00004.vtk"
)

rho = data["rho"]

# rho hat die Reihenfolge (phi, theta, r)
# Schnitt möglichst nahe bei phi = 0
iphi = np.argmin(np.abs(0.5 * (phi[:-1] + phi[1:])))
rho_slice = rho[iphi, :, :]

# r und theta sind Zellgrenzen
R, THETA = np.meshgrid(r, theta)

# Sphärische Polarkoordinaten -> kartesische Koordinaten
# Für phi = 0 gilt y = 0
X = R * np.sin(THETA)
Z = R * np.cos(THETA)

plt.figure(figsize=(7, 6))

plt.pcolormesh(
    X,
    Z,
    rho_slice,
    shading="flat"
)

plt.colorbar(label="Density")

plt.xlabel("x")
plt.ylabel("z")
plt.title("MHD Blast Wave in Spherical Polar Coordinates at t = 0.4")

plt.axis("equal")
plt.tight_layout()
plt.show()
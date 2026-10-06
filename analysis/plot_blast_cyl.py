import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "blast"
sys.path.append(str(ROOT / "vis" / "python"))
import athena_read

# Letzten Output bei t = 0.3 einlesen
x1, x2, x3, data = athena_read.vtk(
    RUN / "Blast.block0.out1.00003.vtk"
)

rho = data["rho"]

# Schnitt durch die Mitte in z-Richtung
rho_slice = rho[32, :, :]

# x1 und x2 sind Zellgrenzen:
# x1 = r, x2 = phi
r, phi = np.meshgrid(x1, x2)

# Zylinderkoordinaten -> kartesische Koordinaten
X = r * np.cos(phi)
Y = r * np.sin(phi)

plt.figure(figsize=(8, 6))

plt.pcolormesh(
    X,
    Y,
    rho_slice,
    shading="flat"
)

plt.colorbar(label="Density")

plt.xlabel("x")
plt.ylabel("y")
plt.title("MHD Blast Wave in Cylindrical Coordinates at t = 0.3")

plt.axis("equal")
plt.tight_layout()
plt.show()
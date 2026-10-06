import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "linear_wave"
sys.path.append(str(ROOT / "vis" / "python"))

import athena_read
import matplotlib.pyplot as plt

# 3D-VTK-Datei einlesen
x, y, z, data = athena_read.vtk(
    RUN / "LinWave.block0.out2.00500.vtk"
)

rho = data["rho"]

# Mittlere Ebene in z-Richtung auswählen
k = rho.shape[0] // 2
rho_slice = rho[k, :, :]

print("3D shape:", rho.shape)
print("2D slice shape:", rho_slice.shape)
print("rho min/max:", rho.min(), rho.max())

# 2D-Schnitt darstellen
plt.figure(figsize=(10, 5))

plt.pcolormesh(
    x,
    y,
    rho_slice,
    shading="flat"
)

plt.colorbar(label="Density")
plt.xlabel("x")
plt.ylabel("y")
plt.title("3D MHD Linear Wave at t = 5.0, central z-slice")

plt.tight_layout()
plt.show()
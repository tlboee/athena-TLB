import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "orszag_tang"
sys.path.append(str(ROOT / "vis" / "python"))

import athena_read
import matplotlib.pyplot as plt

# Athena++ VTK-Datei einlesen
x, y, z, data = athena_read.vtk(
    RUN / "OrszagTang.block0.out2.00100.vtk"
)

rho = data["rho"]

print("rho shape:", rho.shape)
print("rho min/max:", rho.min(), rho.max())

plt.figure(figsize=(8, 7))

plt.imshow(
    rho.squeeze(),
    origin="lower",
    extent=[x[0], x[-1], y[0], y[-1]],
    aspect="equal"
)

plt.colorbar(label="Density")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Orszag-Tang Vortex at t = 1.0")

plt.tight_layout()
plt.show()
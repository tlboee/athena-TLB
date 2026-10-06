import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "blast"
sys.path.append(str(ROOT / "vis" / "python"))
import athena_read

# Gesamtes Dichtefeld: z, y, x
rho = np.zeros((64, 64, 64))

# Die vier MeshBlocks einlesen
for block in range(4):
    filename = RUN / f"Blast.block{block}.out1.00010.vtk"

    x, y, z, data = athena_read.vtk(filename)
    rho_block = data["rho"]

    print(f"Block {block}: rho shape = {rho_block.shape}")

    # Position des Blocks im globalen Gitter
    ix = 0 if x[0] < 0 else 32
    iy = 0 if y[0] < 0 else 32

    rho[:, iy:iy+32, ix:ix+32] = rho_block

# Schnitt durch die Mitte in z-Richtung
rho_slice = rho[32, :, :]

plt.figure(figsize=(7, 6))

plt.imshow(
    rho_slice,
    origin="lower",
    extent=[-1, 1, -1, 1],
    aspect="equal"
)

plt.colorbar(label="Density")
plt.xlabel("x")
plt.ylabel("y")
plt.title("3D MHD Blast Wave at t = 1")

plt.tight_layout()
plt.show()
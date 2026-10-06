import sys
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "gr_blast"
sys.path.append(str(ROOT / "vis" / "python"))
import athena_read

x, y, z, data = athena_read.vtk(
    RUN / "gr_blast.block0.out1.00010.vtk"
)

rho = data["rho"].squeeze()
press = data["press"].squeeze()

plt.figure(figsize=(7, 6))

plt.pcolormesh(x, y, press, shading="flat")
plt.colorbar(label="Gas pressure")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Relativistic Blast Wave at t = 1")

plt.axis("equal")
plt.tight_layout()
plt.show()
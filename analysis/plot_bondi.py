import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "gr_bondi"
sys.path.append(str(ROOT / "vis" / "python"))
import athena_read

r, theta, phi, data = athena_read.vtk(
    RUN / "gr_bondi.block0.out1.00001.vtk"
)

# Cell faces -> cell centers
rc = 0.5 * (r[:-1] + r[1:])

rho = data["rho"].squeeze()
press = data["press"].squeeze()

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(rc, rho, label="Density")
ax1.set_xlabel("r / M")
ax1.set_ylabel("Density")

ax2 = ax1.twinx()
ax2.plot(rc, press, linestyle="--", label="Pressure")
ax2.set_ylabel("Gas pressure")

# Sonic point from athinput.bondi
ax1.axvline(8.0, linestyle=":", label="Sonic point")

ax1.set_title("GR Bondi Accretion at t = 100")
ax1.legend(loc="upper right")

plt.tight_layout()
plt.show()
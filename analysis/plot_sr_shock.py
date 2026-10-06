from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "sr_shock"

data = np.loadtxt(
    RUN / "hydro_shock_rel_1.block0.out1.00001.tab",
    comments="#"
)

x = data[:, 1]
rho = data[:, 2]
press = data[:, 3]

# Athena++ stores u = gamma*v
u = data[:, 4]
gamma = np.sqrt(1.0 + u**2)
vx = u / gamma

fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

ax[0].plot(x, rho)
ax[0].set_ylabel("Density")

ax[1].plot(x, press)
ax[1].set_ylabel("Pressure")

ax[2].plot(x, vx)
ax[2].set_ylabel(r"$v_x/c$")
ax[2].set_xlabel("x")

fig.suptitle("Special Relativistic Shock Tube at t = 0.4")

plt.tight_layout()
plt.show()
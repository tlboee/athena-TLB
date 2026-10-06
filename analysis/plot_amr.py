import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "slotted_cylinder"
sys.path.append(str(ROOT / "vis" / "python"))
import athena_read

d = athena_read.athdf(
    RUN / "SlottedCylinder.out2.00100.athdf",
    raw=True
)

levels = d["Levels"]
x1f = d["x1f"]
x2f = d["x2f"]

fig, ax = plt.subplots(figsize=(10, 5))

for n, level in enumerate(levels):
    x0 = x1f[n, 0]
    x1 = x1f[n, -1]
    y0 = x2f[n, 0]
    y1 = x2f[n, -1]

    rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        linewidth=0.7 if level == 0 else 1.0
    )

    ax.add_patch(rect)

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)

ax.set_aspect("equal")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Static Mesh Refinement")

plt.tight_layout()
plt.show()
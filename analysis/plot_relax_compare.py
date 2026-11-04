import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, os, re

BASE = os.path.expanduser("~/athena/runs")
RUNS = [
    ("instant", r"$\tau=0$ (sofort)",    "#e41a1c"),
    ("0.1yr",   r"$\tau=0.1$ yr",        "#ff7f00"),
    ("0.5yr",   r"$\tau=0.5$ yr",        "#4daf4a"),
    ("1yr",     r"$\tau=1$ yr",          "#377eb8"),
    ("5yr",     r"$\tau=5$ yr",          "#984ea3"),
]
S_PER_YR = 3.156e7

fig, ax = plt.subplots(figsize=(9, 5))

for run_name, label, color in RUNS:
    run_dir = os.path.join(BASE, f"relax_{run_name}")
    prefix  = f"relax_{run_name}"
    pattern = os.path.join(run_dir, f"{prefix}.block0.out1.*.tab")
    b0_files = sorted(glob.glob(pattern))
    times, v_inners = [], []

    for b0 in b0_files:
        ts = b0.split(".")[-2]
        all_block_files = sorted(glob.glob(
            os.path.join(run_dir, f"{prefix}.block*.out1.{ts}.tab")))
        sim_time = None
        rows = []
        for bf in all_block_files:
            with open(bf) as f:
                for line in f:
                    if line.startswith("#"):
                        if sim_time is None and "time=" in line:
                            m = re.search(r"time=([0-9eE.+\-]+)", line)
                            if m:
                                sim_time = float(m.group(1))
                        continue
                    v = line.split()
                    if len(v) < 7:
                        continue
                    rows.append((float(v[1]), float(v[3]), float(v[6])))
        if not rows or sim_time is None:
            continue
        r_min = min(r[0] for r in rows)
        inner = [(r[1], r[2]) for r in rows if abs(r[0] - r_min) < 1e-6 * r_min]
        if not inner:
            continue
        thetas = np.array([c[0] for c in inner])
        vels   = np.array([c[1] for c in inner])
        v_avg  = np.average(vels, weights=np.sin(thetas))
        times.append(sim_time / S_PER_YR)
        v_inners.append(v_avg / 1e5)

    if times:
        ax.plot(times, v_inners, marker="o", ms=5,
                lw=1.8, label=label, color=color)

ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Zeit [yr]", fontsize=12)
ax.set_ylabel(r"$\langle v_r \rangle$ am inneren Rand [km/s]", fontsize=12)
ax.set_title(r"Radialgeschwindigkeit $v_r(r_\mathrm{min})$ – $\tau_\mathrm{relax}$-Studie", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.expanduser("~/athena/relax_compare.png")
plt.savefig(out, dpi=150)
print(f"Gespeichert: {out}")

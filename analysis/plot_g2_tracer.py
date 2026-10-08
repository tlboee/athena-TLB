import numpy as np
import matplotlib.pyplot as plt
import glob, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Run directory can be overridden, e.g. for cluster output copied elsewhere:
#   python analysis/plot_g2_tracer.py ~/Downloads/g2_2d_tracer
RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'runs' / 'g2_2d_tracer')
OUT_DIR = str(ROOT / 'figures' / 'g2_tracer')
os.makedirs(OUT_DIR, exist_ok=True)

def load_snapshot(snap_num):
    """Load all 16 blocks for a given snapshot and return merged arrays."""
    pattern = os.path.join(RUN_DIR, f"g2_2d_tracer.block*.out1.{snap_num:05d}.tab")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files for snapshot {snap_num}")
    
    rows = []
    for f in files:
        data = np.loadtxt(f, comments='#')
        rows.append(data)
    data = np.vstack(rows)
    
    # Sort by (j, i) so grid is ordered
    idx = np.lexsort((data[:,0], data[:,2]))
    data = data[idx]
    
    r   = data[:,1]
    th  = data[:,3]
    rho = data[:,4]
    tr  = data[:,9]   # r0 = passive tracer
    
    return r, th, rho, tr

def read_time(snap_num):
    """Read simulation time from first block header."""
    f = os.path.join(RUN_DIR, f"g2_2d_tracer.block0.out1.{snap_num:05d}.tab")
    with open(f) as fh:
        line = fh.readline()
    # "# Athena++ data at time=X  cycle=Y  variables=..."
    t = float(line.split("time=")[1].split()[0])
    return t

# ---- Plot each snapshot ----
yr = 3.156e7   # seconds per year

for snap in range(11):
    r, th, rho, tr = load_snapshot(snap)
    t_s = read_time(snap)
    t_yr = t_s / yr

    # Cartesian coords (in units of 1e16 cm)
    x = r * np.sin(th) / 1e16
    z = r * np.cos(th) / 1e16

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"G2 simulation — t = {t_yr:.1f} yr", fontsize=13)

    sc1 = axes[0].scatter(x, z, c=np.log10(rho), s=1, cmap='inferno',
                          vmin=-24, vmax=-18)
    plt.colorbar(sc1, ax=axes[0], label=r'$\log_{10}\,\rho$ [g cm$^{-3}$]')
    axes[0].set_xlabel(r'$x$ [$10^{16}$ cm]')
    axes[0].set_ylabel(r'$z$ [$10^{16}$ cm]')
    axes[0].set_title('Density')
    axes[0].set_aspect('equal')

    sc2 = axes[1].scatter(x, z, c=tr, s=1, cmap='RdBu_r', vmin=0, vmax=1)
    plt.colorbar(sc2, ax=axes[1], label='Tracer (1=ambient, 0=G2)')
    axes[1].set_xlabel(r'$x$ [$10^{16}$ cm]')
    axes[1].set_ylabel(r'$z$ [$10^{16}$ cm]')
    axes[1].set_title('Passive Tracer')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"g2_snap_{snap:05d}_t{t_yr:.1f}yr.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}  (t={t_yr:.1f} yr)")

print("Done.")

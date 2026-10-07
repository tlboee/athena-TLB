import numpy as np
import matplotlib.pyplot as plt
import glob, os

RUN_DIR = "/Users/tomleobehrendt/Downloads/g2_2d_tracer"
OUT_DIR = "/Users/tomleobehrendt/Downloads/g2_plots"
os.makedirs(OUT_DIR, exist_ok=True)

def load_snapshot(snap_num):
    pattern = os.path.join(RUN_DIR, f"g2_2d_tracer.block*.out1.{snap_num:05d}.tab")
    files = sorted(glob.glob(pattern))
    rows = []
    for f in files:
        data = np.loadtxt(f, comments='#')
        rows.append(data)
    data = np.vstack(rows)
    idx = np.lexsort((data[:,0], data[:,2]))
    data = data[idx]
    return data[:,1], data[:,3], data[:,4], data[:,9]  # r, th, rho, tracer

def read_time(snap_num):
    f = os.path.join(RUN_DIR, f"g2_2d_tracer.block0.out1.{snap_num:05d}.tab")
    with open(f) as fh:
        line = fh.readline()
    return float(line.split("time=")[1].split()[0])

yr = 3.156e7

for snap in range(11):
    r, th, rho, tr = load_snapshot(snap)
    t_yr = read_time(snap) / yr
    x = r * np.sin(th) / 1e16
    z = r * np.cos(th) / 1e16

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"G2 — t = {t_yr:.1f} yr", fontsize=13)

    sc1 = axes[0].scatter(x, z, c=np.log10(rho), s=1, cmap='inferno', vmin=-24, vmax=-18)
    plt.colorbar(sc1, ax=axes[0], label=r'$\log_{10}\,\rho$ [g cm$^{-3}$]')
    axes[0].set_xlabel(r'$x$ [$10^{16}$ cm]')
    axes[0].set_ylabel(r'$z$ [$10^{16}$ cm]')
    axes[0].set_title('Dichte')
    axes[0].set_aspect('equal')

    sc2 = axes[1].scatter(x, z, c=tr, s=1, cmap='RdBu_r', vmin=0, vmax=1)
    plt.colorbar(sc2, ax=axes[1], label='Tracer (1=Atmosphäre, 0=G2)')
    axes[1].set_xlabel(r'$x$ [$10^{16}$ cm]')
    axes[1].set_ylabel(r'$z$ [$10^{16}$ cm]')
    axes[1].set_title('Passiver Tracer')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"g2_snap_{snap:05d}_t{t_yr:.1f}yr.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"t={t_yr:.1f} yr gespeichert")

print("Fertig!")
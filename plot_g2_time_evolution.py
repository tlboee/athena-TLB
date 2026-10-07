"""
Zeitevolution v_r am inneren Rand (theta-gemittelt) für alle 4 Runs.
Für das Treffen mit Max Grönke, 8. Oktober 2026.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob, os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 11,
    'figure.dpi': 150,
})

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
YR_S     = 3.156e7   # seconds per year
KM_S     = 1e5       # cm/s → km/s

RUNS = [
    ('g2_2d',         'Kein NSC, keine Relaxation',  '#e74c3c'),  # rot
    ('g2_2d_nsc',     'NSC, keine Relaxation',       '#3498db'),  # blau
    ('g2_2d_relax',   'NSC + Relaxation (1 yr)',     '#2ecc71'),  # grün
    ('g2_2d_burkert', 'Burkert+12: relax only',      '#f39c12'),  # orange
]

def read_inner_vr(run_name):
    """Lies für jeden Zeitschritt die theta-gemittelte v_r am innersten Gitterpunkt."""
    pattern = os.path.join(RUNS_DIR, run_name, f'{run_name}.block0.out1.?????.tab')
    files   = sorted(glob.glob(pattern))
    times, vr_inner = [], []

    for fname in files:
        # Zeitstempel aus Header
        with open(fname) as f:
            header = f.readline()
        t_s = float(header.split('time=')[1].split()[0])

        # Daten laden (Spalten: i, x1v, j, x2v, rho, press, vel1, vel2, vel3)
        data = np.loadtxt(fname, comments='#')
        r_all  = data[:, 1]   # x1v
        vr_all = data[:, 6]   # vel1

        r_min  = np.unique(r_all)[0]          # innerstes r
        mask   = np.abs(r_all - r_min) < 1e10 # alle theta bei r_min
        vr_mean = vr_all[mask].mean() / KM_S  # → km/s

        times.append(t_s / YR_S)
        vr_inner.append(vr_mean)

    return np.array(times), np.array(vr_inner)

# ─── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for run_name, label, color in RUNS:
    t, vr = read_inner_vr(run_name)
    ax.plot(t, vr, '-o', color=color, label=label,
            linewidth=2, markersize=5, zorder=3)

ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)

# Beschriftung
ax.set_xlabel('Zeit [yr]')
ax.set_ylabel(r'$\langle v_r \rangle_\theta$ am inneren Rand [km s$^{-1}$]')
ax.set_title('Zeitentwicklung der radialen Einströmgeschwindigkeit\n'
             r'(am inneren Rand, $\theta$-gemittelt)')
ax.legend(loc='lower left', framealpha=0.9)
ax.set_xlim(-0.2, 10.5)

# Anmerkungen
ax.annotate('konvektiver Kollaps\n(NSC, keine Relaxation)',
            xy=(7, -3200), fontsize=9, color='#3498db',
            ha='center', style='italic')
ax.annotate('stabil (Burkert+12)',
            xy=(7.5, -200), fontsize=9, color='#f39c12',
            ha='center', style='italic')

ax.grid(True, alpha=0.25, linestyle=':')
fig.tight_layout()

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'figures', 'g2_time_evolution_vr.png')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Gespeichert: {outpath}')

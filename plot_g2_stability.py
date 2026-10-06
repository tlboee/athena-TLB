"""
plot_g2_stability.py
Liest die 4 Athena++ Ausgabedateien des 1D-Stabilitätstests
und plottet Dichte und Radialgeschwindigkeit vs. Radius.

Zeigt ob (und wie) die Burkert+12-Atmosphäre mit reiner BH-Schwerkraft
stabil bleibt oder kollabiert.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'vis/python')
import athena_read

# -----------------------------------------------------------------------
# Physikalische Konstanten (CGS)
# -----------------------------------------------------------------------
k_B  = 1.380649e-16   # Boltzmann-Konstante [erg/K]
m_H  = 1.6726e-24     # Protonenmasse [g]
mu   = 0.5            # mittleres Molekulargewicht

# Burkert+12 Referenzparameter
rho0 = 9.5e-22        # g/cm^3
T0   = 2.0e8          # K
r0   = 1.0e16         # cm

# Einheitenumrechnung
AU    = 1.496e13      # cm
yr    = 3.156e7       # s

# -----------------------------------------------------------------------
# Ausgabedateien einlesen
# -----------------------------------------------------------------------
files = [
    'g2_1d.block0.out1.00000.tab',   # t = 0
    'g2_1d.block0.out1.00001.tab',   # t ≈ 10 yr
    'g2_1d.block0.out1.00002.tab',   # t ≈ 20 yr
    'g2_1d.block0.out1.00003.tab',   # t ≈ 30 yr
]

datasets = []
times    = []
for f in files:
    d = athena_read.tab(f)
    datasets.append(d)
    times.append(d['time'] / yr)
    print(f"  {f}  →  t = {d['time']/yr:.1f} yr")

# Spaltennamen prüfen
print("Spalten:", list(datasets[0].keys()))

# -----------------------------------------------------------------------
# Analytisches Burkert-Profil
# -----------------------------------------------------------------------
r_ana  = np.linspace(datasets[0]['x1v'].min(),
                     datasets[0]['x1v'].max(), 500)
rho_ana = rho0 * (r0 / r_ana)

# -----------------------------------------------------------------------
# Farben für die 4 Zeitschritte
# -----------------------------------------------------------------------
colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
labels = [f't = {t:.1f} yr' for t in times]

# -----------------------------------------------------------------------
# Figure: 2 Panels (Dichte | Radialgeschwindigkeit)
# -----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('G2-Projekt: 1D Stabilitätstest (Burkert+12-Profil, nur BH-Schwerkraft)',
             fontsize=13)

# --- Panel 1: Dichte ---
for d, col, lab in zip(datasets, colors, labels):
    r   = d['x1v']
    rho = d['rho']
    ax1.loglog(r / AU, rho, color=col, lw=1.8, label=lab)

ax1.loglog(r_ana / AU, rho_ana, 'k--', lw=1.2, label='Burkert+12 analytisch')
ax1.set_xlabel('r  [AU]', fontsize=12)
ax1.set_ylabel(r'$\rho$  [g/cm³]', fontsize=12)
ax1.set_title('Dichte', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Steigung annotieren (Referenz: -1)
r_fit = r_ana[(r_ana > 1e15) & (r_ana < 1e17)]
rho_fit = rho0 * (r0 / r_fit)
ax1.annotate(r'Steigung = $-1$', xy=(r_fit[80]/AU, rho_fit[80]),
             xytext=(0.55, 0.75), textcoords='axes fraction',
             fontsize=9, color='gray',
             arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# --- Panel 2: Radialgeschwindigkeit ---
km_s = 1e5  # cm/s -> km/s

for d, col, lab in zip(datasets, colors, labels):
    r   = d['x1v']
    # Spaltenname kann 'vel1' oder 'rho_v1' sein – wir nehmen was verfügbar
    if 'vel1' in d:
        v_r = d['vel1']
    elif 'rho_v1' in d:
        v_r = d['rho_v1'] / d['rho']
    else:
        # Erste Geschwindigkeitsspalte suchen
        vkey = [k for k in d.keys() if 'vel' in k or 'v1' in k]
        v_r = d[vkey[0]] if vkey else np.zeros_like(r)

    ax2.plot(r / AU, v_r / km_s, color=col, lw=1.8, label=lab)

ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.set_xlabel('r  [AU]', fontsize=12)
ax2.set_ylabel(r'$v_r$  [km/s]', fontsize=12)
ax2.set_title('Radialgeschwindigkeit', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig('figures/g2_stability_1d.png', dpi=150, bbox_inches='tight')
print("\nPlot gespeichert: figures/g2_stability_1d.png")
plt.show()

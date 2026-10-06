import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# Konstanten (CGS)
# ============================================================
k_B  = 1.381e-16   # Boltzmann [erg/K]
m_H  = 1.673e-24   # Protonenmasse [g]
G    = 6.674e-8    # Gravitationskonstante [cm³/g/s²]
M_sun = 1.989e33   # Sonnenmasse [g]
mu   = 0.5         # mittleres Molekulargewicht

# Schwarzes Loch Sgr A*
M_BH = 4.31e6 * M_sun

# Burkert+12 Parameter
r0   = 1.0e16
rho0 = 9.5e-22
T0   = 2.0e8

# ============================================================
# Profile und ihre Ableitungen
# ============================================================
r = np.logspace(-1, 1, 1000) * r0

rho = rho0 * (r0 / r)
T   = T0   * (r0 / r)
P   = rho * k_B * T / (mu * m_H)

# Linke Seite: dP/dr (numerisch)
dPdr = np.gradient(P, r)

# Rechte Seite: -G M_BH rho / r² (Gravitationskraft)
gravity = -G * M_BH * rho / r**2

# Relatives Residuum: wie gut ist das Gleichgewicht?
residuum = (dPdr - gravity) / np.abs(gravity)

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
fig.suptitle('Hydrostatischer Gleichgewichtscheck\n'
             r'$dP/dr = -G M_\bullet \rho / r^2$', fontsize=12)

axes[0].loglog(r/r0, np.abs(dPdr),   label=r'$|dP/dr|$ (Druckgradient)')
axes[0].loglog(r/r0, np.abs(gravity),label=r'$G M_\bullet \rho / r^2$ (Gravitation)',
               linestyle='--')
axes[0].set_ylabel('Kraft pro Volumen [dyn/cm³]')
axes[0].legend()

axes[1].semilogx(r/r0, residuum * 100)
axes[1].axhline(0, color='gray', linestyle='--')
axes[1].set_ylabel('Relatives Residuum [%]')
axes[1].set_xlabel(r'Radius $r/r_0$')
axes[1].set_ylim(-5, 5)

for ax in axes:
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(ROOT / 'figures' / 'hydrostatic_check.png', dpi=150)
plt.show()

# Zahlenwert ausgeben
print(f"Max. Abweichung vom Gleichgewicht: {np.max(np.abs(residuum))*100:.2f} %")

# Analytischer Check: welche BH-Masse wäre nötig für exaktes Gleichgewicht?
M_required = 2 * k_B * T0 * r0 / (mu * m_H * G)
print(f"Verwendete BH-Masse:       {M_BH/M_sun:.2e} Sonnenmassen")
print(f"Nötig für Gleichgewicht:   {M_required/M_sun:.2e} Sonnenmassen")
print(f"Verhältnis:                {M_required/M_BH:.2f}")
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Physikalische Konstanten (CGS-Einheitensystem)
# ============================================================
k_B  = 1.381e-16   # Boltzmann-Konstante [erg/K]
m_H  = 1.673e-24   # Protonenmasse [g]
mu   = 0.5         # mittleres Molekulargewicht (vollständig ionisiertes H)

# ============================================================
# Burkert+12 Parameter für das heiße Umgebungsmedium
# ============================================================
r0   = 1.0e16      # Referenzradius [cm]
rho0 = 9.5e-22     # Referenzdichte bei r0 [g/cm³]
T0   = 2.0e8       # Referenztemperatur bei r0 [K]

# ============================================================
# Radialer Bereich: 0.1 r0 bis 10 r0
# ============================================================
r = np.logspace(-1, 1, 500) * r0   # logarithmisch verteilt

# ============================================================
# Analytische Profile (alle skalieren mit r)
# ============================================================
rho = rho0 * (r0 / r)              # Dichte: 1/r
T   = T0   * (r0 / r)              # Temperatur: 1/r
P   = rho * k_B * T / (mu * m_H)  # Druck über ideales Gas: 1/r²

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
fig.suptitle('Burkert+12: Analytisches Profil des heißen Umgebungsmediums\num Sgr A*', fontsize=12)

r_in_r0 = r / r0   # x-Achse in Einheiten von r0

axes[0].loglog(r_in_r0, rho, color='steelblue')
axes[0].axvline(1, color='gray', linestyle='--', alpha=0.5, label='r = r₀')
axes[0].set_ylabel(r'Dichte $\rho$ [g cm$^{-3}$]')
axes[0].legend()

axes[1].loglog(r_in_r0, T, color='tomato')
axes[1].axvline(1, color='gray', linestyle='--', alpha=0.5)
axes[1].set_ylabel(r'Temperatur $T$ [K]')

axes[2].loglog(r_in_r0, P, color='seagreen')
axes[2].axvline(1, color='gray', linestyle='--', alpha=0.5)
axes[2].set_ylabel(r'Druck $P$ [dyn cm$^{-2}$]')
axes[2].set_xlabel(r'Radius $r / r_0$ mit $r_0 = 10^{16}$ cm')

for ax in axes:
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('burkert_profile.png', dpi=150)
plt.show()
print("Steigungen (erwartet: -1, -1, -2):")
print(f"  rho: {np.polyfit(np.log10(r_in_r0), np.log10(rho), 1)[0]:.2f}")
print(f"  T:   {np.polyfit(np.log10(r_in_r0), np.log10(T),   1)[0]:.2f}")
print(f"  P:   {np.polyfit(np.log10(r_in_r0), np.log10(P),   1)[0]:.2f}")
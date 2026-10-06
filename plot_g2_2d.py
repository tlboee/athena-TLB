#!/usr/bin/env python3
"""
Plot 2D adiabatic ambient medium simulation results (Burkert+12 profile).
Produces:
  figures/g2_2d_density_t10.png  -- 2D density map at t=10 yr (r-theta plane)
  figures/g2_2d_radprof.png      -- theta-averaged radial profiles vs Burkert
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, glob

os.makedirs('figures', exist_ok=True)

# ---- physical constants (CGS) ----
k_B  = 1.380649e-16
m_H  = 1.6726e-24
rho0 = 9.5e-22
T0   = 2.0e8
r0   = 1.0e16
mu   = 0.5
yr   = 3.156e7   # s

# ---- helper: read one 2D tab file ----
def read_tab2d(fname):
    data = np.loadtxt(fname, comments='#')
    # cols: i x1v j x2v rho press vel1 vel2 vel3
    i_idx = data[:,0].astype(int)
    r     = data[:,1]
    j_idx = data[:,2].astype(int)
    th    = data[:,3]
    rho   = data[:,4]
    press = data[:,5]
    vr    = data[:,6]
    return i_idx, j_idx, r, th, rho, press, vr

# ---- determine grid dimensions ----
f0 = 'g2_2d.block0.out1.00000.tab'
ii, jj, rr, th, rho0_d, p0, v0 = read_tab2d(f0)
nr = len(np.unique(ii))
nt = len(np.unique(jj))
r_vals  = np.unique(rr)
th_vals = np.unique(th)

def reshape2d(arr):
    # data is ordered (j changes fastest for fixed i) -- check
    return arr.reshape(nr, nt)

# ---- Burkert analytical profile ----
r_ana  = np.logspace(np.log10(r_vals.min()), np.log10(r_vals.max()), 300)
rho_ana = rho0 * (r0 / r_ana)
P_ana   = rho_ana * k_B * T0 * (r0/r_ana) / (mu * m_H)

# ---- Figure 1: 2D density map at t=10 yr ----
fname_last = sorted(glob.glob('g2_2d.block0.out1.*.tab'))[-1]
# get time from header
with open(fname_last) as f:
    hdr = f.readline()
t_yr = float(hdr.split('time=')[1].split()[0]) / yr

ii, jj, rr, th, rho2d, p2d, vr2d = read_tab2d(fname_last)
RHO = reshape2d(rho2d)   # shape (nr, nt)
VR  = reshape2d(vr2d)

# meshgrid for pcolormesh (theta on x-axis, r on y-axis in log scale)
TH, R = np.meshgrid(th_vals, r_vals)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# density
im = axes[0].pcolormesh(np.degrees(TH), R/r0, np.log10(RHO),
                         cmap='inferno', shading='nearest')
axes[0].set_yscale('log')
axes[0].set_xlabel('θ [deg]')
axes[0].set_ylabel('r / r₀')
axes[0].set_title(f'log₁₀ ρ [g/cm³]  at  t = {t_yr:.1f} yr')
plt.colorbar(im, ax=axes[0])

# radial velocity in km/s
vkm = VR / 1e5
vmax = max(abs(vkm).max(), 100)
im2 = axes[1].pcolormesh(np.degrees(TH), R/r0, vkm,
                          cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          shading='nearest')
axes[1].set_yscale('log')
axes[1].set_xlabel('θ [deg]')
axes[1].set_ylabel('r / r₀')
axes[1].set_title(f'v_r [km/s]  at  t = {t_yr:.1f} yr')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('figures/g2_2d_density_t10.png', dpi=150)
plt.close()
print("Saved figures/g2_2d_density_t10.png")

# ---- Figure 2: theta-averaged radial profiles ----
files = sorted(glob.glob('g2_2d.block0.out1.*.tab'))
times_yr = []
for f in files:
    with open(f) as fh: hdr = fh.readline()
    times_yr.append(float(hdr.split('time=')[1].split()[0]) / yr)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
for fname, t, col in zip(files, times_yr, colors):
    ii, jj, rr, th, rho_d, p_d, vr_d = read_tab2d(fname)
    RHO_f = reshape2d(rho_d)
    VR_f  = reshape2d(vr_d)
    rho_avg = RHO_f.mean(axis=1)   # theta-average
    vr_avg  = VR_f.mean(axis=1)
    axes[0].loglog(r_vals/r0, rho_avg, color=col, alpha=0.8, label=f't={t:.1f} yr')
    axes[1].semilogx(r_vals/r0, vr_avg/1e5, color=col, alpha=0.8)

axes[0].loglog(r_ana/r0, rho_ana, 'k--', lw=2, label='Burkert analytical')
axes[0].set_xlabel('r / r₀')
axes[0].set_ylabel('ρ [g/cm³]')
axes[0].set_title('θ-averaged density  (2D adiabatic)')
axes[0].legend(fontsize=7, ncol=2)

axes[1].axhline(0, color='k', lw=0.8, ls='--')
axes[1].set_xlabel('r / r₀')
axes[1].set_ylabel('⟨v_r⟩ [km/s]')
axes[1].set_title('θ-averaged radial velocity  (2D adiabatic)')

plt.tight_layout()
plt.savefig('figures/g2_2d_radprof.png', dpi=150)
plt.close()
print("Saved figures/g2_2d_radprof.png")

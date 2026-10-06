#!/usr/bin/env python3
"""
2D adiabatic ambient medium (Burkert+12) – diagnostic plots.
  figures/g2_2d_density_t10.png  -- 2D maps at t=10 yr
  figures/g2_2d_radprof.png      -- theta-averaged radial profiles
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / 'runs' / 'g2_2d'
FIG = ROOT / 'figures'

os.makedirs(FIG, exist_ok=True)

# ---- constants (CGS) ----
k_B  = 1.380649e-16;  m_H = 1.6726e-24
rho0 = 9.5e-22;  T0 = 2.0e8;  r0 = 1.0e16;  mu = 0.5
yr   = 3.156e7

def read_tab2d(fname):
    d = np.loadtxt(fname, comments='#')
    # cols: i x1v(r)  j x2v(th)  rho press vel1 vel2 vel3
    r, th = d[:,1], d[:,3]
    rho, press, vr = d[:,4], d[:,5], d[:,6]
    nr = len(np.unique(d[:,0].astype(int)))
    nt = len(np.unique(d[:,2].astype(int)))
    # file ordered: outer=j(theta), inner=i(r)  → shape (nt, nr)
    R   = r.reshape(nt, nr)[0,:]          # (nr,)  r values
    TH  = th.reshape(nt, nr)[:,0]         # (nt,)  theta values
    RHO = rho.reshape(nt, nr)             # (nt, nr)
    VR  = vr.reshape(nt, nr)              # (nt, nr)
    return R, TH, RHO, VR

def get_time(fname):
    with open(fname) as f: h = f.readline()
    return float(h.split('time=')[1].split()[0]) / yr

files = sorted(glob.glob(str(RUN / 'g2_2d.block0.out1.*.tab')))
times = [get_time(f) for f in files]

# ---- Burkert analytical ----
R0, TH0, _, _ = read_tab2d(files[0])
r_ana = np.logspace(np.log10(R0.min()), np.log10(R0.max()), 300)
rho_ana = rho0 * (r0 / r_ana)

# ==========================================================
# Fig 1: 2D maps at t=10 yr
# ==========================================================
R, TH, RHO, VR = read_tab2d(files[-1])
t_last = times[-1]

# meshgrid for pcolormesh: x=theta, y=r/r0
TH2, R2 = np.meshgrid(np.degrees(TH), R/r0)   # both (nr, nt)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

im0 = axes[0].pcolormesh(TH2, R2, np.log10(RHO.T),
                          cmap='inferno', shading='nearest')
axes[0].set_yscale('log')
axes[0].set_xlabel('θ [deg]', fontsize=12)
axes[0].set_ylabel('r / r₀', fontsize=12)
axes[0].set_title(f'log₁₀ ρ [g/cm³]   t = {t_last:.1f} yr', fontsize=12)
plt.colorbar(im0, ax=axes[0])

vkm = VR / 1e5
vlim = np.percentile(np.abs(vkm), 99)
im1 = axes[1].pcolormesh(TH2, R2, vkm.T,
                          cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                          shading='nearest')
axes[1].set_yscale('log')
axes[1].set_xlabel('θ [deg]', fontsize=12)
axes[1].set_ylabel('r / r₀', fontsize=12)
axes[1].set_title(f'v_r [km/s]   t = {t_last:.1f} yr', fontsize=12)
plt.colorbar(im1, ax=axes[1])

plt.suptitle('2D adiabatic ambient medium (Burkert+12 profile)', fontsize=13)
plt.tight_layout()
plt.savefig(FIG / 'g2_2d_density_t10.png', dpi=150)
plt.close()
print("Saved figures/g2_2d_density_t10.png")

# ==========================================================
# Fig 2: theta-averaged radial profiles over time
# ==========================================================
colors = plt.cm.viridis(np.linspace(0, 1, len(files)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for fname, t, col in zip(files, times, colors):
    R, TH, RHO, VR = read_tab2d(fname)
    rho_avg = RHO.mean(axis=0)   # average over theta → shape (nr,)
    vr_avg  = VR.mean(axis=0)
    lbl = f't = {t:.1f} yr'
    axes[0].loglog(R/r0, rho_avg, color=col, lw=1.5, label=lbl)
    axes[1].semilogx(R/r0, vr_avg/1e5, color=col, lw=1.5, label=lbl)

axes[0].loglog(r_ana/r0, rho_ana, 'k--', lw=2, label='Burkert analytical')
axes[0].set_xlabel('r / r₀', fontsize=12)
axes[0].set_ylabel('⟨ρ⟩_θ  [g/cm³]', fontsize=12)
axes[0].set_title('θ-averaged density', fontsize=12)
axes[0].legend(fontsize=8, ncol=2)

axes[1].axhline(0, color='k', lw=0.8, ls='--')
axes[1].set_xlabel('r / r₀', fontsize=12)
axes[1].set_ylabel('⟨v_r⟩_θ  [km/s]', fontsize=12)
axes[1].set_title('θ-averaged radial velocity', fontsize=12)
axes[1].legend(fontsize=8, ncol=2)

plt.suptitle('2D adiabatic ambient medium – time evolution', fontsize=13)
plt.tight_layout()
plt.savefig(FIG / 'g2_2d_radprof.png', dpi=150)
plt.close()
print("Saved figures/g2_2d_radprof.png")

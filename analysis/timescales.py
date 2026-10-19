"""Analytische Zeitskalen entlang der Keplerbahn von CC01 (Burkert+12, Tabelle 1).

t_cc  = chi^(1/2) R / v            (Cloud-Crushing-Zeit)
t_KH  = chi^(1/2) / (k v), k = 2 pi / lambda, lambda = R   (KH-Anwachszeit)
t_tid = sqrt(r^3 / (G M))          (Gezeiten- bzw. dynamische Zeit)
Atmosphaere ruht; chi = T_hot(r) / T_c (Druckgleichgewicht), T_hot = T0 r0 / r.
N_x(t) = int_0^t dt' / t_x(t')     (Zahl der durchlaufenen Zeitskalen)

Aufruf:  python3 analysis/timescales.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

G, MSUN, KB, MH = 6.674e-8, 1.989e33, 1.380649e-16, 1.6726e-24
YR = 3.156e7
GM = G * 4.31e6 * MSUN
MU, R0 = 0.6139, 1.0e16
T0 = MU * MH * GM / (2 * KB * R0)          # hydrostatisch, ~2.13e8 K
T_C = 1.0e4                                # Wolkentemperatur
R_C = 1.87e15                              # Wolkenradius (konstant angenommen)
T_START, T_END = 1995.5, 2060.0

# Keplerbahn (Leapfrog), Startwerte aus Burkert+12 Tabelle 1
pos = np.array([-7.22e16, 2.21e16])
vel = np.array([794.59e5, 48.45e5])
n = 400000
dt = (T_END - T_START) * YR / n
t, r, v = np.empty(n + 1), np.empty(n + 1), np.empty(n + 1)
for k in range(n + 1):
    t[k], r[k], v[k] = k * dt, np.linalg.norm(pos), np.linalg.norm(vel)
    if k == n:
        break
    vel += 0.5 * dt * (-GM * pos / np.linalg.norm(pos)**3)
    pos += dt * vel
    vel += 0.5 * dt * (-GM * pos / np.linalg.norm(pos)**3)

year = T_START + t / YR
chi  = (T0 * R0 / r) / T_C
t_cc = np.sqrt(chi) * R_C / v
t_kh = np.sqrt(chi) / ((2 * np.pi / R_C) * v)
t_td = np.sqrt(r**3 / GM)
N = {name: np.concatenate([[0], np.cumsum(0.5 * (1/x[1:] + 1/x[:-1]) * np.diff(t))])
     for name, x in [("cc", t_cc), ("KH", t_kh), ("tid", t_td)]}

i_peri = np.argmin(r)
print(f"T0 = {T0:.3e} K; Perizentrum {year[i_peri]:.2f} bei r = {r[i_peri]:.2e} cm, v = {v[i_peri]/1e5:.0f} km/s")
print(f"{'Jahr':>7s} {'r [cm]':>9s} {'v [km/s]':>9s} {'chi':>7s} {'t_cc [yr]':>10s} {'t_KH [yr]':>10s}"
      f" {'t_tid [yr]':>11s} {'N_cc':>6s} {'N_KH':>6s} {'N_tid':>6s}")
for y in [1995.5, 2005.5, 2010.5, 2012.5, year[i_peri], 2015.5, 2025.0, 2040.0, 2059.0]:
    i = np.argmin(np.abs(year - y))
    print(f"{year[i]:7.1f} {r[i]:9.2e} {v[i]/1e5:9.0f} {chi[i]:7.0f} {t_cc[i]/YR:10.2f} {t_kh[i]/YR:10.2f}"
          f" {t_td[i]/YR:11.2f} {N['cc'][i]:6.2f} {N['KH'][i]:6.2f} {N['tid'][i]:6.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
ax = axes[0]
ax.semilogy(year, t_cc / YR, label=r"$t_{\rm cc} = \chi^{1/2} R/v$")
ax.semilogy(year, t_kh / YR, label=r"$t_{\rm KH}$ ($\lambda = R$)")
ax.semilogy(year, t_td / YR, label=r"$t_{\rm tid} = \sqrt{r^3/GM}$")
ax.semilogy(year, (year[-1] - year) + 1e-3, "k:", lw=0.8, label="verbleibende Zeit bis 2060")
ax.set_ylabel("Zeitskala [yr]")
ax.set_ylim(1e-2, 1e3)
ax.legend(fontsize=8)
ax = axes[1]
for name, label in [("cc", r"$N_{\rm cc}$"), ("KH", r"$N_{\rm KH}$"), ("tid", r"$N_{\rm tid}$")]:
    ax.plot(year, N[name], label=label)
ax.set_ylabel(r"$N = \int dt / t_x$")
ax.set_yscale("log")
ax.set_ylim(1e-2, 1e3)
ax.legend(fontsize=8)
for ax in axes:
    ax.axvline(year[i_peri], color="grey", ls="--", lw=0.8)
    ax.axvline(2013.0, color="C3", ls=":", lw=0.8)      # Kern (tr > 0.9999) weg / Akkretionsbeginn
    ax.axvline(2059.0, color="C4", ls=":", lw=0.8)      # tr > 0.9 unter 50 % der Anfangsmasse
    ax.set_xlabel("Jahr")
out = os.path.expanduser("~/athena/analysis/timescales.png")
fig.savefig(out, dpi=150)
print("Gespeichert:", out)

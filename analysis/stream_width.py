"""Breite des Wolkenstroms und lokale Cloud-Crushing-Zeit entlang der Bahn (Max' Spekulation 12.11.: r_cl << r_cl0 -> t_KH << t_KH0).

Zwei Masken: Tracer > 0.5 (geometrische Ausdehnung) und rho/rho_eq > sqrt(chi0) (noch dichtes Gas, Max' Mass fuer chi >= 100).
r_eff = 2 A / P (A Flaeche, P Umfang der Maskenkontur): runde Wolke -> R, langer Streifen -> Breite.
chi_loc = Median rho/rho_eq(r) im Wolkengas (Tracer > 0.5);  v = massengewichteter Mittelwert von |v| (nicht |<v>|).
t_cc,loc = chi_loc^(1/2) r_eff / v, je Maske.  Laeufe: CC01 1024 (0-50 yr) und fein um das Perizentrum (1024, 2048).
Cache: <lauf>/stream_width.csv.   Ausgabe: runs/stream_width.png.   Aufruf (Notebook):  %run ~/athena/analysis/stream_width.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io
try:
    from skimage.measure import find_contours
except ImportError:
    find_contours = None

BASE = os.path.expanduser("~/athena/runs")
YR, KM, R0, RHO0, RR0, R_IN = 3.156e7, 1e5, 1.87e15, 1.702e-21, 1.0e16, 1.9e15
CHI0, T_PERI = 2755.0, 18.0
RUNS = [("1024, whole orbit", "ps_ref_big_x2_t50", "#21918c", 1.0),
        ("1024, fine", "ps_ref_big_x2_peri", "#0b5d55", 2.0),
        ("2048, fine", "ps_ref_big_x4_peri", "#bddf26", 2.0)]

def perimeter(mask, dx):
    if find_contours is not None:
        return sum(np.sum(np.hypot(*np.diff(c, axis=0).T)) for c in find_contours(mask.astype(float), 0.5)) * dx
    edges = np.sum(mask[1:, :] != mask[:-1, :]) + np.sum(mask[:, 1:] != mask[:, :-1])
    return edges * dx * np.pi / 4                                   # Treppen-Korrektur

def r_eff(mask, dx):
    if mask.sum() < 10:
        return np.nan
    return 2 * mask.sum() * dx * dx / perimeter(mask, dx)

def series(run):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "stream_width.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        x, y = s["x"], s["y"]; dx = x[1] - x[0]
        X, Y = np.meshgrid(x, y)
        rho, tr = s["rho"], np.clip(s["r0"], 0, 1)
        c = rho / (RHO0 * RR0 / np.maximum(np.hypot(X, Y), R_IN))
        m_tr, m_dq = tr > 0.5, c > np.sqrt(CHI0)
        if m_tr.sum() < 10:
            continue
        r_tr, r_dq = r_eff(m_tr, dx), r_eff(m_dq, dx)
        chi = float(np.median(c[m_tr]))
        w = (rho * tr)[m_tr]
        v = float((w * np.hypot(s[[k for k in ("vel1", "vx", "v1") if k in s][0]], s[[k for k in ("vel2", "vy", "v2") if k in s][0]])[m_tr]).sum() / w.sum())
        rows.append([t / YR, r_tr / R0, r_dq / R0, chi, v / KM,
                     np.sqrt(chi) * r_tr / v / YR, np.sqrt(chi) * r_dq / v / YR])
        print(f"  {run}: t = {t/YR:6.2f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t", "r_tr_R0", "r_dq_R0", "chi_loc", "v_kms", "tcc_tr_yr", "tcc_dq_yr"])
    df.to_csv(cache, index=False)
    return df

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, constrained_layout=True)
data = {}
for lab, run, c, lw in RUNS:
    if not os.path.isdir(os.path.join(BASE, run)):
        print(f"{run} fehlt"); continue
    df = data[lab] = series(run)
    axes[0, 0].plot(df["t"], df["r_tr_R0"], color=c, lw=lw, label=f"{lab}: tracer > 0.5")
    axes[0, 0].plot(df["t"], df["r_dq_R0"], color=c, lw=lw, ls="--", label=f"{lab}: ρ/ρ_eq > √χ₀")
    axes[0, 1].plot(df["t"], df["chi_loc"], color=c, lw=lw, label=lab)
    axes[1, 0].plot(df["t"], df["v_kms"], color=c, lw=lw, label=lab)
    axes[1, 1].plot(df["t"], df["tcc_tr_yr"], color=c, lw=lw, label=f"{lab}: tracer > 0.5")
    axes[1, 1].plot(df["t"], df["tcc_dq_yr"], color=c, lw=lw, ls="--", label=f"{lab}: ρ/ρ_eq > √χ₀")
for ax, ttl in zip(axes.flat, ("effective width r_eff / R0 = 2A/(P R0)", "local density contrast χ_loc = median ρ/ρ_eq (tracer > 0.5)",
                               "mass-weighted |v| [km/s]", "local cloud-crushing time χ_loc^½ r_eff / v [yr]")):
    ax.set_yscale("log"); ax.axvline(T_PERI, color="k", ls=":", lw=0.8); ax.set_title(ttl, fontsize=10)
axes[1, 1].axhline(0.25, color="0.5", lw=1); axes[1, 1].text(1, 0.28, "pericentre passage r_p / v_p ≈ 0.25 yr", fontsize=9, color="0.4")
for ax in axes[1]:
    ax.set_xlabel("t [yr]")
axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=8)
fig.suptitle("CC01 adiabatic: width of the cloud stream and local cloud-crushing time along the orbit")
fig.savefig(os.path.join(BASE, "stream_width.png"), dpi=150)
plt.show()

print(f"\n{'Lauf':18s} {'t [yr]':>7s} {'r_tr/R0':>8s} {'r_dq/R0':>8s} {'chi_loc':>8s} {'v [km/s]':>9s} {'tcc_tr':>8s} {'tcc_dq':>8s}")
for lab, df in data.items():
    for tt in (0.0, 10.0, 16.0, 17.5, 18.0, 18.5, 19.0, 25.0):
        if df["t"].min() - 0.03 <= tt <= df["t"].max() + 0.03:
            r = df.iloc[int(np.argmin(np.abs(df["t"] - tt)))]
            print(f"{lab:18s} {r['t']:7.2f} {r['r_tr_R0']:8.3f} {r['r_dq_R0']:8.3f} {r['chi_loc']:8.1f} {r['v_kms']:9.0f} "
                  f"{r['tcc_tr_yr']:8.3g} {r['tcc_dq_yr']:8.3g}")

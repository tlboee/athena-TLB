"""Kelvin-Helmholtz-Zeitskala entlang der Bahn: idealisiert vs. gemessen, integriert vs. ein Wert.

t_KH = sqrt(chi) R / (2 pi v)  (lambda = R, = t_cc / 2 pi)
idealisiert: Keplerbahn, chi = chi0 r_start / r (Druckgleichgewicht bei fester T_c)
gemessen:    chi = medianer lokaler Kontrast der Wolke (chic), v = |v_Schwerpunkt| (Atmosphaere ruht), aus mX.csv
N_KH(t) = int dt / t_KH; Vergleich mit t / t_KH(Start) und t / t_KH(Perizentrum).
Zerstoerungszeiten t_50 (Tracer > 0.9, dichter Strom) aus study_table.csv.
Aufruf (Notebook):  %run ~/athena/analysis/kh_plot.py [lauf:bezeichnung ...]
"""
import json, math, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/athena/runs")
YR, KM = 3.156e7, 1.0e5
GM = 6.674e-8 * 4.31e6 * 1.989e33
X0, Y0 = -7.22e16, 2.21e16
R_START = math.hypot(X0, Y0)
R_C = 1.87e15
DEFAULT = ["g_chi10_rp4:χ = 10", "g_chi100_rp4:χ = 100", "g_chi1000_rp4:χ = 1000", "ps_ref_big:CC01 (χ ≈ 2760)"]
RUNS = [a.split(":", 1) if ":" in a else (a, a) for a in (sys.argv[1:] or DEFAULT)]
tab = pd.read_csv(os.path.join(BASE, "study_table.csv"))

def kepler_rv(vx, vy, t_yr):
    ux, uy = vx * KM, vy * KM
    E = 0.5 * (ux * ux + uy * uy) - GM / R_START
    h = X0 * uy - Y0 * ux
    a = -GM / (2 * E)
    e = math.sqrt(1 + 2 * E * h * h / GM**2)
    n = math.sqrt(GM / a**3)
    Ea = math.acos(max(-1.0, min(1.0, (1 - R_START / a) / e)))
    M0 = -(Ea - e * math.sin(Ea)) if X0 * ux + Y0 * uy < 0 else Ea - e * math.sin(Ea)
    M = M0 + n * np.asarray(t_yr) * YR
    Ek = M + 0.85 * e * np.sign(np.sin(M))
    for _ in range(50):
        Ek = Ek - (Ek - e * np.sin(Ek) - M) / (1 - e * np.cos(Ek))
    r = a * (1 - e * np.cos(Ek))
    return r, np.sqrt(GM * (2 / r - 1 / a))

def cumint(t, tx):
    return np.concatenate([[0.0], np.cumsum(0.5 * (1 / tx[1:] + 1 / tx[:-1]) * np.diff(t))])

fig, axes = plt.subplots(2, len(RUNS), figsize=(5.4 * len(RUNS), 8), sharex="col", constrained_layout=True,
                         squeeze=False)
print(f"{'Lauf':16s} {'tKH(0)':>7s} {'tKH(p)':>7s} {'Mass':>12s} {'t_d':>6s} {'N_int':>6s} {'N_mess':>7s} {'t/tKH0':>7s} {'t/tKHp':>7s}")
for col, (run, lab) in enumerate(RUNS):
    with open(os.path.join(BASE, run, "manifest.json")) as f:
        man = json.load(f)
    sub = tab[tab["run"] == run]
    chi0, tp = sub["chi"].iloc[0], man.get("t_peri_yr", 18.0)
    t_end = min(sub["t_end"].iloc[0], 100.0)
    R = man.get("R_cloud", R_C) or R_C
    # idealisiert
    t = np.linspace(0, t_end, int(t_end / 0.002) + 1)
    r, v = kepler_rv(man["vx_kms"], man["vy_kms"], t)
    tkh = np.sqrt(chi0 * R_START / r) * R / v / (2 * np.pi) / YR
    N_id = cumint(t, tkh)
    tkh0, tkhp = tkh[0], tkh[np.argmin(r)]
    # gemessen
    mx = pd.read_csv(os.path.join(BASE, run, "mX.csv")).dropna(subset=["xc"])
    tm = mx["t_yr"].to_numpy()
    vm = np.hypot(mx["vxc"], mx["vyc"]).to_numpy()
    tkh_m = np.sqrt(np.maximum(mx["chic"].to_numpy(), 1.0)) * R / np.maximum(vm, 1.0) / (2 * np.pi) / YR
    N_m = cumint(tm, tkh_m)
    # Zeilen zeichnen
    ax = axes[0, col]
    ax.plot(t - tp, tkh, "k-", lw=1.5, label="idealised (χ ∝ 1/r, Kepler)")
    ax.plot(tm - tp, tkh_m, "o", color="C1", ms=3, label="measured (cloud contrast and velocity)")
    ax.set_yscale("log"); ax.set_ylabel("t_KH [yr]")
    ax.set_title(f"{lab}: idealised t_KH {tkh0:.1f} yr (start) → {tkhp:.1f} yr (pericentre),\n"
                 f"only ×{tkh0/tkhp:.1f} before pericentre; rises towards apocentre", fontsize=9)
    ax = axes[1, col]
    ax.plot(t - tp, N_id, "k-", lw=1.5, label="∫ dt / t_KH, idealised")
    ax.plot(tm - tp, N_m, "-", color="C1", label="∫ dt / t_KH, measured")
    ax.plot(t - tp, t / tkh0, "--", color="C0", label="t / t_KH(start)")
    ax.plot(t - tp, t / tkhp, ":", color="C2", lw=2, label="t / t_KH(pericentre)")
    for key, ls, nm in [("0.9", "-", "mixing"), ("ds", "--", "dense stream")]:
        s = sub[(sub["X"] == key) & (sub["Y"] == "1/2") & (sub["norm"] == "init")]["t"]
        if len(s) and np.isfinite(s.iloc[0]):
            td = s.iloc[0]
            ax.axvline(td - tp, color="grey", ls=ls, lw=1)
            ax.text(td - tp, ax.get_ylim()[1] if False else 0.02, f" t_50 {nm}", rotation=90, fontsize=7,
                    color="grey", va="bottom", transform=ax.get_xaxis_transform())
            print(f"{run:16s} {tkh0:7.2f} {tkhp:7.2f} {nm:>12s} {td:6.1f} {np.interp(td, t, N_id):6.1f} "
                  f"{np.interp(td, tm, N_m):7.1f} {td / tkh0:7.1f} {td / tkhp:7.1f}")
    ax.set_yscale("log"); ax.set_ylim(0.1, 300); ax.set_xlabel("t − t_peri [yr]"); ax.set_ylabel("N_KH")
    for a_ in axes[:, col]:
        a_.axvline(0, color="grey", lw=0.6)
axes[0, 0].legend(fontsize=7); axes[1, 0].legend(fontsize=7, loc="upper left")
out = os.path.join(BASE, "kh_timescales.png")
fig.savefig(out, dpi=150)
print("Gespeichert:", out)

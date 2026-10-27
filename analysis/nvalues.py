"""N-Werte zum Zeitpunkt der Disruption: welche Zeitskala ordnet das Gitter?

N_x(t) = Integral_0^t dt'/t_x(t') entlang der Keplerbahn jedes Laufs (Startwerte aus manifest.json):
  t_cc  = chi(r)^(1/2) R / v,  chi(r) = chi0 * r_start / r  (Wolke bei konstanter T im Druckgleichgewicht)
  t_KH  = t_cc / (2 pi)        (lambda = R; proportional zu t_cc, gleiche Streuung)
  t_tid = sqrt(r^3 / GM)
Disruptionszeiten aus runs/study_table.csv: t_50 (Y = 1/2), bezogen auf m_tot(0).

Aufruf (Notebook, nach study_table.py):  %run ~/athena/analysis/nvalues.py
"""
import json, math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/athena/runs")
YR, KM = 3.156e7, 1.0e5
GM = 6.674e-8 * 4.31e6 * 1.989e33
X0, Y0 = -7.22e16, 2.21e16
R_START = math.hypot(X0, Y0)
R_C = 1.87e15
MEAS = [("0.9", "Tracer > 0.9"), ("1/3", "Tracer > 1/3"), ("da_1/3", "rho > rho0/3"), ("ek_1/3", "K < 3^g K0")]


def kepler_rv(vx, vy, t_yr):
    """Abstand r und Geschwindigkeit v auf der Keplerbahn zu den Zeiten t_yr (yr seit Start)."""
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


def n_curves(man, chi0, t_end):
    t = np.linspace(0.0, t_end, int(t_end / 0.005) + 1)
    r, v = kepler_rv(man["vx_kms"], man["vy_kms"], t)
    tcc = np.sqrt(chi0 * R_START / r) * R_C / v / YR
    ttid = np.sqrt(r**3 / GM) / YR
    cum = lambda tx: np.concatenate([[0.0], np.cumsum(0.5 * (1 / tx[1:] + 1 / tx[:-1]) * np.diff(t))])
    return t, cum(tcc), cum(tcc / (2 * np.pi)), cum(ttid), r


tab = pd.read_csv(os.path.join(BASE, "study_table.csv"))
rows = []
for run, sub in tab.groupby("run"):
    with open(os.path.join(BASE, run, "manifest.json")) as f:
        man = json.load(f)
    chi0, t_end = sub["chi"].iloc[0], sub["t_end"].iloc[0]
    t, Ncc, Nkh, Ntid, r = n_curves(man, chi0, t_end)
    for key, label in MEAS:
        s = sub[(sub["X"] == key) & (sub["Y"] == "1/2") & (sub["norm"] == "init")]["t"]
        td = s.iloc[0] if len(s) else np.nan
        row = {"run": run, "chi": chi0, "rp": sub["rp"].iloc[0], "t_peri": sub["t_peri"].iloc[0],
               "x2": run.endswith("_x2_t30"), "grid": run.startswith("g_"), "measure": label, "t_d": td}
        if np.isfinite(td):
            row.update(N_cc=np.interp(td, t, Ncc), N_KH=np.interp(td, t, Nkh),
                       N_tid=np.interp(td, t, Ntid), r_d=np.interp(td, t, r))
        rows.append(row)
nv = pd.DataFrame(rows)
nv.to_csv(os.path.join(BASE, "nvalues.csv"), index=False)

pd.set_option("display.width", 200)
for label in [m[1] for m in MEAS]:
    sub = nv[(nv["measure"] == label)].sort_values(["x2", "chi", "rp"])
    print(f"\n=== {label}: t_50 [yr seit 1995.5] und N-Werte zu diesem Zeitpunkt (NaN = nicht erreicht)")
    print(sub[["run", "t_d", "N_cc", "N_KH", "N_tid", "r_d"]].round(3).to_string(index=False))

print("\nStreuung ueber die Gitterlaeufe (std von log10, in dex; kleiner = ordnet besser).")
print("Nur Laeufe, in denen das Mass erreicht wurde; t_d zum Vergleich.")
for x2, tag in [(False, "512"), (True, "doppelte Aufloesung (bis 30 yr)")]:
    print(f"\n  {tag}:")
    for label in [m[1] for m in MEAS]:
        sub = nv[(nv["measure"] == label) & nv["grid"] & (nv["x2"] == x2)].dropna(subset=["t_d"])
        if len(sub) < 3:
            print(f"    {label:14s}: zu wenige Laeufe ({len(sub)})")
            continue
        sd = {q: np.log10(sub[q]).std() for q in ("t_d", "N_cc", "N_tid")}
        gm = {q: 10**np.log10(sub[q]).mean() for q in ("N_cc", "N_tid")}
        print(f"    {label:14s} ({len(sub):2d} Laeufe): t_d {sd['t_d']:.2f}   N_cc {sd['N_cc']:.2f} "
              f"(typ. {gm['N_cc']:.2f})   N_tid {sd['N_tid']:.2f} (typ. {gm['N_tid']:.2f})")

fig, axes = plt.subplots(1, len(MEAS), figsize=(4.2 * len(MEAS), 4), constrained_layout=True)
colors = {10: "C0", 100: "C1", 1000: "C2", 2755: "k", 10000: "C3"}
for ax, (_, label) in zip(axes, MEAS):
    for x2, mk in [(False, "o"), (True, "s")]:
        sub = nv[(nv["measure"] == label) & (nv["x2"] == x2)].dropna(subset=["t_d"])
        for chi, ss in sub.groupby("chi"):
            c = colors.get(int(round(chi)), "grey")
            ax.scatter(ss["N_cc"], ss["N_tid"], marker=mk, edgecolors=c, facecolors="none" if x2 else c,
                       s=35, label=f"chi={chi:g}" + (" (x2)" if x2 else ""))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N_cc bei t_50"); ax.set_ylabel("N_tid bei t_50"); ax.set_title(label, fontsize=10)
axes[0].legend(fontsize=6)
out = os.path.join(BASE, "nvalues.png")
fig.savefig(out, dpi=140)
print("\nGespeichert:", out, os.path.join(BASE, "nvalues.csv"))

"""Momentane Keplerbahn des Wolkenschwerpunkts (Tracer > 0.5) vor und nach dem Perizentrum.

Aus Schwerpunkt (xc, yc) und Geschwindigkeit (vxc, vyc) in runs/<lauf>/mX.csv:
  E = v^2/2 - GM/r,  a = -GM/(2E),  h = x vy - y vx,  e-Vektor = ((v^2 - GM/r) r - (r.v) v) / GM
Auf der ungestoerten Keplerbahn waeren a, e, omega konstant; Aenderungen kommen von der Hydrodynamik.
Vorbehalt: nach der Zerstoerung beschreibt der Schwerpunkt nur noch die mittlere Bewegung des Gases.
Aufruf (Notebook):  %run ~/athena/analysis/orbit_elements.py [lauf ...]
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/athena/runs")
GM = 6.674e-8 * 4.31e6 * 1.989e33
RUNS = sys.argv[1:] or ["g_chi10_rp4", "g_chi100_rp4", "g_chi1000_rp4", "g_chi1000_rp32",
                        "ps_ref_big", "g_chi1e4_rp4_big"]
tab = pd.read_csv(os.path.join(BASE, "study_table.csv"))

def elements(df):
    x, y, vx, vy = (df[c].to_numpy() for c in ("xc", "yc", "vxc", "vyc"))
    r, v2 = np.hypot(x, y), vx**2 + vy**2
    E = 0.5 * v2 - GM / r
    h = x * vy - y * vx
    a = -GM / (2 * E)
    rv = x * vx + y * vy
    ex = ((v2 - GM / r) * x - rv * vx) / GM
    ey = ((v2 - GM / r) * y - rv * vy) / GM
    e = np.hypot(ex, ey)
    om = np.degrees(np.unwrap(np.arctan2(ey, ex)))
    return dict(t=df["t_yr"].to_numpy(), x=x, y=y, r=r, E=E, h=h, a=a, e=e, om=om,
                rp=a * (1 - e), ra=np.where(E < 0, a * (1 + e), np.nan))

fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
print(f"{'Lauf':18s} {'t-t_p':>6s} {'a/a0':>7s} {'e':>6s} {'d_omega':>8s} {'r_p [cm]':>9s} {'r_a [cm]':>9s} {'h/h0':>6s}")
for k, run in enumerate(RUNS):
    df = pd.read_csv(os.path.join(BASE, run, "mX.csv")).dropna(subset=["xc"])
    el = elements(df)
    tp = tab[tab["run"] == run]["t_peri"].iloc[0]
    c = f"C{k}"
    ts = el["t"] - tp
    th = np.linspace(0, 2 * np.pi, 400)
    a0, e0, om0 = el["a"][0], el["e"][0], np.radians(el["om"][0])
    rk = a0 * (1 - e0**2) / (1 + e0 * np.cos(th - om0))
    ax[0, 0].plot(rk * np.cos(th) / 1e16, rk * np.sin(th) / 1e16, ":", color=c, lw=0.8)
    ax[0, 0].plot(el["x"] / 1e16, el["y"] / 1e16, "-", color=c, label=run)
    ax[0, 1].plot(ts, el["a"] / a0, color=c, label=run)
    ax[1, 0].plot(ts, el["e"], color=c)
    ax[1, 1].plot(ts, el["om"] - el["om"][0], color=c)
    for dt in (-5, 5, 20, 50):
        i = np.argmin(np.abs(ts - dt))
        if abs(ts[i] - dt) < 1.5:
            print(f"{run:18s} {ts[i]:6.1f} {el['a'][i]/a0:7.3f} {el['e'][i]:6.3f} {el['om'][i]-el['om'][0]:8.1f} "
                  f"{el['rp'][i]:9.2e} {el['ra'][i]:9.2e} {el['h'][i]/el['h'][0]:6.3f}")
ax[0, 0].plot(0, 0, "k+", ms=10)
ax[0, 0].set_aspect("equal"); ax[0, 0].set_xlim(-30, 8); ax[0, 0].set_ylim(-20, 6)
ax[0, 0].set_xlabel("x [1e16 cm]"); ax[0, 0].set_ylabel("y [1e16 cm]")
ax[0, 0].set_title("Schwerpunkt (durchgezogen) und urspruengliche Keplerbahn (gepunktet)", fontsize=9)
ax[0, 0].legend(fontsize=7)
for a_, lab in [(ax[0, 1], "a / a(0)"), (ax[1, 0], "Exzentrizitaet e"), (ax[1, 1], "Drehung des Perizentrums [Grad]")]:
    a_.axvline(0, color="grey", lw=0.8); a_.set_xlabel("t - t_peri [yr]"); a_.set_ylabel(lab)
ax[0, 1].set_yscale("log")
out = os.path.join(BASE, "orbit_elements.png")
fig.savefig(out, dpi=140)
print("Gespeichert:", out)

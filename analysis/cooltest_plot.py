"""Kuehltest: Wolke mit 1e5 K ohne Schwarzes Loch -> T(t) muss auf 1e4 K fallen und dort bleiben.

Erwartete Kurve: isochore Integration von dT/dt = -C rho Lambda(T) mit der Tabelle aus
src/utils/townsend_cooling.hpp (gleiche Konstanten wie dort), unabhaengig vom Simulationscode.
Aufruf:  python3 analysis/cooltest_plot.py runs/cool_test2
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/cool_test2"
YR, KB, MH_SIM, MU_SIM = 3.156e7, 1.380649e-16, 1.6726e-24, 0.6139

# --- Tabelle aus Max' Header lesen ---------------------------------------------------------
hdr = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "utils",
                        "townsend_cooling.hpp")).read()
def table(name):
    vals = dict((int(i), float(v)) for i, v in re.findall(rf"{name}\((\d+)\)\s*=\s*([-0-9.eE+]+);", hdr))
    return np.array([vals[i] for i in range(len(vals))])
cool_t, cool_coef, cool_index = table("cool_t"), table("cool_coef"), table("cool_index")
g, X, Z, mh_t, kb_t = 5/3, 0.7, 0.02, 1.6605e-24, 1.380648e-16
mu = 1 / (2*X + 0.75*(1 - X - Z) + Z/2); mu_e, mu_h = 2 / (1 + X), 1 / X
C = 1e-23 * (g - 1) * mu / (kb_t * mu_e * mu_h * mh_t)       # wie const_factor im Header

def Lam(T):   # in Einheiten von 1e-23 erg cm^3/s, wie im Header
    i = max(0, min(np.searchsorted(cool_t, T) - 1, len(cool_t) - 1))
    return cool_coef[i] * (T / cool_t[i])**cool_index[i]

def expected(T0, rho, t_end):
    t, T, ts, Ts = 0.0, T0, [0.0], [T0]
    while t < t_end and T > cool_t[0] * 1.0001:
        dTdt = -C * rho * Lam(T)
        h = min(1e-3 * T / abs(dTdt), t_end - t)
        T = max(T + h * dTdt, cool_t[0]); t += h
        ts.append(t); Ts.append(T)
    ts.append(t_end); Ts.append(T)
    return np.array(ts), np.array(Ts)

# --- Simulation einlesen ---------------------------------------------------------------------
ts, Tc, Tm, rhom, Pm = [], [], [], [], []
for n in g2_io.snapshot_numbers(RUN):
    t, s = g2_io.load_snapshot(RUN, n)
    rho, P, tr = s["rho"], s["press"], s["r0"]
    T = P * MU_SIM * MH_SIM / (rho * KB)
    j, i = np.unravel_index(np.argmin(np.add.outer(s["y"]**2, s["x"]**2)), rho.shape)
    cl = tr > 0.9
    w = (rho * tr)[cl]
    ts.append(t / YR); Tc.append(T[j, i]); Tm.append((T[cl] * w).sum() / w.sum())
    rhom.append((rho[cl] * w).sum() / w.sum()); Pm.append((P[cl] * w).sum() / w.sum())
ts, Tc, Tm, rhom, Pm = map(np.array, (ts, Tc, Tm, rhom, Pm))

rho0 = rhom[0]
te, Te = expected(Tc[0], rho0, ts[-1] * YR)
tcool0 = Tc[0] / (C * rho0 * Lam(Tc[0])) / YR
def t_reach(t, T, lev=1.1e4):
    k = np.where(T <= lev)[0]
    return t[k[0]] if len(k) else np.nan
print(f"T_Start = {Tc[0]:.3g} K, rho_Wolke = {rho0:.3g} g/cm3, t_cool(T_Start) = {tcool0:.3f} yr")
print(f"T <= 1.1e4 K erreicht: Simulation (Zentrum) {t_reach(ts, Tc):.3f} yr, erwartet {t_reach(te / YR, Te):.3f} yr")
print(f"T am Ende: Zentrum {Tc[-1]:.4g} K, Massenmittel {Tm[-1]:.4g} K")

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, constrained_layout=True)
a1.plot(ts, Tc, "C0-", lw=2, label="Simulation: Zentrum der Wolke")
a1.plot(ts, Tm, "C1--", lw=1.5, label="Simulation: Massenmittel (Tracer > 0.9)")
a1.plot(te / YR, Te, "k:", lw=2, label="erwartet (isochor, Tabelle aus townsend_cooling.hpp)")
a1.axhline(1e4, color="grey", lw=0.8)
a1.axvline(tcool0, color="C3", lw=0.8, ls="--", label=f"t_cool(T_Start) = {tcool0:.2f} yr")
a1.set_yscale("log"); a1.set_ylabel("T [K]"); a1.legend(fontsize=8)
a1.set_title("Kuehltest: Wolke mit 1e5 K, ohne Schwarzes Loch, gleichfoermige Umgebung", fontsize=10)
a2.plot(ts, rhom / rhom[0], "C2-", label="Dichte der Wolke / Anfang")
a2.plot(ts, Pm / Pm[0], "C4-", label="Druck der Wolke / Anfang")
a2.set_yscale("log"); a2.set_xlabel("t [yr]"); a2.legend(fontsize=8)
os.makedirs(os.path.join(RUN, "plots"), exist_ok=True)
out = os.path.join(RUN, "plots", "cooltest_T.png")
fig.savefig(out, dpi=150)
print("Gespeichert:", out)

"""Ueberlebenskriterium nach Gronke & Oh (2018): t_cool,mix / t_cc < 1 -> Wolke ueberlebt.

Kuehlung: Tabelle und Konstanten aus src/utils/townsend_cooling.hpp (Max Gronke), tcool wie dort:
          tcool = T / (C rho Lambda(T)), C = 1e-23 (gamma-1) mu / (k_B mu_e mu_H m_u).
Mischgas: T_mix = sqrt(T_cl T_hot), rho_mix = sqrt(rho_cl rho_hot) (Wolke isobar mit lokaler Atmosphaere).
Atmosphaere: Schartmann-Normierung rho0 = 1.702e-21 g/cm^3 bei 1e16 cm, T0 = 2.128e8 K, rho, T ~ 1/r.
Aufruf:  python3 analysis/gronke_oh.py
"""
import math, os, re
import numpy as np

YR, R_C = 3.156e7, 1.87e15
GM = 6.674e-8 * 4.31e6 * 1.989e33
RHO0, T0, R0 = 1.702e-21, 2.128e8, 1e16
X0, Y0 = -7.22e16, 2.21e16
R_START = math.hypot(X0, Y0)
VEL = {4e15: (794.59, 48.45), 8e15: (826.88, 153.94), 1.6e16: (870.88, 297.69), 3.2e16: (934.48, 505.46)}

hdr = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "utils",
                        "townsend_cooling.hpp")).read()
def table(name):
    vals = dict((int(i), float(v)) for i, v in re.findall(rf"{name}\((\d+)\)\s*=\s*([-0-9.eE+]+);", hdr))
    return np.array([vals[i] for i in range(len(vals))])
cool_t, cool_coef, cool_index = table("cool_t"), table("cool_coef"), table("cool_index")
g, X, Z, mh, kb = 5/3, 0.7, 0.02, 1.6605e-24, 1.380648e-16
mu = 1 / (2*X + 0.75*(1 - X - Z) + Z/2); mu_e, mu_h = 2 / (1 + X), 1 / X
C = 1e-23 * (g - 1) * mu / (kb * mu_e * mu_h * mh)

def Lam(T):
    if T < cool_t[0]:
        return 0.0
    i = min(np.searchsorted(cool_t, T) - 1, len(cool_t) - 1)
    return cool_coef[i] * (T / cool_t[i])**cool_index[i]

def tcool(T, rho, fac=1.0):          # [yr]; ausserhalb der Tabelle oben: letztes Stueck fortgesetzt
    L = Lam(T) * fac
    return T / (C * rho * L) / YR if L > 0 else float("inf")

def v_peri(vx, vy, rp):
    E = 0.5 * ((vx * 1e5)**2 + (vy * 1e5)**2) - GM / R_START
    a = -GM / (2 * E)
    return math.sqrt(GM * (2 / rp - 1 / a))

def state(r, chi_start, fac=1.0):
    rho_h, T_h = RHO0 * R0 / r, T0 * R0 / r
    T_cl = T0 * R0 / R_START / chi_start           # Wolkentemperatur fest (wie im Gitter)
    rho_cl = rho_h * T_h / T_cl                     # isobar
    T_mix, rho_mix = math.sqrt(T_cl * T_h), math.sqrt(rho_cl * rho_h)
    return T_cl, rho_cl, rho_h, T_h, T_mix, tcool(T_mix, rho_mix, fac), tcool(T_cl, rho_cl, fac)

print("Kuehltabelle aus townsend_cooling.hpp. Zeiten in yr. Verh. = t_cool,mix / t_cc (< 1: Kuehlung gewinnt).\n")
print(f"{'chi':>6s} {'r_p':>8s} | {'Ort':>8s} {'T_cl':>8s} {'T_mix':>8s} {'t_cool,mix':>10s} {'t_cc':>8s} "
      f"{'Verh.':>8s} {'t_cool,cl':>10s}")
rows = [(chi, rp) for chi in (10, 100, 1000, 1e4) for rp in VEL] + [(2755, 4e15)]
for chi, rp in rows:
    vx, vy = VEL[rp]
    for where, r, v in [("Start", R_START, math.hypot(vx, vy) * 1e5), ("Perizent", rp, v_peri(vx, vy, rp))]:
        T_cl, rho_cl, rho_h, T_h, T_mix, tcm, tcl = state(r, chi)
        tcc = math.sqrt(rho_cl / rho_h) * R_C / v / YR
        print(f"{chi:6g} {rp:8.1e} | {where:>8s} {T_cl:8.2e} {T_mix:8.2e} {tcm:10.3g} {tcc:8.3g} "
              f"{tcm / tcc:8.3g} {tcl:10.3g}")

print("\nCC01 (chi = 2755, T_cl = 1.0e4 K) mit Lambda_fac:")
vx, vy = VEL[4e15]
for fac in (0.01, 0.1, 1.0, 10.0):
    out = []
    for where, r, v in [("Start", R_START, math.hypot(vx, vy) * 1e5), ("Perizent", 4e15, v_peri(vx, vy, 4e15))]:
        T_cl, rho_cl, rho_h, T_h, T_mix, tcm, tcl = state(r, 2755, fac)
        out.append(f"{where}: t_cool,mix/t_cc = {tcm / (math.sqrt(rho_cl / rho_h) * R_C / v / YR):.3g}")
    print(f"  Lambda_fac = {fac:5g}:  " + ",  ".join(out))

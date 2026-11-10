"""Zeitskalen-Tabelle aller Gitterlaeufe (512) + R-Varianten + CC01: Zeitskalen, Modell und gemessenes Regime.

t_cc = chi^(1/2) R / v (Start, aus study_table.csv), t_KH (Start), t_sc = R / c_s,cl (Schallzeit der Wolke),
t_cool,cl und t_cool,mix (Start; Wolke isobar, T_mix = sqrt(T_cl T_hot), rho_mix = sqrt(rho_cl rho_hot), Townsend-Tabelle),
r/r_crit = t_cc / t_cool,mix.  Modell x = ALPHA t_cc / t_peri (ALPHA fuer rho/rho_eq > sqrt(chi0), 512).
Gemessenes Regime aus t_50 (dq): vorher (< 0.85 t_peri) / Perizentrum (<= 1 + t_tid/t_peri + 0.3) / ueberlebt.
Ausgabe: runs/timescales.csv, runs/timescales.md, Tabelle sortiert nach x.
Aufruf:  python3 analysis/timescale_table.py   (oder %run im Notebook)
"""
import json, math, os, re
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/athena/runs")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "utils", "townsend_cooling.hpp")
ALPHA = 6.77
R_C, R0_PROF, RHO0, T0 = 1.87e15, 1.0e16, 1.702e-21, 2.128e8
R_START = math.hypot(-7.22e16, 2.21e16)
g, X, Z, mh, kb = 5 / 3, 0.7, 0.02, 1.6605e-24, 1.380648e-16
mu = 1 / (2 * X + 0.75 * (1 - X - Z) + Z / 2); mu_e, mu_h = 2 / (1 + X), 1 / X
C = 1e-23 * (g - 1) * mu / (kb * mu_e * mu_h * mh)
YR = 3.156e7

hdr = open(SRC).read()
def table(name):
    vals = dict((int(i), float(v)) for i, v in re.findall(rf"{name}\((\d+)\)\s*=\s*([-0-9.eE+]+);", hdr))
    return np.array([vals[i] for i in range(len(vals))])
cool_t, cool_coef, cool_index = table("cool_t"), table("cool_coef"), table("cool_index")
def Lam(T):
    if T < cool_t[0]:
        return 0.0
    i = min(np.searchsorted(cool_t, T) - 1, len(cool_t) - 1)
    return cool_coef[i] * (T / cool_t[i])**cool_index[i]
def tcool(T, rho):
    L = Lam(T)
    return T / (C * rho * L) / YR if L > 0 else float("inf")

def rfac(run):
    try:
        man = json.load(open(os.path.join(BASE, run, "manifest.json")))
        return float(man.get("params", {}).get("problem/R_cloud", R_C)) / R_C
    except Exception:
        return 1.0

tab = pd.read_csv(os.path.join(BASE, "study_table.csv"))
tab = tab[(tab["Y"] == "1/2") & (tab["norm"] == "init")]
keep = tab["run"].str.match(r"^g_chi[0-9e]+_rp\d+(_big)?$") | tab["run"].str.contains(r"_rp\d+_R") | (tab["run"] == "ps_ref_big")
tab = tab[keep]
rows = []
for run, sub in tab.groupby("run"):
    r = sub.iloc[0]
    chi, f = float(r["chi"]), rfac(run)
    R = R_C * f
    T_h, rho_h = T0 * R0_PROF / R_START, RHO0 * R0_PROF / R_START
    T_cl, rho_cl = T_h / chi, rho_h * chi
    t_cc = r["t_cc0"] * f
    t_sc = R / math.sqrt(g * kb * T_cl / (mu * mh)) / YR
    tcm, tcl = tcool(math.sqrt(T_cl * T_h), math.sqrt(rho_cl * rho_h)), tcool(T_cl, rho_cl)
    t50 = {X_: sub.loc[sub["X"] == X_, "t"].iloc[0] if (sub["X"] == X_).any() else np.nan for X_ in ("0.9", "dq")}
    x, plat = ALPHA * t_cc / r["t_peri"], 1 + r["t_tid_peri"] / r["t_peri"]
    y = t50["dq"] / r["t_peri"]
    reg = ("überlebt" if not np.isfinite(y) else "vorher" if y < 0.85 else "Perizentrum" if y <= plat + 0.3 else "überlebt")
    rows.append(dict(run=run, chi=chi, rp=r["rp"], R_R0=f, t_peri=r["t_peri"], t_tid=r["t_tid_peri"], t_cc=t_cc,
                     t_KH=r["t_KH0"] * f, t_sc=t_sc, t_cool_cl=tcl, t_cool_mix=tcm, r_rcrit=t_cc / tcm, x=x,
                     model="vorher" if x < 1 else "Perizentrum", t50_mix=t50["0.9"], t50_dq=t50["dq"], y_dq=y,
                     regime=reg, t_end=r["t_end"]))
df = pd.DataFrame(rows).sort_values("x")
df.to_csv(os.path.join(BASE, "timescales.csv"), index=False)

cols = ["run", "chi", "rp", "R_R0", "t_peri", "t_tid", "t_cc", "t_KH", "t_sc", "t_cool_mix", "r_rcrit", "x", "y_dq", "model", "regime"]
fmt = lambda v: f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(fmt(v) for v in rr) + " |" for rr in df[cols].values]
open(os.path.join(BASE, "timescales.md"), "w").write("\n".join(md) + "\n")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print(f"\n{len(df)} Laeufe. Gespeichert: runs/timescales.csv, runs/timescales.md")

"""Abtragung nach dem Perizentrum: tau_1/2 = Zeit ab t_peri, bis die Masse auf die Haelfte ihres Werts bei t_peri faellt.

Masse aus <lauf>/mX.csv: unvermischt (m_0.9) und dichter Strom (dq), Laeufe aus runs/timescales.csv mit Regime
Perizentrum oder ueberlebt.  Anpassung (nur Laeufe mit erreichtem tau_1/2):  log tau = a log chi + b log R + c log r_p + k.
Vergleich: t_cc Start (1/2, 1, 0); t_cc Perizentrum, Dichte konstant (1/2, 1, 1); isobar (1/2, 1, 0).
Ausgabe: runs/erosion.csv, runs/erosion.png.   Aufruf:  python3 analysis/erosion.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/athena/runs")
ts = pd.read_csv(os.path.join(BASE, "timescales.csv"))
ts = ts[ts["regime"].isin(["Perizentrum", "überlebt"])]
rows = []
for _, r in ts.iterrows():
    p = os.path.join(BASE, r["run"], "mX.csv")
    if not os.path.exists(p):
        print(f"{r['run']}: keine mX.csv"); continue
    df = pd.read_csv(p)
    t = df["t_yr"].to_numpy()
    row = dict(run=r["run"], chi=r["chi"], R=r["R_R0"], rp=r["rp"], x=r["x"], regime=r["regime"],
               t_left=t[-1] - r["t_peri"])
    for col, lab in (("m_0.9", "mix"), ("dq", "dq")):
        m = df[col].to_numpy() / df["m_tot"].iloc[0]
        m1 = np.interp(r["t_peri"], t, m)
        after = (t > r["t_peri"]) & (m < 0.5 * m1)
        row[f"m_peri_{lab}"] = m1
        row[f"tau_{lab}"] = (t[after][0] - r["t_peri"]) if after.any() else np.nan
    rows.append(row)
E = pd.DataFrame(rows).sort_values("x")
E.to_csv(os.path.join(BASE, "erosion.csv"), index=False)
pd.set_option("display.width", 200)
print(E[["run", "chi", "R", "rp", "x", "regime", "m_peri_mix", "tau_mix", "m_peri_dq", "tau_dq", "t_left"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
for ax, lab in zip(axes, ("mix", "dq")):
    ok = E[f"tau_{lab}"].notna() & (E[f"tau_{lab}"] > 0)
    A = np.column_stack([np.log10(E["chi"][ok]), np.log10(E["R"][ok]), np.log10(E["rp"][ok]), np.ones(ok.sum())])
    y = np.log10(E[f"tau_{lab}"][ok])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    rms = np.sqrt(np.mean((pred - y)**2))
    print(f"\n[{lab}] {ok.sum()} Laeufe:  tau ~ chi^{coef[0]:.2f} R^{coef[1]:.2f} r_p^{coef[2]:.2f}   (Streuung {10**rms:.2f}x)")
    fit = lambda d: 10**(coef[0]*np.log10(d["chi"]) + coef[1]*np.log10(d["R"]) + coef[2]*np.log10(d["rp"]) + coef[3])
    ax.loglog(fit(E[ok]), E[f"tau_{lab}"][ok], "o", color="#d9822b", label="τ½ reached")
    cens = ~ok
    ax.loglog(fit(E[cens]), E["t_left"][cens], "^", mfc="none", color="#1d2733", label="lower limit (not reached)")
    lim = [1, 300]; ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel(f"fit: χ^{coef[0]:.2f} R^{coef[1]:.2f} r_p^{coef[2]:.2f}  [yr]")
    ax.set_ylabel("τ½ after pericentre [yr]")
    ax.set_title({"mix": "unmixed (tracer > 0.9)", "dq": "dense stream (ρ/ρ_eq > √χ₀)"}[lab])
    ax.legend(fontsize=10)
fig.savefig(os.path.join(BASE, "erosion.png"), dpi=150)
print("\nGespeichert: runs/erosion.csv, runs/erosion.png")

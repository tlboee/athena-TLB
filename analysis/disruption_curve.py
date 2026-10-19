"""Disruptionskurve: Wolkenmasse oberhalb von Tracer-Schwellen gegen die Zeit.

Aufruf:  %run ~/athena/analysis/disruption_curve.py s12       (vordefinierter Satz)
         %run ~/athena/analysis/disruption_curve.py run1 run2 (Run-Namen in ~/athena/runs)

Fuer jede Tracer-Schwelle zwei Anteile:
  init(t) = M(tr > Schwelle) / M_tot(0)   -> Mischung + Akkretion
  rem(t)  = M(tr > Schwelle) / M_tot(t)   -> nur Mischung (bezogen auf vorhandenes Wolkengas)
und die Zeitpunkte, an denen sie unter 50 % bzw. 10 % fallen.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
RUN_SETS = {
    "hr": {   # 1024 x 768, Burkert-Normierung, bis 2015.5
        "g2_cart_hr":        r"$\chi \approx 2800$, $\gamma = 5/3$",
        "g2_cart_hr_rho621": r"$\chi \approx 4900$, $\gamma = 5/3$",
        "g2_cart_hr_g101":   r"$\chi \approx 2800$, $\gamma = 1.01$",
    },
    "lr": {   # 256 x 192, Mac
        "g2_cart_cc01_v2":     r"$\chi \approx 2800$, $\gamma = 5/3$",
        "g2_cart_cc01_rho621": r"$\chi \approx 4900$, $\gamma = 5/3$",
        "g2_cart_cc01_g1.01":  r"$\chi \approx 2800$, $\gamma = 1.01$",
    },
    "s12": {  # CC01 mit Schartmann-Normierung bis 2060, zwei Rechenwege
        "g2_cart_hr_s12":     "CC01 bis 2060 (Lauf 1)",
        "g2_cart_hr_s12_acc": "CC01 bis 2060 (Lauf 2, mit Zaehlern)",
    },
    "conv": { # Konvergenz bis 2015.5, Schartmann-Normierung
        "g2_cart_hr_s12_acc": "1024 x 768",
        "g2_cart_xhr_s12":    "2048 x 1536",
    },
}
args = sys.argv[1:] or ["hr"]
if args[0] in RUN_SETS:
    SET, RUNS = args[0], RUN_SETS[args[0]]
else:
    SET, RUNS = "custom", {a: a for a in args}
THRESH  = [0.9, 0.99, 0.9999]
LEVELS  = [0.5, 0.1]   # Kriterium: Anteil, unter den die Wolkenmasse faellt
T_PERI  = 18.0         # Perizentrum 2013.5
T_CONV  = 20.0         # Vergleichszeitpunkt fuer Konvergenz (2015.5)
YR      = g2_io.YR
COLORS  = ["C0", "C1", "C2", "C3", "C4"]


def cloud_masses(run, n):
    """Zeit und Massen (gleiche Zellvolumina -> Summe rho*tr); None bei nan."""
    t, g = g2_io.load_snapshot(os.path.join(BASE, run), n)
    m, tr = g["rho"] * g["r0"], g["r0"]
    if not np.all(np.isfinite(m)):
        return None
    return t, [m.sum()] + [m[tr > th].sum() for th in THRESH]


def t_below(t, f, level):
    """Erster Zeitpunkt, an dem f unter level faellt (linear interpoliert), sonst None."""
    below = np.where(f < level)[0]
    if len(below) == 0 or below[0] == 0:
        return None
    i = below[0]
    return t[i-1] + (level - f[i-1]) * (t[i] - t[i-1]) / (f[i] - f[i-1])


results = {}
for run in RUNS:
    rows = []
    for n in g2_io.snapshot_numbers(os.path.join(BASE, run)):
        r = cloud_masses(run, n)
        if r is not None:
            rows.append([r[0] / YR] + r[1])
    if not rows:
        print(f"{run}: keine Snapshots gefunden -> uebersprungen")
        continue
    df = pd.DataFrame(rows, columns=["t_yr", "M_tot"] + [f"M_tr>{th}" for th in THRESH])
    for th in THRESH:
        df[f"rem_tr>{th}"] = df[f"M_tr>{th}"] / df["M_tot"]
    M0 = df["M_tot"].iloc[0]
    for col in ["M_tot"] + [f"M_tr>{th}" for th in THRESH]:
        df[col] /= M0
    df.to_csv(os.path.join(BASE, run, "disruption.csv"), index=False)
    results[run] = df
    print(f"{run}: {len(df)} Snapshots bis t = {df['t_yr'].iloc[-1]:.1f} yr")

print(f"\nSatz '{SET}': Zeitpunkt [yr seit 1995.5], an dem M(tr > Schwelle) unter das Kriterium faellt")
print("  'init' = bezogen auf die Anfangsmasse (Mischung + Akkretion), "
      "'rem' = bezogen auf die vorhandene Masse (nur Mischung); '-' = nicht erreicht")
for run, df in results.items():
    t = df["t_yr"].to_numpy()
    print(f"\n  {run}  (bis {t[-1]:.1f} yr, M_tot am Ende {df['M_tot'].iloc[-1]:.3f})")
    header = f"    {'Schwelle':>9s}"
    for L in LEVELS:
        header += f"{'init ' + str(int(L * 100)) + '%':>11s}{'rem ' + str(int(L * 100)) + '%':>11s}"
    print(header + f"{'init(2015.5)':>14s}")
    for th in THRESH:
        fi = df[f"M_tr>{th}"].to_numpy()
        fr = df[f"rem_tr>{th}"].to_numpy()
        cells = ""
        for L in LEVELS:
            for f in (fi, fr):
                x = t_below(t, f, L)
                cells += f"{x:11.1f}" if x is not None else f"{'-':>11s}"
        c20 = f"{np.interp(T_CONV, t, fi):14.3f}" if t[-1] >= T_CONV - 0.05 else f"{'-':>14s}"
        print(f"    {'> ' + str(th):>9s}{cells}{c20}")

# Abbildung 1: alle Schwellen relativ zur Anfangsmasse
fig, axes = plt.subplots(1, len(THRESH), figsize=(5 * len(THRESH), 4.2), sharey=True,
                         constrained_layout=True)
for ax, th in zip(axes, THRESH):
    for (run, label), c in zip(RUNS.items(), COLORS):
        if run not in results:
            continue
        df = results[run]
        ax.plot(df["t_yr"], df[f"M_tr>{th}"], color=c, label=label)
        ax.plot(df["t_yr"], df["M_tot"], color=c, ls=":", lw=1)
    ax.axvline(T_PERI, color="grey", ls="--", lw=0.8)
    ax.set_title(f"tr > {th}  ({SET})")
    ax.set_xlabel("t [yr] seit 1995.5")
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("Masse / Anfangsmasse")
axes[0].legend(loc="lower left", fontsize=8)
out1 = os.path.expanduser(f"~/athena/analysis/disruption_curve_{SET}.png")
fig.savefig(out1, dpi=150)

# Abbildung 2: Tracer > 0.9 relativ zur Anfangsmasse und zur vorhandenen Masse
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
for (run, label), c in zip(RUNS.items(), COLORS):
    if run not in results:
        continue
    df = results[run]
    axes[0].plot(df["t_yr"], df["M_tr>0.9"], color=c, label=label)
    axes[0].plot(df["t_yr"], df["M_tot"], color=c, ls=":", lw=1)
    axes[1].plot(df["t_yr"], df["rem_tr>0.9"], color=c, label=label)
axes[0].set_title("tr > 0.9 / Anfangsmasse  (Mischung + Akkretion)")
axes[1].set_title("tr > 0.9 / vorhandene Masse  (nur Mischung)")
for ax in axes:
    ax.axvline(T_PERI, color="grey", ls="--", lw=0.8)
    for L in LEVELS:
        ax.axhline(L, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("t [yr] seit 1995.5")
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("Anteil")
axes[0].legend(loc="lower left", fontsize=8)
out2 = os.path.expanduser(f"~/athena/analysis/disruption_tr09_{SET}.png")
fig.savefig(out2, dpi=150)
print("\nGespeichert:", out1, "\n            ", out2)

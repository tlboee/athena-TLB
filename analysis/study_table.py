"""Auswertung der Parameterstudie: m_X(t), t_X,Y, Randverlust, Vergleich mit Zeitskalen.

Aufruf (Notebook):  %run ~/athena/analysis/study_table.py 'g_*' ps_ref
                    %run ~/athena/analysis/study_table.py '*_t25'      (Pilotlaeufe)
Liest alle Run-Ordner mit manifest.json, deren Name auf ein Muster passt.
"""
import fnmatch, json, math, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io
sys.path.insert(0, os.path.expanduser("~/athena/vis/python"))
import athena_read

BASE  = os.path.expanduser("~/athena/runs")
YR    = g2_io.YR
GM    = 6.674e-8 * 4.31e6 * 1.989e33
R_C   = 1.87e15
V0    = math.hypot(794.59, 48.45) * 1e5      # Startgeschwindigkeit CC01 [cm/s]
XS    = [0.1, 1/3, 0.5, 0.9, 0.99, 0.9999]
XN    = {0.1: "0.1", 1/3: "1/3", 0.5: "0.5", 0.9: "0.9", 0.99: "0.99", 0.9999: "0.9999"}
YS    = [0.5, 1/3, 0.1]
YN    = {0.5: "1/2", 1/3: "1/3", 0.1: "1/10"}
FLAG  = 0.01      # Warnung, wenn > 1 % der Anfangsmasse mit Tracer > 1/3 die Box verlassen hat

patterns = sys.argv[1:] or ["g_*", "ps_ref*"]


def mx_curve(run_dir):
    """m_X(t) / m_tot(0) aus den Snapshots; Zwischenspeicher mX.csv."""
    cache = os.path.join(run_dir, "mX.csv")
    nums = g2_io.snapshot_numbers(run_dir)
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        if len(df) == len(nums):
            return df
    rows = []
    for n in nums:
        t, g = g2_io.load_snapshot(run_dir, n)
        m, tr = g["rho"] * g["r0"], g["r0"]
        rows.append([t / YR, m.sum()] + [m[tr > x].sum() for x in XS])
    df = pd.DataFrame(rows, columns=["t_yr", "m_tot"] + [f"m_{XN[x]}" for x in XS])
    m0 = df["m_tot"].iloc[0]
    for c in df.columns[1:]:
        df[c] /= m0
    df.to_csv(cache, index=False)
    return df


def t_below(t, f, level):
    below = np.where(f < level)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    return t[i-1] + (level - f[i-1]) * (t[i] - t[i-1]) / (f[i] - f[i-1])


def boundary(run_dir):
    """Randverlust (Anteil der Anfangsmasse): gesamt, Tracer > 1/3, Tracer > 1/2, Senke."""
    h = athena_read.hst(os.path.join(run_dir, "g2_cart.hst"))
    t = np.asarray(h["time"])
    _, idx = np.unique(t[::-1], return_index=True)          # Duplikate nach Restarts
    keep = np.sort(len(t) - 1 - idx)
    t = t[keep] / YR
    s = np.asarray(h["0-scalar"])[keep]
    loss = (s[0] - s) / s[0]
    cs, cb, cr = (np.asarray(h[k])[keep] for k in ("M_sink", "M_buffer", "M_reset"))
    c3, c2 = np.asarray(h["M_buf_tr>1/3"])[keep], np.asarray(h["M_buf_tr>1/2"])[keep]
    csum = cs + cb + cr
    with np.errstate(divide="ignore", invalid="ignore"):
        buf  = loss * np.where(np.abs(csum) > 0, cb / csum, 0.0)
        sink = loss * np.where(np.abs(csum) > 0, cs / csum, 0.0)
        b3   = buf * np.where(cb > 0, c3 / cb, 0.0)
        b2   = buf * np.where(cb > 0, c2 / cb, 0.0)
    return pd.DataFrame({"t_yr": t, "sink": sink, "buf": buf, "buf_1/3": b3, "buf_1/2": b2})


# --- Laeufe einlesen ---------------------------------------------------------------------
runs = []
for d in sorted(os.listdir(BASE)):
    if not any(fnmatch.fnmatch(d, p) for p in patterns):
        continue
    mf = os.path.join(BASE, d, "manifest.json")
    if not os.path.exists(mf) or not g2_io.snapshot_numbers(os.path.join(BASE, d)):
        continue
    with open(mf) as f:
        m = json.load(f)
    m["dir"] = os.path.join(BASE, d)
    m["chi_eff"] = m["chi"] if m["chi"] else 2755.0         # Baseline: Tabellendichte
    m["rp_eff"] = m["orbit"]["rp"]
    runs.append(m)
print(f"{len(runs)} Laeufe: " + ", ".join(r["name"] for r in runs))

rows, curves, bounds = [], {}, {}
for r in runs:
    df = mx_curve(r["dir"])
    bd = boundary(r["dir"])
    curves[r["name"]], bounds[r["name"]] = df, bd
    t, tp = df["t_yr"].to_numpy(), r["t_peri_yr"]
    v = math.hypot(r["vx_kms"], r["vy_kms"]) * 1e5
    t_cc = math.sqrt(r["chi_eff"]) * R_C / v / YR
    t_tid = math.sqrt(r["rp_eff"] ** 3 / GM) / YR
    flag = bd.loc[bd["buf_1/3"] > FLAG, "t_yr"]
    base = {"run": r["name"], "chi": r["chi_eff"], "rp": r["rp_eff"], "t_peri": tp,
            "t_end": t[-1], "t_cc0": t_cc, "t_KH0": t_cc / (2 * math.pi), "t_tid_peri": t_tid,
            "sink_end": bd["sink"].iloc[-1], "buf_end": bd["buf"].iloc[-1],
            "buf13_end": bd["buf_1/3"].iloc[-1],
            "t_flag": flag.iloc[0] if len(flag) else np.nan}
    for x in XS:
        fi = df[f"m_{XN[x]}"].to_numpy()
        fr = fi / df["m_tot"].to_numpy()
        for y in YS:
            for norm, f in (("init", fi), ("rem", fr)):
                tx = t_below(t, f, y)
                rows.append(dict(base, X=XN[x], Y=YN[y], norm=norm, t=tx, t_rel=tx - tp))

tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(BASE, "study_table.csv"), index=False)

# --- Kompakte Uebersicht: X = 1/3 und 0.9, Y = 1/3, bezogen auf die Anfangsmasse ------------
pd.set_option("display.width", 200)
sel = tab[(tab["Y"] == "1/3") & (tab["norm"] == "init") & (tab["X"].isin(["1/3", "0.9"]))]
wide = sel.pivot_table(index=["run", "chi", "rp", "t_peri", "t_end", "t_cc0", "t_tid_peri",
                              "sink_end", "buf13_end", "t_flag"],
                       columns="X", values="t_rel", dropna=False).reset_index()
wide = wide.rename(columns={"1/3": "t_rel(1/3,1/3)", "0.9": "t_rel(0.9,1/3)"})
print("\nt_rel = t_X,Y - t_peri [yr] (Y = 1/3, bezogen auf die Anfangsmasse); NaN = nicht erreicht")
print("t_flag: Zeitpunkt, ab dem > 1 % der Anfangsmasse mit Tracer > 1/3 die Box verlassen hat")
print(wide.round(3).to_string(index=False))

# --- Abbildung 1: m_X(t) fuer alle Laeufe -------------------------------------------------
n = len(runs)
nc = min(4, n)
nr = math.ceil(n / nc)
fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), sharex=True, sharey=True,
                         squeeze=False, constrained_layout=True)
for ax, r in zip(axes.flat, runs):
    df = curves[r["name"]]
    tr = df["t_yr"] - r["t_peri_yr"]
    ax.plot(tr, df["m_tot"], "k", lw=1.2)
    for x in XS:
        ax.plot(tr, df[f"m_{XN[x]}"], lw=1, label=f"X = {XN[x]}")
    for y in YS:
        ax.axhline(y, color="grey", ls=":", lw=0.7)
    ax.axvline(0, color="grey", ls="--", lw=0.7)
    ax.set_title(r["name"], fontsize=9)
    ax.set_ylim(0, 1.05)
for ax in axes[-1]:
    ax.set_xlabel("t - t_peri [yr]")
for ax in axes[:, 0]:
    ax.set_ylabel("m_X / m_tot(0)")
axes.flat[0].legend(fontsize=6)
for ax in list(axes.flat)[n:]:
    ax.set_visible(False)
out1 = os.path.join(BASE, "study_mX.png")
fig.savefig(out1, dpi=130)

# --- Abbildung 2: t_X,Y gegen chi und r_p, mit Zeitskalen ----------------------------------
grid = [r for r in runs if r["chi"]]
if len(grid) > 1:
    g = tab[(tab["run"].isin([r["name"] for r in grid])) & (tab["X"] == "0.9") &
            (tab["Y"] == "1/3") & (tab["norm"] == "init")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for rp, sub in g.groupby("rp"):
        sub = sub.sort_values("chi")
        axes[0].loglog(sub["chi"], sub["t_rel"].where(sub["t_rel"] > 0), "o-", label=f"r_p = {rp:.1e}")
    chis = np.array(sorted(g["chi"].unique()))
    axes[0].loglog(chis, np.sqrt(chis) * R_C / V0 / YR, "k--", label="t_cc(Start)")
    axes[0].set_xlabel("chi")
    for chi, sub in g.groupby("chi"):
        sub = sub.sort_values("rp")
        axes[1].loglog(sub["rp"], sub["t_rel"].where(sub["t_rel"] > 0), "o-", label=f"chi = {chi:g}")
    rps = np.array(sorted(g["rp"].unique()))
    axes[1].loglog(rps, np.sqrt(rps ** 3 / GM) / YR, "k--", label="t_tid(Perizentrum)")
    axes[1].set_xlabel("r_p [cm]")
    for ax in axes:
        ax.set_ylabel("t_0.9,1/3 - t_peri [yr]")
        ax.legend(fontsize=7)
    out2 = os.path.join(BASE, "study_tXY.png")
    fig.savefig(out2, dpi=150)
    print("Gespeichert:", out1, out2)
else:
    print("Gespeichert:", out1)

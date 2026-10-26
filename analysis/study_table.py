"""Auswertung der Parameterstudie: m_X(t), t_X,Y, Randverlust, Vergleich mit Zeitskalen.

Aufruf (Notebook):  %run ~/athena/analysis/study_table.py 'g_chi*_rp4' ... ps_ref ps_ref_ppm ...
Liest alle Run-Ordner mit manifest.json, deren Name auf ein Muster passt.

Normierungen fuer t_X,Y (Zeitpunkt, an dem m_X unter Y faellt; Zeiten in yr seit 1995.5):
  init     m_X / m_tot(0)                         (Mischung + Akkretion + Randverlust)
  init_rc  (m_X + Randverlust von Gas mit Tracer > X) / m_tot(0), nur X = 1/3 und 1/2
           (ueber den Rand verlorenes Wolkengas zaehlt als noch nicht zerstoert)
  rem      m_X / m_tot(t)                         (nur Mischung des noch vorhandenen Gases)
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
FLAG  = 0.01      # Warnung, wenn > 1 % der Anfangs-Tracermasse die Box verlassen hat (Gesamtzaehler)

patterns = sys.argv[1:] or ["g_*", "ps_ref*"]


RHO0, R_IN = 1.702e-21, 1.9e15          # Schartmann-Normierung bei 1e16 cm, Senkenradius
DF = {"1": 1.0, "1/3": 1/3, "1/10": 0.1}  # Dichteschwellen (Scannapieco & Brueggen 2015: F_1, F_1/3, F_1/10)
DCOLS = [f"dl_{k}" for k in DF] + [f"da_{k}" for k in DF]


def mx_curve(run_dir):
    """m_X(t) / m_tot(0) aus den Snapshots; Zwischenspeicher mX.csv.

    Tracer:  m_X        = Summe rho*tr ueber Zellen mit tr > X
    Dichte:  dl_f (lokal)    = Summe rho*tr ueber Zellen mit rho/rho_eq(r) > f * chi0
             da_f (absolut)  = Summe rho*tr ueber Zellen mit rho > f * rho_cl0   (wie SB15)
    chi0 und rho_cl0: Median ueber die Zellen mit tr > 0.99 im ersten Snapshot.
    """
    cache = os.path.join(run_dir, "mX.csv")
    nums = g2_io.snapshot_numbers(run_dir)
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        if len(df) == len(nums) and all(c in df.columns for c in DCOLS):
            return df
    rows, rho_eq, chi0, rho_cl0 = [], None, None, None
    for n in nums:
        t, g = g2_io.load_snapshot(run_dir, n)
        rho, tr = g["rho"], g["r0"]
        if rho_eq is None:
            X, Y = np.meshgrid(g["x"], g["y"])
            rho_eq = RHO0 * 1e16 / np.maximum(np.hypot(X, Y), R_IN)
            cl = tr > 0.99
            rho_cl0, chi0 = np.median(rho[cl]), np.median((rho / rho_eq)[cl])
            print(f"  {os.path.basename(run_dir)}: rho_cl0 = {rho_cl0:.3e} g/cm3, chi0 (lokal) = {chi0:.1f}")
        m, contrast = rho * tr, rho / rho_eq
        rows.append([t / YR, m.sum()] + [m[tr > x].sum() for x in XS]
                    + [m[contrast > f * chi0].sum() for f in DF.values()]
                    + [m[rho > f * rho_cl0].sum() for f in DF.values()])
    df = pd.DataFrame(rows, columns=["t_yr", "m_tot"] + [f"m_{XN[x]}" for x in XS] + DCOLS)
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
    """Verluste als Anteil der Anfangsmasse: Senke, Rand gesamt, Rand mit Tracer > 1/3 bzw. > 1/2."""
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
    if not any(fnmatch.fnmatch(d, p) for p in patterns if not p.startswith("!")):
        continue
    if any(fnmatch.fnmatch(d, p[1:]) for p in patterns if p.startswith("!")):
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

rows, curves = [], {}
for r in runs:
    df = mx_curve(r["dir"])
    bd = boundary(r["dir"])
    t, tp = df["t_yr"].to_numpy(), r["t_peri_yr"]
    corr = {"1/3": np.interp(t, bd["t_yr"], bd["buf_1/3"]),
            "0.5": np.interp(t, bd["t_yr"], bd["buf_1/2"])}
    curves[r["name"]] = (df, corr)
    v = math.hypot(r["vx_kms"], r["vy_kms"]) * 1e5
    t_cc = math.sqrt(r["chi_eff"]) * R_C / v / YR
    flag = bd.loc[bd["buf"] > FLAG, "t_yr"]
    base = {"run": r["name"], "chi": r["chi_eff"], "rp": r["rp_eff"], "t_peri": tp,
            "t_end": t[-1], "t_cc0": t_cc, "t_KH0": t_cc / (2 * math.pi),
            "t_tid_peri": math.sqrt(r["rp_eff"] ** 3 / GM) / YR,
            "sink_end": bd["sink"].iloc[-1], "buf_end": bd["buf"].iloc[-1],
            "buf13_end": bd["buf_1/3"].iloc[-1],
            "t_flag": flag.iloc[0] if len(flag) else np.nan}
    for x in XS:
        fi = df[f"m_{XN[x]}"].to_numpy()
        variants = [("init", fi), ("rem", fi / df["m_tot"].to_numpy())]
        if XN[x] in corr:
            variants.append(("init_rc", fi + corr[XN[x]]))
        for y in YS:
            for norm, f in variants:
                tx = t_below(t, f, y)
                rows.append(dict(base, X=XN[x], Y=YN[y], norm=norm, t=tx, t_rel=tx - tp))
    for key in DCOLS:
        f = df[key].to_numpy()
        for y in YS:
            tx = t_below(t, f, y)
            rows.append(dict(base, X=key, Y=YN[y], norm="init", t=tx, t_rel=tx - tp))

tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(BASE, "study_table.csv"), index=False)


# --- Uebersicht: Zeiten in yr seit 1995.5 -----------------------------------------------------
def pick(run, X, norm, Y="1/3"):
    s = tab[(tab["run"] == run) & (tab["X"] == X) & (tab["Y"] == Y) & (tab["norm"] == norm)]["t"]
    return s.iloc[0]


def fmt(v, t_end):
    return f"{v:7.1f}" if np.isfinite(v) else f" >{t_end:5.0f}"


print("\nZeiten in yr seit 1995.5; '>100' = bis Laufende nicht erreicht")
print("t(1/3)    = t_1/3,1/3 bezogen auf m_tot(0);  t(1/3)rc = mit Randkorrektur (nur Diagnose,\n"
      "            unterschaetzt den Verlust bei duennem Gas); Rand = gesamter Randverlust")
print("t(0.9)    = t_0.9,1/3 bezogen auf m_tot(0);  Rand! = t_flag liegt vor t(1/3) bzw. t(0.9)")
print("dl_1/3    = t_50 mit rho/rho_eq(r) > chi0/3 (lokal);  da_1/3 = t_50 mit rho > rho_cl0/3 (SB15)")
print(f"{'run':15s} {'chi':>6s} {'r_p':>8s} {'t_peri':>6s} {'t_cc0':>6s} {'Senke':>6s} {'Rand':>8s} "
      f"{'t_flag':>7s} {'t(1/3)':>7s} {'t(1/3)rc':>8s} {'t(0.9)':>7s} {'dl_1/3':>7s} {'da_1/3':>7s}  Warnung")
for r in runs:
    n = r["name"]
    b = tab[tab["run"] == n].iloc[0]
    t13, t13c, t09 = pick(n, "1/3", "init"), pick(n, "1/3", "init_rc"), pick(n, "0.9", "init")
    tdl, tda = pick(n, "dl_1/3", "init", Y="1/2"), pick(n, "da_1/3", "init", Y="1/2")
    fl = b["t_flag"]
    warn = ""
    if np.isfinite(fl) and ((not np.isfinite(t13)) or fl < t13 or (not np.isfinite(t09)) or fl < t09):
        warn = "Rand!"
    print(f"{n:15s} {b['chi']:6.0f} {b['rp']:8.1e} {b['t_peri']:6.1f} {b['t_cc0']:6.1f} "
          f"{b['sink_end']:6.3f} {b['buf_end']:8.3f} {fmt(fl, np.inf) if np.isfinite(fl) else '      -'} "
          f"{fmt(t13, b['t_end'])} {fmt(t13c, b['t_end']):>8s} {fmt(t09, b['t_end'])} "
          f"{fmt(tdl, b['t_end'])} {fmt(tda, b['t_end'])}  {warn}")

# --- Abbildung 1: m_X(t) fuer alle Laeufe ---------------------------------------------------
n = len(runs)
nc = min(4, n)
nr = math.ceil(n / nc)
fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), sharex=True, sharey=True,
                         squeeze=False, constrained_layout=True)
for ax, r in zip(axes.flat, runs):
    df, corr = curves[r["name"]]
    tr = df["t_yr"] - r["t_peri_yr"]
    ax.plot(tr, df["m_tot"], "k", lw=1.2)
    for x in XS:
        ax.plot(tr, df[f"m_{XN[x]}"], lw=1, label=f"X = {XN[x]}")
    ax.plot(tr, df["dl_1/3"], "k--", lw=1.4, label="rho/rho_eq > chi0/3")
    ax.plot(tr, df["da_1/3"], "k:", lw=1.4, label="rho > rho_cl0/3 (SB15)")
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

# --- Abbildung 2: t_1/3,1/3 (randkorrigiert) gegen chi und r_p ------------------------------
g = tab[(tab["X"] == "1/3") & (tab["Y"] == "1/3") & (tab["norm"] == "init")].copy()
lo = tab[(tab["X"] == "1/3") & (tab["Y"] == "1/3") & (tab["norm"] == "init")].set_index("run")["t"]
g["t_lo"] = g["run"].map(lo)        # unkorrigiert = untere Grenze
grid = g[g["run"].str.startswith("g_")]
ref = g[g["run"].str.startswith("ps_ref")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

ax = axes[0]
for k, (rp, sub) in enumerate(grid.groupby("rp")):
    sub = sub.sort_values("chi")
    ok = np.isfinite(sub["t"]).to_numpy()
    ax.plot(sub["chi"][ok], sub["t"][ok], "o-", color=f"C{k}", label=f"r_p = {rp:.1e} cm")
    ax.plot(sub["chi"][~ok], sub["t_end"][~ok], "^", color=f"C{k}", mfc="none", ms=9)
    for xv, a, b in zip(sub["chi"], sub["t_lo"], sub["t"].fillna(sub["t_end"])):
        if np.isfinite(a) and a < b:
            ax.plot([xv, xv], [a, b], color=f"C{k}", lw=1)
            ax.plot(xv, a, "_", color=f"C{k}", ms=10)
chis = np.logspace(1, 4, 50)
ax.plot(chis, np.sqrt(chis) * R_C / V0 / YR, "k--", lw=1, label="t_cc (Start, CC01-Bahn)")
ax.axhspan(18.0, 21.4, color="grey", alpha=0.15, label="Perizentrum")
ax.plot(ref["chi"], ref["t"], "kx", ms=8, label="CC01-Kontrollen")
ax.set_xlabel("chi")

ax = axes[1]
for k, (chi, sub) in enumerate(grid.groupby("chi")):
    sub = sub.sort_values("rp")
    ok = np.isfinite(sub["t"]).to_numpy()
    ax.plot(sub["rp"][ok], sub["t"][ok], "o-", color=f"C{k}", label=f"chi = {chi:g}")
    ax.plot(sub["rp"][~ok], sub["t_end"][~ok], "^", color=f"C{k}", mfc="none", ms=9)
    for xv, a, b in zip(sub["rp"], sub["t_lo"], sub["t"].fillna(sub["t_end"])):
        if np.isfinite(a) and a < b:
            ax.plot([xv, xv], [a, b], color=f"C{k}", lw=1)
            ax.plot(xv, a, "_", color=f"C{k}", ms=10)
peri = grid.drop_duplicates("rp").sort_values("rp")
ax.plot(peri["rp"], peri["t_peri"], "k--", lw=1, label="Perizentrum")
ax.plot(peri["rp"], peri["t_peri"] + peri["t_tid_peri"], "k:", lw=1, label="Perizentrum + t_tid")
ax.set_xlabel("r_p [cm]")

for ax in axes:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("t_1/3,1/3 [yr seit 1995.5]")
    ax.legend(fontsize=7, title="offene Dreiecke: bis 100 yr nicht erreicht", title_fontsize=7)
out2 = os.path.join(BASE, "study_tXY.png")
fig.savefig(out2, dpi=150)
print("Gespeichert:", out1, out2)

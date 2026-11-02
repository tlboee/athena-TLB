"""Aufloesungstest mit Kuehlung (CC01, grosse Domain, PLM): 512 / 1024 / 2048, mit und ohne Kuehlung.

Pro Snapshot: kaltes Wolkengas (tr > 0.5, T < 3e4 K), unvermischt (tr > 0.9), Wolkengas in der Domain,
jeweils als Anteil von sum(rho tr) zu Beginn; akkretiert aus der History (Verlust nach Zaehlerverhaeltnis).
Ergebnisse je Lauf in <lauf>/res_cool.csv (Cache).  Aufruf (Notebook):  %run ~/athena/analysis/res_cooling.py
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, KB, MH, MU = 3.156e7, 1.380649e-16, 1.6726e-24, 0.6139
T_END = 50.0                                   # yr seit Start (gemeinsame Endzeit)
RES = [("512", "ps_ref_big", "ps_ref_big_cool_t50"),
       ("1024", "ps_ref_big_x2_t50", "ps_ref_big_cool_x2_t50"),
       ("2048", "ps_ref_big_x4_t50", "ps_ref_big_cool_x4_t50")]

def series(run):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "res_cool.csv")
    nums = g2_io.snapshot_numbers(d)
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        if len(df) >= sum(1 for n in nums if True) or df["t_yr"].max() >= T_END - 0.5:
            return df
    rows = []
    for n in nums:
        t, s = g2_io.load_snapshot(d, n)
        if t / YR > T_END + 0.5:
            break
        rho, tr = s["rho"], s["r0"]
        T = s["press"] * MU * MH / (rho * KB)
        m = rho * tr
        rows.append([t / YR, m.sum(), m[(tr > 0.5) & (T < 3e4)].sum(), m[tr > 0.9].sum()])
        print(f"  {run}: t = {t/YR:5.1f} yr", end="\r")
    df = pd.DataFrame(rows, columns=["t_yr", "m_dom", "m_cold", "m_unmixed"])
    m0 = df["m_dom"].iloc[0]
    for c in ("m_dom", "m_cold", "m_unmixed"):
        df[c] = df[c] / m0
    df.to_csv(cache, index=False)
    print()
    return df

def accreted(run):
    h = np.loadtxt(os.path.join(BASE, run, "g2_cart.hst"))
    t, tr, sink, buf, res = h[:, 0] / YR, h[:, 10], h[:, 11], h[:, 12], h[:, 13]
    den = sink + buf + np.abs(res)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(den > 0, (tr[0] - tr) / tr[0] * sink / den, 0.0)
    return t, acc

def t_peri(run):
    with open(os.path.join(BASE, run, "manifest.json")) as f:
        return json.load(f).get("t_peri_yr", 18.0)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
cols = {"512": "#440154", "1024": "#21918c", "2048": "#bddf26"}
print(f"{'Aufl.':5s} {'Kuehl.':6s} {'t-t_p':>6s} {'kalt':>6s} {'unverm.':>8s} {'Domain':>7s} {'akkr.':>6s}")
for lab, adia, cool in RES:
    for run, ls, kind in [(cool, "-", "an"), (adia, "--", "aus")]:
        df = series(run)
        tp = t_peri(run)
        df = df[df["t_yr"] <= T_END + 0.01]
        ta, acc = accreted(run)
        sel = ta <= T_END + 0.01
        x = df["t_yr"] - tp
        axes[0].plot(x, df["m_cold"], ls, color=cols[lab], label=f"{lab}, cooling {kind}")
        axes[1].plot(ta[sel] - tp, acc[sel], ls, color=cols[lab])
        axes[2].plot(x, df["m_unmixed"], ls, color=cols[lab])
        for dt in (5, 10, 20, 32):
            i = np.argmin(np.abs(x.to_numpy() - dt))
            if abs(x.iloc[i] - dt) < 0.6:
                j = np.argmin(np.abs(ta - tp - dt))
                print(f"{lab:5s} {kind:6s} {x.iloc[i]:6.1f} {df['m_cold'].iloc[i]:6.3f} {df['m_unmixed'].iloc[i]:8.3f} "
                      f"{df['m_dom'].iloc[i]:7.3f} {acc[j]:6.3f}")
titles = ["cold cloud gas (tracer > 0.5, T < 3·10⁴ K)", "accreted (sink)", "unmixed (tracer > 0.9)"]
for ax, ti in zip(axes, titles):
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("t − t_peri [yr]"); ax.set_title(ti, fontsize=10); ax.set_ylim(0, 1.02)
axes[0].set_ylabel("fraction of initial cloud mass")
axes[0].legend(fontsize=7, title="resolution (cells, x)", title_fontsize=7)
fig.suptitle("CC01, large domain, PLM: resolution test with and without cooling", fontsize=11)
out = os.path.join(BASE, "res_cooling.png")
fig.savefig(out, dpi=150)
print("Gespeichert:", out)

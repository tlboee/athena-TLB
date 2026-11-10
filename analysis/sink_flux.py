"""Akkretiert unvermischtes Gas? Massenfluss des Wolkengases nach innen durch einen Kreis knapp ausserhalb der Senke.

Ring um R_F = 1.25 R_IN (Breite 3 Zellen); Mdot_in = sum rho*tr*(-v_r)*dA / Ringbreite ueber Zellen mit v_r < 0,
getrennt nach tr > 0.9 (unvermischt) und tr <= 0.9 (vermischt). Zeitintegral (Trapez) / anfaengliche Wolkenmasse.
CC01 1024 doppelt: Snapshots alle 1 yr (0-50 yr) und alle 0.05 yr (17-19.5 yr) -> Test, ob 1 yr Abstand reicht.
Cache: <lauf>/sink_flux.csv.   Aufruf (Notebook):  %run ~/athena/analysis/sink_flux.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

BASE = os.path.expanduser("~/athena/runs")
YR, R_IN = 3.156e7, 1.9e15
R_F = 1.25 * R_IN
RUNS = [("χ=1000, r_p=4e15 (512)", "g_chi1000_rp4", "g_chi1000_rp4", 18.0),
        ("χ=100, r_p=4e15 (512)", "g_chi100_rp4", "g_chi100_rp4", 18.0),
        ("CC01 1024, Δt = 1 yr", "ps_ref_big_x2_t50", "ps_ref_big_x2_t50", 18.0),
        ("CC01 1024, Δt = 0.05 yr", "ps_ref_big_x2_peri", "ps_ref_big_x2_t50", 18.0)]

def key(s, cands):
    for k in cands:
        if k in s:
            return k
    raise SystemExit(f"Schluessel {cands} fehlt, vorhanden: {list(s.keys())}")

def m0(orig):
    d = os.path.join(BASE, orig)
    _, s = g2_io.load_snapshot(d, list(g2_io.snapshot_numbers(d))[0])
    dx, dy = s["x"][1] - s["x"][0], s["y"][1] - s["y"][0]
    return float((s["rho"] * np.clip(s["r0"], 0, 1)).sum() * dx * dy)

def series(run, orig):
    d = os.path.join(BASE, run)
    cache = os.path.join(d, "sink_flux.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    M0, rows = m0(orig), []
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        x, y = s["x"], s["y"]
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = np.meshgrid(x, y)
        r = np.hypot(X, Y)
        hw = 1.0 * dx
        ring = np.abs(r - (R_IN + 3 * dx)) < hw              # R_IN + 3 Zellen, +-1 Zelle
        vr = (X * s[key(s, ("vel1", "vx", "v1"))] + Y * s[key(s, ("vel2", "vy", "v2"))]) / np.maximum(r, 1.0)
        tr = np.clip(s["r0"], 0.0, 1.0)
        f = np.where(ring, s["rho"] * tr * (-vr) * dx * dy / (2 * hw), 0.0)   # [g/s pro Laengeneinheit]
        un = tr > 0.9
        rows.append([t / YR, f[un].sum() * YR / M0, f[~un].sum() * YR / M0])           # [Anteil M0 pro yr]
        print(f"  {run}: t = {t/YR:6.2f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t", "mdot_unmixed", "mdot_mixed"])
    df.to_csv(cache, index=False)
    return df

def integ(df, t0=-np.inf, t1=np.inf):
    m = (df["t"] >= t0) & (df["t"] <= t1)
    return np.trapezoid(df["mdot_unmixed"][m], df["t"][m]), np.trapezoid(df["mdot_mixed"][m], df["t"][m])

fig, axes = plt.subplots(1, len(RUNS), figsize=(18, 4.2), constrained_layout=True)
print(f"{'Lauf':26s} {'Fenster [yr]':>13s} {'akkr. unverm.':>13s} {'akkr. verm.':>11s} {'Anteil unverm.':>14s}")
data = {}
for ax, (lab, run, orig, tp) in zip(axes, RUNS):
    if not os.path.isdir(os.path.join(BASE, run)):
        print(f"{run} fehlt"); continue
    df = series(run, orig); data[lab] = df
    for (t0, t1) in ((-np.inf, np.inf), (17.0, 19.5)):
        u, m = integ(df, t0, t1)
        if u + m > 0:
            w = "alle" if np.isinf(t0) else f"{t0:g}-{t1:g}"
            print(f"{lab:26s} {w:>13s} {u:13.3f} {m:11.3f} {u/(u+m):14.2f}")
    ax.plot(df["t"], df["mdot_unmixed"], color="#1f4e79", label="unmixed (tr > 0.9)")
    ax.plot(df["t"], df["mdot_mixed"], color="#7fb3e6", label="mixed (tr ≤ 0.9)")
    ax.axvline(tp, color="k", ls="--", lw=0.8)
    ax.set_title(lab, fontsize=10); ax.set_xlabel("t [yr]")
axes[0].set_ylabel("net inflow at r = R_in + 3 cells  [M₀ / yr]")
axes[0].legend(fontsize=8)
fig.suptitle("Cloud gas flowing into the sink: unmixed vs. mixed (dashed: pericentre)")
fig.savefig(os.path.join(BASE, "sink_flux.png"), dpi=150)
plt.show()

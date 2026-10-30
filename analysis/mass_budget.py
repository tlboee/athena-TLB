"""Massenbilanz des Wolkengases: unvermischt (tr > 0.9) / vermischt / akkretiert / ueber den Rand.

Gesamtmasse des Wolkengases in der Domain exakt aus der History (Spalte 0-scalar). Der Verlust wird
nach dem Verhaeltnis der Zaehler M_sink, M_buffer, M_reset aufgeteilt (die Zaehler ueberzaehlen ueber die
RK-Stufen um einen fuer alle Kanaele gleichen Faktor; siehe g2_cartesian.cpp). Unvermischt aus mX.csv
(Spalte m_0.9) bzw. direkt aus den Snapshots, falls es keine mX.csv gibt.
Aufruf (Notebook):  %run ~/athena/analysis/mass_budget.py [lauf:bezeichnung ...]
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.expanduser("~/athena/runs")
YR = 3.156e7
DEFAULT = ["g_chi10_rp4:χ = 10", "g_chi100_rp4:χ = 100", "g_chi1000_rp4:χ = 1000",
           "g_chi1e4_rp4_big:χ = 10⁴", "ps_ref_big:CC01, adiabatic", "ps_ref_big_cool_t50:CC01, with cooling"]
RUNS = [a.split(":", 1) if ":" in a else (a, a) for a in (sys.argv[1:] or DEFAULT)]

def unmixed(run_dir):
    p = os.path.join(run_dir, "mX.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        if "m_0.9" in df.columns:
            return df["t_yr"].to_numpy(), df["m_0.9"].to_numpy()
    import g2_io
    ts, m9 = [], []
    for n in g2_io.snapshot_numbers(run_dir):
        t, s = g2_io.load_snapshot(run_dir, n)
        m = s["rho"] * s["r0"]
        ts.append(t / YR); m9.append(m[s["r0"] > 0.9].sum())
    m9 = np.array(m9)
    return np.array(ts), m9 / (m9[0] if m9[0] > 0 else 1.0)

def budget(run):
    d = os.path.join(BASE, run)
    h = np.loadtxt(os.path.join(d, "g2_cart.hst"))
    t, tr, sink, buf, res = h[:, 0] / YR, h[:, 10], h[:, 11], h[:, 12], h[:, 13]
    loss = (tr[0] - tr) / tr[0]
    den = sink + buf + np.abs(res)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(den > 0, loss * sink / den, 0.0)
        bnd = np.where(den > 0, loss * buf / den, 0.0)
    ts, m9 = unmixed(d)
    um = np.interp(t, ts, m9)
    dom = tr / tr[0]
    mix = np.clip(dom - um, 0, None)
    with open(os.path.join(d, "manifest.json")) as f:
        tp = json.load(f).get("t_peri_yr", 18.0)
    return dict(t=t - tp, unmixed=um, mixed=mix, accreted=acc, boundary=bnd, rest=1 - dom - acc - bnd)

n = len(RUNS)
fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5.2 * ((n + 1) // 2), 8.4), sharex=False, sharey=True,
                         constrained_layout=True, squeeze=False)
cols = {"unmixed": "#1f4e79", "mixed": "#7fb3e6", "accreted": "#404040", "boundary": "#d62728"}
print(f"{'Lauf':24s} {'t-t_p':>6s} {'unverm.':>8s} {'verm.':>7s} {'akkr.':>7s} {'Rand':>6s} {'Rest':>6s}")
for ax, (run, lab) in zip(axes.flat, RUNS):
    b = budget(run)
    ax.stackplot(b["t"], b["unmixed"], b["mixed"], b["accreted"], b["boundary"],
                 colors=[cols[k] for k in ("unmixed", "mixed", "accreted", "boundary")],
                 labels=["unmixed (tracer > 0.9)", "mixed (tracer < 0.9)", "accreted (sink)", "lost at boundary"])
    ax.axvline(0, color="white", lw=1, ls="--")
    i = min(np.searchsorted(b["t"], 50.0), len(b["t"]) - 1)
    ax.set_title(f"{lab}: at t − t_peri = {b['t'][i]:.0f} yr\n"
                 f"{100*b['accreted'][i]:.0f} % accreted, {100*b['mixed'][i]:.0f} % mixed, "
                 f"{100*b['unmixed'][i]:.0f} % unmixed", fontsize=9)
    ax.set_xlim(b["t"][0], min(b["t"][-1], 82)); ax.set_ylim(0, 1)
    ax.set_xlabel("t − t_peri [yr]")
    for dt in (5, 20, 50):
        k = np.argmin(np.abs(b["t"] - dt))
        if abs(b["t"][k] - dt) < 1:
            print(f"{run:24s} {b['t'][k]:6.1f} {b['unmixed'][k]:8.3f} {b['mixed'][k]:7.3f} {b['accreted'][k]:7.3f} "
                  f"{b['boundary'][k]:6.3f} {b['rest'][k]:6.3f}")
for ax in axes[:, 0]:
    ax.set_ylabel("fraction of initial cloud mass")
for ax in axes.flat[n:]:
    ax.axis("off")
axes.flat[0].legend(loc="lower left", fontsize=7)
out = os.path.join(BASE, "mass_budget.png")
fig.savefig(out, dpi=150)
print("Gespeichert:", out)

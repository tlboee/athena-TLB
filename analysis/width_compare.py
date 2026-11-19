"""Max (19.11.): besseres Breitenmass als r_eff = 2A/P.

Lokale Dicke: Distanztransformation dt der Maske; fuer jedes lokale Maximum (i, j) von dt wird ein Kreis mit Radius dt
gezeichnet, jede Zelle bekommt den groessten Radius eines Kreises, der sie enthaelt. Breite = massengewichteter Median
(rho*tr) dieser lokalen Dicke ueber die Maske. Runde Wolke -> R, Strom -> halbe Breite; duenne Fetzen zaehlen nur mit ihrer Masse.
Masken: tracer > 0.5 (Kern) und > 0.1 (inkl. Mischschicht). Vergleich mit 2A/P; t_cc,loc mit chi_loc, v aus stream_width.csv.
Cache: <lauf>/width_compare2.csv.   Ausgabe: runs/width_compare.png.   Aufruf:  %run ~/athena/analysis/width_compare.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
sys.path.insert(0, os.path.expanduser("~/athena/analysis"))
import g2_io
try:
    from skimage.measure import find_contours
except ImportError:
    find_contours = None

BASE = os.path.expanduser("~/athena/runs")
YR, R0, T_PERI, T_PASS = 3.156e7, 1.87e15, 18.0, 0.25
RUNS = [("adiabatic, whole orbit", "ps_ref_big_x2_t50", "#21918c"),
        ("adiabatic, fine", "ps_ref_big_x2_peri", "#0b5d55"),
        ("with cooling, whole orbit", "ps_ref_big_cool_x2_t50", "#c0392b")]

def wmedian(x, w):
    i = np.argsort(x); c = np.cumsum(w[i])
    return float(x[i][np.searchsorted(c, 0.5 * c[-1])])

def local_thickness(m):
    """Lokale Dicke in Zellen: groesster eingeschriebener Kreis, der die Zelle enthaelt."""
    dt = ndimage.distance_transform_edt(m)
    ridge = m & (dt >= ndimage.maximum_filter(dt, size=3))
    lt = np.zeros_like(dt)
    ii, jj = np.nonzero(ridge); rr = dt[ii, jj]
    for k in np.argsort(rr):                      # kleine Kreise zuerst, grosse ueberschreiben
        i, j, r = ii[k], jj[k], rr[k]; R = int(np.ceil(r))
        i0, i1, j0, j1 = max(i - R, 0), min(i + R + 1, m.shape[0]), max(j - R, 0), min(j + R + 1, m.shape[1])
        yy, xx = np.ogrid[i0 - i:i1 - i, j0 - j:j1 - j]
        disc = (yy**2 + xx**2 <= r * r) & m[i0:i1, j0:j1]
        sub = lt[i0:i1, j0:j1]; sub[disc] = np.maximum(sub[disc], r)
    return lt

def thickness(mask, w, dx):
    if mask.sum() < 20:
        return np.nan
    lt = local_thickness(mask)
    return wmedian(lt[mask], w[mask]) * dx

def two_a_over_p(m, dx):
    if m.sum() < 20:
        return np.nan
    if find_contours is not None:
        P = sum(np.sum(np.hypot(*np.diff(c, axis=0).T)) for c in find_contours(m.astype(float), 0.5)) * dx
    else:
        P = (np.sum(m[1:, :] != m[:-1, :]) + np.sum(m[:, 1:] != m[:, :-1])) * dx * np.pi / 4
    return 2 * m.sum() * dx * dx / P

def series(run):
    d = os.path.join(BASE, run); cache = os.path.join(d, "width_compare2.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for n in g2_io.snapshot_numbers(d):
        t, s = g2_io.load_snapshot(d, n)
        dx = float(s["x"][1] - s["x"][0]); tr = np.clip(s["r0"], 0, 1); w = s["rho"] * tr
        rows.append([t / YR, two_a_over_p(tr > 0.5, dx) / R0, thickness(tr > 0.5, w, dx) / R0, thickness(tr > 0.1, w, dx) / R0])
        print(f"  {run}: t = {t/YR:6.2f} yr", end="\r")
    print()
    df = pd.DataFrame(rows, columns=["t", "r_2AP", "h_05", "h_01"]); df.to_csv(cache, index=False)
    return df

fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)
print(f"{'Lauf':27s} {'t':>5s} {'2A/P':>6s} {'h(0.5)':>7s} {'h(0.1)':>7s} | t_cc,loc [yr]: {'2A/P':>6s} {'h(0.5)':>7s} {'h(0.1)':>7s}")
for lab, run, c in RUNS:
    if not os.path.isdir(os.path.join(BASE, run)):
        print(f"{run} fehlt"); continue
    w = series(run)
    sw = pd.read_csv(os.path.join(BASE, run, "stream_width.csv"))
    chi = np.interp(w["t"], sw["t"], sw["chi_loc"]); v = np.interp(w["t"], sw["t"], sw["v_kms"]) * 1e5
    tcc = {k: np.sqrt(chi) * w[k] * R0 / v / YR for k in ("r_2AP", "h_05", "h_01")}
    for k, ls, lw, nm in (("r_2AP", ":", 1.2, "2A/P (tracer > 0.5)"), ("h_05", "-", 1.6, "local thickness (tracer > 0.5)"),
                          ("h_01", "--", 1.6, "local thickness (tracer > 0.1)")):
        axes[0].semilogy(w["t"], w[k], color=c, ls=ls, lw=lw, label=f"{lab}: {nm}")
        axes[1].semilogy(w["t"], tcc[k], color=c, ls=ls, lw=lw, label=lab if k == "h_05" else None)
    for tt in (0, 10, 17, 18, 19, 25):
        i = int(np.argmin(np.abs(w["t"] - tt)))
        if abs(w["t"][i] - tt) > 0.3:
            continue
        print(f"{lab:27s} {w['t'][i]:5.1f} {w['r_2AP'][i]:6.2f} {w['h_05'][i]:7.2f} {w['h_01'][i]:7.2f} |"
              f" {tcc['r_2AP'][i]:20.3g} {tcc['h_05'][i]:7.3g} {tcc['h_01'][i]:7.3g}")
for ax in axes:
    ax.axvline(T_PERI, color="k", ls=":", lw=0.8); ax.set_xlabel("t [yr]"); ax.set_xlim(0, 50)
axes[0].axhline(1, color="0.6", lw=0.8); axes[0].set_ylabel("width / R0")
axes[0].set_title("Width: 2A/P vs. mass-weighted local thickness"); axes[0].legend(fontsize=7)
axes[1].axhline(T_PASS, color="0.5", lw=1); axes[1].set_ylabel("t_cc,loc [yr]")
axes[1].set_title("Local cloud-crushing time (dotted: 2A/P, solid: tracer > 0.5, dashed: tracer > 0.1)")
axes[1].legend(fontsize=9)
fig.savefig(os.path.join(BASE, "width_compare.png"), dpi=150)
plt.show()

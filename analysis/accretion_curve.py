"""Akkretionskurve aus der History-Datei: exakte Tracer-Abnahme, aufgeteilt nach Kanaelen.

Aufruf:  %run ~/athena/analysis/accretion_curve.py <run_dir>
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~/athena/vis/python"))
import athena_read

RUN     = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/athena/runs/g2_cart_hr_s12_acc")
YR      = 3.156e7
T_START = 1995.5
M_OBS   = 1.7e28          # beobachtete Wolkenmasse [g] (Burkert+12, Schartmann+12)
M_SUN   = 1.989e33
T_PERI  = 18.0

h = athena_read.hst(os.path.join(RUN, "g2_cart.hst"))
t = np.asarray(h["time"])

# Doppelte Zeiten nach Neustarts entfernen (jeweils letzte Zeile behalten)
_, idx = np.unique(t[::-1], return_index=True)
keep = np.sort(len(t) - 1 - idx)
t = t[keep] / YR
s = np.asarray(h["0-scalar"])[keep]
cnt = {}
for name in ["M_sink", "M_buffer", "M_reset"]:
    c = np.asarray(h[name])[keep].copy()
    # Falls ein Zaehler bei einem Neustart auf 0 zurueckgesetzt wurde: fortsetzen
    for i in range(1, len(c)):
        if c[i] < 0.5 * c[i-1] and c[i-1] > 0:
            print(f"Hinweis: {name} springt bei t = {t[i]:.2f} yr zurueck -> wird fortgesetzt")
            c[i:] += c[i-1] - c[i]
    cnt[name] = c

M0   = s[0]
loss = (M0 - s) / M0                                   # exakter Gesamtverlust (Anteil)
csum = cnt["M_sink"] + cnt["M_buffer"] + cnt["M_reset"]
with np.errstate(divide="ignore", invalid="ignore"):
    frac = {k: np.where(np.abs(csum) > 0, v / csum, 0.0) for k, v in cnt.items()}
acc  = {k: loss * frac[k] for k in cnt}                # Anteil der Anfangsmasse je Kanal

# Akkretionsrate (ueber 1 yr geglaettet), umgerechnet auf die beobachtete Wolkenmasse
dt_h  = np.median(np.diff(t))
w     = max(1, int(round(1.0 / dt_h)))
rate  = np.gradient(acc["M_sink"], t) * M_OBS / M_SUN      # M_sun / yr
rate_s = np.convolve(rate, np.ones(w) / w, mode="same")


def at(year, arr):
    return np.interp(year - T_START, t, arr)


print(f"Lauf: {RUN}")
print(f"History: {len(t)} Zeilen, t = {t[0]:.2f} ... {t[-1]:.2f} yr")
for year in [2013.5, 2015.5, 2025.0, 2040.0, 2060.0]:
    if year - T_START <= t[-1] + 1e-6:
        print(f"  {year:6.1f}: Verlust gesamt {at(year, loss)*100:5.1f} %   "
              f"Senke {at(year, acc['M_sink'])*100:5.1f} %   "
              f"Rand {at(year, acc['M_buffer'])*100:5.1f} %   "
              f"Zuruecksetzen {at(year, acc['M_reset'])*100:5.2f} %")
i0 = np.argmax(acc["M_sink"] > 1e-4)
if T_START + t[-1] >= 2025.0 and i0 > 0:
    mean_rate = (at(2025.0, acc["M_sink"]) - acc["M_sink"][i0]) * M_OBS / M_SUN / (2025.0 - T_START - t[i0])
    print(f"Beginn der Akkretion: {T_START + t[i0]:.1f}; mittlere Rate bis 2025: {mean_rate:.2e} M_sun/yr"
          f"  (Schartmann+12: ~6e-8)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
ax = axes[0]
ax.plot(T_START + t, loss * 100, "k", label="Gesamtverlust (exakt)")
ax.plot(T_START + t, acc["M_sink"] * 100, "C0", label="Senke (Akkretion)")
ax.plot(T_START + t, acc["M_buffer"] * 100, "C1", label="Rand")
ax.plot(T_START + t, acc["M_reset"] * 100, "C2", label="Zuruecksetzen")
ax.plot([2060], [40], "rs", label="Schartmann+12: ~40 % bis 2060")
ax.axvline(T_START + T_PERI, color="grey", ls="--", lw=0.8)
ax.set_xlabel("Jahr")
ax.set_ylabel("Anteil der Anfangsmasse [%]")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(T_START + t, rate_s, "C0", label="Akkretionsrate (1 yr gemittelt)")
ax.axhline(6e-8, color="r", ls=":", label="Schartmann+12: ~6e-8 (Mittel bis 2025)")
ax.axvline(T_START + T_PERI, color="grey", ls="--", lw=0.8)
ax.set_xlabel("Jahr")
ax.set_ylabel(r"$\dot M$ [M$_\odot$/yr]  (skaliert auf $1.7\times10^{28}$ g)")
ax.set_yscale("log")
ax.set_ylim(1e-9, 1e-6)
ax.legend(fontsize=8)

out = os.path.join(RUN, "plots", "accretion.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
np.savetxt(os.path.join(RUN, "accretion.csv"),
           np.column_stack([T_START + t, loss, acc["M_sink"], acc["M_buffer"], acc["M_reset"], rate_s]),
           header="year loss_total acc_sink acc_buffer acc_reset rate_Msun_per_yr", fmt="%.6e")
print("Gespeichert:", out)

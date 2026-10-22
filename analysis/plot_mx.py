"""m_X(t): Wolkenmasse mit Tracer > X ueber die Zeit, aus disruption.csv.

Aufruf:  python3 analysis/plot_mx.py <run_dir> [<run_dir> ...]
         (Standard: Sicherung des Referenzlaufs auf dem Mac)
"""
import os, sys
import pandas as pd
import matplotlib.pyplot as plt

T_START = 1995.5
runs = [os.path.expanduser(a) for a in sys.argv[1:]] or \
       [os.path.expanduser("~/athena_backup/runs/g2_cart_hr_s12_acc")]

for run in runs:
    df = pd.read_csv(os.path.join(run, "disruption.csv"))
    cols = [c for c in df.columns if c.startswith("M_tr>")]
    year = T_START + df["t_yr"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, log in zip(axes, [False, True]):
        ax.plot(year, df["M_tot"], "k", lw=1.5, label="Tracer-Masse gesamt")
        for c in cols:
            ax.plot(year, df[c], label="tr > " + c[len("M_tr>"):])
        for Y, ls in [(0.5, "--"), (1/3, "-."), (0.1, ":")]:
            ax.axhline(Y, color="grey", ls=ls, lw=0.8)
        ax.axvline(2013.5, color="grey", lw=0.8)
        ax.set_xlabel("Jahr")
        ax.set_ylabel(r"$m_X(t)\,/\,m_{\rm tot}(0)$")
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1.5)
        else:
            ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{os.path.basename(run.rstrip('/'))}: Wolkenmasse mit Tracer > X")
    out = os.path.join(run, "m_X.png")
    fig.savefig(out, dpi=150)
    print("Gespeichert:", out)

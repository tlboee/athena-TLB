"""Parameterstudie G2/CC01: erzeugt und schickt die Laeufe als Slurm-Jobs ab.

Jeder Lauf weicht in genau einem Parameter von CC01 ab (Schartmann-Normierung,
athinput.g2_cart_cc01_s12). Drei Abschnitte wie im Referenzlauf:
  A: PPM, CFL 0.3 bis zum Perizentrum
  B: PPM, CFL 0.2 bis 2 yr nach dem Perizentrum
  C: PLM, CFL 0.3 bis T_END

Aufruf (Cluster, im Repo):
  python3 jobs/param_study.py --dry-run          # nur anzeigen
  python3 jobs/param_study.py                    # alle Laeufe abschicken
  python3 jobs/param_study.py ps_chi10 ps_rp8    # nur ausgewaehlte Laeufe
"""
import argparse, datetime, os, subprocess

HOME    = os.path.expanduser("~")
ATHENA  = os.path.join(HOME, "athena")
BINARY  = os.path.join(ATHENA, "bin", "athena_g2_cart")
INPUT   = os.path.join(ATHENA, "inputs", "g2", "athinput.g2_cart_cc01_s12")
RUNS    = os.path.join(ATHENA, "runs")
LOGS    = os.path.join(ATHENA, "logs")
MODULES = "compiler/gnu/14.2 mpi/openmpi/5.0-gnu-14.2 lib/hdf5/1.14-gnu-14.2-openmpi-5.0"
YR      = 3.156e7
T_END   = 100.0      # yr seit 1995.5 (CC01 erreicht das 1/3-Kriterium nach ~76 yr)
T_HOT0  = 2.818e7    # K, Atmosphaere am Startort der Wolke

# Gemeinsame Einstellungen fuer alle Laeufe (512 x 384: Kern und Akkretion konvergiert)
COMMON = {
    "mesh/nx1": 512, "mesh/nx2": 384, "meshblock/nx1": 64, "meshblock/nx2": 64,
    "output1/dt": 3.156e7,
}
NTASKS = 48          # 512 x 384 / (64 x 64) = 48 Meshblocks


def chi_run(chi):
    """Dichtekontrast ueber die Wolkentemperatur im Druckgleichgewicht."""
    T_c = T_HOT0 / chi
    p = {"problem/rho_cloud": 0, "problem/T_cloud": f"{T_c:.4g}"}
    if T_c < 1e4:
        p["problem/T_floor"] = f"{T_c:.4g}"
    return {"params": p, "t_peri": 18.0, "info": f"chi = {chi:g}, T_cloud = {T_c:.3g} K"}


def rp_run(vx, vy, t_peri, rp):
    return {"params": {"problem/vx_cloud": vx, "problem/vy_cloud": vy}, "t_peri": t_peri,
            "info": f"Perizentrum {rp:.1e} cm"}


def size_run(factor):
    return {"params": {"problem/R_cloud": f"{1.87e15 * factor:.4g}"}, "t_peri": 18.0,
            "info": f"R = {factor:g} x R_CC01"}


# --- Die Laeufe: Werte nach dem Treffen mit Max hier anpassen ---------------------------
STUDY = {
    "ps_ref":     {"params": {}, "t_peri": 18.0, "info": "CC01 (Referenz bei 512 x 384)"},
    "ps_chi10":   chi_run(10),
    "ps_chi100":  chi_run(100),
    "ps_chi1000": chi_run(1000),
    "ps_chi1e4":  chi_run(1e4),
    "ps_rp8":     rp_run(826.88, 153.94, 19.00, 8e15),
    "ps_rp16":    rp_run(870.88, 297.69, 20.60, 1.6e16),
    "ps_rp32":    rp_run(934.48, 505.46, 21.41, 3.2e16),
    "ps_R2":      size_run(2),
    "ps_R4":      size_run(4),
}
# ------------------------------------------------------------------------------------------


def overrides(d):
    return " ".join(f"{k}={v}" for k, v in d.items())


def commands(name, run):
    """Die drei Athena++-Aufrufe fuer einen Lauf, verknuepft mit &&."""
    t1, t2 = run["t_peri"] * YR, (run["t_peri"] + 2.0) * YR
    a = (f"mpirun -np $SLURM_NTASKS {BINARY} -i {INPUT} -d . "
         f"{overrides(COMMON)} {overrides(run['params'])} time/tlim={t1:.6g}")
    b = f"mpirun -np $SLURM_NTASKS {BINARY} -r g2_cart.final.rst -d . time/cfl_number=0.2 time/tlim={t2:.6g}"
    c = (f"mpirun -np $SLURM_NTASKS {BINARY} -r g2_cart.final.rst -d . "
         f"time/xorder=2 time/cfl_number=0.3 time/tlim={T_END * YR:.6g}")
    return [a, b, c]


def write_run_info(run_dir, name, run, cmds):
    head = subprocess.run(["git", "-C", ATHENA, "log", "-1", "--format=%h %s"],
                          capture_output=True, text=True).stdout.strip()
    with open(os.path.join(run_dir, "RUN_INFO.md"), "w") as f:
        f.write(f"# RUN_INFO: {name} (Parameterstudie)\n\n{run['info']}\n\n")
        f.write(f"- Abgeschickt: {datetime.date.today()}\n- Repo-HEAD: `{head}`\n")
        f.write(f"- Input: `{INPUT}`\n- Gemeinsam: `{overrides(COMMON)}`\n")
        f.write(f"- Abweichung von CC01: `{overrides(run['params']) or '-'}`\n")
        f.write(f"- Abschnitte: PPM/CFL 0.3 bis {run['t_peri']} yr, PPM/CFL 0.2 bis "
                f"{run['t_peri'] + 2} yr, PLM/CFL 0.3 bis {T_END} yr\n\n## Befehle\n\n")
        for c in cmds:
            f.write(f"    {c}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="Laeufe (Standard: alle)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    names = args.names or list(STUDY)
    for name in names:
        run = STUDY[name]
        run_dir = os.path.join(RUNS, name)
        cmds = commands(name, run)
        wrap = (f"module purge; module load {MODULES}; cd {run_dir} && "
                + " && ".join(c.replace("$SLURM_NTASKS", "\\$SLURM_NTASKS") for c in cmds))
        sb = ["sbatch", "--partition=dev_cpu_il", "--time=00:30:00", "--nodes=1",
              f"--ntasks={NTASKS}", "--mem=16gb", f"--job-name={name}",
              f"--output={LOGS}/{name}_%j.out", f"--wrap={wrap}"]
        print(f"\n== {name}: {run['info']}")
        if args.dry_run:
            for c in cmds:
                print("   ", c)
            continue
        if os.path.exists(run_dir):
            print(f"   {run_dir} existiert schon -> uebersprungen")
            continue
        os.makedirs(run_dir)
        write_run_info(run_dir, name, run, cmds)
        print("  ", subprocess.run(sb, capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    main()

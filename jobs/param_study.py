"""Parameterstudie G2: Gitter aus Dichtekontrast chi und Perizentrumsabstand r_p.

Basis: CC01 (Schartmann-Normierung, inputs/g2/athinput.g2_cart_cc01_s12), 512 x 384.
chi ueber die Wolkentemperatur im Druckgleichgewicht: T_cloud = T_hot(Start) / chi.
r_p ueber die tangentiale Startgeschwindigkeit (Startort und radiale Geschwindigkeit wie CC01).
Drei Abschnitte wie im Referenzlauf: PPM/CFL 0.3 bis t_peri, PPM/CFL 0.2 bis t_peri + 2 yr,
danach PLM/CFL 0.3 bis T_END.

Aufruf (im Repo):
  python3 jobs/param_study.py --list                 Laeufe auflisten
  python3 jobs/param_study.py --preflight            Vorabpruefung (auch auf dem Mac)
  python3 jobs/param_study.py --dry-run              Job-Befehle anzeigen
  python3 jobs/param_study.py --tend 25 'g_*rp16'    Pilot bis 25 yr (eigene Ordner *_t25)
  python3 jobs/param_study.py                        alle Laeufe abschicken (Cluster)
Namen koennen Muster sein ('g_chi1e*'); ohne Namen: alle.
"""
import argparse, datetime, fnmatch, json, math, os, subprocess

HOME    = os.path.expanduser("~")
ATHENA  = os.path.join(HOME, "athena")
BINARY  = os.path.join(ATHENA, "bin", "athena_g2_cart")
INPUT   = os.path.join(ATHENA, "inputs", "g2", "athinput.g2_cart_cc01_s12")
RUNS    = os.path.join(ATHENA, "runs")
LOGS    = os.path.join(ATHENA, "logs")
MODULES = "compiler/gnu/14.2 mpi/openmpi/5.0-gnu-14.2 lib/hdf5/1.14-gnu-14.2-openmpi-5.0"

YR, KM  = 3.156e7, 1.0e5
GM      = 6.674e-8 * 4.31e6 * 1.989e33
X0, Y0  = -7.22e16, 2.21e16          # Startort CC01
VX0, VY0 = 794.59, 48.45             # km/s, CC01
R_CLOUD = 1.87e15
R_IN    = 1.9e15
T_HOT0  = 2.818e7                    # K, Atmosphaere am Startort
T_END   = 100.0                      # yr seit 1995.5
SCHEME  = "ppm"     # "ppm": PPM/CFL 0.3 durchgehend (mit Reparatur im pgen); "plm"; "mixed" (alt)
RUN_TIMEOUT = 1500  # s pro mpirun-Aufruf; bei Stillstand wird nur dieser Lauf abgebrochen
# Groessere Domain fuer das Gitter (Bahnen bis r_p = 3.2e16 cm), gleiche Zellgroesse wie
# 512 x 384 in der CC01-Domain: dx = 1.42e17 / 512 = 2.7734e14 cm
NX1, NX2, NTASKS = 768, 896, 56      # 12 x 14 = 168 Meshblocks zu 64 x 64, 3 pro Prozess
DOMAIN  = {"x1min": -1.53e17, "x1max": 6.0e16, "x2min": -2.0e17, "x2max": 4.85e16}
COMMON  = {"mesh/nx1": NX1, "mesh/nx2": NX2, "meshblock/nx1": 64, "meshblock/nx2": 64,
           **{f"mesh/{k}": f"{v:.6g}" for k, v in DOMAIN.items()},
           "output1/dt": 3.156e7, "output2/dt": 3.156e8}   # Restart alle 10 yr

CHI_GRID = [10, 100, 1000, 1e4]
RP_GRID  = [4e15, 8e15, 1.6e16, 3.2e16]


# ---------------------------------------------------------------- Bahn ------------------
def orbit(vx, vy):
    """Keplerelemente aus Startort (X0, Y0) und Geschwindigkeit (vx, vy) in km/s."""
    ux, uy = vx * KM, vy * KM
    r = math.hypot(X0, Y0)
    E = 0.5 * (ux * ux + uy * uy) - GM / r
    h = X0 * uy - Y0 * ux
    a = -GM / (2 * E)
    ex, ey = uy * h / GM - X0 / r, -ux * h / GM - Y0 / r
    e = math.hypot(ex, ey)
    n = math.sqrt(GM / a**3)
    Ean = math.acos(max(-1.0, min(1.0, (1 - r / a) / e)))
    M = Ean - e * math.sin(Ean)
    inbound = X0 * ux + Y0 * uy < 0
    t_peri = M / n if inbound else 2 * math.pi / n - M / n
    return {"a": a, "e": e, "rp": a * (1 - e), "ra": a * (1 + e), "P_yr": 2 * math.pi / n / YR,
            "t_peri_yr": t_peri / YR, "ex": ex, "ey": ey}


_R0 = math.hypot(X0, Y0)
_RH = (X0 / _R0, Y0 / _R0)
_VR = VX0 * _RH[0] + VY0 * _RH[1]
_VT = (VX0 - _VR * _RH[0], VY0 - _VR * _RH[1])


def velocity_for_rp(rp):
    """Tangentiale Startgeschwindigkeit so skalieren, dass das Perizentrum rp ist (Bisektion)."""
    def v(f):
        return _VR * _RH[0] + f * _VT[0], _VR * _RH[1] + f * _VT[1]
    lo, hi = 0.05, 3.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if orbit(*v(mid))["rp"] > rp:
            hi = mid
        else:
            lo = mid
    vx, vy = v(0.5 * (lo + hi))
    return round(vx, 2), round(vy, 2)


def min_margin_to_domain(orb, dom):
    """Kleinster Abstand der vollstaendigen Bahnellipse zum Domainrand (negativ = ausserhalb)."""
    w = math.atan2(orb["ey"], orb["ex"])
    p = orb["a"] * (1 - orb["e"] ** 2)
    m = float("inf")
    for k in range(720):
        th = 2 * math.pi * k / 720
        rr = p / (1 + orb["e"] * math.cos(th))
        x, y = rr * math.cos(th + w), rr * math.sin(th + w)
        m = min(m, x - dom["x1min"], dom["x1max"] - x, y - dom["x2min"], dom["x2max"] - y)
    return m


def path_check(vx, vy, t_end, dom, buf, n=4000):
    """Keplerbahn bis t_end: kleinster Abstand zum Rand (minus Puffer) und Austrittszeit."""
    ux, uy = vx * KM, vy * KM
    r = math.hypot(X0, Y0)
    E = 0.5 * (ux * ux + uy * uy) - GM / r
    h = X0 * uy - Y0 * ux
    a = -GM / (2 * E)
    ex, ey = uy * h / GM - X0 / r, -ux * h / GM - Y0 / r
    e, w = math.hypot(ex, ey), math.atan2(ey, ex)
    nn = math.sqrt(GM / a**3)
    Ea = math.acos(max(-1.0, min(1.0, (1 - r / a) / e)))
    M0 = -(Ea - e * math.sin(Ea)) if X0 * ux + Y0 * uy < 0 else Ea - e * math.sin(Ea)
    s = 1.0 if h > 0 else -1.0
    margin, t_exit = float("inf"), None
    for k in range(n + 1):
        t = t_end * k / n
        M = M0 + nn * t * YR
        Ek = M + 0.85 * e * math.copysign(1.0, math.sin(M))
        for _ in range(50):
            Ek -= (Ek - e * math.sin(Ek) - M) / (1 - e * math.cos(Ek))
        nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(Ek / 2), math.sqrt(1 - e) * math.cos(Ek / 2))
        rr = a * (1 - e * math.cos(Ek))
        x, y = rr * math.cos(w + s * nu), rr * math.sin(w + s * nu)
        m = min(x - dom["x1min"], dom["x1max"] - x, y - dom["x2min"], dom["x2max"] - y) - buf
        margin = min(margin, m)
        if m < 0 and t_exit is None:
            t_exit = t
    return margin, t_exit


def read_mesh(path):
    vals, block = {}, None
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("<"):
            block = line.strip("<>").strip()
        elif block == "mesh" and "=" in line:
            k, v = (t.strip() for t in line.split("=", 1))
            vals[k] = v
    return {k: float(vals[k]) for k in ("x1min", "x1max", "x2min", "x2max")}


# ---------------------------------------------------------------- Laeufe ----------------
def fmt_chi(chi):
    return {10: "10", 100: "100", 1000: "1000", 1e4: "1e4"}.get(chi, f"{chi:g}")


def make_runs():
    runs = {}
    base = {"vx": VX0, "vy": VY0, "chi": None, "rp": None}
    runs["ps_ref"] = dict(base, params={}, info="CC01 (Tabellendichte), Baseline")
    runs["ps_ref_plm"] = dict(base, params={}, scheme="plm",
                              info="Kontrolle: Baseline mit PLM durchgehend")
    runs["ps_ref_mixed"] = dict(base, params={}, scheme="mixed",
                                info="Kontrolle: Baseline mit altem Schema (PPM bis t_peri, dann PLM)")
    runs["ps_ref_cfl025"] = dict(base, params={"time/cfl_number": 0.25},
                                 info="Wiederholung: CFL 0.25 im ersten Abschnitt")
    runs["ps_ref_shift"] = dict(base, params={"problem/x_cloud": f"{X0 + 1.39e14:.6g}"},
                                info="Wiederholung: Wolke um eine halbe Zelle verschoben")
    for chi in CHI_GRID:
        T_c = T_HOT0 / chi
        for rp in RP_GRID:
            vx, vy = (VX0, VY0) if rp == 4e15 else velocity_for_rp(rp)
            p = {"problem/rho_cloud": 0, "problem/T_cloud": f"{T_c:.4g}",
                 "problem/vx_cloud": vx, "problem/vy_cloud": vy}
            if T_c < 1e4:
                p["problem/T_floor"] = f"{T_c:.4g}"
            name = f"g_chi{fmt_chi(chi)}_rp{rp / 1e15:g}"
            runs[name] = {"vx": vx, "vy": vy, "chi": chi, "rp": rp, "params": p,
                          "info": f"chi = {chi:g} (T_cloud = {T_c:.3g} K), r_p = {rp:.2g} cm"}
    for run in runs.values():
        run["orbit"] = orbit(run["vx"], run["vy"])
        run["t_peri"] = round(run["orbit"]["t_peri_yr"], 2)
    return runs


def ov(d):
    return " ".join(f"{k}={v}" for k, v in d.items())


def commands(run, t_end):
    t1, t2 = run["t_peri"] * YR, (run["t_peri"] + 2.0) * YR
    mp = f"mpirun -np $SLURM_NTASKS {BINARY}"
    scheme = run.get("scheme", SCHEME)
    if scheme in ("ppm", "plm"):
        xo = 3 if scheme == "ppm" else 2
        return [f"{mp} -i {INPUT} -d . {ov(COMMON)} time/xorder={xo} time/cfl_number=0.3 "
                f"{ov(run['params'])} time/tlim={t_end * YR:.6g}"]
    return [f"{mp} -i {INPUT} -d . {ov(COMMON)} {ov(run['params'])} time/tlim={t1:.6g}",
            f"{mp} -r g2_cart.final.rst -d . time/cfl_number=0.2 time/tlim={t2:.6g}",
            f"{mp} -r g2_cart.final.rst -d . time/xorder=2 time/cfl_number=0.3 time/tlim={t_end * YR:.6g}"]


def preflight(name, run, dom, t_end):
    """Liste von (ok, Text) fuer einen Lauf."""
    o, p, out = run["orbit"], run["params"], []
    T_c = float(p.get("problem/T_cloud", 1e4))
    T_f = float(p.get("problem/T_floor", 1e4))
    out.append((T_f <= T_c * 1.0001, f"T_floor {T_f:.3g} K <= T_cloud {T_c:.3g} K"))
    dx = (dom["x1max"] - dom["x1min"]) / NX1
    x0 = float(p.get("problem/x_cloud", X0))
    inside = (x0 - R_CLOUD > dom["x1min"] + 2 * dx and x0 + R_CLOUD < dom["x1max"] - 2 * dx and
              Y0 - R_CLOUD > dom["x2min"] + 2 * dx and Y0 + R_CLOUD < dom["x2max"] - 2 * dx)
    out.append((inside, f"Wolke innerhalb Domain und Puffer (R = {R_CLOUD / dx:.1f} Zellen)"))
    out.append((o["rp"] > 2 * R_IN, f"r_p = {o['rp']:.2e} cm > 2 r_in"))
    mm, t_exit = path_check(run["vx"], run["vy"], t_end, dom, 2 * dx)
    if t_exit is not None:
        out.append((False, f"Keplerbahn verlaesst die Domain bei t = {t_exit:.1f} yr"))
    else:
        out.append((mm > 1e16, f"Keplerbahn bis {t_end:g} yr in der Domain, Abstand min. "
                               f"{mm / 1e16:.2f} e16 cm" + ("" if mm > 1e16 else " (knapp)")))
    nb = (NX1 // 64) * (NX2 // 64)
    out.append((nb % NTASKS == 0, f"{nb} Meshblocks auf {NTASKS} Prozesse ({nb / NTASKS:g} pro Prozess)"))
    return out


# ---------------------------------------------------------------- Hauptprogramm ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="Laeufe oder Muster (Standard: alle)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tend", type=float, default=T_END, help="Endzeit in yr (Pilot: z. B. 25)")
    ap.add_argument("--bundle", type=int, default=7, help="Laeufe pro Slurm-Job")
    ap.add_argument("--partition", default="cpu_il")
    ap.add_argument("--time", default="02:00:00")
    args = ap.parse_args()

    runs = make_runs()
    names = [n for n in runs if not args.names or any(fnmatch.fnmatch(n, pat) for pat in args.names)]
    dom = dict(DOMAIN)
    suffix = "" if args.tend == T_END else f"_t{args.tend:g}"

    if args.list or args.preflight:
        for n in names:
            r, o = runs[n], runs[n]["orbit"]
            print(f"{n:22s} vx={r['vx']:8.2f} vy={r['vy']:7.2f}  e={o['e']:.3f} r_p={o['rp']:.2e} "
                  f"t_peri={r['t_peri']:6.2f} yr  P={o['P_yr']:6.1f} yr  | {r['info']}")
            if args.preflight:
                for ok, txt in preflight(n, r, dom, args.tend):
                    print(f"      {'ok  ' if ok else 'WARN'} {txt}")
        return

    todo = []
    for n in names:
        run_dir = os.path.join(RUNS, n + suffix)
        if os.path.exists(run_dir):
            print(f"{n + suffix}: Ordner existiert schon -> uebersprungen")
            continue
        todo.append((n + suffix, run_dir, runs[n], commands(runs[n], args.tend)))

    head = subprocess.run(["git", "-C", ATHENA, "log", "-1", "--format=%h %s"],
                          capture_output=True, text=True).stdout.strip()
    for i in range(0, len(todo), args.bundle):
        group = todo[i:i + args.bundle]
        parts = [f"( cd {d} && " + " && ".join(f"timeout {RUN_TIMEOUT} {x}" for x in c) + " )"
                 for _, d, _, c in group]
        wrap = f"module purge; module load {MODULES}; " + "; ".join(parts)
        jobname = f"{group[0][0]}+{len(group) - 1}" if len(group) > 1 else group[0][0]
        sb = ["sbatch", f"--partition={args.partition}", f"--time={args.time}", "--nodes=1",
              f"--ntasks={NTASKS}", "--mem=16gb", f"--job-name={jobname}",
              f"--output={LOGS}/{jobname}_%j.out", f"--wrap={wrap}"]
        print(f"\n== Job {i // args.bundle + 1}: " + ", ".join(g[0] for g in group))
        if args.dry_run:
            for name, d, r, c in group:
                print(f"   {name}: {r['info']}")
                for cmd in c:
                    print(f"      {cmd}")
            continue
        for name, d, r, c in group:
            os.makedirs(d)
            manifest = {"name": name, "info": r["info"], "date": str(datetime.date.today()),
                        "git": head, "input": INPUT, "binary": BINARY, "ntasks": NTASKS,
                        "common": COMMON, "params": r["params"], "chi": r["chi"], "rp_target": r["rp"],
                        "vx_kms": r["vx"], "vy_kms": r["vy"], "orbit": r["orbit"],
                        "t_peri_yr": r["t_peri"], "t_end_yr": args.tend,
                        "scheme": {"ppm": "PPM/CFL0.3 durchgehend, Reparatur im pgen", "plm": "PLM/CFL0.3 durchgehend",
                                   "mixed": "PPM/CFL0.3 bis t_peri; PPM/CFL0.2 bis t_peri+2; PLM/CFL0.3 danach"}[r.get("scheme", SCHEME)],
                        "commands": c}
            with open(os.path.join(d, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
            with open(os.path.join(d, "RUN_INFO.md"), "w") as f:
                f.write(f"# RUN_INFO: {name}\n\n{r['info']}\n\n- Abgeschickt: {manifest['date']}\n"
                        f"- Repo-HEAD: `{head}`\n- Abweichung von CC01: `{ov(r['params']) or '-'}`\n"
                        f"- Perizentrum nach {r['t_peri']} yr; Ende {args.tend} yr\n"
                        f"- Details: `manifest.json`\n")
        res = subprocess.run(sb, capture_output=True, text=True)
        print("  ", (res.stdout or res.stderr).strip())


if __name__ == "__main__":
    main()

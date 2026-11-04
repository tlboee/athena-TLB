"""Tabelle aller Laeufe (Anhang A): Parameter aus manifest.json, erreichte Zeit und Reparaturen aus g2_cart.hst.
Ausgabe: runs/run_table.csv, runs/run_table.md, runs/run_table.tex.  Aufruf:  python3 analysis/run_table.py
"""
import glob, json, os, re
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/athena/runs")
YR, R0 = 3.156e7, 1.87e15

def num(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

rows = []
for d in sorted(glob.glob(os.path.join(BASE, "*"))):
    name = os.path.basename(d)
    hst = os.path.join(d, "g2_cart.hst")
    if not os.path.isdir(d) or "UNGUELTIG" in name or not os.path.exists(hst):
        continue
    man = {}
    if os.path.exists(os.path.join(d, "manifest.json")):
        with open(os.path.join(d, "manifest.json")) as f:
            man = json.load(f)
    p = dict(man.get("params", {}) or {})
    com = man.get("common", {})
    if isinstance(com, dict):
        p = {**com, **p}
    elif isinstance(com, (list, str)):
        for kv in (com.split() if isinstance(com, str) else com):
            if "=" in str(kv):
                k, v = str(kv).split("=", 1)
                p.setdefault(k, v)
    try:
        last = np.loadtxt(hst, ndmin=2)[-1]
        t_end = last[0] / YR
        reps = int(last[16] + last[17] + last[18]) if len(last) > 18 else np.nan
    except Exception:
        t_end, reps = np.nan, np.nan
    is3d = name.startswith("d3_")
    nx1 = int(num(p.get("mesh/nx1"), 512 if is3d else 768))
    nx2 = int(num(p.get("mesh/nx2"), 384 if is3d else 896))
    grid = f"{nx1} x {nx2}" + (f" x {int(num(p.get('mesh/nx3'), 144))}" if is3d else "")
    xos = []
    for c in man.get("commands", []) or []:
        found = re.findall(r"time/xorder=(\d)", c)
        if found:
            xos.append(int(found[-1]))
    if not xos:
        xos = [int(num(p.get("time/xorder"), 3 if "_ppm" in name else 2))]
    xos = [x for i, x in enumerate(xos) if i == 0 or x != xos[i - 1]]
    recon = "→".join({2: "PLM", 3: "PPM"}.get(x, str(x)) for x in xos)
    m = re.search(r"rp(\d+)", name)
    rp = float(m.group(1)) * 1e15 if m else 4e15
    rp = num((man.get("orbit") or {}).get("rp"), rp)
    chi = man.get("chi", np.nan)
    if not np.isfinite(num(chi)) and (name.startswith("ps_ref") or name.startswith("d3_cc01")):
        chi = 2755.0
    cooling = "_cool" in name or str(p.get("problem/cooling", "")).lower() in ("true", "1")
    lam = num(p.get("problem/Lambda_fac"), 1.0 if cooling else np.nan)
    used = not (name.startswith(("dbg_", "rep_", "g2_cart_")) or name.endswith("_t25") or name == "d3_cc01_test")
    rows.append(dict(
        run=name, used=used, info=man.get("info", ""),
        chi=num(chi), r_p=rp, R_over_R0=num(p.get("problem/R_cloud"), R0) / R0,
        grid=grid, recon=recon,
        cooling="yes" if cooling else "no", Lambda_fac=lam,
        t_lim_yr=num(p.get("time/tlim")) / YR, t_end_yr=round(t_end, 1), repairs=reps,
    ))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(BASE, "run_table.csv"), index=False)
with open(os.path.join(BASE, "run_table.md"), "w") as f:
    md = df.drop(columns=["info"]).astype(object).where(df.drop(columns=["info"]).notna(), "--")
    f.write("| " + " | ".join(md.columns) + " |\n|" + "---|" * len(md.columns) + "\n")
    for _, r in md.iterrows():
        f.write("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in r) + " |\n")
cols = ["run", "chi", "r_p", "R_over_R0", "grid", "recon", "cooling", "Lambda_fac", "t_end_yr"]
with open(os.path.join(BASE, "run_table.tex"), "w") as f:
    f.write(df[df["used"]][cols].to_latex(index=False, float_format="%.3g", na_rep="--", escape=True,
                              caption="Overview of all simulations.", label="tab:runs"))
print(f"{len(df)} Laeufe")
print(df[["run", "used", "chi", "r_p", "R_over_R0", "grid", "recon", "cooling", "Lambda_fac", "t_end_yr", "repairs"]].to_string(index=False))

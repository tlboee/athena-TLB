"""Einlesen von Athena++-Snapshots der G2-Laeufe.

HDF5 (.athdf, ein File pro Snapshot) wird bevorzugt; aeltere Laeufe mit .tab
(ein File pro Meshblock) werden weiterhin gelesen.
"""
import glob, os, re, sys
import numpy as np

VIS_PY = os.path.expanduser("~/athena/vis/python")
if VIS_PY not in sys.path:
    sys.path.insert(0, VIS_PY)

PREFIX = "g2_cart"
YR = 3.156e7
_TIME_RE = re.compile(r"time=([0-9.eE+-]+)")


def snapshot_numbers(run_dir, prefix=PREFIX):
    """Sortierte Liste der vorhandenen Snapshot-Nummern."""
    h5 = glob.glob(os.path.join(run_dir, f"{prefix}.out1.*.athdf"))
    if h5:
        return sorted(int(f.split(".")[-2]) for f in h5)
    tab = glob.glob(os.path.join(run_dir, f"{prefix}.block0.out1.*.tab"))
    return sorted(int(f.split(".")[-2]) for f in tab)


def snapshot_time(run_dir, n, prefix=PREFIX):
    """Simulationszeit von Snapshot n in Sekunden."""
    h5 = os.path.join(run_dir, f"{prefix}.out1.{n:05d}.athdf")
    if os.path.exists(h5):
        import h5py
        with h5py.File(h5, "r") as f:
            return float(f.attrs["Time"])
    tab = os.path.join(run_dir, f"{prefix}.block0.out1.{n:05d}.tab")
    with open(tab) as f:
        return float(_TIME_RE.search(f.readline()).group(1))


def load_snapshot(run_dir, n, prefix=PREFIX):
    """Liest Snapshot n. Rueckgabe (t, g): t in s, g[name][j, i] als 2D-Array (y, x)
    fuer rho, press, vx, vy, vz, r0 sowie 1D-Zellmitten g['x'], g['y']."""
    h5 = os.path.join(run_dir, f"{prefix}.out1.{n:05d}.athdf")
    if os.path.exists(h5):
        import athena_read
        d = athena_read.athdf(h5)
        g = {out: np.asarray(d[name][0], dtype=float) for out, name in
             [("rho", "rho"), ("press", "press"), ("vx", "vel1"), ("vy", "vel2"),
              ("vz", "vel3"), ("r0", "r0")]}
        g["x"], g["y"] = np.asarray(d["x1v"]), np.asarray(d["x2v"])
        return float(d["Time"]), g

    import pandas as pd
    files = sorted(glob.glob(os.path.join(run_dir, f"{prefix}.block*.out1.{n:05d}.tab")))
    if not files:
        raise FileNotFoundError(f"Snapshot {n} nicht gefunden in {run_dir}")
    t = snapshot_time(run_dir, n, prefix)
    d = pd.concat([pd.read_csv(f, sep=r"\s+", comment="#", header=None)
                   for f in files]).to_numpy()
    # Spalten: 0 i, 1 x, 2 j, 3 y, 4 rho, 5 press, 6 vx, 7 vy, 8 vz, 9 r0
    x, y = np.unique(d[:, 1]), np.unique(d[:, 3])
    i, j = np.searchsorted(x, d[:, 1]), np.searchsorted(y, d[:, 3])
    g = {"x": x, "y": y}
    for k, name in enumerate(["rho", "press", "vx", "vy", "vz", "r0"], start=4):
        a = np.full((len(y), len(x)), np.nan)
        a[j, i] = d[:, k]
        g[name] = a
    return t, g


def nearest_snapshots(run_dir, times_yr, prefix=PREFIX):
    """Snapshot-Nummern, deren Zeit den gewuenschten Zeiten (in Jahren) am naechsten liegt."""
    nums = snapshot_numbers(run_dir, prefix)
    if len(nums) < 2:
        return nums
    t0 = snapshot_time(run_dir, nums[0], prefix)
    dt = (snapshot_time(run_dir, nums[1], prefix) - t0) / (nums[1] - nums[0])
    return [min(max(int(round((ty * YR - t0) / dt)) + nums[0], nums[0]), nums[-1])
            for ty in times_yr]

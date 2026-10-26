"""Film aus den Snapshots eines G2-Laufs: Dichte und Tracer nebeneinander.

Aufruf (JupyterHub-Notebook):
  %run ~/athena/analysis/make_movie.py <run_dir> [--fps 6] [--zoom=x0,x1,y0,y1] [--out name]
  --zoom in Einheiten von 1e16 cm, z.B. --zoom=-4,1.5,-2.5,3 (mit "=", weil der Wert mit "-" beginnt) fuer das Perizentrum
"""
import argparse, json, os, shutil, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g2_io

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--fps", type=float, default=6)
p.add_argument("--zoom", default=None, help="x0,x1,y0,y1 in 1e16 cm")
p.add_argument("--out", default=None, help="Dateiname ohne Endung")
p.add_argument("--dpi", type=int, default=150)
p.add_argument("--logtr", action="store_true", help="Tracer logarithmisch (1e-3 ... 1)")
p.add_argument("--contrast", action="store_true", help="Dichte relativ zur Gleichgewichtsatmosphaere")
a = p.parse_args()

RUN     = os.path.expanduser(a.run)
YR      = g2_io.YR
T_START = 1995.5
nums    = g2_io.snapshot_numbers(RUN)
MF      = os.path.join(RUN, "manifest.json")
MAN     = json.load(open(MF)) if os.path.exists(MF) else None
LABEL   = (MAN["name"] + ": " + MAN["info"].split(" |")[0]) if MAN else "CC01"
VX, VY  = (MAN["vx_kms"], MAN["vy_kms"]) if MAN else (794.59, 48.45)
CHI     = (MAN["chi"] or 2755.0) if MAN else 2755.0       # Baseline: Tabellendichte
print(f"{len(nums)} Snapshots in {RUN}")


def kepler_orbit(t_end, n=50000):
    """Ballistische Bahn aus Burkert+12 Tabelle 1 (Leapfrog-Integration)."""
    GM  = 6.674e-8 * 4.31e6 * 1.989e33
    pos = np.array([-7.22e16, 2.21e16])
    vel = np.array([VX * 1e5, VY * 1e5])
    dt  = t_end / n
    path = [pos.copy()]
    for _ in range(n):
        vel += 0.5 * dt * (-GM * pos / np.linalg.norm(pos)**3)
        pos += dt * vel
        vel += 0.5 * dt * (-GM * pos / np.linalg.norm(pos)**3)
        path.append(pos.copy())
    return np.array(path) / 1e16


def get_writer(fps):
    """MP4 ueber ffmpeg (System oder imageio-ffmpeg), sonst GIF."""
    ff = shutil.which("ffmpeg")
    if ff is None:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ff = None
    if ff:
        plt.rcParams["animation.ffmpeg_path"] = ff
        return animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=6000,
                                      extra_args=["-pix_fmt", "yuv420p"]), ".mp4"
    print("Kein ffmpeg gefunden -> GIF")
    return animation.PillowWriter(fps=fps), ".gif"


# Erstes Bild aufbauen
t, g = g2_io.load_snapshot(RUN, nums[0])
dx, dy = g["x"][1] - g["x"][0], g["y"][1] - g["y"][0]
ext = [(g["x"][0] - dx/2) / 1e16, (g["x"][-1] + dx/2) / 1e16,
       (g["y"][0] - dy/2) / 1e16, (g["y"][-1] + dy/2) / 1e16]
orbit = kepler_orbit(g2_io.snapshot_time(RUN, nums[-1]))
XX, YY = np.meshgrid(g["x"], g["y"])
RHO_EQ = 1.702e-21 * 1e16 / np.maximum(np.hypot(XX, YY), 1.9e15)   # Schartmann-Normierung, konstant in r < r_in


def rho_img(rho):
    return rho / RHO_EQ if a.contrast else rho


fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
im0 = axes[0].imshow(rho_img(g["rho"]), origin="lower", extent=ext,
                     norm=LogNorm(1, 3 * CHI) if a.contrast else LogNorm(1e-23, 1e-17),
                     cmap="viridis", interpolation="nearest")
def tr_img(r0):
    return np.clip(r0, 1e-4, 1) if a.logtr else r0


im1 = axes[1].imshow(tr_img(g["r0"]), origin="lower", extent=ext,
                     norm=LogNorm(1e-3, 1) if a.logtr else None,
                     vmin=None if a.logtr else 0, vmax=None if a.logtr else 1,
                     cmap="magma", interpolation="nearest")
for ax in axes:
    ax.plot(orbit[:, 0], orbit[:, 1], "w--", lw=0.7)
    ax.plot(0, 0, "w+", ms=8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"x [$10^{16}$ cm]")
    if a.zoom:
        x0, x1, y0, y1 = (float(s) for s in a.zoom.split(","))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
axes[0].set_ylabel(r"y [$10^{16}$ cm]")
fig.colorbar(im0, ax=axes[0], label=(r"$\rho\,/\,\rho_{\rm eq}(r)$" if a.contrast else r"$\rho$ [g/cm$^3$]"), shrink=0.8)
fig.colorbar(im1, ax=axes[1], label="Tracer r0", shrink=0.8)
title = fig.suptitle("")


def update(k):
    t, g = g2_io.load_snapshot(RUN, nums[k])
    im0.set_data(rho_img(g["rho"]))
    im1.set_data(tr_img(g["r0"]))
    title.set_text(f"{LABEL}   t = {t / YR:5.1f} yr   ({T_START + t / YR:.1f})")
    if k % 10 == 0:
        print(f"  Bild {k + 1}/{len(nums)}")
    return im0, im1, title


writer, ext_out = get_writer(a.fps)
name = a.out or ("movie_zoom" if a.zoom else "movie")
out = os.path.join(RUN, "plots", name + ext_out)
os.makedirs(os.path.dirname(out), exist_ok=True)
anim = animation.FuncAnimation(fig, update, frames=len(nums), blit=False)
anim.save(out, writer=writer, dpi=a.dpi)
plt.close(fig)
print("Gespeichert:", out)

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "rasterio", "matplotlib", "scipy"]
# ///
"""
Cartes de contours du cumul annuel de la durée d'ensoleillement potentiel sur la France,
à partir des GeoTIFF produits par `duree_jour.py --annee AAAA` :

  1. sans masque (horizon plat, formule continue) ;
  2. avec masque du relief ;
  3. réduction due au relief (%).

Usage :
    uv run cartes_contours_france.py --annee 2026 [--in sorties] [--facteur 8]
"""
import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LightSource
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq

A3 = 3.0 / 3600.0


def declinaison(j):
    g = 2 * math.pi / 365.0 * j
    return (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))


def plat_analytique(lat_deg, annee):
    """Durée annuelle (h) sans relief : bord supérieur apparent (Sæmundsson) à 0°,
    même critère que duree_jour.py, sans la quantification au pas de 1 min."""
    refr = lambda h: 1.02 / math.tan(math.radians(h + 10.3 / (h + 5.11))) / 60.0
    h0 = math.radians(brentq(lambda h: h + refr(h) + 0.2666, -2.0, 0.0))
    phi = np.radians(lat_deg)
    tot = np.zeros_like(phi, dtype=float)
    njours = 366 if (annee % 4 == 0 and annee % 100) or annee % 400 == 0 else 365
    for j in range(njours):
        dec = declinaison(j)
        c = (math.sin(h0) - np.sin(phi) * math.sin(dec)) / (np.cos(phi) * math.cos(dec))
        tot += 2 * np.degrees(np.arccos(np.clip(c, -1, 1))) / 15.0
    return tot


def reduire(a, f):
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a[:h, :w].reshape(h // f, f, w // f, f), axis=(1, 3))


def lire(path, f):
    with rasterio.open(path) as s:
        b = s.bounds
        a = s.read(1)
    a = reduire(a, f)
    ny, nx = a.shape
    ext = (b.left, b.left + nx * f * A3, b.top - ny * f * A3, b.top)
    return a, ext


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annee", type=int, default=2026)
    p.add_argument("--in", dest="src", type=Path, default=Path("sorties"))
    p.add_argument("--out", type=Path, default=None, help="défaut : <in>/cartes_france")
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--facteur", type=int, default=8,
                   help="agrégation des pixels 90 m (8 ≈ 700 m) avant contours")
    a = p.parse_args()
    out = a.out or a.src / "cartes_france"
    out.mkdir(parents=True, exist_ok=True)
    f = a.facteur

    relief, ext = lire(a.src / f"jour_relief_annee_{a.annee}_h.tif", f)
    plat_g, _ = lire(a.src / f"jour_plat_annee_{a.annee}_h.tif", f)
    ny, nx = relief.shape
    xs = ext[0] + (np.arange(nx) + 0.5) * f * A3
    ys = ext[3] - (np.arange(ny) + 0.5) * f * A3
    X, Y = np.meshgrid(xs, ys)
    hors = ~np.isfinite(relief)

    plat = np.broadcast_to(plat_analytique(ys, a.annee)[:, None], relief.shape).copy()
    plat[hors] = np.nan
    # réduction calculée avec la référence plate du même calcul (même échantillonnage des jours)
    reduc = 100.0 * (1.0 - relief / plat_g)
    print(f"écart plat analytique / grille : max {np.nanmax(abs(plat - plat_g)):.2f} h")

    # ombrage à partir du MNT France en cache (s'il existe)
    ombre = None
    dems = sorted(a.cache.glob("dem_-5.8*_40.9*_10.2*_51.5*.npy"))
    if dems:
        b = [float(v) for v in dems[0].stem.split("_")[1:]]
        dem = np.load(dems[0], mmap_mode="r")
        r0 = int(round((b[3] - ext[3]) / A3))
        c0 = int(round((ext[0] - b[0]) / A3))
        d = np.asarray(dem[r0:r0 + ny * f:f, c0:c0 + nx * f:f], dtype=float)[:ny, :nx]
        dx = f * A3 * 111320 * math.cos(math.radians(46.2))
        ombre = LightSource(azdeg=315, altdeg=40).hillshade(d, vert_exag=3, dx=dx, dy=f * A3 * 111320)
        ombre = np.where(hors, np.nan, ombre)

    aspect = 1.0 / math.cos(math.radians(0.5 * (ext[2] + ext[3])))

    def carte(champ, niveaux, cmap, titre, unite, fichier, fmt, lignes=None, labels=None,
              lissage=0.7, lissage_lignes=1.5):
        fig, ax = plt.subplots(figsize=(14, 13), constrained_layout=True)
        if ombre is not None:
            ax.imshow(ombre, cmap="gray", extent=ext, aspect=aspect, vmin=0, vmax=1)
        z = gaussian_filter(np.nan_to_num(champ, nan=np.nanmedian(champ)), lissage)
        z = np.where(hors, np.nan, z)
        cm = plt.get_cmap(cmap)
        norm = BoundaryNorm(niveaux, cm.N, extend="both")   # couleurs réparties par palier
        cf = ax.contourf(X, Y, z, levels=niveaux, cmap=cm, norm=norm, alpha=0.75, extend="both")
        lignes = niveaux if lignes is None else lignes
        zl = z if lissage_lignes <= lissage else np.where(hors, np.nan, gaussian_filter(
            np.nan_to_num(champ, nan=np.nanmedian(champ)), lissage_lignes))
        cs = ax.contour(X, Y, zl, levels=lignes, colors="#222", linewidths=0.3)
        if labels is not None:
            ax.clabel(cs, levels=labels, fmt=fmt, fontsize=7, inline=True)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect(aspect)
        ax.set_xlabel("longitude (°)"); ax.set_ylabel("latitude (°)")
        ax.set_title(titre, fontsize=15)
        ax.grid(color="0.5", lw=0.3, alpha=0.4)
        fig.colorbar(cf, ax=ax, shrink=0.7, label=unite,
                     format=FuncFormatter(lambda v, _: fmt(v)))
        fig.savefig(out / fichier, dpi=150)
        plt.close(fig)
        print(f"  {out / fichier}")

    for k, v in (("plat", plat), ("relief", relief), ("réduction %", reduc)):
        q = np.nanpercentile(v, [0, 1, 50, 99, 100])
        print(f"{k:12s} min {q[0]:.1f}  p1 {q[1]:.1f}  médiane {q[2]:.1f}  p99 {q[3]:.1f}  max {q[4]:.1f}")

    hfmt = lambda v: f"{v:,.0f} h".replace(",", " ")
    lo, hi = math.floor(np.nanmin(plat)), math.ceil(np.nanmax(plat))
    lv = np.arange(lo, hi + 1, 1.0)
    carte(plat, lv, "YlOrBr",
          f"Durée annuelle d'ensoleillement potentiel SANS masque (horizon plat), {a.annee}",
          "heures / an", f"france_1_sans_masque_{a.annee}.png", hfmt,
          labels=lv[::2], lissage=0, lissage_lignes=0)
    lv = np.array([1600, 2400, 3000, 3400, 3700, 3900, 4050, 4150, 4250, 4325, 4400, 4450, 4500])
    carte(relief, lv, "YlOrBr",
          f"Durée annuelle d'ensoleillement potentiel AVEC masque du relief, {a.annee}",
          "heures / an", f"france_2_avec_masque_{a.annee}.png", hfmt,
          lignes=[3000, 3700, 4150], labels=None)
    lv = np.array([0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60])
    carte(reduc, lv, "YlGnBu",
          f"Réduction annuelle de la durée d'ensoleillement due au relief, {a.annee}",
          "% de réduction", f"france_3_reduction_pct_{a.annee}.png", lambda v: f"{v:.0f} %",
          lignes=[10, 20, 40], labels=None)


if __name__ == "__main__":
    main()

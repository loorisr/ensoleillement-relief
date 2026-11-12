# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "numba",
#   "rasterio",
#   "matplotlib",
#   "requests",
#   "pyogrio",
#   "shapely",
#   "scipy",
# ]
# ///
"""
Carte de France de la durée réelle du jour (masque du relief) comparée à la
durée astronomique du jour (horizon plat), à ~90 m de résolution.

MNT : Copernicus GLO-90 (3 secondes d'arc ≈ 90 m), bucket AWS public.

Méthode
-------
1. Mosaïque du MNT sur une grille régulière 3" × 3" (lat/lon) couvrant la France
   + une marge (pour les montagnes voisines qui masquent l'horizon : Alpes
   suisses/italiennes, Pyrénées espagnoles...).
2. Pour chaque pixel : horizon (angle d'élévation) calculé dans N azimuts en
   marchant le long du rayon jusqu'à --max-dist km, avec courbure terrestre et
   réfraction (k = 0.13). Un sommet a un horizon négatif (dépression de
   l'horizon), il peut donc avoir un jour PLUS long que l'horizon plat.
3. Pour chaque ligne (latitude constante), la trajectoire du Soleil
   (azimut, élévation) est calculée pas à pas (1 min par défaut).
   Le Soleil est « levé » quand son bord supérieur, réfraction comprise
   (Sæmundsson), dépasse l'horizon local interpolé à son azimut.
   La durée du jour réel = somme des instants où le Soleil est visible
   (les éclipses par le relief en cours de journée sont donc comptées).
4. Durée « sans relief » = même calcul avec un horizon à 0° (définition
   standard : centre du Soleil à -0.833°).

Usage
-----
    uv run duree_jour.py --date 2026-12-21
    uv run duree_jour.py --date 2026-12-21 --date 2026-06-21
    uv run duree_jour.py --bbox 6.0 45.5 7.2 46.3 --date 2026-12-21   # test Mont-Blanc
    uv run duree_jour.py --annee 2026                  # cumul annuel (heures)
    uv run duree_jour.py --annee 2026 --pas-jours 7    # cumul annuel rapide (1 jour/7)
    uv run duree_jour.py --point 45.9237 6.8694 --nom "Chamonix"

Sorties (dans --out) : GeoTIFF relief / plat / différence (minutes, ou heures
pour un cumul annuel) + pourcentage relief/plat, et PNG 4 panneaux.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from numba import njit, prange

ARCSEC3 = 3.0 / 3600.0
R_EARTH = 6371000.0
K_REFR = 0.13                       # coefficient de réfraction terrestre
FRANCE_BBOX = (-5.25, 41.3, 9.65, 51.15)   # lon_min, lat_min, lon_max, lat_max
COP_URL = ("https://copernicus-dem-90m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
NE_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"


# --------------------------------------------------------------------------
# Données
# --------------------------------------------------------------------------
def download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    if dest.with_suffix(dest.suffix + ".missing").exists():
        return False
    r = requests.get(url, timeout=120, stream=True)
    if r.status_code in (403, 404):          # tuile océan : inexistante
        dest.with_suffix(dest.suffix + ".missing").touch()
        return False
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return True


def build_dem(bbox, cache: Path):
    """Mosaïque Copernicus GLO-90 rééchantillonnée sur une grille 3"."""
    import rasterio
    from rasterio.merge import merge

    lon0, lat0, lon1, lat1 = bbox
    tiles = [(la, lo) for la in range(math.floor(lat0), math.ceil(lat1))
             for lo in range(math.floor(lon0), math.ceil(lon1))]
    cache.mkdir(parents=True, exist_ok=True)

    def job(t):
        la, lo = t
        url = COP_URL.format(ns="N" if la >= 0 else "S", lat=abs(la),
                             ew="E" if lo >= 0 else "W", lon=abs(lo))
        dest = cache / Path(url).name
        return dest if download(url, dest) else None

    print(f"MNT : {len(tiles)} tuiles Copernicus GLO-90 à vérifier/télécharger…")
    with ThreadPoolExecutor(8) as ex:
        paths = [p for p in ex.map(job, tiles) if p is not None]
    print(f"      {len(paths)} tuiles terrestres")

    srcs = [rasterio.open(p) for p in paths]
    arr, transform = merge(srcs, bounds=bbox, res=(ARCSEC3, ARCSEC3),
                           nodata=0.0, dtype="float32",
                           resampling=rasterio.enums.Resampling.bilinear)
    for s in srcs:
        s.close()
    dem = arr[0]
    dem[~np.isfinite(dem)] = 0.0
    dem[dem < -100] = 0.0
    return dem, transform


def france_mask(shape, transform, cache: Path):
    import pyogrio
    import shapely
    from rasterio.features import rasterize

    z = cache / "ne_10m_admin_0_countries.zip"
    download(NE_URL, z)
    meta, _, geoms, fields = pyogrio.raw.read(z, columns=["ADM0_A3"])
    idx = [i for i, v in enumerate(fields[0]) if v == "FRA"]
    geom = shapely.from_wkb(geoms[idx[0]])
    return rasterize([(geom, 1)], out_shape=shape, transform=transform,
                     fill=0, dtype="uint8").astype(bool)


# --------------------------------------------------------------------------
# Astronomie
# --------------------------------------------------------------------------
def declination(date: dt.date) -> float:
    """Déclinaison solaire (rad) à midi UTC, formule de Spencer."""
    g = 2 * math.pi / 365.0 * (date.timetuple().tm_yday - 1)
    return (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))


# --------------------------------------------------------------------------
# Noyau numba
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _refraction_deg(h):
    """Réfraction (°) pour une élévation vraie h (°), Sæmundsson."""
    if h < -2.0:
        h = -2.0
    return 1.02 / math.tan(math.radians(h + 10.3 / (h + 5.11))) / 60.0


@njit(cache=True)
def _first_above(e, n, x):
    """Premier indice m dans [0, n) tel que e[m] > x (e croissant)."""
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) >> 1
        if e[mid] > x:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(parallel=True, cache=True, fastmath=True)
def compute(dem, mask, lats, dy, dx0, decls, weights, didx, dists, nazim, dt_min,
            obs_h, row0, row1, col0, col1, out_real, out_flat):
    """
    dem        : MNT (m) grille lat/lon, ligne 0 au nord
    lats       : latitude (°) du centre de chaque ligne
    dy         : taille pixel N-S (m) ; dx0 : taille pixel E-O à l'équateur (m)
    decls      : déclinaisons (rad) des jours calculés
    weights    : poids (nb de jours représentés) de chaque jour calculé
    didx       : indice de la sortie dans laquelle chaque jour est cumulé
    dists      : distances d'échantillonnage le long des rayons (m)
    out_real   : (nout, row1-row0, col1-col0) float32, minutes de Soleil visible (cumul)
    out_flat   : (nout, row1-row0) float32, minutes avec horizon plat (cumul)
    """
    nd = decls.shape[0]
    nds = dists.shape[0]
    nrow, ncol = dem.shape
    daz = 360.0 / nazim
    nt = int(round(1440.0 / dt_min))
    tn = nt // 2                                   # midi solaire ; matin = [0, tn)
    reff = R_EARTH / (1.0 - K_REFR)
    semi = 0.2666                                  # demi-diamètre solaire (°)

    for i in prange(row0, row1):
        r = i - row0
        phi = math.radians(lats[i])
        dx = dx0 * math.cos(phi)

        # --- décalages entiers des rayons pour cette ligne -----------------
        di = np.empty((nazim, nds), np.int32)
        dj = np.empty((nazim, nds), np.int32)
        for k in range(nazim):
            a = math.radians(k * daz)
            ca, sa = math.cos(a), math.sin(a)
            for s in range(nds):
                di[k, s] = int(round(-dists[s] * ca / dy))
                dj[k, s] = int(round(dists[s] * sa / dx))
        curv = dists * dists / (2.0 * reff)
        invd = 1.0 / dists

        # --- trajectoire solaire (matin) pour cette latitude ---------------
        # élévation symétrique autour de midi : e(nt-1-m) = e(m),
        # azimut miroir : az(nt-1-m) = 360 - az(m).
        el_app = np.empty((nd, tn), np.float64)    # élévation bord sup. apparente
        paz = np.empty((nd, tn), np.float64)       # azimut / daz
        for d in range(nd):
            dec = decls[d]
            flat = 0
            for m in range(tn):
                H = math.radians(-180.0 + (m + 0.5) * dt_min * 0.25)
                se = (math.sin(phi) * math.sin(dec)
                      + math.cos(phi) * math.cos(dec) * math.cos(H))
                el = math.degrees(math.asin(min(1.0, max(-1.0, se))))
                az = math.degrees(math.atan2(
                    math.sin(H),
                    math.cos(H) * math.sin(phi) - math.tan(dec) * math.cos(phi))) + 180.0
                e = el + _refraction_deg(el) + semi
                el_app[d, m] = e
                paz[d, m] = (az % 360.0) / daz
                if e > 0.0:
                    flat += 2
            out_flat[didx[d], r] += flat * dt_min * weights[d]

        hz = np.empty(nazim + 1, np.float64)
        for j in range(col0, col1):
            if not mask[i, j]:
                continue
            z0 = dem[i, j] + obs_h
            # horizon initial : dépression de l'horizon géométrique (sommets)
            t0 = -math.tan(math.acos(reff / (reff + z0))) if z0 > 0 else 0.0
            hmin = 90.0
            hmax = -90.0
            for k in range(nazim):
                best = -1e9
                for s in range(nds):
                    ii = i + di[k, s]
                    jj = j + dj[k, s]
                    if ii < 0 or ii >= nrow or jj < 0 or jj >= ncol:
                        break
                    tg = (dem[ii, jj] - z0 - curv[s]) * invd[s]
                    if tg > best:
                        best = tg
                if best < t0:
                    best = t0
                h = math.degrees(math.atan(best))
                hz[k] = h
                if h < hmin:
                    hmin = h
                if h > hmax:
                    hmax = h
            hz[nazim] = hz[0]

            for d in range(nd):
                e = el_app[d]
                # au-dessus de hmax : visible à coup sûr ; sous hmin : jamais.
                a = _first_above(e, tn, hmin)
                b = _first_above(e, tn, hmax)
                cnt = 2 * (tn - b)
                for m in range(a, b):
                    em = e[m]
                    p = paz[d, m]                      # matin
                    k0 = min(int(p), nazim - 1)
                    f = p - k0
                    if em > hz[k0] * (1.0 - f) + hz[k0 + 1] * f:
                        cnt += 1
                    p = nazim - p                      # après-midi (miroir)
                    if p >= nazim:
                        p -= nazim
                    k0 = min(int(p), nazim - 1)
                    f = p - k0
                    if em > hz[k0] * (1.0 - f) + hz[k0 + 1] * f:
                        cnt += 1
                out_real[didx[d], r, j - col0] += cnt * dt_min * weights[d]


def sample_distances(max_dist, step0, growth):
    d, out = step0, []
    while d <= max_dist:
        out.append(d)
        d += max(step0, d * growth)
    return np.array(out, np.float64)


# --------------------------------------------------------------------------
# Sorties
# --------------------------------------------------------------------------
def write_tif(path, arr, transform, nodata=np.nan):
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="float32",
                       crs="EPSG:4326", transform=transform, nodata=nodata,
                       compress="deflate", predictor=3, tiled=True,
                       BIGTIFF="IF_SAFER") as dst:
        dst.write(arr.astype("float32"), 1)


def block_reduce(a, f):
    if f <= 1:
        return a
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a[:h, :w].reshape(h // f, f, w // f, f), axis=(1, 3))


def fmt_hm(m, _=None):
    m = int(round(m))
    sign = "-" if m < 0 else ""
    m = abs(m)
    return f"{sign}{m // 60}h{m % 60:02d}"


def plot(real, flat2d, diff, pct, extent, title, unit, path, max_px):
    """unit : 'min' (une date, valeurs en minutes) ou 'h' (cumul, en heures)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import FuncFormatter

    f = max(1, int(math.ceil(max(real.shape) / max_px)))
    r, fl, df, pc = (block_reduce(x, f) for x in (real, flat2d, diff, pct))
    latc = 0.5 * (extent[2] + extent[3])
    aspect = 1.0 / math.cos(math.radians(latc))
    kw = dict(extent=extent, origin="upper", aspect=aspect, interpolation="antialiased")
    fmt = FuncFormatter(fmt_hm) if unit == "min" else FuncFormatter(lambda v, _: f"{v:,.0f} h".replace(",", " "))

    fig, axes = plt.subplots(2, 2, figsize=(20, 18), constrained_layout=True)
    axes = axes.ravel()
    vmin = np.nanpercentile(r, 0.5)
    vmax = np.nanmax(fl)

    im = axes[0].imshow(fl, cmap="cividis", vmin=vmin, vmax=vmax, **kw)
    axes[0].set_title("Durée du jour sans relief (horizon plat)")
    fig.colorbar(im, ax=axes[0], shrink=0.8, format=fmt)
    im = axes[1].imshow(r, cmap="cividis", vmin=vmin, vmax=vmax, **kw)
    axes[1].set_title("Durée réelle du jour (masque du relief)")
    fig.colorbar(im, ax=axes[1], shrink=0.8, format=fmt)

    lo = min(np.nanpercentile(df, 0.2), -1.0)
    hi = max(np.nanpercentile(df, 99.9), 1.0)
    im = axes[2].imshow(df, cmap="RdBu", norm=TwoSlopeNorm(0.0, lo, hi), **kw)
    axes[2].set_title(f"Différence relief − plat ({'minutes' if unit == 'min' else 'heures'})\n"
                      "bleu : jour plus long (sommets)  ·  rouge : jour raccourci (vallées)")
    cb = fig.colorbar(im, ax=axes[2], shrink=0.8, label="minutes" if unit == "min" else "heures")
    cb.ax.set_yscale("linear")            # échelle proportionnelle de part et d'autre de 0

    lo = min(np.nanpercentile(pc, 0.2), 99.0)
    hi = max(np.nanpercentile(pc, 99.9), 101.0)
    im = axes[3].imshow(pc, cmap="RdBu", norm=TwoSlopeNorm(100.0, lo, hi), **kw)
    axes[3].set_title("Durée réelle en % de la durée sans relief\n"
                      "100 % = aucun effet du relief")
    cb = fig.colorbar(im, ax=axes[3], shrink=0.8, format=FuncFormatter(lambda v, _: f"{v:.0f} %"))
    cb.ax.set_yscale("linear")

    for ax in axes:
        ax.set_xlabel("longitude (°)")
        ax.set_ylabel("latitude (°)")
        ax.grid(color="0.5", lw=0.3, alpha=0.4)
    fig.suptitle(f"{title} — MNT Copernicus GLO-90 (~90 m)", fontsize=18)
    fig.savefig(path, dpi=130)
    plt.close(fig)



# --------------------------------------------------------------------------
# Mode point : horizon fin + éphémérides jour par jour
# --------------------------------------------------------------------------
def eq_time_min(date: dt.date) -> float:
    """Équation du temps (minutes), Spencer."""
    g = 2 * math.pi / 365.0 * (date.timetuple().tm_yday - 1)
    return 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                     - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))


def point_horizon(dem, transform, lat, lon, obs_h, max_dist, naz=720, step=30.0):
    """Horizon (°) tous les 360/naz degrés, interpolation bilinéaire du MNT."""
    from scipy.ndimage import map_coordinates
    reff = R_EARTH / (1.0 - K_REFR)
    inv = ~transform
    c0, r0 = inv * (lon, lat)
    z0 = float(map_coordinates(dem, [[r0 - 0.5], [c0 - 0.5]], order=1)[0]) + obs_h
    d = np.arange(step, max_dist + step, step)
    az = np.arange(naz) * 360.0 / naz
    A, D = np.meshgrid(np.radians(az), d, indexing="ij")
    la = lat + np.degrees(D * np.cos(A) / R_EARTH)
    lo = lon + np.degrees(D * np.sin(A) / (R_EARTH * math.cos(math.radians(lat))))
    cc, rr = inv * (lo, la)
    z = map_coordinates(dem, [rr - 0.5, cc - 0.5], order=1, mode="nearest")
    tg = (z - z0 - D * D / (2 * reff)) / D
    t0 = -math.tan(math.acos(reff / (reff + max(z0, 0.0))))
    h = np.degrees(np.arctan(np.maximum(tg.max(axis=1), t0)))
    dist = d[tg.argmax(axis=1)]
    return az, h, dist, z0 - obs_h


def sun_track(lat, dec, dt_min):
    """Hour angle (°), élévation apparente du bord sup. (°), azimut (°) sur 24 h."""
    H = -180.0 + (np.arange(int(round(1440 / dt_min))) + 0.5) * dt_min * 0.25
    phi, Hr = math.radians(lat), np.radians(H)
    el = np.degrees(np.arcsin(np.clip(math.sin(phi) * math.sin(dec)
                                      + math.cos(phi) * math.cos(dec) * np.cos(Hr), -1, 1)))
    az = (np.degrees(np.arctan2(np.sin(Hr), np.cos(Hr) * math.sin(phi)
                                - math.tan(dec) * math.cos(phi))) + 180.0) % 360.0
    hh = np.maximum(el, -2.0)
    refr = 1.02 / np.tan(np.radians(hh + 10.3 / (hh + 5.11))) / 60.0
    return H, el + refr + 0.2666, az


def run_point(a, lat, lon, name):
    import csv
    from zoneinfo import ZoneInfo
    from rasterio.transform import from_origin
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = a.max_dist * 1000 / 111320.0 + 0.02
    ml = m / math.cos(math.radians(lat))
    big = (math.floor((lon - ml) / ARCSEC3) * ARCSEC3, math.floor((lat - m) / ARCSEC3) * ARCSEC3,
           math.ceil((lon + ml) / ARCSEC3) * ARCSEC3, math.ceil((lat + m) / ARCSEC3) * ARCSEC3)
    demf = a.cache / ("dem_" + "_".join(f"{v:.4f}" for v in big) + ".npy")
    if demf.exists():
        dem = np.load(demf)
    else:
        dem, _ = build_dem(big, a.cache / "copernicus")
        np.save(demf, dem)
    transform = from_origin(big[0], big[3], ARCSEC3, ARCSEC3)

    az, h, hdist, z = point_horizon(dem, transform, lat, lon, a.obs_height, a.max_dist * 1000)
    year = (a.annee or [dt.date.today().year])[0]
    tz = ZoneInfo("Europe/Paris")
    dtm = 0.5
    rows = []
    d = dt.date(year, 1, 1)
    while d.year == year:
        H, e, saz = sun_track(lat, declination(d), dtm)
        hz = np.interp(saz, az, h, period=360.0)
        vis, flat = e > hz, e > 0.0

        def clock(Hdeg):
            utc_h = 12.0 - lon / 15.0 - eq_time_min(d) / 60.0 + Hdeg / 15.0
            t = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc) + dt.timedelta(hours=utc_h)
            return t.astimezone(tz).strftime("%H:%M")

        def edges(v):
            if not v.any():
                return "", ""
            i = np.flatnonzero(v)
            return clock(H[i[0]] - dtm * 0.125), clock(H[i[-1]] + dtm * 0.125)

        lr, cr = edges(vis)
        lp, cp = edges(flat)
        nseg = int(np.sum(np.diff(vis.astype(int)) == 1) + (1 if vis[0] else 0))
        rows.append(dict(date=d.isoformat(), plat_min=flat.sum() * dtm, reel_min=vis.sum() * dtm,
                         lever_plat=lp, coucher_plat=cp, lever_reel=lr, coucher_reel=cr,
                         interruptions=max(0, nseg - 1)))
        d += dt.timedelta(days=1)

    tag = "".join(c if c.isalnum() else "_" for c in name).strip("_") or "point"
    with open(a.out / f"point_{tag}_{year}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # --- résumé console ------------------------------------------------------
    print(f"\n{name}  ({lat:.5f} N, {lon:.5f} E), altitude MNT {z:.0f} m, année {year}")
    print(f"Horizon : max {h.max():.1f}° à l'azimut {az[h.argmax()]:.0f}° "
          f"({hdist[h.argmax()] / 1000:.1f} km)")
    for lab, a0, a1 in (("E (60-120°)", 60, 120), ("SE (120-160°)", 120, 160),
                        ("S (160-200°)", 160, 200), ("SO (200-240°)", 200, 240),
                        ("O (240-300°)", 240, 300)):
        sel = (az >= a0) & (az < a1)
        print(f"   {lab:14s} {h[sel].min():5.1f}° … {h[sel].max():5.1f}°")
    print(f"\n{'date':10s}  {'plat':>6s}  {'réel':>6s}  {'écart':>6s}   "
          f"{'lever plat→réel':>16s}  {'coucher réel←plat':>18s}")
    for r in rows:
        if r["date"][8:] == "21" or r["date"][5:] in ("06-21", "12-21"):
            if r["date"][8:] != "21":
                continue
            print(f"{r['date']}  {fmt_hm(r['plat_min']):>6s}  {fmt_hm(r['reel_min']):>6s}  "
                  f"{fmt_hm(r['reel_min'] - r['plat_min']):>6s}   "
                  f"{r['lever_plat']:>6s} → {r['lever_reel'] or '—':>6s}   "
                  f"{r['coucher_reel'] or '—':>6s} ← {r['coucher_plat']:>6s}")
    fp = sum(r["plat_min"] for r in rows) / 60
    fr = sum(r["reel_min"] for r in rows) / 60
    print(f"\nCumul {year} : plat {fp:.0f} h, réel {fr:.0f} h, écart {fr - fp:.0f} h "
          f"({100 * fr / fp:.1f} % du jour sans relief)")

    # --- figure ----------------------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(13, 15), constrained_layout=True)
    ax[0].fill_between(az, -5, h, color="0.45", lw=0)
    for dd, lab, col in ((dt.date(year, 12, 21), "21 déc.", "#1f5fa8"),
                         (dt.date(year, 3, 20), "équinoxes", "#888"),
                         (dt.date(year, 6, 21), "21 juin", "#c4581b")):
        _, e, saz = sun_track(lat, declination(dd), 4.0)
        o = np.argsort(saz)
        ax[0].plot(saz[o], e[o], color=col, lw=2, label=lab)
    ax[0].set_xlim(30, 330)
    ax[0].set_ylim(-2, max(70, h.max() + 5))
    ax[0].set_xticks([45, 90, 135, 180, 225, 270, 315], ["NE", "E", "SE", "S", "SO", "O", "NO"])
    ax[0].set_ylabel("élévation (°)")
    ax[0].set_title("Horizon (gris) et course du Soleil")
    ax[0].legend(loc="upper right")
    ax[0].grid(alpha=0.3)

    days = [dt.date.fromisoformat(r["date"]) for r in rows]
    ax[1].plot(days, [r["plat_min"] / 60 for r in rows], color="0.55", lw=2, label="sans relief")
    ax[1].plot(days, [r["reel_min"] / 60 for r in rows], color="#1f5fa8", lw=2, label="avec relief")
    ax[1].set_ylabel("durée du jour (h)")
    ax[1].set_title("Durée du jour")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    def hours(sv):
        return [int(x[:2]) + int(x[3:]) / 60 if x else np.nan for x in sv]
    for key, col, ls, lab in (("lever_plat", "0.55", "--", "lever sans relief"),
                              ("lever_reel", "#c4581b", "-", "lever réel"),
                              ("coucher_plat", "0.55", ":", "coucher sans relief"),
                              ("coucher_reel", "#1f5fa8", "-", "coucher réel")):
        ax[2].plot(days, hours([r[key] for r in rows]), color=col, ls=ls, lw=2, label=lab)
    ax[2].set_ylabel("heure légale (Europe/Paris)")
    ax[2].set_title("Lever et coucher du Soleil")
    ax[2].legend(ncol=2)
    ax[2].grid(alpha=0.3)
    fig.suptitle(f"{name} — {lat:.4f} N, {lon:.4f} E, {z:.0f} m — {year}", fontsize=15)
    out = a.out / f"point_{tag}_{year}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Sorties : {out} et {out.with_suffix('.csv')}")

# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", action="append", type=dt.date.fromisoformat,
                   help="date AAAA-MM-JJ (répétable). Défaut : solstice d'hiver "
                        "(si --annee n'est pas donné).")
    p.add_argument("--annee", type=int, action="append",
                   help="cumul sur l'année entière (répétable), en heures")
    p.add_argument("--pas-jours", type=int, default=1,
                   help="pour --annee : un jour calculé tous les N jours, pondéré par N "
                        "(défaut 1 = tous les jours)")
    p.add_argument("--bbox", nargs=4, type=float, default=FRANCE_BBOX,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   help="zone de calcul (défaut : France métropolitaine)")
    p.add_argument("--max-dist", type=float, default=40.0,
                   help="distance max de recherche de l'horizon, km (défaut 40)")
    p.add_argument("--nazim", type=int, default=72,
                   help="nombre d'azimuts de l'horizon (défaut 72 = tous les 5°)")
    p.add_argument("--dt", type=float, default=1.0, help="pas de temps, minutes")
    p.add_argument("--obs-height", type=float, default=2.0,
                   help="hauteur de l'observateur au-dessus du MNT, m")
    p.add_argument("--no-mask", action="store_true",
                   help="ne pas découper selon la frontière française (toute la bbox terrestre)")
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--out", type=Path, default=Path("sorties"))
    p.add_argument("--png-max-px", type=int, default=3000,
                   help="taille max (px) des panneaux PNG (sous-échantillonnage)")
    p.add_argument("--point", nargs=2, type=float, metavar=("LAT", "LON"),
                   help="calcul en un point : horizon, lever/coucher réels jour par jour "
                        "(année --annee, sinon l'année en cours)")
    p.add_argument("--nom", default="point", help="nom du point (titres, fichiers)")
    a = p.parse_args()
    if int(round(1440.0 / a.dt)) % 2:
        p.error("--dt doit diviser 720 minutes")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.point:
        return run_point(a, a.point[0], a.point[1], a.nom)

    # --- produits demandés : (étiquette, titre, unité, [(jour, poids), ...]) --
    products = []
    dates = a.date or ([] if a.annee else [dt.date(dt.date.today().year, 12, 21)])
    for d in dates:
        products.append((d.isoformat(), f"Durée du jour le {d.isoformat()}", "min", [(d, 1.0)]))
    for y in a.annee or []:
        start, end = dt.date(y, 1, 1), dt.date(y + 1, 1, 1)
        days, d = [], start
        while d < end:
            w = min(a.pas_jours, (end - d).days)
            days.append((d + dt.timedelta(days=w // 2), float(w)))   # jour central du pas
            d += dt.timedelta(days=w)
        title = f"Cumul annuel {y} de la durée du jour"
        if a.pas_jours > 1:
            title += f" (1 jour sur {a.pas_jours})"
        products.append((f"annee_{y}", title, "h", days))

    decls = np.array([declination(d) for _, _, _, days in products for d, _ in days])
    weights = np.array([w for _, _, _, days in products for _, w in days])
    didx = np.array([k for k, (_, _, _, days) in enumerate(products) for _ in days], np.int32)

    # --- MNT avec marge pour les horizons ----------------------------------
    lon0, lat0, lon1, lat1 = a.bbox
    mlat = a.max_dist * 1000 / 111320.0
    mlon = mlat / math.cos(math.radians(max(abs(lat0), abs(lat1))))
    big = (math.floor((lon0 - mlon) / ARCSEC3) * ARCSEC3,
           math.floor((lat0 - mlat) / ARCSEC3) * ARCSEC3,
           math.ceil((lon1 + mlon) / ARCSEC3) * ARCSEC3,
           math.ceil((lat1 + mlat) / ARCSEC3) * ARCSEC3)
    key = "dem_" + "_".join(f"{v:.4f}" for v in big) + ".npy"
    demf = a.cache / key
    from rasterio.transform import from_origin
    transform = from_origin(big[0], big[3], ARCSEC3, ARCSEC3)
    if demf.exists():
        dem = np.load(demf)
    else:
        dem, transform = build_dem(big, a.cache / "copernicus")
        np.save(demf, dem)
    nrow, ncol = dem.shape
    print(f"Grille MNT : {nrow} × {ncol} = {nrow * ncol / 1e6:.0f} Mpx")

    # fenêtre utile (sans marge)
    row0 = max(0, int(round((big[3] - lat1) / ARCSEC3)))
    row1 = min(nrow, int(round((big[3] - lat0) / ARCSEC3)))
    col0 = max(0, int(round((lon0 - big[0]) / ARCSEC3)))
    col1 = min(ncol, int(round((lon1 - big[0]) / ARCSEC3)))

    land = dem > 0.5
    if a.no_mask:
        mask = land
    else:
        mask = france_mask(dem.shape, transform, a.cache) & land
    print(f"Pixels à calculer : {mask[row0:row1, col0:col1].sum() / 1e6:.1f} M")

    lats = big[3] - (np.arange(nrow) + 0.5) * ARCSEC3
    dy = R_EARTH * math.radians(ARCSEC3)
    dists = sample_distances(a.max_dist * 1000.0, 90.0, 0.04)
    print(f"{a.nazim} azimuts × {len(dists)} pas jusqu'à {a.max_dist} km, "
          f"{len(products)} produit(s), {len(decls)} jour(s) calculé(s), {os.cpu_count()} cœurs")

    out_real = np.zeros((len(products), row1 - row0, col1 - col0), np.float32)
    out_flat = np.zeros((len(products), row1 - row0), np.float32)

    import time
    t = time.time()
    nb = 40                       # bandes, pour afficher la progression
    edges = np.linspace(row0, row1, nb + 1).astype(int)
    for b in range(nb):
        r0, r1 = edges[b], edges[b + 1]
        if r1 <= r0:
            continue
        sub_r = np.zeros((len(products), r1 - r0, col1 - col0), np.float32)
        sub_f = np.zeros((len(products), r1 - r0), np.float32)
        compute(dem, mask, lats, dy, dy, decls, weights, didx, dists, a.nazim, a.dt,
                a.obs_height, r0, r1, col0, col1, sub_r, sub_f)
        out_real[:, r0 - row0:r1 - row0] = sub_r
        out_flat[:, r0 - row0:r1 - row0] = sub_f
        el = time.time() - t
        print(f"  {100 * (b + 1) / nb:5.1f} %  {el / 60:5.1f} min écoulées, "
              f"reste ≈ {el / (b + 1) * (nb - b - 1) / 60:5.1f} min", flush=True)

    otr = from_origin(big[0] + col0 * ARCSEC3, big[3] - row0 * ARCSEC3, ARCSEC3, ARCSEC3)
    extent = (big[0] + col0 * ARCSEC3, big[0] + col1 * ARCSEC3,
              big[3] - row1 * ARCSEC3, big[3] - row0 * ARCSEC3)
    m = mask[row0:row1, col0:col1]
    for k, (tag, title, unit, _) in enumerate(products):
        scale = 1.0 if unit == "min" else 1.0 / 60.0
        real = out_real[k] * scale
        real[~m] = np.nan
        flat2d = np.broadcast_to(out_flat[k][:, None] * scale, real.shape).astype(np.float32)
        flat2d[~m] = np.nan
        diff = real - flat2d
        with np.errstate(all="ignore"):
            pct = 100.0 * real / flat2d
        u = "min" if unit == "min" else "h"
        write_tif(a.out / f"jour_relief_{tag}_{u}.tif", real, otr)
        write_tif(a.out / f"jour_plat_{tag}_{u}.tif", flat2d, otr)
        write_tif(a.out / f"jour_difference_{tag}_{u}.tif", diff, otr)
        write_tif(a.out / f"jour_pourcentage_{tag}.tif", pct, otr)
        plot(real, flat2d, diff, pct, extent, title, unit,
             a.out / f"carte_duree_jour_{tag}.png", a.png_max_px)
        v, q = diff[m], pct[m]
        print(f"{tag} : relief − plat  min {np.nanmin(v):.0f} {u}, médiane {np.nanmedian(v):.1f} {u}, "
              f"max {np.nanmax(v):.0f} {u} | % du jour plat : min {np.nanmin(q):.1f} %, "
              f"médiane {np.nanmedian(q):.1f} %, moyenne {np.nanmean(q):.1f} %")
        del real, flat2d, diff, pct
    print(f"Sorties dans {a.out.resolve()}")


if __name__ == "__main__":
    sys.exit(main())

import sys
import time
import logging
import numpy as np
import pandas as pd
import healpy as hp
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
import astropy.units as u
from sqlalchemy.exc import OperationalError
from lsst.daf.butler import Butler
from lsst.sphgeom import Angle, Circle, LonLat, UnitVector3d

sys.path.append("/sdf/data/rubin/user/pai/projects/processing/catalogs/")
from healpix_utils import healpix_to_radec, radec_to_healpix
from utils import get_catalog_refs_cone

logger = logging.getLogger(__name__)


R_x = {
    'u': 4.757217815396922,
    'g': 3.6605664439892625,
    'r': 2.70136780871597,
    'i': 2.0536599130965882,
    'z': 1.5900964472616756,
    'y': 1.3077049588254708,
}


def build_object_columns(bands, band_1=None, band_2=None, *, cols):
    band_cols = []
    for b in bands:
        band_cols.append(f"{b}_psfFlux")

    if band_1 is not None and f"{band_1}_psfFlux" not in band_cols:
       band_cols.append(f"{band_1}_psfFlux")
    if band_2 is not None and f"{band_2}_psfFlux" not in band_cols:
       band_cols.append(f"{band_2}_psfFlux")

    columns = ['coord_ra', 'coord_dec', 'objectId', 'ebv', cols.ext] + band_cols
    return columns


def build_quality_mask(df, flag_cols):
    """True where the row is GOOD (none of the flags set), False if any flag is True."""
    present = [c for c in flag_cols if c in df.columns]
    missing = [c for c in flag_cols if c not in df.columns]
    if missing:
        logger.warning(f"{len(missing)} flag columns not found in df, skipping: {missing}")

    if not present:
        return pd.Series(True, index=df.index)

    any_flag_set = df[present].any(axis=1)
    return ~any_flag_set


def query_object_by_healpixel(hpx, nside, bands,
                              band_1, band_2,
                              color_min, color_max,
                              repo='dp2_prep',
                              collections=['LSSTCam/runs/DRP/DP2'],
                              instrument='LSSTCam', *, cols):
    logger.info(f"\n[hpx={hpx}] Starting object query (nside={nside}, repo={repo}, collections={collections})")
    hpx_ra, hpx_dec = healpix_to_radec(hpx, nside=nside)
    radius_deg = 1.1 * hp.nside2resol(nside) * 180 / np.pi
    logger.info(f"[hpx={hpx}] Search radius: {radius_deg:.5f} deg (1.1x pixel resolution)")

    try:
        object_refs, butler = get_catalog_refs_cone(
            "object",
            hpx_ra,
            hpx_dec,
            radius_deg,
            repo,
            collections,
            instrument,
            return_butler=True,
        )
    except Exception as e:
        logger.error(f"[hpx={hpx}] ERROR: get_catalog_refs_cone failed: {e}")
        raise

    object_cols = build_object_columns(bands, band_1, band_2, cols=cols)
    object_table = pd.concat([butler.get(r, parameters={"columns": object_cols}).to_pandas() for r in object_refs])
    healpixels = radec_to_healpix(object_table['coord_ra'],
                                  object_table['coord_dec'],
                                  nside=nside)
    object_table['healpixel'] = healpixels
    healpixel_mask = healpixels == hpx
    len_before_hpx = len(object_table)
    object_table = object_table[healpixel_mask]
    logger.info(f"[hpx={hpx}] Filtered to exact pixel: {len_before_hpx} -> {len(object_table)} rows "
               f"({len_before_hpx - len(object_table)} dropped as cone-search padding outside this pixel)")

    for col in object_table.columns:
        if col.endswith("_psfFlux"):
            b = col.split('_')[0]
            object_table[f'{b}_psfMag'] = (object_table[f'{b}_psfFlux'].values*u.nJy).to(u.ABmag).value
            object_table[f'{b}_psfMag'] -= (object_table['ebv'] * 0.86) * R_x[b] ## extinction correction

    len_before_color_mask = len(object_table)
    color = object_table[f"{band_1}_psfMag"] - object_table[f"{band_2}_psfMag"]
    color_mask = (color >= color_min) & (color <= color_max)
    object_table = object_table[color_mask]
    logger.info(f"[hpx={hpx}] Filtered color range {band_1}-{band_2} in ({color_min}, {color_max}): {len_before_color_mask} -> {len(object_table)} rows "
                f"({len_before_color_mask - len(object_table)} dropped)")

    return object_table


def select_objects(object_table, bands, min_mag, max_mag, max_extendedness=0.5, *, cols):
    """Apply the magnitude-range (every band) and extendedness cuts to the object table."""
    n_obj = len(object_table)

    mag_mask = np.ones(n_obj, dtype=bool)
    for b in bands:
        col = f'{b}_psfMag'
        band_pass = object_table[col].between(min_mag, max_mag).to_numpy()  # NaN -> False
        logger.info(f"Band '{b}' {col} in [{min_mag}, {max_mag}]: "
                    f"{n_obj - band_pass.sum()} / {n_obj} objects fail (out of range or NaN)")
        mag_mask &= band_pass
    logger.info(f"Mag range cut (all bands): {n_obj} -> {mag_mask.sum()} objects")

    ext_mask = (object_table[cols.ext] < max_extendedness).to_numpy()  # NaN -> False
    logger.info(f"Extendedness cut ({cols.ext} < {max_extendedness}): "
                f"{n_obj - ext_mask.sum()} / {n_obj} objects fail (extended or NaN)")

    keep = mag_mask & ext_mask
    logger.info(f"Kept {keep.sum()} / {n_obj} objects after mag + extendedness cuts")
    return object_table[keep]


def build_source_columns(flag_cols, *, cols):
    data_cols = [cols.id, cols.visit, cols.band, 'ra', 'dec',
                 cols.flux, cols.flux_err]
    columns = data_cols + flag_cols
    return columns


def _with_retries(fn, what, hpx, tries=4, wait=30):
    """Call fn(), retrying with backoff on dropped database connections."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except OperationalError as e:
            if attempt == tries:
                logger.error(f"[hpx={hpx}] {what} failed after {tries} attempts: {e.orig}")
                raise
            delay = wait * attempt
            logger.warning(f"[hpx={hpx}] {what} attempt {attempt}/{tries} failed "
                           f"({e.orig}); retrying in {delay}s")
            time.sleep(delay)


def query_source_refs_cone(ra, dec, radius_deg, repo, collections, instrument, hpx=None):
    """Find per-visit `source` datasets whose visit region overlaps a cone.

    Constrains on visit.region directly instead of going through patches,
    which avoids the expensive patch-visit overlap join in the registry.
    """
    butler = Butler(repo, collections=collections, instrument=instrument)
    region = Circle(UnitVector3d(LonLat.fromDegrees(ra, dec)),
                    Angle.fromDegrees(radius_deg))

    def run():
        return butler.query_datasets(
            "source",
            where=f"instrument = '{instrument}' AND visit.region OVERLAPS cone",
            bind={"cone": region},
            limit=None,
        )

    refs = _with_retries(run, "source ref query", hpx)
    return list(refs), butler


def query_source_by_healpixel(hpx, nside, bands,
                              repo='dp2_prep',
                              collections=['LSSTCam/runs/DRP/DP2'],
                              instrument='LSSTCam', *, cols):
    logger.info(f"\n[hpx={hpx}] Starting source query (nside={nside}, repo={repo}, collections={collections})")
    hpx_ra, hpx_dec = healpix_to_radec(hpx, nside=nside)
    radius_deg = 1.1 * hp.nside2resol(nside) * 180 / np.pi
    logger.info(f"[hpx={hpx}] Search radius: {radius_deg:.5f} deg (1.1x pixel resolution)")

    source_refs, butler = query_source_refs_cone(
        hpx_ra, hpx_dec, radius_deg, repo, collections, instrument, hpx=hpx)
    logger.info(f"[hpx={hpx}] Found {len(source_refs)} source refs")
    source_cols = build_source_columns(cols.flags, cols=cols)
    source = pd.concat(
        [butler.get(r, parameters={"columns": source_cols}).to_pandas() for r in source_refs],
        ignore_index=True
    )

    #filter out all sources in other healpixels
    source['healpixel'] = radec_to_healpix(source['ra'], source['dec'], nside=nside)
    len_before_hpx = len(source)
    source = source[source['healpixel'] == hpx]
    logger.info(f"[hpx={hpx}] Filtered to exact pixel: {len_before_hpx} -> {len(source)} rows "
               f"({len_before_hpx - len(source)} dropped as cone-search padding outside this pixel)")

    visit_refs = list(butler.query_datasets("visit_table"))
    visit_df = pd.concat([butler.get(r, parameters={"columns": [cols.visit, cols.mjd]}).to_pandas() for r in visit_refs],
                        ignore_index=True).drop_duplicates(subset=cols.visit)

    #join source to visit table to get MJD
    source = source.merge(visit_df[[cols.visit, cols.mjd]], on=cols.visit, how='left')
    del visit_refs, visit_df

    #filter out sources with a quality flag set
    quality_mask = build_quality_mask(source, cols.flags)
    len_before_quality_mask = len(source)
    source = source[quality_mask]
    logger.info(f"[hpx={hpx}] Applied quality flags: {len_before_quality_mask} -> {len(source)} rows "
               f"({len_before_quality_mask - len(source)} dropped for a set quality flag)")

    source = source.drop(columns=[c for c in cols.flags if c in source.columns])

    #filter source to bands
    band_mask = np.isin(source[cols.band], [b for b in bands])
    len_before_band_mask = len(source)
    source = source[band_mask]
    logger.info(f"[hpx={hpx}] Filtered to {bands} bands: {len_before_band_mask} -> {len(source)} rows "
               f"({len_before_band_mask - len(source)} dropped)")

    #filter negative fluxes
    flux_mask = source[cols.flux] > 0
    len_before_flux_mask = len(source)
    source = source[flux_mask].copy()
    logger.info(f"[hpx={hpx}] Filtered out negative fluxes: {len_before_flux_mask} -> {len(source)} rows "
               f"({len_before_flux_mask - len(source)} dropped)")

    #convert Flux/Flux error to mag/mag error
    source[cols.mag] = (source[cols.flux].values * u.nJy).to(u.ABmag).value
    source[cols.mag_err] = 2.5 / np.log(10) * (source[cols.flux_err].values / source[cols.flux].values)

    return source


def filter_source_to_objects(source, object_table, objects, max_sep_arcsec=1, *, cols):
    """Keep sources whose per-object position matches a selected object.

    The nanmedian RA/Dec of each cols.id group in `source` is crossmatched to
    the full `object_table` (as match_object_and_dia_object did), and an ID is
    kept only if its nearest object lies within `max_sep_arcsec` and is in
    `objects` (the rows of `object_table` that passed the selection cuts).
    Both tables must share the same index.
    """
    n_before = len(source)
    if n_before == 0 or len(objects) == 0:
        logger.warning("Empty input; returning empty frame")
        return source.iloc[0:0]

    # ---- per-object median position ----
    pos = source.groupby(cols.id)[['ra', 'dec']].agg(np.nanmedian)
    n_ids = len(pos)

    # ---- positional match to the full object table ----
    src_coords = SkyCoord(pos['ra'].values * u.deg, pos['dec'].values * u.deg)
    obj_coords = SkyCoord(object_table['coord_ra'].values * u.deg,
                          object_table['coord_dec'].values * u.deg)
    idx, sep2d, _ = src_coords.match_to_catalog_sky(obj_coords)
    matched = sep2d < max_sep_arcsec * u.arcsec
    n_matched = int(matched.sum())
    median_sep = np.median(sep2d[matched].arcsec) if n_matched else np.nan
    logger.info(f"Matched {n_matched} / {n_ids} source IDs within {max_sep_arcsec} arcsec "
                f"({n_ids - n_matched} unmatched, {len(np.unique(idx[matched]))} unique objects, "
                f"median sep {median_sep:.3f} arcsec)")

    # ---- keep IDs whose matched object passed the selection ----
    selected = object_table.index.isin(objects.index)
    keep = matched & selected[idx]
    logger.info(f"Kept {keep.sum()} / {n_ids} source IDs matched to a selected object")

    source = source[source[cols.id].isin(pos.index[keep])]
    logger.info(f"Filtered to selected objects: {n_before} -> {len(source)} rows "
                f"({n_before - len(source)} dropped)")
    return source


def max_error_cut(lightcurves, max_mag_err, *, cols):
    n_before = len(lightcurves)
    lightcurves = lightcurves[lightcurves[cols.mag_err] <= max_mag_err].copy()
    logger.info(f"Max error cut (<= {max_mag_err:.2f}): {n_before} -> {len(lightcurves)} rows "
                f"({n_before - len(lightcurves)} dropped)")
    return lightcurves


def sigma_clip_cut(lightcurves, sigma=3, maxiters=1, *, cols):
    """Drop points more than `sigma` from the median, clipped per object and band."""
    n_before = len(lightcurves)
    clipped = lambda m: ~np.ma.getmaskarray(sigma_clip(m.to_numpy(), sigma=sigma, maxiters=maxiters, masked=True))
    keep = lightcurves.groupby([cols.id, cols.band])[cols.mag].transform(clipped).astype(bool)
    lightcurves = lightcurves[keep]
    logger.info(f"Sigma clip ({sigma} sigma, maxiters={maxiters}): {n_before} -> {len(lightcurves)} rows "
                f"({n_before - len(lightcurves)} dropped)")
    return lightcurves


def min_point_cut(lightcurves, bands, min_points, *, cols):
    """Drop every object that has fewer than min_points[band] points in any band in `bands`."""
    n_before, n_obj_before = len(lightcurves), lightcurves[cols.id].nunique()

    # points per (object, band); a band with no rows counts as 0
    counts = (lightcurves.groupby([cols.id, cols.band]).size()
              .unstack(fill_value=0).reindex(columns=list(bands), fill_value=0))

    passes = pd.Series(True, index=counts.index)
    for b in bands:
        band_ok = counts[b] >= min_points[b]
        logger.info(f"Band '{b}' min_points={min_points[b]}: {(~band_ok).sum()} / {len(counts)} objects fail")
        passes &= band_ok

    lightcurves = lightcurves[lightcurves[cols.id].isin(passes.index[passes])]
    logger.info(f"Min points cut: {n_before} -> {len(lightcurves)} rows, "
                f"{n_obj_before} -> {lightcurves[cols.id].nunique()} objects")
    return lightcurves

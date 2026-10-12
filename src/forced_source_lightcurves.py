import sys
import logging
import numpy as np
import pandas as pd
import healpy as hp
from astropy.stats import sigma_clip
import astropy.units as u

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


def build_forced_source_columns(flag_cols, *, cols):
    data_cols = [cols.id, cols.visit, cols.band,
                 cols.flux, cols.flux_err]
    columns = data_cols + flag_cols
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


def query_forced_source_by_healpixel(objects, hpx, nside, bands,
                                     dataset_type='object_forced_source',
                                     repo='dp2_prep',
                                     collections=['LSSTCam/runs/DRP/DP2'],
                                     instrument='LSSTCam', *, cols):
    """Forced-photometry lightcurves for the selected objects in one HEALPix pixel.

    `objects` is the selected object table (already restricted to this pixel),
    so filtering forced sources to its objectIds also removes the cone-search
    padding. Rows are filtered per ref while loading to keep memory down.
    """
    logger.info(f"\n[hpx={hpx}] Starting {dataset_type} query (nside={nside}, repo={repo}, collections={collections})")
    hpx_ra, hpx_dec = healpix_to_radec(hpx, nside=nside)
    radius_deg = 1.1 * hp.nside2resol(nside) * 180 / np.pi
    logger.info(f"[hpx={hpx}] Search radius: {radius_deg:.5f} deg (1.1x pixel resolution)")

    try:
        frc_src_refs, butler = get_catalog_refs_cone(
            dataset_type,
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
    logger.info(f"[hpx={hpx}] Found {len(frc_src_refs)} {dataset_type} refs")

    # ---- load, keeping only rows for the selected objects ----
    frc_src_cols = build_forced_source_columns(cols.flags, cols=cols)
    selected_ids = objects['objectId'].to_numpy()
    n_read = 0
    parts = []
    for r in frc_src_refs:
        t = butler.get(r, parameters={"columns": frc_src_cols}).to_pandas()
        if cols.id not in t.columns:  # ID stored as the index
            t = t.reset_index()
        n_read += len(t)
        t = t[t[cols.id].isin(selected_ids)]
        if len(t):
            parts.append(t)

    if not parts:
        logger.info(f"[hpx={hpx}] No forced sources for the selected objects")
        return pd.DataFrame(columns=frc_src_cols + [cols.mjd, 'ra', 'dec', cols.mag, cols.mag_err])

    frc_src = pd.concat(parts, ignore_index=True)
    logger.info(f"[hpx={hpx}] Filtered to selected {cols.id}: {n_read} -> {len(frc_src)} rows "
               f"({n_read - len(frc_src)} dropped), {frc_src[cols.id].nunique()} / {len(objects)} objects")

    # ---- join visit table to get MJD ----
    visit_refs = list(butler.query_datasets("visit_table"))
    visit_df = pd.concat([butler.get(r, parameters={"columns": [cols.visit, cols.mjd]}).to_pandas() for r in visit_refs],
                        ignore_index=True).drop_duplicates(subset=cols.visit)
    frc_src = frc_src.merge(visit_df[[cols.visit, cols.mjd]], on=cols.visit, how='left')
    del visit_refs, visit_df

    # ---- join object table to get RA, Dec ----
    positions = objects[['objectId', 'coord_ra', 'coord_dec']].rename(
        columns={'objectId': cols.id, 'coord_ra': 'ra', 'coord_dec': 'dec'})
    frc_src = frc_src.merge(positions, on=cols.id, how='left')

    # ---- quality flags ----
    quality_mask = build_quality_mask(frc_src, cols.flags)
    len_before_quality_mask = len(frc_src)
    frc_src = frc_src[quality_mask]
    logger.info(f"[hpx={hpx}] Applied quality flags: {len_before_quality_mask} -> {len(frc_src)} rows "
               f"({len_before_quality_mask - len(frc_src)} dropped for a set quality flag)")
    frc_src = frc_src.drop(columns=[c for c in cols.flags if c in frc_src.columns])

    # ---- bands ----
    band_mask = np.isin(frc_src[cols.band], list(bands))
    len_before_band_mask = len(frc_src)
    frc_src = frc_src[band_mask]
    logger.info(f"[hpx={hpx}] Filtered to {bands} bands: {len_before_band_mask} -> {len(frc_src)} rows "
               f"({len_before_band_mask - len(frc_src)} dropped)")

    # ---- negative fluxes ----
    flux_mask = frc_src[cols.flux] > 0
    len_before_flux_mask = len(frc_src)
    frc_src = frc_src[flux_mask].copy()
    logger.info(f"[hpx={hpx}] Filtered out negative fluxes: {len_before_flux_mask} -> {len(frc_src)} rows "
               f"({len_before_flux_mask - len(frc_src)} dropped)")

    # ---- flux -> mag ----
    frc_src[cols.mag] = (frc_src[cols.flux].values * u.nJy).to(u.ABmag).value
    frc_src[cols.mag_err] = 2.5 / np.log(10) * (frc_src[cols.flux_err].values / frc_src[cols.flux].values)

    return frc_src


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
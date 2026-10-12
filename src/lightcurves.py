import sys
import logging
import numpy as np
import pandas as pd
import healpy as hp
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
from lsst.analysis.tools import fluxToMag

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.stats import sigma_clip

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

def build_dia_object_columns(bands, *, cols):
    band_cols = []
    for b in bands:
        band_cols.append(f"{b}_scienceFluxMean")
    columns = ['ra', 'dec', cols.id] + band_cols
    return columns

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
    
def build_dia_obj_frc_src_columns(flag_cols, *, cols):
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

def correct_diff_flux(dia_obj_frc_src, dia_object, bands, *, cols):
    mean_flux_cols = [f'{b}_scienceFluxMean' for b in bands]
    dia_obj_frc_src = dia_obj_frc_src.merge(dia_object[[cols.id] + mean_flux_cols],
                                                on=cols.id,
                                                how='left')
    corrected_flux = np.zeros(len(dia_obj_frc_src))
    for b in bands:
        band_mask = dia_obj_frc_src[cols.band] == b
        corrected_flux[band_mask] = (dia_obj_frc_src[band_mask][cols.flux] +
                                     dia_obj_frc_src[band_mask][f'{b}_scienceFluxMean'])

    dia_obj_frc_src['correctedDiffFlux'] = corrected_flux
    dia_obj_frc_src = dia_obj_frc_src.drop(columns=[f'{b}_scienceFluxMean' for b in bands if f'{b}_scienceFluxMean' in dia_obj_frc_src.columns])
    return dia_obj_frc_src

def query_dia_object_by_healpixel(hpx, nside, bands,
                                  repo='dp2_prep',
                                  collections=['LSSTCam/runs/DRP/DP2'],
                                  instrument='LSSTCam', *, cols):
    logger.info(f"[hpx={hpx}] Starting dia_object query (nside={nside}, repo={repo}, collections={collections})")
    hpx_ra, hpx_dec = healpix_to_radec(hpx, nside=nside)
    radius_deg = 1.1 * hp.nside2resol(nside) * 180 / np.pi
    logger.info(f"[hpx={hpx}] Search radius: {radius_deg:.5f} deg (1.1x pixel resolution)")

    try:
        dia_object_refs, butler = get_catalog_refs_cone(
            "dia_object",
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

    logger.info(f"[hpx={hpx}] Found {len(dia_object_refs)} dia_object refs")
    dia_object_cols = build_dia_object_columns(bands, cols=cols)

    dia_object_tables = [butler.get(r, parameters={"columns": dia_object_cols}) for r in dia_object_refs]

    non_empty_tables = [t for t in dia_object_tables if len(t) > 0]
    n_empty = len(dia_object_tables) - len(non_empty_tables)
    if n_empty > 0:
        logger.info(f"[hpx={hpx}] {n_empty} / {len(dia_object_tables)} refs returned empty tables, excluding them")
    
    dia_object = pd.concat(non_empty_tables)
    dia_object.reset_index(inplace=True) # diaObjectId is index, converting it to a regular column
    
    #filter out all objects in other healpixels
    healpixels = radec_to_healpix(dia_object['ra'],
                                  dia_object['dec'],
                                  nside=nside)
    dia_object['healpixel'] = healpixels
    healpixel_mask = healpixels == hpx
    len_before_hpx = len(dia_object)
    dia_object = dia_object[healpixel_mask]
    logger.info(f"[hpx={hpx}] Filtered to exact pixel: {len_before_hpx} -> {len(dia_object)} rows "
               f"({len_before_hpx - len(dia_object)} dropped as cone-search padding outside this pixel)")

    return dia_object

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
    
def match_object_and_dia_object(object_table, dia_object, hpx, bands,
                                min_mag, max_mag, max_sep_arcsec=1, max_extendedness=0.5, *, cols):
    dia_object = dia_object.copy()  # don't mutate the caller's frame
    n_dia = len(dia_object)

    psf_cols = [f'{b}_psfMag' for b in bands]
    cols = [cols.ext] + psf_cols
    missing = [c for c in cols if c not in object_table.columns]
    if missing:
        raise KeyError(f"\nobject_table is missing columns: {missing}")

    logger.info(f"Matching {n_dia} dia_objects to {len(object_table)} objects "
                f"(max_sep={max_sep_arcsec} arcsec)")
    if n_dia == 0 or len(object_table) == 0:
        logger.warning(f"Empty input catalogue; returning empty frame")
        return dia_object.iloc[0:0]

    # ---- positional match ----
    dia_coords = SkyCoord(dia_object['ra'].values * u.deg, dia_object['dec'].values * u.deg)
    obj_coords = SkyCoord(object_table['coord_ra'].values * u.deg,
                          object_table['coord_dec'].values * u.deg)
    idx, sep2d, _ = dia_coords.match_to_catalog_sky(obj_coords)
    matched = (sep2d < max_sep_arcsec * u.arcsec)
    n_matched = int(matched.sum())
    median_sep = np.median(sep2d[matched].arcsec) if n_matched else np.nan
    logger.info(f"Matched {n_matched} / {n_dia} dia_objects within {max_sep_arcsec} arcsec "
                f"({n_dia - n_matched} unmatched, {len(np.unique(idx[matched]))} unique objects, "
                f"median sep {median_sep:.3f} arcsec)")

    for c in cols:
        vals = object_table[c].to_numpy(dtype=float)[idx]
        dia_object[c] = np.where(matched, vals, np.nan)  # NaN where unmatched

    # ---- magnitude range cut (must pass in every band) ----
    mag_mask = np.ones(n_dia, dtype=bool)
    for b, col in zip(bands, psf_cols):
        band_pass = dia_object[col].between(min_mag, max_mag).to_numpy()  # NaN -> False
        logger.info(f"Band '{b}' {col} in [{min_mag}, {max_mag}]: "
                    f"{n_dia - band_pass.sum()} / {n_dia} dia_objects fail "
                    f"(out of range, unmatched, or NaN)")
        mag_mask &= band_pass
    logger.info(f"Mag range cut (all bands): {n_dia} -> {mag_mask.sum()} dia_objects")

    # ---- extendedness cut ----
    ext_mask = (dia_object[cols.ext] < max_extendedness).to_numpy()  # NaN -> False
    logger.info(f"Extendedness cut ({cols.ext} < {max_extendedness}): "
                f"{n_dia - ext_mask.sum()} / {n_dia} dia_objects fail (extended, unmatched, or NaN)")

    keep = mag_mask & ext_mask
    logger.info(f"Kept {keep.sum()} / {n_dia} dia_objects after match + mag + extendedness cuts")
    return dia_object[keep]
    
    
def query_dia_object_forced_source_by_healpixel(dia_object, nside, bands,
                                                repo='dp2_prep',
                                                collections=['LSSTCam/runs/DRP/DP2'],
                                                instrument='LSSTCam', *, cols):
    hpx = dia_object['healpixel'].values[0]
    logger.info(f"\n[hpx={hpx}] Starting query (nside={nside}, repo={repo}, collections={collections})")
    hpx_ra, hpx_dec = healpix_to_radec(hpx, nside=nside)
    radius_deg = 1.1 * hp.nside2resol(nside) * 180 / np.pi
    logger.info(f"[hpx={hpx}] Search radius: {radius_deg:.5f} deg (1.1x pixel resolution)")

    try:
        dia_obj_frc_src_refs, butler = get_catalog_refs_cone(
            "dia_object_forced_source",
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
    logger.info(f"[hpx={hpx}] Found {len(dia_obj_frc_src_refs)} dia_object_forced_source refs")
    dia_obj_frc_src_cols = build_dia_obj_frc_src_columns(cols.flags, cols=cols)
    dia_obj_frc_src = pd.concat(
        [butler.get(r, parameters={"columns": dia_obj_frc_src_cols}).to_pandas() for r in dia_obj_frc_src_refs],
        ignore_index=True
    )

    visit_refs = list(butler.query_datasets("visit_table"))
    visit_df = pd.concat([butler.get(r, parameters={"columns": [cols.visit, cols.mjd]}).to_pandas() for r in visit_refs],
                        ignore_index=True).drop_duplicates(subset=cols.visit)

    #join dia_obj_frc_src to visit table to get MJD
    dia_obj_frc_src = dia_obj_frc_src.merge(visit_df[[cols.visit, cols.mjd]], on=cols.visit, how='left')
    del visit_refs, visit_df

    #join dia_obj_frc_src to dia_object to get RA, Dec
    dia_obj_frc_src = dia_obj_frc_src.merge(dia_object[[cols.id, 'ra', 'dec']], on=cols.id, how='left')

    #filter out objects with a quality flag set
    quality_mask = build_quality_mask(dia_obj_frc_src, cols.flags)
    len_before_quality_mask = len(dia_obj_frc_src)
    dia_obj_frc_src = dia_obj_frc_src[quality_mask]
    logger.info(f"[hpx={hpx}] Applied quality flags: {len_before_quality_mask} -> {len(dia_obj_frc_src)} rows "
               f"({len_before_quality_mask - len(dia_obj_frc_src)} dropped for a set quality flag)")

    dia_obj_frc_src = dia_obj_frc_src.drop(columns=[c for c in cols.flags if c in dia_obj_frc_src.columns])

    #filter dia_obj_frc_src to diaObjectIds in dia_object
    id_mask = np.isin(dia_obj_frc_src[cols.id], dia_object[cols.id])
    len_before_id_mask = len(dia_obj_frc_src)
    dia_obj_frc_src = dia_obj_frc_src[id_mask]
    logger.info(f"[hpx={hpx}] Filtered to diaObjectId: {len_before_id_mask} -> {len(dia_obj_frc_src)} rows "
               f"({len_before_id_mask - len(dia_obj_frc_src)} dropped)")

    #filter dia_obj_frc_src to bands
    band_mask = np.isin(dia_obj_frc_src[cols.band], [b for b in bands])
    len_before_band_mask = len(dia_obj_frc_src)
    dia_obj_frc_src = dia_obj_frc_src[band_mask]
    logger.info(f"[hpx={hpx}] Filtered to {bands} bands: {len_before_band_mask} -> {len(dia_obj_frc_src)} rows "
               f"({len_before_band_mask - len(dia_obj_frc_src)} dropped)")
    
    #convert psfDiffFlux to correctedDiffFlux by adding {band}_scienceFluxMean from dia_object table if using difference imaging flux
    if 'diff' in cols.flux.lower():
        dia_obj_frc_src = correct_diff_flux(dia_obj_frc_src, dia_object, bands, cols=cols)
        target_flux_col = 'correctedDiffFlux'
    else:
        target_flux_col = cols.flux

    #filter negative fluxes
    flux_mask = dia_obj_frc_src[target_flux_col] > 0
    len_before_flux_mask = len(dia_obj_frc_src)
    dia_obj_frc_src = dia_obj_frc_src[flux_mask]
    logger.info(f"[hpx={hpx}] Filtered out negative fluxes: {len_before_flux_mask} -> {len(dia_obj_frc_src)} rows "
               f"({len_before_flux_mask - len(dia_obj_frc_src)} dropped)")

    #convert Flux/Flux error to mag/mag error
    dia_obj_frc_src[cols.mag] = (dia_obj_frc_src[target_flux_col].values * u.nJy).to(u.ABmag).value

    ratio = (dia_obj_frc_src[cols.flux_err].values * u.nJy)/(dia_obj_frc_src[target_flux_col].values * u.nJy)
    dia_obj_frc_src[cols.mag_err] =  2.5 / np.log(10) * ratio.value

    
    #filter data with errors larger than max_mag_err
    # err_mask = dia_obj_frc_src[cols.mag_err] < max_mag_err
    # len_before_err_mask = len(dia_obj_frc_src)
    # dia_obj_frc_src = dia_obj_frc_src[err_mask]
    # logger.info(f"[hpx={hpx}] Filtered out magnitude errors larger than {max_mag_err:.2f}: {len_before_err_mask} -> {len(dia_obj_frc_src)} rows "
    #            f"({len_before_err_mask - len(dia_obj_frc_src)} dropped)")

    # # filter data with data points less than the min_points per band
    # all_object_ids = dia_obj_frc_src[cols.id].unique()
    # n_total_objects = len(all_object_ids)
    
    # min_point_mask = np.ones(len(dia_obj_frc_src), dtype=bool)
    # len_before_min_point_mask = len(dia_obj_frc_src)
    # counts = dia_obj_frc_src.groupby([cols.id, cols.band]).size()
    # diaObjectId = dia_obj_frc_src[cols.id]
    
    # for band, value in min_points.items():
    #     if value <= 0: # no filtering on band required
    #         continue
    #     band_counts = counts.xs(band, level=cols.band)
    #     mapped_counts = diaObjectId.map(band_counts)
    #     band_pass = mapped_counts.values >= value
    
    #     failing_object_ids = diaObjectId[~band_pass].unique()
    #     logger.info(f"[hpx={hpx}] Band '{band}' min_points={value}: "
    #                f"{len(failing_object_ids)} / {n_total_objects} objects dropped "
    #                f"for having < {value} points in this band")
    
    #     min_point_mask &= band_pass
    
    # dia_obj_frc_src = dia_obj_frc_src[min_point_mask]
    # logger.info(f"[hpx={hpx}] Filtered out diaObjects with data points less than {min_points}: {len_before_min_point_mask} -> {len(dia_obj_frc_src)} rows "
    #            f"({len_before_min_point_mask - len(dia_obj_frc_src)} dropped)")

    return dia_obj_frc_src

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
    

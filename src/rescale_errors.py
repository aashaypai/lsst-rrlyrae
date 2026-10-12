import sys
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

from lsst.daf.butler import Butler

sys.path.append("/sdf/data/rubin/user/pai/projects/processing/catalogs/")
from healpix_utils import healpix_to_radec, radec_to_healpix
from utils import get_catalog_refs_cone

from schema import (
    VISIT_COL, MJD_COL, BAND_COL,
    ID_COL, FLUX_COL, FLUX_ERR_COL, MAG_COL, MAG_ERR_COL, FLAG_COLS,
)


def calculate_reduced_chi2_mean_mag(table):
    group = table.groupby([ID_COL, BAND_COL])
    num_points = group.size()

    diff2_by_err2 = (table[MAG_COL] - group[MAG_COL].transform('mean')) ** 2 / (table[MAG_ERR_COL] ** 2)
    chi2 = diff2_by_err2.groupby([table[ID_COL], table[BAND_COL]]).sum()

    reduced_chi2 = (chi2 / (num_points - 1)).where(num_points >= 2) # NaN for single-point groups
    mean_mags = group[MAG_COL].mean()
    return reduced_chi2, mean_mags


def log_chi2_mag_model(m, c0, c1, c2):
    x = m - 20.0
    return c0 + c1 * x + c2 * x ** 2


def fit_chi2_quadratic(mag, reduced_chi2, sigma, maxiters):
    """
    Fit: log10(reduced_chi2) = c0 + c1*(mag-20) + c2*(mag-20)^2
    Excludes outliers via an iterative sigma-clip on log10(reduced_chi2).
    """
    mag = np.asarray(mag, dtype=float)
    reduced_chi2 = np.asarray(reduced_chi2, dtype=float)

    finite_mask = np.isfinite(mag) & np.isfinite(reduced_chi2) & (reduced_chi2 > 0)

    log_chi2_finite = np.log10(reduced_chi2[finite_mask])
    clipped = sigma_clip(log_chi2_finite, sigma=sigma, maxiters=maxiters, cenfunc='median', stdfunc='std')
    sigma_keep = ~clipped.mask

    keep_mask = np.zeros(len(mag), dtype=bool)
    keep_mask[finite_mask] = sigma_keep

    popt, _ = curve_fit(log_chi2_mag_model, mag[keep_mask], np.log10(reduced_chi2[keep_mask]))
    return popt, keep_mask


def error_scaling_factor(mag, popt):
    """Multiplicative factor for reported mag errors that flattens
    reduced_chi2 vs magnitude, using the fitted log10(reduced_chi2) quadratic."""
    log_chi2_fit = log_chi2_mag_model(mag, *popt)
    return np.sqrt(10 ** log_chi2_fit)


def fit_chi2_by_healpixel(reduced_chi2, mean_mags, bands, 
                          min_objects, sigma, maxiters):
    fits = {}
    for b in bands:
        if b not in reduced_chi2.index.get_level_values(BAND_COL):
            fits[b] = None
            continue

        band_chi2 = reduced_chi2.xs(b, level=BAND_COL)
        band_mag = mean_mags.xs(b, level=BAND_COL)

        joined = pd.DataFrame({'mean_mag': band_mag, 'reduced_chi2': band_chi2}).dropna()
        joined = joined[joined['reduced_chi2'] > 0]

        if len(joined) < min_objects:
            fits[b] = None
            continue

        popt, _ = fit_chi2_quadratic(joined['mean_mag'].values, 
                                     joined['reduced_chi2'].values,
                                     sigma = sigma,
                                     maxiters = maxiters)
        fits[b] = popt
        
    return fits


def rescale_errors_by_healpixel(table, bands, *, min_objects, sigma, maxiters):
    reduced_chi2, mean_mags = calculate_reduced_chi2_mean_mag(table)
    fits = fit_chi2_by_healpixel(reduced_chi2, mean_mags, bands,
                                 min_objects, sigma, maxiters)

    rescaled_table = table.copy()
    rescaled_table['chi2_rescaling'] = np.nan
    rescaled_table[MAG_ERR_COL] = rescaled_table[MAG_ERR_COL].astype(float)

    for b in bands:
        popt = fits.get(b)
        if popt is None:
            continue

        band_mean_mag = mean_mags.xs(b, level=BAND_COL)

        row_mask = table[BAND_COL] == b
        obj_ids_this_band = table.loc[row_mask, ID_COL]
        mapped_mean_mag = obj_ids_this_band.map(band_mean_mag)
        has_mean_mag = mapped_mean_mag.notna()
        idx = row_mask[row_mask].index[has_mean_mag.values]

        scale = error_scaling_factor(mapped_mean_mag[has_mean_mag].values, popt)
        rescaled_table.loc[idx, MAG_ERR_COL] = rescaled_table.loc[idx, MAG_ERR_COL].values * scale
        rescaled_table.loc[idx, 'chi2_rescaling'] = scale # debugging

    return rescaled_table

def rescale_errors_from_zeropoints(dia_obj_frc_src, bands,
                                   repo="dp2_prep",
                                   collections=["LSSTCam/runs/DRP/DP2"],
                                   instrument="LSSTCam"):
    butler = Butler(repo, 
                    collections=collections,
                    instrument=instrument)
        
    refs = butler.query_datasets('visit_detector_table')
    visit_tab = butler.get(refs[0], parameters={'columns':['visitId', 'zeroPoint', 'band']}).to_pandas()
    visit_tab = visit_tab[np.isin(visit_tab['band'], [b for b in bands])]
    
    zp_scatter = visit_tab.groupby('visitId')['zeroPoint'].std(ddof=0)
    zp_err = dia_obj_frc_src[VISIT_COL].map(zp_scatter).fillna(0)
    dia_obj_frc_src[MAG_ERR_COL] = np.sqrt(dia_obj_frc_src[MAG_ERR_COL]**2 + zp_err**2)
    dia_obj_frc_src['zp_err'] = zp_err # debugging
    return dia_obj_frc_src


        
    
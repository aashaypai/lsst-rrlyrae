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



def calculate_reduced_chi2_mean_mag(table, *, cols):
    group = table.groupby([cols.id, cols.band])
    num_points = group.size()

    diff2_by_err2 = (table[cols.mag] - group[cols.mag].transform('mean')) ** 2 / (table[cols.mag_err] ** 2)
    chi2 = diff2_by_err2.groupby([table[cols.id], table[cols.band]]).sum()

    reduced_chi2 = (chi2 / (num_points - 1)).where(num_points >= 2) # NaN for single-point groups
    mean_mags = group[cols.mag].mean()
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
                          min_objects, sigma, maxiters, *, cols):
    fits = {}
    for b in bands:
        if b not in reduced_chi2.index.get_level_values(cols.band):
            fits[b] = None
            continue

        band_chi2 = reduced_chi2.xs(b, level=cols.band)
        band_mag = mean_mags.xs(b, level=cols.band)

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


def rescale_errors_by_healpixel(table, bands, *, min_objects, sigma, maxiters, cols):
    reduced_chi2, mean_mags = calculate_reduced_chi2_mean_mag(table, cols=cols)
    fits = fit_chi2_by_healpixel(reduced_chi2, mean_mags, bands,
                                 min_objects, sigma, maxiters, cols=cols)

    rescaled_table = table.copy()
    rescaled_table['chi2_rescaling'] = np.nan
    rescaled_table[cols.mag_err] = rescaled_table[cols.mag_err].astype(float)

    for b in bands:
        popt = fits.get(b)
        if popt is None:
            continue

        band_mean_mag = mean_mags.xs(b, level=cols.band)

        row_mask = table[cols.band] == b
        obj_ids_this_band = table.loc[row_mask, cols.id]
        mapped_mean_mag = obj_ids_this_band.map(band_mean_mag)
        has_mean_mag = mapped_mean_mag.notna()
        idx = row_mask[row_mask].index[has_mean_mag.values]

        scale = error_scaling_factor(mapped_mean_mag[has_mean_mag].values, popt)
        rescaled_table.loc[idx, cols.mag_err] = rescaled_table.loc[idx, cols.mag_err].values * scale
        rescaled_table.loc[idx, 'chi2_rescaling'] = scale # debugging

    return rescaled_table

def rescale_errors_from_zeropoints(dia_obj_frc_src, bands,
                                   repo="dp2_prep",
                                   collections=["LSSTCam/runs/DRP/DP2"],
                                   instrument="LSSTCam", *, cols):
    butler = Butler(repo, 
                    collections=collections,
                    instrument=instrument)
        
    refs = butler.query_datasets('visit_detector_table')
    visit_tab = butler.get(refs[0], parameters={'columns':['visitId', 'zeroPoint', 'band']}).to_pandas()
    visit_tab = visit_tab[np.isin(visit_tab['band'], [b for b in bands])]
    
    zp_scatter = visit_tab.groupby('visitId')['zeroPoint'].std(ddof=0)
    zp_err = dia_obj_frc_src[cols.visit].map(zp_scatter).fillna(0)
    dia_obj_frc_src[cols.mag_err] = np.sqrt(dia_obj_frc_src[cols.mag_err]**2 + zp_err**2)
    dia_obj_frc_src['zp_err'] = zp_err # debugging
    return dia_obj_frc_src


        
    
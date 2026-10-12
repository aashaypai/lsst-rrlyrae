import logging

import pandas as pd
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

logger = logging.getLogger(__name__)


def match_catalogs(cat1, cat2, max_sep_arcsec):  # cat1 is stats, cat2 is external
    if len(cat1) == 0 or len(cat2) == 0:   # match_to_catalog_sky fails on empty catalogues
        return np.zeros(len(cat1), dtype=bool)
    cat1_coords = SkyCoord(ra=cat1['ra'].values * u.deg, dec=cat1['dec'].values * u.deg)
    cat2_coords = SkyCoord(ra=cat2['ra'].values * u.deg, dec=cat2['dec'].values * u.deg)

    _, sep2d, _ = cat1_coords.match_to_catalog_sky(cat2_coords)
    return sep2d.arcsec < max_sep_arcsec


def _add_match(cat1, cat2, name, max_sep_arcsec):
    is_match = match_catalogs(cat1, cat2, max_sep_arcsec)
    already = cat1['match'].to_numpy() if 'match' in cat1.columns else np.zeros(len(cat1), dtype=bool)
    cat1['match'] = already | is_match
    if 'catalog' not in cat1.columns:
        cat1['catalog'] = pd.Series([None] * len(cat1), index=cat1.index, dtype=object)
    prev = cat1['catalog'].fillna('')
    joined = np.where(prev == '', name, prev + ',' + name)
    cat1['catalog'] = np.where(is_match, joined, cat1['catalog'])
    return cat1


def match_muraveva2025(cat1, max_sep_arcsec, filepath='../data/sagittarius/muraveva2025_table_a3.dat'):
    cols = ['gaia_source_id', 'ra', 'dec', 'pmra', 'pmdec', 'feh', 'feh_err', 'd_helio', 'd_helio_err']
    rrl_cat = pd.read_csv(filepath, sep=r'\s+', names=cols)
    return _add_match(cat1, rrl_cat, 'muraveva2025', max_sep_arcsec)


def match_gaia_vari_rrlyrae(cat1, gaia_table, max_sep_arcsec):
    return _add_match(cat1, gaia_table, 'gaia_dr3_vari_rrlyrae', max_sep_arcsec)

def match_ogle4_blg(cat1, max_sep_arcsec, filepath='../data/sagittarius/ogle4_galactic_bulge_rrl.dat'):
    cols = ['ogle_id', 'type', 'RA', 'Dec', 'ogle4_id', 'ogle3_id', 'ogle2_id', 'other_designation', 'extra']
    rrl_cat = pd.read_csv(filepath,
            sep='\s+', names=cols)
    c = SkyCoord(rrl_cat['RA'].to_numpy(), rrl_cat['Dec'].to_numpy(), unit=(u.hourangle, u.deg))
    rrl_cat['ra'] = c.ra.deg
    rrl_cat['dec'] = c.dec.deg
    return _add_match(cat1, rrl_cat, 'OGLE-IV-BLG', max_sep_arcsec)

    


    
    
    

    
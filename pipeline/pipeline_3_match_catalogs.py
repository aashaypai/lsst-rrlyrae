import os
import sys
import time
import glob
import re
import argparse
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord

sys.path.append("/sdf/data/rubin/user/pai/projects/processing/catalogs/")
from healpix_utils import healpix_to_radec
import healpy as hp

from pipe_utils import extract_healpixel, get_filepaths
from config import load_config, override, save_config

from match_catalogs import match_muraveva2025, match_gaia_vari_rrlyrae, match_ogle4_blg
logger = logging.getLogger(__name__)

def query_gaia_by_healpixel(hpx, nside):
    """
    Query the Gaia archive for RR Lyrae sources within a cone around the
    center of HEALPix pixel `hpx`, matching the same 1.1x-pixel-resolution
    padding convention used throughout this project's Butler queries.
    """
    hpx_ra, hpx_dec = healpix_to_radec(hpx, nside=nside)
    radius_deg = 1.1 * hp.nside2resol(nside) * 180 / np.pi
    
    logger.info(f"Querying Gaia for hpx={hpx} at RA={hpx_ra}, Dec={hpx_dec}, radius={radius_deg}...")
    if not (np.isfinite(hpx_ra) and np.isfinite(hpx_dec) and np.isfinite(radius_deg)):
        raise ValueError(f"Invalid query parameters for hpx={hpx}: RA={hpx_ra}, Dec={hpx_dec}, radius={radius_deg}")
    
    query = f"""
    SELECT
        rr.source_id, rr.best_classification,
        gs.ra, gs.dec
    FROM gaiadr3.vari_rrlyrae AS rr
    JOIN gaiadr3.gaia_source AS gs
        ON rr.source_id = gs.source_id
    WHERE 1=CONTAINS(
        POINT('ICRS', gs.ra, gs.dec),
        CIRCLE('ICRS', {hpx_ra}, {hpx_dec}, {radius_deg})
    )
    """

    job = Gaia.launch_job_async(query)
    gaia_table = job.get_results().to_pandas()
    return gaia_table
    
def match_gaia_by_healpixel(hpx, filepath, outdir, nside, max_sep_arcsec, *, cols):
    logdir = os.path.join(outdir, "pipeline_3_match_logs")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    log_path = os.path.join(logdir, f"hpx{hpx}.log")

    logger_root = logging.getLogger()
    handler = logging.FileHandler(log_path, mode='w')
    handler.setFormatter(logging.Formatter(f'[hpx={hpx}] %(asctime)s %(levelname)s %(message)s'))
    logger_root.addHandler(handler)
    logger_root.setLevel(logging.INFO)

    t0 = time.time()
    try:
        stats = pd.read_parquet(filepath)
        logger.info(f"read {len(stats)} rows from {filepath}")

        logger.info(f"Querying Gaia for hpx={hpx}...")
        gaia_table = query_gaia_by_healpixel(hpx, nside=nside)
        logger.info(f"  retrieved {len(gaia_table)} Gaia sources")

        stats = stats[[cols.id, 'ra', 'dec']].copy()
        stats = match_gaia_vari_rrlyrae(stats, gaia_table, max_sep_arcsec=max_sep_arcsec)
        stats = match_muraveva2025(stats, max_sep_arcsec=max_sep_arcsec)
        stats = match_ogle4_blg(stats, max_sep_arcsec=max_sep_arcsec)

        n_matched = np.nansum(stats['match'])
        
        outfile = os.path.join(outdir, f'match_hpx{hpx}.parquet')
        stats.to_parquet(outfile)
        logger.info(f"match table saved to {outfile}")

        elapsed = time.time() - t0
        logger.info(f"took {elapsed:.1f}s")

        return {'hpx': hpx, 'status': 'ok', 'n_objects': len(stats),
               'n_matched': n_matched, 'outfile': outfile, 'elapsed': elapsed}

    except Exception as e:
        logger.error(f"FAILED: {e}")
        raise

    finally:
        logger_root.removeHandler(handler)
        handler.close()


def run_pipeline(filedir, cfg):
    run, max_sep_arcsec = cfg.run, cfg.matching.max_sep_arcsec
    filepaths = get_filepaths(filedir, fn="stats_hpx*.parquet")
    logger.info(f"Found {len(filepaths)} lightcurve files in {filedir}")

    os.makedirs(run.outdir, exist_ok=True)
    config_path = save_config(cfg, run.outdir, "config_pipeline_3.yaml")

    logger.info(f"Processing {len(filepaths)} files with {run.max_workers} parallel workers")
    logger.info(f"  nside={run.nside}, max_sep_arcsec={max_sep_arcsec}")
    logger.info(f"  output -> {run.outdir}/ (config saved to {config_path})")

    t0 = time.time()
    results = {}
    failures = {}
    n_completed = 0

    with ProcessPoolExecutor(max_workers=run.max_workers) as executor:
        futures = {}
        for fp in filepaths:
            hpx = extract_healpixel(os.path.basename(fp))
            futures[executor.submit(match_gaia_by_healpixel, hpx, fp, run.outdir, run.nside,
                                    max_sep_arcsec, cols=cfg.columns)] = hpx

        for future in as_completed(futures):
            hpx = futures[future]
            n_completed += 1
            try:
                result = future.result()
                results[hpx] = result
                status = result['status']
                if status == 'ok':
                    logger.info(f"[{n_completed}/{len(filepaths)}] hpx={hpx}: "
                               f"{result['n_matched']}/{result['n_objects']} matched "
                               f"in {result['elapsed']:.1f}s -> {result['outfile']}")
                else:
                    logger.info(f"[{n_completed}/{len(filepaths)}] hpx={hpx}: {status} (no output written)")
            except Exception as e:
                failures[hpx] = str(e)
                logger.error(f"[{n_completed}/{len(filepaths)}] hpx={hpx}: FAILED -- {e} "
                            f"(see {run.outdir}/pipeline_3_match_logs/hpx{hpx}.log)")

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results.values() if r['status'] == 'ok')
    n_empty = sum(1 for r in results.values() if r['status'] == 'empty')
    n_matched_total = sum(r.get('n_matched', 0) for r in results.values())

    logger.info(f"Done in {elapsed/60:.1f} min.")
    logger.info(f"  Succeeded: {n_ok}/{len(filepaths)}")
    logger.info(f"  Empty: {n_empty}/{len(filepaths)}")
    logger.info(f"  Failed: {len(failures)}/{len(filepaths)}")
    logger.info(f"  Total Gaia matches: {n_matched_total}")
    if failures:
        logger.info(f"  Failed pixels: {sorted(failures.keys())}")

    return results, failures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Match lightcurve objects to Gaia sources, per HEALPix pixel.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--filedir", type=str, required=True)
    # optional overrides of config values
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--nside", type=int, default=None)
    parser.add_argument("--max-sep-arcsec", type=float, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = override(cfg, 'run', outdir=args.outdir, nside=args.nside, max_workers=args.max_workers)
    cfg = override(cfg, 'matching', max_sep_arcsec=args.max_sep_arcsec)

    run_pipeline(args.filedir, cfg)

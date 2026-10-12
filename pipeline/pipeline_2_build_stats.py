import os
import sys
import time
import glob
import argparse
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from schema import (
    ID_COL, MAG_COL, MAG_ERR_COL, BAND_COL,
    NSIDE, BANDS, OUTDIR, MAX_WORKERS, MIN_POINTS, MIN_MAG,
    MAX_MAG, MAX_MAG_ERR, MIN_OBJECTS_FOR_FIT, SIGMA, MAXITERS
)

from pipe_utils import extract_healpixel, get_filepaths

logger = logging.getLogger(__name__)


def calculate_reduced_chi2(band_table):
    band_group = band_table.groupby(ID_COL)
    num_points = band_group.size()
    diff2 = (band_table[MAG_COL] - band_group[MAG_COL].transform('mean')) ** 2
    diff2_by_err2 = diff2 / (band_table[MAG_ERR_COL] ** 2)
    chi2 = diff2_by_err2.groupby(band_table[ID_COL]).sum()
    reduced_chi2 = (chi2 / (num_points - 1)).where(num_points >= 2)  # NaN for single-point groups
    return reduced_chi2


def calculate_significance(band_table):
    band_group = band_table.groupby(ID_COL)
    num_points = band_group.size()

    idxmax = band_group[MAG_COL].idxmax()
    max_mags = band_table[MAG_COL].loc[idxmax].set_axis(idxmax.index)
    max_mag_errs = band_table[MAG_ERR_COL].loc[idxmax].set_axis(idxmax.index)

    idxmin = band_group[MAG_COL].idxmin()
    min_mags = band_table[MAG_COL].loc[idxmin].set_axis(idxmin.index)
    min_mag_errs = band_table[MAG_ERR_COL].loc[idxmin].set_axis(idxmin.index)

    significance = (max_mags - min_mags) / np.sqrt(max_mag_errs**2 + min_mag_errs**2)
    significance = significance.where(num_points >= 2)  # NaN for single-point groups
    return significance

def calculate_n_obs(band_table):
    band_group = band_table.groupby(ID_COL)
    num_points = band_group.size()

    return num_points
    
def build_stats_table_by_healpixel(hpx, filepath, bands, outdir):
    logdir = os.path.join(outdir, "pipeline_2_stats_logs")
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
        lightcurve = pd.read_parquet(filepath)
        logger.info(f"read {len(lightcurve)} rows from {filepath}")

        group = lightcurve.groupby(ID_COL)[['ra', 'dec']].mean()
        file_stats = group.reset_index()

        for b in bands:
            band_mask = lightcurve[BAND_COL] == b
            band_lightcurve = lightcurve[band_mask]

            reduced_chi2 = calculate_reduced_chi2(band_lightcurve)
            reduced_chi2.name = f'{b}_reduced_chi2'
            file_stats = file_stats.merge(reduced_chi2, on=ID_COL, how='left')

            significance = calculate_significance(band_lightcurve)
            significance.name = f'{b}_significance'
            file_stats = file_stats.merge(significance, on=ID_COL, how='left')

            n_obs = calculate_n_obs(band_lightcurve)
            n_obs.name = f'{b}_n_obs'
            file_stats = file_stats.merge(n_obs, on=ID_COL, how='left')
            file_stats[f'{b}_n_obs'] = file_stats[f'{b}_n_obs'].fillna(0)
            

        n_objects = file_stats[ID_COL].nunique() if len(file_stats) else 0

        if n_objects == 0:
            logger.info("no objects to save -- skipping")
            return {'hpx': hpx, 'status': 'empty', 'n_objects': 0,
                   'outfile': None, 'elapsed': time.time() - t0}

        outfile = os.path.join(outdir, f'stats_hpx{hpx}.parquet')
        file_stats.to_parquet(outfile)
        logger.info(f"stats saved to {outfile}")

        elapsed = time.time() - t0
        logger.info(f"took {elapsed:.1f}s")

        return {'hpx': hpx, 'status': 'ok', 'n_objects': n_objects,
               'outfile': outfile, 'elapsed': elapsed}

    except Exception as e:
        logger.error(f"FAILED: {e}")
        raise

    finally:
        logger_root.removeHandler(handler)
        handler.close()


def run_pipeline(filedir, bands=BANDS, outdir=OUTDIR, max_workers=MAX_WORKERS):
    filepaths = get_filepaths(filedir, fn="lightcurve_hpx*.parquet")
    logger.info(f"Found {len(filepaths)} lightcurve files in {filedir}")

    os.makedirs(outdir, exist_ok=True)

    logger.info(f"Processing {len(filepaths)} files with {max_workers} parallel workers")
    logger.info(f"  bands={bands}")
    logger.info(f"  output -> {outdir}/")

    t0 = time.time()
    results = {}
    failures = {}
    n_completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for fp in filepaths:
            hpx = extract_healpixel(os.path.basename(fp))
            futures[executor.submit(build_stats_table_by_healpixel, hpx, fp, bands, outdir)] = hpx

        for future in as_completed(futures):
            hpx = futures[future]
            n_completed += 1
            try:
                result = future.result()
                results[hpx] = result
                status = result['status']
                if status == 'ok':
                    logger.info(f"[{n_completed}/{len(filepaths)}] hpx={hpx}: "
                               f"{result['n_objects']} objects in {result['elapsed']:.1f}s "
                               f"-> {result['outfile']}")
                else:
                    logger.info(f"[{n_completed}/{len(filepaths)}] hpx={hpx}: {status} "
                               f"(no output written)")
            except Exception as e:
                failures[hpx] = str(e)
                logger.error(f"[{n_completed}/{len(filepaths)}] hpx={hpx}: FAILED -- {e} "
                            f"(see {outdir}/pipeline_2_stats_logs/hpx{hpx}.log)")

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results.values() if r['status'] == 'ok')
    n_empty = sum(1 for r in results.values() if r['status'] == 'empty')

    logger.info(f"Done in {elapsed/60:.1f} min.")
    logger.info(f"  Succeeded: {n_ok}/{len(filepaths)}")
    logger.info(f"  Empty (no objects): {n_empty}/{len(filepaths)}")
    logger.info(f"  Failed: {len(failures)}/{len(filepaths)}")
    if failures:
        logger.info(f"  Failed pixels: {sorted(failures.keys())}")
        logger.info(f"  Check individual logs in {outdir}/pipeline_2_stats_logs/ for details.")

    return results, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build statistics table for lightcurves")
    parser.add_argument("--filedir", type=str, required=True, help="directory where lightcurve parquet files are stored")
    parser.add_argument("--bands", type=str, default=BANDS)
    parser.add_argument("--outdir", type=str, default=OUTDIR)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    run_pipeline(
        args.filedir,
        bands=args.bands,
        outdir=args.outdir,
        max_workers=args.max_workers,
    )
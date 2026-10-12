import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from schema import (
    ID_COL, MAG_ERR_COL,
    NSIDE, BANDS, OUTDIR, MAX_WORKERS, MIN_POINTS,
    MIN_MAG, MAX_MAG, MAX_MAG_ERR,
    BAND_1, BAND_2, COLOR_MIN, COLOR_MAX,
    MIN_OBJECTS_FOR_FIT, SIGMA, MAXITERS
)

from source_lightcurves import (
    query_object_by_healpixel,
    query_source_by_healpixel,
    select_objects,
    filter_source_to_objects,
    max_error_cut, sigma_clip_cut,
    min_point_cut
)

from rescale_errors import rescale_errors_by_healpixel, rescale_errors_from_zeropoints

sys.path.append("/sdf/data/rubin/user/pai/projects/processing/catalogs/")
sys.path.append("/sdf/data/rubin/user/pai/projects/rrlyrae/analysis")
from healpix_utils import radec_to_healpix, healpix_to_radec

logger = logging.getLogger(__name__)

def validate_min_points(min_points, bands):
    if type(min_points) == int:
        min_points = {b: min_points for b in bands}
        return min_points
    elif type(min_points) == dict:
        missing_bands = [b for b in bands if b not in min_points]
        if missing_bands:
            for b in missing_bands:
                min_points[b] = 0
        else:
            min_points = {b:min_points[b] for b in bands}
        return min_points
    else:
        msg = f"min_points must be an int or a dict, got {type(min_points).__name__}"
        logger.error(msg)
        raise TypeError(msg)
        
def build_lightcurves_by_healpixel(hpx, nside, bands,
                                   min_points, min_mag, max_mag,
                                   band_1, band_2, color_min, color_max,
                                   max_mag_err, outdir, **kwargs):
    logdir = os.path.join(outdir, "pipeline_1_lightcurves_logs")
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
        min_points = validate_min_points(min_points, bands)
        
        object_table = query_object_by_healpixel(hpx, nside, bands,
                                                 band_1 = band_1, 
                                                 band_2 = band_2,
                                                 color_min = color_min,
                                                 color_max = color_max, 
                                                 **kwargs)
        object_table = object_table.reset_index(drop=True)  # unique index for matching
        objects = select_objects(object_table, bands, min_mag, max_mag,
                                 max_extendedness=0.5)

        if len(objects) == 0:
            logger.info("no objects passed selection -- skipping source query")
            return {'hpx': hpx, 'status': 'empty', 'n_lightcurves': 0,
                    'outfile': None, 'elapsed': time.time() - t0}
        
        source = query_source_by_healpixel(hpx, nside, bands, **kwargs)
        source = source = filter_source_to_objects(source, object_table, objects,
                                  max_sep_arcsec=1)
        
        #source = rescale_errors_from_zeropoints(source, bands, **kwargs)
        lightcurves = rescale_errors_by_healpixel(source, bands,
                                                  min_objects=MIN_OBJECTS_FOR_FIT,
                                                  sigma=SIGMA,
                                                  maxiters=MAXITERS)
        
        lightcurves = max_error_cut(lightcurves, max_mag_err)
        lightcurves = sigma_clip_cut(lightcurves)
        lightcurves = min_point_cut(lightcurves, bands, min_points)
        
        n_objects = lightcurves[ID_COL].nunique() if len(lightcurves) else 0

        if n_objects == 0:
            logger.info("no objects survived -- skipping save")
            return {'hpx': hpx, 'status': 'empty', 'n_lightcurves': 0,
                   'outfile': None, 'elapsed': time.time() - t0}

        outfile = f"lightcurve_hpx{hpx}.parquet"
        savepath = os.path.join(outdir, outfile)
        lightcurves.to_parquet(savepath)
        logger.info(f"lightcurves saved to {savepath}")

        elapsed = time.time() - t0
        logger.info(f"took {elapsed:.1f}s")

        return {'hpx': hpx, 'status': 'ok', 'n_lightcurves': n_objects,
               'outfile': savepath, 'elapsed': elapsed}

    except Exception as e:
        logger.error(f"FAILED: {e}")
        raise

    finally:
        logger_root.removeHandler(handler)
        handler.close()


def run_pipeline(healpix_footprint_fp, nside=NSIDE, bands=BANDS, outdir=OUTDIR,
                 min_points=MIN_POINTS, min_mag=MIN_MAG, max_mag=MAX_MAG,
                 band_1=BAND_1, band_2=BAND_2, color_min=COLOR_MIN, color_max=COLOR_MAX,
                 max_mag_err=MAX_MAG_ERR, max_workers=MAX_WORKERS, **kwargs):
    healpixels = np.load(healpix_footprint_fp)
    healpixels = np.array([h for h in healpixels if ((healpix_to_radec(h, nside=32)[0] > 300) &
                                      (healpix_to_radec(h, nside=32)[0] < 360))])
    logger.info(f"Loaded {len(healpixels)} healpixels, dtype={healpixels.dtype}")

    os.makedirs(outdir, exist_ok=True)

    logger.info(f"Processing {len(healpixels)} HEALPix pixels with {max_workers} parallel workers")
    logger.info(f"  nside={nside}, bands={bands}")
    logger.info(f"  min_points={min_points}, min_mag={min_mag}, max_mag={max_mag}, max_mag_err={max_mag_err}")
    logger.info(f"  output -> {outdir}/")

    t0 = time.time()
    results = {}
    failures = {}
    n_completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(build_lightcurves_by_healpixel,
                    hpx=hpx, nside=nside, bands=bands,
                    min_points=min_points, min_mag=min_mag, max_mag=max_mag,
                    band_1=band_1, band_2=band_2, color_min=color_min, color_max=color_max,
                    max_mag_err=max_mag_err, outdir=outdir, **kwargs): hpx
                for hpx in healpixels
            }

        for future in as_completed(futures):
            hpx = futures[future]
            n_completed += 1
            try:
                result = future.result()
                results[hpx] = result
                status = result['status']
                if status == 'ok':
                    logger.info(f"[{n_completed}/{len(healpixels)}] hpx={hpx}: "
                               f"{result['n_lightcurves']} lightcurves in {result['elapsed']:.1f}s "
                               f"-> {result['outfile']}")
                else:
                    logger.info(f"[{n_completed}/{len(healpixels)}] hpx={hpx}: {status} "
                               f"(no output written)")
            except Exception as e:
                failures[hpx] = str(e)
                logger.error(f"[{n_completed}/{len(healpixels)}] hpx={hpx}: FAILED -- {e} "
                            f"(see {outdir}/pipeline_1_lightcurves_logs/hpx{hpx}.log)")

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results.values() if r['status'] == 'ok')
    n_empty = sum(1 for r in results.values() if r['status'] == 'empty')

    logger.info(f"Done in {elapsed/60:.1f} min.")
    logger.info(f"  Succeeded: {n_ok}/{len(healpixels)}")
    logger.info(f"  Empty (no lightcurves): {n_empty}/{len(healpixels)}")
    logger.info(f"  Failed: {len(failures)}/{len(healpixels)}")
    if failures:
        logger.info(f"  Failed pixels: {sorted(failures.keys())}")
        logger.info(f"  Check individual logs in {outdir}/pipeline_1_lightcurves_logs/ for details.")

    return results, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save rescaled source lightcurves per HEALPix pixel, in parallel.")
    parser.add_argument("--hpxfp", type=str, required=True, help="HEALPix pixel filepath to process")
    parser.add_argument("--nside", type=int, default=NSIDE)
    parser.add_argument("--bands", type=str, default=BANDS)
    parser.add_argument("--outdir", type=str, default=OUTDIR)
    parser.add_argument("--min-mag", type=float, default=MIN_MAG)
    parser.add_argument("--max-mag", type=float, default=MAX_MAG)
    parser.add_argument("--max-mag-err", type=float, default=MAX_MAG_ERR)
    parser.add_argument("--band-1", type=str, default=BAND_1)
    parser.add_argument("--band-2", type=str, default=BAND_2)
    parser.add_argument("--color-min", type=float, default=COLOR_MIN)
    parser.add_argument("--color-max", type=float, default=COLOR_MAX)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    run_pipeline(
        args.hpxfp,
        nside=args.nside,
        bands=args.bands,
        outdir=args.outdir,
        min_points=MIN_POINTS,
        min_mag=args.min_mag,
        max_mag=args.max_mag,
        max_mag_err=args.max_mag_err,
        band_1=args.band_1,
        band_2=args.band_2,
        color_min=args.color_min,
        color_max=args.color_max,
        max_workers=args.max_workers,
    )
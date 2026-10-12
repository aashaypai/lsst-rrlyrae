import os
import sys
import glob
import time
import argparse
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from pipe_utils import extract_healpixel, get_filepaths

from schema import (
    ID_COL, MJD_COL, MAG_COL, MAG_ERR_COL, BAND_COL, BANDS,
    NSIDE, MAX_WORKERS, N_OBS_MIN, N_OBS_MAX,
    LOG_REDUCED_CHI2_MIN, LOG_REDUCED_CHI2_MAX,
    LOG_SIGNIFICANCE_MIN, LOG_SIGNIFICANCE_MAX
)

from period_fitting import period_search, avg_template_fit, fit_vonmises_fisher_kappa
#from plotting import make_phased_lightcurve_plot

sys.path.append('/sdf/data/rubin/user/pai/projects/processing/catalogs')

from healpix_utils import radec_to_healpix

logger = logging.getLogger(__name__)

def validate_cut(cut, bands):
    if isinstance(cut, (int, float)):
        cut = {b:cut for b in bands}
        return cut
    elif isinstance(cut, dict):
        missing_bands = [b for b in bands if b not in cut]
        if missing_bands:
            msg = (f"cut dict is missing entries for bands: {missing_bands} "
                  f"(bands requested: {list(bands)}, cut keys: {list(cut.keys())})")
            logger.error(msg)
            raise ValueError(msg)
        else:
            cut = {b:cut[b] for b in bands}
        return cut
    else:
        msg = f"cut must be an int or a dict, got {type(cut).__name__}"
        logger.error(msg)
        raise TypeError(msg)

def stats_cut(stats_table, bands,
              n_obs_min, n_obs_max,
              log_reduced_chi2_min, log_reduced_chi2_max,
              log_significance_min, log_significance_max):

    # (label, column suffix, take log10?, lower limits, upper limits)
    cuts = [
        ('n_obs', 'n_obs', False, validate_cut(n_obs_min, bands), validate_cut(n_obs_max, bands)),
        ('log10(reduced_chi2)', 'reduced_chi2', True, validate_cut(log_reduced_chi2_min, bands), validate_cut(log_reduced_chi2_max, bands)),
        ('log10(significance)', 'significance', True, validate_cut(log_significance_min, bands), validate_cut(log_significance_max, bands)),
    ]

    n_total = len(stats_table)
    full_mask = np.ones(n_total, dtype=bool)

    logger.info(f"STATS CUT ({n_total} rows):\n" +
                "\n".join(f"{lo} < {label} < {hi}" for label, _, _, lo, hi in cuts))

    for b in bands:
        for label, suffix, use_log, lo, hi in cuts:
            if suffix == 'n_obs' and lo[b] <= 0:   # n_obs cut is skipped entirely when its min is 0
                continue
            x = stats_table[f'{b}_{suffix}'].to_numpy(dtype=float)
            if use_log:
                with np.errstate(divide='ignore', invalid='ignore'):   # log10 of <=0 -> NaN, fails the cut
                    x = np.log10(x)
            mask = (x > lo[b]) & (x < hi[b])

            logger.info(f"  [{b}] {label}: {(~mask).sum()} / {n_total} fail on their own, "
                        f"{(full_mask & ~mask).sum()} newly dropped")
            full_mask &= mask

    logger.info(f"STATS CUT: kept {full_mask.sum()} / {n_total} rows ({n_total - full_mask.sum()} dropped)")
    return full_mask
    
def build_full_stats_table(stats_dir, nside=NSIDE):
    filepaths = get_filepaths(stats_dir, fn="stats_hpx*.parquet")
    stats = pd.concat([pd.read_parquet(fp) for fp in filepaths], ignore_index=True)
    stats['healpixel'] = radec_to_healpix(stats['ra'].values, stats['dec'].values, nside=nside)

    logger.info(f"generated stats table for the full footprint containing {stats[ID_COL].nunique()} objects")

    return stats

def get_match_table(hpx, matchdir, match_flag='match'):
    try:
        filepath = get_filepaths(matchdir, fn=f"match_hpx{hpx}.parquet")[0]
        table = pd.read_parquet(filepath)
        table = table[table[match_flag] == True]
    except FileNotFoundError:
        return None
    return table
    
def fit_lightcurve_period_by_healpixel(fp, stats_table, outdir, matchdir=None):
    hpx = extract_healpixel(fp)

    if matchdir is not None:
        match_table = get_match_table(hpx, matchdir)
    else:
        match_table = None
        
    logdir = os.path.join(outdir, "pipeline_4_period_fit_logs")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    log_path = os.path.join(logdir, f"hpx{hpx}.log")

    logger_root = logging.getLogger()
    handler = logging.FileHandler(log_path, mode='w')
    handler.setFormatter(logging.Formatter(f'[hpx={hpx}] %(asctime)s %(levelname)s %(message)s'))
    logger_root.addHandler(handler)
    logger_root.setLevel(logging.INFO)

    t0 = time.time()

    result_coeff_keys = None
    try:
        object_ids = np.unique(stats_table[stats_table['healpixel'] == hpx][ID_COL].values)
        lightcurves = pd.read_parquet(fp)
        id_mask = np.isin(lightcurves[ID_COL].values, object_ids)
        lightcurves = lightcurves[id_mask]

        rows = []
        for obj_id, group in lightcurves.groupby(ID_COL):
            
            match_flag = False
            if match_table is not None:
                if obj_id in match_table[ID_COL].values:  
                    match_flag = True
                    
            epoch_time = group[MJD_COL].values
            mags = group[MAG_COL].values
            mag_errs = group[MAG_ERR_COL].values
            bands = group[BAND_COL].values

            try:
                #period_search_results = period_search(epoch_time, mags, mag_errs, bands)
                result, template = avg_template_fit(epoch_time, mags, mag_errs, bands)
                
                ## RSS per degree of freedom
                n_obs_cols = [c for c in stats_table.columns if c.endswith('_n_obs')]
                band_n_obs = stats_table.loc[stats_table[ID_COL] == obj_id, n_obs_cols]
                n_obs = band_n_obs.sum(axis=1).item()
                
                DOF = 4
                rss_dof = result.rss/(n_obs - DOF)
    
                ## Von Mises-Fisher Kappa
                try:
                    mu, mu_err, kappa, kappa_err = fit_vonmises_fisher_kappa(epoch_time, result)
                except Exception as e:
                    logger.exception(f"{ID_COL} {obj_id}: Von-Mises Fisher Kappa fit FAILED")
                    mu, mu_err, kappa, kappa_err = np.nan, np.nan, np.nan, np.nan
                    
                row = {ID_COL: obj_id,
                      'match': match_flag,
                      #'P_search': period_search_results.best_period,
                      'P_template': result.best_period,
                      'template_min_rss': np.nanmin(result.rss),
                      'template_min_rss_dof': np.nanmin(rss_dof),
                      'vmf_kappa': kappa,
                      'vmf_kappa_err': kappa_err,
                      'vmf_mu': mu,
                      'vmf_mu_err': mu_err,
                      }
                row.update(result.best_coeffs)

                if result_coeff_keys is None:
                    result_coeff_keys = list(result.best_coeffs.keys())

            except Exception as e:
                logger.exception(f"{ID_COL} {obj_id}: FAILED; {e}") 
                row = {ID_COL: obj_id,
                      'match': match_flag,
                      'P_template': np.nan,
                      'template_min_rss': np.nan,
                      'template_min_rss_dof': np.nan,
                      'vmf_kappa': np.nan,
                      'vmf_kappa_err': np.nan,
                      'vmf_mu': np.nan,
                      'vmf_mu_err': np.nan,
                    }
                if result_coeff_keys is not None:
                    row.update({k: np.nan for k in result_coeff_keys})
        
            rows.append(row)
            logger.info(f"{row}")
            
        period_fit_table = pd.DataFrame(rows)
        n_objects = len(period_fit_table)

        outfile = f"period_fit_hpx{hpx}.parquet"
        savepath = os.path.join(outdir, outfile)
        period_fit_table.to_parquet(savepath)
        logger.info(f"period fits saved to {savepath}")

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

def run_pipeline(lightcurve_dir, stats_dir, outdir, 
                 matchdir=None,
                 bands=BANDS, nside=NSIDE, max_workers=MAX_WORKERS,
                 n_obs_min=N_OBS_MIN, n_obs_max=N_OBS_MAX,
                 log_reduced_chi2_min=LOG_REDUCED_CHI2_MIN,
                 log_reduced_chi2_max=LOG_REDUCED_CHI2_MAX,
                 log_significance_min=LOG_SIGNIFICANCE_MIN,
                 log_significance_max=LOG_SIGNIFICANCE_MAX):

    stats_table = build_full_stats_table(stats_dir, nside=nside)
    stats_mask = stats_cut(stats_table, bands,
                           n_obs_min, n_obs_max,
                           log_reduced_chi2_min, log_reduced_chi2_max,
                           log_significance_min, log_significance_max)

    stats_table = stats_table[stats_mask]
    kept_healpixels = set(np.unique(stats_table['healpixel']))
    logger.info(f"{stats_table[ID_COL].nunique()} pass the stats cut from {len(kept_healpixels)} unique healpixels (nside {nside})")

    filepaths = get_filepaths(lightcurve_dir, fn="lightcurve_hpx*.parquet")
    filepaths = [fp for fp in filepaths if extract_healpixel(os.path.basename(fp)) in kept_healpixels]
    logger.info(f"{len(filepaths)} lightcurve files match a kept healpixel")

    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    results = {}
    failures = {}
    n_completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fit_lightcurve_period_by_healpixel, fp, stats_table, outdir, matchdir): fp
            for fp in filepaths
        }

        for future in as_completed(futures):
            fp = futures[future]
            hpx = extract_healpixel(os.path.basename(fp))
            n_completed += 1
            try:
                result = future.result()
                results[hpx] = result
                logger.info(f"[{n_completed}/{len(filepaths)}] hpx={hpx}: "
                           f"{result['n_lightcurves']} objects in {result['elapsed']:.1f}s "
                           f"-> {result['outfile']}")
            except Exception as e:
                failures[hpx] = str(e)
                logger.error(f"[{n_completed}/{len(filepaths)}] hpx={hpx}: FAILED -- {e} "
                            f"(see {outdir}/pipeline_4_period_fit_logs/hpx{hpx}.log)")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed/60:.1f} min.")
    logger.info(f"  Succeeded: {len(results)}/{len(filepaths)}")
    logger.info(f"  Failed: {len(failures)}/{len(filepaths)}")
    if failures:
        logger.info(f"  Failed pixels: {sorted(failures.keys())}")

    return results, failures


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Fit periods for lightcurves passing the stats cut.")
    parser.add_argument("--lightcurve-dir", type=str, required=True)
    parser.add_argument("--stats-dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    # parser.add_argument("--plotdir", type=str, default=None,
    #                    help="If given, save phased lightcurve plots here (skipped entirely if not set)")
    parser.add_argument("--matchdir", type=str, default=None,
                       help="If given, checks if the object matches to Gaia DR3")
    parser.add_argument("--bands", type=str, default=BANDS)
    parser.add_argument("--nside", type=int, default=NSIDE)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    run_pipeline(args.lightcurve_dir, args.stats_dir, args.outdir, 
                 matchdir=args.matchdir, bands=args.bands, 
                 nside=args.nside, max_workers=args.max_workers)
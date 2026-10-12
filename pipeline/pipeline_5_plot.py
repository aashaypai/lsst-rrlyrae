import os
import sys
import time
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                 # headless workers; must be set before plotting is imported
from concurrent.futures import ProcessPoolExecutor, as_completed

from pipe_utils import extract_healpixel, get_filepaths
from config import load_config, override, save_config
from period_fitting import full_template_fit
from plotting import make_phased_lightcurve_cutout_plot, get_preliminary_visit_image_cutouts, make_pvi_cutout_gif

logger = logging.getLogger(__name__)


def plot_candidates_by_file(fp, candidates, plotdir, *, cols):
    """Fit and plot every candidate found in one lightcurve_hpx*.parquet file."""
    hpx = extract_healpixel(os.path.basename(fp))
    t0 = time.time()

    lc = pd.read_parquet(fp)
    lc = lc[lc[cols.id].isin(candidates.index)]                # only candidates in this file
    if lc.empty:                                              # most files: nothing to do, no log file
        return {'hpx': hpx, 'status': 'empty', 'n_plotted': 0, 'failures': {},
                'found_ids': [], 'elapsed': time.time() - t0}

    logdir = os.path.join(plotdir, "pipeline_5_plot_logs")
    os.makedirs(logdir, exist_ok=True)
    logger_root = logging.getLogger()
    handler = logging.FileHandler(os.path.join(logdir, f"hpx{hpx}.log"), mode='w')
    handler.setFormatter(logging.Formatter(f'[hpx={hpx}] %(asctime)s %(levelname)s %(message)s'))
    logger_root.addHandler(handler)
    logger_root.setLevel(logging.INFO)

    try:
        n_plotted, failures = 0, {}
        logger.info(f"{lc[cols.id].nunique()} candidates in {fp}")
        for obj_id, lightcurves in lc.groupby(cols.id):
            try:
                mjd = lightcurves[cols.mjd].to_numpy()          # not `time`: that would shadow the time module
                mag = lightcurves[cols.mag].to_numpy()
                err = lightcurves[cols.mag_err].to_numpy()
                band = lightcurves[cols.band].to_numpy()
                visits = lightcurves[cols.visit].to_numpy()
                ra, dec = lightcurves['ra'].iloc[0], lightcurves['dec'].iloc[0]
                result, tmpl = full_template_fit(mjd, mag, err, band, rrl_type='ab')

                row = candidates.loc[obj_id]
                title_text = f"{cols.id}: {obj_id} RRL prob. {row['rrl_prob']:.2f}"
                if 'match' in candidates.columns and row['match'] == True:
                    title_text += " [known RRL]"

                outfile = make_phased_lightcurve_cutout_plot(
                    mjd, mag, err, band, ra=ra, dec=dec,
                    result=result, template=tmpl,
                    plotdir=plotdir, plotname=f"{obj_id}_lc_cutout.png",
                    title_text=title_text)
                n_plotted += 1
                logger.info(f"lightcurve plot saved to {outfile}")

                # cutouts_dict, cutout_bands_dict, coords_dict = get_preliminary_visit_image_cutouts(ra, dec, visits=visits)

                # # not every requested visit is guaranteed to come back (chip gaps, edges); drop the missing ones
                # found_visits = list(cutouts_dict.keys())
                # missing = set(visits) - set(found_visits)
                # if missing:
                #     logger.warning(f"{cols.id} {obj_id}: {len(missing)} visit(s) had no PVI cutout: {sorted(missing)}")
            
                # sort_idx = np.argsort([cutout_bands_dict[v] for v in found_visits])   # sort by band, via the dict's own values
                # sorted_visits = [found_visits[i] for i in sort_idx]
            
                # all_cutouts = [cutouts_dict[v] for v in sorted_visits]
                # all_coords = [coords_dict[v] for v in sorted_visits]
                # all_bands = [cutout_bands_dict[v] for v in sorted_visits]
            
                # gif_outfile = make_pvi_cutout_gif(all_cutouts, all_coords, sorted_visits, all_bands, obj_id, plotdir)
                # logger.info(f"GIF plot saved to {gif_outfile}")

            except Exception as e:                             # one bad object shouldn't stop the rest
                failures[obj_id] = str(e)
                logger.exception(f"{cols.id} {obj_id}: FAILED")

        elapsed = time.time() - t0
        logger.info(f"{n_plotted} plotted, {len(failures)} failed, took {elapsed:.1f}s")
        return {'hpx': hpx, 'status': 'ok', 'n_plotted': n_plotted, 'failures': failures,
                'found_ids': lc[cols.id].unique().tolist(), 'elapsed': elapsed}

    finally:
        logger_root.removeHandler(handler)
        handler.close()


def plot_candidates(candidates_filepath, filedir, plotdir, cfg, top_n=None):
    cols, max_workers = cfg.columns, cfg.run.max_workers
    """Plot every candidate, one lightcurve file per worker process.

    candidates_filepath : parquet with cols.id and 'rrl_prob' (and optionally 'match')
    filedir             : directory of lightcurve_hpx*.parquet files
    top_n               : only plot the top_n candidates by rrl_prob (default: all)
    """
    candidates = pd.read_parquet(candidates_filepath)
    if top_n is not None:
        candidates = candidates.nlargest(top_n, 'rrl_prob')
    candidates = candidates.drop_duplicates(cols.id).set_index(cols.id)
    candidates = candidates[[c for c in ('rrl_prob', 'match') if c in candidates.columns]]

    filepaths = get_filepaths(filedir, fn="lightcurve_hpx*.parquet")
    os.makedirs(plotdir, exist_ok=True)
    save_config(cfg, plotdir, "config_pipeline_5.yaml")
    logger.info(f"{len(candidates)} candidates, {len(filepaths)} lightcurve files in {filedir}")
    logger.info(f"Processing with {max_workers} parallel workers; plots -> {plotdir}/")

    t0 = time.time()
    results, failures = {}, {}
    n_completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(plot_candidates_by_file, fp, candidates, plotdir, cols=cols): fp for fp in filepaths}

        for future in as_completed(futures):
            fp = futures[future]
            hpx = extract_healpixel(os.path.basename(fp))
            n_completed += 1
            try:
                result = future.result()
                results[hpx] = result
                if result['status'] == 'ok':
                    logger.info(f"[{n_completed}/{len(filepaths)}] hpx={hpx}: {result['n_plotted']} plotted, "
                                f"{len(result['failures'])} failed in {result['elapsed']:.1f}s")
            except Exception as e:
                failures[hpx] = str(e)
                logger.error(f"[{n_completed}/{len(filepaths)}] hpx={hpx}: FAILED -- {e}")

    elapsed = time.time() - t0
    ok = [r for r in results.values() if r['status'] == 'ok']
    n_plotted = sum(r['n_plotted'] for r in ok)
    object_failures = {i: msg for r in ok for i, msg in r['failures'].items()}
    found = {i for r in ok for i in r['found_ids']}
    n_missing = len(candidates) - len(found)

    logger.info(f"Done in {elapsed/60:.1f} min.")
    logger.info(f"  Files with candidates: {len(ok)}/{len(filepaths)}; files failed: {len(failures)}")
    logger.info(f"  Plotted: {n_plotted}/{len(candidates)}; objects failed: {len(object_failures)}; "
                f"not found in any file: {n_missing}")
    if failures:
        logger.info(f"  Failed files (healpixel): {sorted(failures.keys())}")
    if object_failures:
        logger.info(f"  Failed objects (first 20): {sorted(object_failures)[:20]} "
                    f"(see {plotdir}/plot_candidates_logs/)")

    return results, failures


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Plot phased lightcurves + cutouts for RRL candidates.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--candidates", type=str, required=True, help="parquet file with the id column and rrl_prob")
    parser.add_argument("--lightcurve-dir", type=str, required=True)
    parser.add_argument("--plotdir", type=str, default="./candidates")
    parser.add_argument("--top-n", type=int, default=None, help="only plot the top N candidates by rrl_prob")
    # optional override of config value
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = override(cfg, 'run', max_workers=args.max_workers)

    plot_candidates(args.candidates, args.lightcurve_dir, args.plotdir, cfg, top_n=args.top_n)

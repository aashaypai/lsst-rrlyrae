# lsst-rrlyrae

RR Lyrae search pipeline for Rubin data, run per HEALPix pixel in five stages.

| Stage | Script | Input | Output |
| --- | --- | --- | --- |
| 1 | `pipeline_1_build_lightcurves.py` (DIA), `pipeline_1.1_build_source_lightcurves.py` (Source), `pipeline_1.2_build_forced_source_lightcurves.py` (ForcedSource) | HEALPix footprint `.npy` | `lightcurve_hpx*.parquet` |
| 2 | `pipeline_2_build_stats.py` | lightcurves | `stats_hpx*.parquet` |
| 3 | `pipeline_3_match_catalogs.py` | stats | `match_hpx*.parquet` |
| 4 | `pipeline_4_fit_period.py` | lightcurves, stats, (matches) | `period_fit_hpx*.parquet` |
| 5 | `pipeline_5_plot.py` | candidates, lightcurves | plots |

## Configuration

Every stage takes `--config`, a YAML file in `configs/`:

- `forced_source.yaml`: ForcedSource lightcurves (stage 1.2), grouped by `objectId`
- `source.yaml`: Source lightcurves (stage 1.1). `sourceId` is unique per detection, so it cannot group epochs into lightcurves on its own.
- `dia.yaml`: DIA forced-source lightcurves (stage 1). Flux columns and flags are placeholders; check them before use.

**Use the same config for every stage of a run**, so column names and cuts stay consistent. Each stage saves the resolved config (including any command-line overrides) next to its output as `config_pipeline_<N>.yaml`.

Command-line flags such as `--outdir`, `--bands` or `--max-mag-err` override the config for that run only.

## Running

Requires the LSST Science Pipelines environment (developed with `lsst-scipipe-14.0.0`), plus `pycycle`, `healpix_utils` and `utils.get_catalog_refs_cone`, which are not yet part of this repo.

From the repository root:

```bash
export PYTHONPATH=$PWD/src:$PWD/utils:$PYTHONPATH

python pipeline/pipeline_1.2_build_forced_source_lightcurves.py \
    --config configs/forced_source.yaml --hpxfp footprint_nside32.npy \
    --outdir out/lightcurves --bands griz

python pipeline/pipeline_2_build_stats.py \
    --config configs/forced_source.yaml --filedir out/lightcurves \
    --outdir out/stats --bands griz

python pipeline/pipeline_3_match_catalogs.py \
    --config configs/forced_source.yaml --filedir out/stats --outdir out/match

python pipeline/pipeline_4_fit_period.py \
    --config configs/forced_source.yaml --lightcurve-dir out/lightcurves \
    --stats-dir out/stats --matchdir out/match --outdir out/periods --bands riz

python pipeline/pipeline_5_plot.py \
    --config configs/forced_source.yaml --candidates candidates.parquet \
    --lightcurve-dir out/lightcurves --plotdir out/plots
```

Stage 4 needs `--bands` to match the bands listed in the `stats_cut` section of the config.

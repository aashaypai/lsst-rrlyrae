"""Pipeline configuration: YAML file -> frozen dataclasses.

Every pipeline stage takes ``--config path/to/config.yaml``. Use the same
config file for all stages of one run, so that column names and cuts stay
consistent from lightcurve building through plotting.
"""
import os
import dataclasses
from dataclasses import dataclass, field, asdict

import yaml


@dataclass(frozen=True)
class Columns:
    """Column names in the per-epoch (lightcurve) tables."""
    id: str          # per-object grouping key (diaObjectId, objectId, ...)
    visit: str
    mjd: str
    band: str
    ext: str         # extendedness column in the Object table
    flux: str
    flux_err: str
    mag: str
    mag_err: str
    flags: list = field(default_factory=list)


@dataclass(frozen=True)
class ButlerConfig:
    repo: str
    collections: list
    instrument: str

    def kwargs(self):
        """Keyword arguments for the query functions."""
        return {'repo': self.repo, 'collections': list(self.collections),
                'instrument': self.instrument}


@dataclass(frozen=True)
class Run:
    nside: int
    bands: str
    outdir: str
    max_workers: int


@dataclass(frozen=True)
class LightcurveCuts:
    min_points: dict
    min_mag: float
    max_mag: float
    max_mag_err: float
    band_1: str
    band_2: str
    color_min: float
    color_max: float
    max_extendedness: float
    max_sep_arcsec: float


@dataclass(frozen=True)
class Rescaling:
    min_objects_for_fit: int
    sigma: float
    maxiters: int


@dataclass(frozen=True)
class StatsCut:
    n_obs_min: object               # int or {band: int}
    n_obs_max: object
    log_reduced_chi2_min: object    # float or {band: float}
    log_reduced_chi2_max: object
    log_significance_min: object
    log_significance_max: object


@dataclass(frozen=True)
class Matching:
    max_sep_arcsec: float


@dataclass(frozen=True)
class Config:
    columns: Columns
    butler: ButlerConfig
    run: Run
    lightcurves: LightcurveCuts
    rescaling: Rescaling
    stats_cut: StatsCut
    matching: Matching


_SECTIONS = {
    'columns': Columns,
    'butler': ButlerConfig,
    'run': Run,
    'lightcurves': LightcurveCuts,
    'rescaling': Rescaling,
    'stats_cut': StatsCut,
    'matching': Matching,
}


def load_config(path):
    """Load and validate a YAML config. Missing or unknown keys raise immediately."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    missing = set(_SECTIONS) - set(raw)
    unknown = set(raw) - set(_SECTIONS)
    if missing or unknown:
        raise ValueError(f"{path}: missing sections {sorted(missing)}, unknown sections {sorted(unknown)}")

    sections = {}
    for name, cls in _SECTIONS.items():
        try:
            sections[name] = cls(**raw[name])
        except TypeError as e:
            raise ValueError(f"{path}: bad '{name}' section: {e}") from None
    return Config(**sections)


def override(cfg, section, **values):
    """Return a copy of cfg with fields in one section replaced.

    Values that are None are ignored, so CLI flags left at their default
    of None fall through to the config file.
    """
    values = {k: v for k, v in values.items() if v is not None}
    if not values:
        return cfg
    new_section = dataclasses.replace(getattr(cfg, section), **values)
    return dataclasses.replace(cfg, **{section: new_section})


def save_config(cfg, outdir, name):
    """Write the resolved config (after CLI overrides) next to a stage's output."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    with open(path, 'w') as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
    return path

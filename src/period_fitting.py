import numpy as np

from scipy.special import i0
from scipy.optimize import curve_fit

from pycycle import PeriodSearch, load_multiband_templates, average_multiband_templates, TemplateFitter
from pycycle.lsdb_utils import apply_des_to_lsst_correction

def vonmises_fisher(x, mu, kappa):
    f = np.exp(kappa * np.cos(x - mu)) / (2 * np.pi * i0(kappa))
    return f

def fit_vonmises_fisher_kappa(time, result):
    phase = ((time) / result.best_period) % 1
    angle = 2 * np.pi * phase   

    counts, edges = np.histogram(angle, bins=50, density=True) 
    centers = 0.5 * (edges[1:] + edges[:-1])

    popt, pcov = curve_fit(vonmises_fisher, centers, counts,
                           p0=[np.mean(angle), 1],       # starting guess
                           bounds=([-np.inf, 1e-9], [np.inf, np.inf]))   # keeps sigma positive
    mu, kappa = popt
    mu_err, kappa_err = np.sqrt(np.diag(pcov))
    return mu, mu_err, kappa, kappa_err

def get_period_bounds(rrl_type):
    if rrl_type == 'ab':
        pmin = 0.2
        pmax = 1
    elif rrl_type == 'c':
        pmin = 0.44
        pmax = 0.89
    else:
        raise ValueError(f"rrl_type must be either 'ab' or 'c', currently: {rrl_type}")
    return pmin, pmax
    
def avg_template_fit(time, mags, mag_errs, bands, rrl_type='ab'):
    pmin, pmax = get_period_bounds(rrl_type)

    templates = load_multiband_templates(f'/sdf/data/rubin/user/pai/Multiband-templates/RR{rrl_type}_normalized.zip')
    avg = average_multiband_templates(templates)
    apply_des_to_lsst_correction(avg, method='empirical')
    
    template_bands = avg.bands  # ['g', 'i', 'r', 'z']
    template_band_mask = np.isin(bands, template_bands)

    time = time[template_band_mask]
    mags = mags[template_band_mask]
    mag_errs = mag_errs[template_band_mask]
    bands = bands[template_band_mask]

    band_to_idx = {b:i for i, b in enumerate(np.unique(bands))}
    bands_int = np.array([band_to_idx[b] for b in bands])
    filtnams = [b for b in np.unique(bands)]

    fitter = TemplateFitter(avg)
    result = fitter.fit(time, mags, mag_errs, bands_int, filtnams=filtnams, 
                        pmin = pmin, pmax = pmax, dphi=0.01)

    return result, avg

def full_template_fit(time, mags, mag_errs, bands, rrl_type='ab'):
    pmin, pmax = get_period_bounds(rrl_type)

    templates = load_multiband_templates(f'/sdf/data/rubin/user/pai/Multiband-templates/RR{rrl_type}_normalized.zip')

    best_rss = np.inf
    best_result = None
    best_template = None
    
    for tmpl in templates:
        
        template_bands = tmpl.bands  # ['g', 'i', 'r', 'z']
        template_band_mask = np.isin(bands, template_bands)
    
        time = time[template_band_mask]
        mags = mags[template_band_mask]
        mag_errs = mag_errs[template_band_mask]
        bands = bands[template_band_mask]
    
        band_to_idx = {b:i for i, b in enumerate(np.unique(bands))}
        bands_int = np.array([band_to_idx[b] for b in bands])
        filtnams = [b for b in np.unique(bands)]
    
        apply_des_to_lsst_correction(tmpl, method='empirical')
        fitter = TemplateFitter(tmpl)
        result = fitter.fit(time, mags, mag_errs, bands_int, filtnams=filtnams, 
                            pmin = pmin, pmax = pmax, dphi=0.01)

        if result.rss.min() < best_rss:
            best_rss = result.rss.min()
            best_result = result
            best_template = tmpl

    return best_result, tmpl
    
def period_search(time, mags, mag_errs, bands, rrl_type='ab'):
    pmin, pmax = get_period_bounds(rrl_type)
    band_to_idx = {b:i for i, b in enumerate(np.unique(bands))}
    bands_int = np.array([band_to_idx[b] for b in bands])
    filtnams = [b for b in np.unique(bands)]
    ps = PeriodSearch(time, mags, mag_errs, bands_int, filtnams=filtnams)
    result = ps.run(pmin = pmin, pmax = pmax, dphi=0.01)
    return result

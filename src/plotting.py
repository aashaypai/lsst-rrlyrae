import os
import sys
import numpy as np
import pandas as pd
import glob
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch

from pycycle import load_multiband_templates, average_multiband_templates
from pycycle.lsdb_utils import apply_des_to_lsst_correction

import lsst.geom as geom
from lsst.daf.butler import Butler


plt.style.use(os.path.abspath("/sdf/data/rubin/user/pai/matplotlib_style.mplstyle"))

BAND_COLORS = {'u': '#56b4e9', 'g': '#008060', 'r': '#ff4000',
              'i': '#850000', 'z': '#6600cc', 'y': '#000000'}

DARK_BAND_COLORS = {'u':'#269EE3', 'g': '#00523D', 'r':'#CC3300',
                    'i': '#520000', 'z': '#47008F', 'y': '#000000'}

def get_preliminary_visit_image_cutouts(ra, dec, visits=None, size=151):
    """size x size pixel preliminary_visit_image cutout centred on (ra, dec) [deg].
    Returns (cutout exposure, (x, y) of the star in the cutout's array pixels)."""
    
    butler = Butler('dp2_prep', collections=['LSSTCam/runs/DRP/DP2'], instrument='LSSTCam')
    skymap_name = 'lsst_cells_v2'
    skymap = butler.get('skyMap', skymap=skymap_name)
    
    sp = geom.SpherePoint(ra * geom.degrees, dec * geom.degrees)
    pvi_refs = butler.query_datasets("preliminary_visit_image", 
                                     where=f"visit_detector_region.region OVERLAPS POINT(:ra, :dec)",
                                     bind={"ra": ra, "dec": dec})
    if visits is not None:
        final_refs = []
        for ref in pvi_refs:
            if ref.dataId['visit'] in visits:
                final_refs.append(ref)
    else:
        final_refs = pvi_refs

    cutouts = {}
    coords = {}
    bands = {}

    for i, ref in enumerate(final_refs):
        data_id = ref.dataId
        key = data_id['visit']
        
        wcs = butler.get('preliminary_visit_image.wcs', data_id)
        pix = wcs.skyToPixel(sp)                                   # in the tract's pixel frame
        cx, cy = int(round(pix.getX())), int(round(pix.getY()))
        half = size // 2
        bbox = geom.Box2I(geom.Point2I(cx - half, cy - half), geom.Extent2I(size, size))
        bbox = bbox.clippedTo(butler.get('preliminary_visit_image.bbox', data_id))   # stay inside the patch

        cutout = butler.get(ref, parameters={'bbox': bbox})   # reads only the subimage
        x0, y0 = cutout.getBBox().getMinX(), cutout.getBBox().getMinY()

        cutouts[key] = cutout
        bands[key] = data_id['band']
        coords[key] = (pix.getX() - x0, pix.getY() - y0)
    return cutouts, bands, coords
    
def get_deep_coadd_cutout(ra, dec, band, size=151):
    """size x size pixel deep_coadd cutout centred on (ra, dec) [deg].
    Returns (cutout exposure, (x, y) of the star in the cutout's array pixels)."""
    
    butler = Butler('dp2_prep', collections=['LSSTCam/runs/DRP/DP2'], instrument='LSSTCam')
    skymap_name = 'lsst_cells_v2'
    skymap = butler.get('skyMap', skymap=skymap_name)
    
    sp = geom.SpherePoint(ra * geom.degrees, dec * geom.degrees)
    tract_info = skymap.findTract(sp)
    patch_info = tract_info.findPatch(sp)
    data_id = dict(skymap=skymap_name, tract=tract_info.getId(),
                   patch=patch_info.sequential_index, band=band)

    wcs = butler.get('deep_coadd.wcs', data_id)
    pix = wcs.skyToPixel(sp)                                   # in the tract's pixel frame
    cx, cy = int(round(pix.getX())), int(round(pix.getY()))
    half = size // 2
    bbox = geom.Box2I(geom.Point2I(cx - half, cy - half), geom.Extent2I(size, size))
    bbox = bbox.clippedTo(butler.get('deep_coadd.bbox', data_id))   # stay inside the patch

    cutout = butler.get('deep_coadd', data_id, parameters={'bbox': bbox})   # reads only the subimage
    x0, y0 = cutout.getBBox().getMinX(), cutout.getBBox().getMinY()
    return cutout, (pix.getX() - x0, pix.getY() - y0)


def add_scale_bar(ax, arcsec=5, pix_scale=0.2, color='k', loc='lower left', fontsize=13):
    """Scale bar anchored to a corner of the axes; arcsec = bar length, pix_scale = arcsec/pixel."""
    ax.add_artist(AnchoredSizeBar(ax.transData, arcsec / pix_scale, f'{arcsec:g}"', loc,
                                  pad=0.6, sep=4, color=color, frameon=False, size_vertical=1,
                                  fontproperties=fm.FontProperties(size=fontsize)))

def make_phased_lightcurve_cutout_plot(time, mag, err, band, ra, dec,
                                       *, result, template, 
                                       plotdir, plotname,
                                       title_text=None):
    band_to_idx = {b:i for i, b in enumerate(template.bands)}
    
    fig, ax = plt.subplots(ncols=2, figsize=(12, 3),
                           gridspec_kw=dict(width_ratios=[1, 2]))
    
    t_max = time[np.argmin(mag)]                    
    phase = ((time) / result.best_period) % 1
    
    cutout, (x, y) = get_deep_coadd_cutout(ra, dec, 'r')
    img = cutout.image.array
    ax[0].imshow(cutout.image.array, origin='lower', cmap='gray',
                norm=ImageNormalize(img, interval=ZScaleInterval(), stretch=AsinhStretch()))
    add_scale_bar(ax[0], arcsec=5) # 5" = 25 px at 0.2"/px
    ax[0].scatter(x, y, s=100, facecolors='none', edgecolors='cyan', lw=1)
    ax[0].set_xticks([]) 
    ax[0].set_yticks([])
    ax[0].set_box_aspect(1)
    for b in np.unique(band):
        m = band == b
        
        for alpha, offset in zip([0.7, 0.4], [0, 1]):
            ax[1].errorbar(phase[m] + offset, mag[m], yerr=err[m], fmt='o', ms=4,
                    mfc=BAND_COLORS[b], c=DARK_BAND_COLORS[b], label=f'{b}', alpha=alpha)
    
            phi_grid = np.linspace(0, 1, 200)
            time_grid = phi_grid * result.best_period
            filts_grid = np.full(len(time_grid), band_to_idx[b], dtype=int)
            
            mag_template = result.predict(time_grid, filts_grid, template.bands)
            template_t_max = time_grid[np.argmin(mag_template)]
            template_phase = ((time_grid - template_t_max) / result.best_period) % 1
            ax[1].plot(phi_grid + offset, mag_template, 
                    c=BAND_COLORS[b], lw=2, zorder=-3, alpha=alpha)
    
    title_string = ""
    if title_text is not None:
        title_string += title_text
    if title_string != "":
        ax[1].set_title(title_string, fontsize=12)
    
    ax[1].set(xlabel='Phase', ylabel='mag', xlim=(0, 2), ylim=(mag.min()-0.1, mag.max()+0.1))
    ax[1].invert_yaxis()
    #ax.legend(frameon=False, ncol=len(np.unique(band)), loc='upper right')
    ax[1].minorticks_on()
    
    for a in ax:
        for spine in a.spines.values():
            spine.set_linewidth(2)
    
    os.makedirs(os.path.join(plotdir, 'lightcurves'), exist_ok=True)
    outfile = os.path.join(plotdir, 'lightcurves',  plotname)
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return outfile


def get_scaled_img(img, norm=None, interval=None, stretch=None):
    """Apply an ImageNormalize (built fresh from img if norm isn't given) and return a 0-1 float array."""
    if norm is None:
        interval = interval or ZScaleInterval()
        stretch = stretch or AsinhStretch()
        norm = ImageNormalize(img, interval=interval, stretch=stretch)
    scaled = np.asarray(norm(img))
    return np.nan_to_num(scaled, nan=0.0)
 
 
def make_pvi_cutout_gif(cutouts, coords, visits, bands, object_id, plotdir):
    if not (len(cutouts) == len(coords) == len(visits) == len(bands)):
        raise ValueError(f"length mismatch: cutouts={len(cutouts)}, coords={len(coords)}, "
                         f"visits={len(visits)}, bands={len(bands)}")   # BUG FIX: catch silent zip() truncation early
 
    os.makedirs(os.path.join(plotdir, 'tmp'), exist_ok=True)
    gif_dir = os.path.join(plotdir, 'pvi_gifs')
    os.makedirs(gif_dir, exist_ok=True)   # BUG FIX: this directory was never created before Image.save()
 
    stack = np.stack([c.image.array for c in cutouts])
    norm = ImageNormalize(stack, interval=ZScaleInterval(), stretch=AsinhStretch())
    frames = [get_scaled_img(c.image.array, norm=norm) for c in cutouts]
 
    for frame, coord, visit, band in zip(frames, coords, visits, bands):
        x, y = coord
        fig, ax = plt.subplots()
 
        ax.imshow(frame, origin='lower', cmap='gray')
        add_scale_bar(ax, arcsec=5, color='cyan')   # 5" = 25 px at 0.2"/px
        ax.scatter(x, y, s=100, facecolors='none', edgecolors='cyan', lw=1)
        ax.set_title(f"{visit}, band: {band}", fontsize=12, color=BAND_COLORS[band])   # BUG FIX: was band_colors (undefined)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        savepath = os.path.join(plotdir, 'tmp', f"{object_id}_{visit}_{band}.png")
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
 
    gif_frames = []
    for visit, band in zip(visits, bands):
        fp = os.path.join(plotdir, 'tmp', f"{object_id}_{visit}_{band}.png")
        im = Image.open(fp)
        gif_frames.append(im.convert('RGB'))
 
    gif_savepath = os.path.join(gif_dir, f"{object_id}_pvi.gif")
    gif_frames[0].save(gif_savepath, format='GIF',
                       append_images=gif_frames[1:], save_all=True, duration=500,
                       loop=0, optimize=False)

    for im in gif_frames:                                   # release file handles before deleting
        im.close()

        frame_paths = glob.glob(os.path.join(plotdir, 'tmp', f"{object_id}_*.png"))
    for fp in frame_paths:                                  # NEW: remove only this object's PNGs
        os.remove(fp)
 
    return gif_savepath

    
def make_phased_lightcurve_plot(time, mags, mag_errs, bands, *,
                                period_search_period, template_result, template,
                                plotdir, object_id=None, 
                                match=False, id_col='ID'):
    mb_bands = template.bands
    band_to_idx_template = {b: i for i, b in enumerate(mb_bands)}

    unique_bands = np.unique(bands)
    fig, axes = plt.subplots(len(unique_bands), 3,
                             figsize=(16, 3.2 * len(unique_bands)),
                             tight_layout=True, squeeze=False)

    for row_num, b in enumerate(unique_bands):
        band_mask = bands == b
        t, mag, err = time[band_mask], mags[band_mask], mag_errs[band_mask]
        color = BAND_COLORS.get(b, 'k')

        # --- col 1: raw ---
        ax = axes[row_num, 0]
        ax.errorbar(t, mag, yerr=err, fmt='o', ms=3, color=color, ecolor='gray', capsize=2)
        ax.invert_yaxis()
        ax.set_ylabel(f'{b} mag')
        ax.set_xlabel('MJD')
        if row_num == 0:
            ax.set_title('Raw lightcurve')

        # --- col 2: PeriodSearch fold ---
        phase = (t / period_search_period) % 1
        ax2 = axes[row_num, 1]
        ax2.errorbar(phase, mag, yerr=err, fmt='o', ms=3, color=color, capsize=2, alpha=0.7)
        ax2.errorbar(phase + 1, mag, yerr=err, fmt='o', ms=3, color=color, capsize=2, alpha=0.2)
        ax2.invert_yaxis()
        ax2.set_xlabel('Phase')
        ax2.set_ylabel(f'{b} mag')
        if row_num == 0:
            ax2.set_title(f'PeriodSearch fold (P = {period_search_period:.5f} d)')

        # --- col 3: TemplateFitter fold + model overlay ---
        ax3 = axes[row_num, 2]
        phase_t = (t / template_result.best_period) % 1
        ax3.errorbar(phase_t, mag, yerr=err,
                    fmt='o', ms=3, color=color, capsize=2, alpha=0.7)
        ax3.errorbar(phase_t + 1, mag, yerr=err,
                    fmt='o', ms=3, color=color, capsize=2, alpha=0.2)

        if b in band_to_idx_template:
            phi_grid = np.linspace(0, 1, 200)
            time_grid = phi_grid * template_result.best_period
            filts_grid = np.full(len(time_grid), band_to_idx_template[b], dtype=int)
            mag_template = template_result.predict(time_grid, filts_grid, mb_bands)
            ax3.plot(phi_grid, mag_template, c='#003049', lw=2, zorder=5)
            ax3.plot(phi_grid + 1, mag_template, c='#003049', lw=2, zorder=5, alpha=0.4)
        else:
            ax3.text(0.5, 0.5, f'{b}-band not in template', transform=ax3.transAxes,
                     ha='center', va='center')

        ax3.invert_yaxis()
        ax3.set_xlabel('Phase')
        ax3.set_ylabel(f'{b} mag')
        if row_num == 0:
            ax3.set_title(f'Template fit (P = {template_result.best_period:.5f} d)')

    for ax in axes.flat:
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    title_string = ""
    if object_id is not None:
        title_string += f"{id_col}: {object_id}"
    if match:
        title_string += f" known RRL"
    if title_string != "":
        fig.suptitle(title_string)
        
    os.makedirs(plotdir, exist_ok=True)
    outfile = os.path.join(plotdir, f"{object_id}_lc.png")
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return outfile
#!/usr/bin/env python
from __future__ import division, print_function

import numpy
import pylab
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from numpy import arange, histogram, log10, newaxis, ones, sort, sum as npsum
from os import makedirs
from os.path import isdir, join

# my code - just using my preferred settings
try:
    import plottools
    plottools.update_rcParams()
except ImportError:
    pass


def main(maxdist=0.4*u.arcsec, plot_path='plots'):
    # create plot folder if it doesn't exist
    if not isdir(plot_path):
        makedirs(plot_path)
    cosmos_galaxies = cosmos_objects(mu_class=1)
    cosmos_stars = cosmos_objects()
    cosmos = {}
    good = {}
    stars = {}
    galaxies = {}
    fig, ax = pylab.subplots()
    for seeing in ('best', 'median', 'worst'):
        cosmos[seeing], good[seeing] = cosmos_sources(seeing)
        stars[seeing] = \
            match(cosmos[seeing]['ira'][good[seeing]],
                  cosmos[seeing]['idec'][good[seeing]], *cosmos_stars,
                  label='stars', seeing=seeing, maxdist=maxdist,
                  fig=fig, ax=ax)[2]
        galaxies[seeing] = \
            match(cosmos[seeing]['ira'][good[seeing]],
                  cosmos[seeing]['idec'][good[seeing]], *cosmos_galaxies,
                  label='galaxies', seeing=seeing, maxdist=maxdist,
                  fig=fig, ax=ax)[2]
        print('')
    print('names = {0}'.format(
            sort([i for i in cosmos['best'].names if 'regauss' in i])))
    sigma = cosmos['best']['ishape_hsm_regauss_sigma']
    print('sigma = {0:.3f} +/- {1:.3f}'.format(
            numpy.average(sigma), numpy.std(sigma)))
    print('')
    ax.axvline(log10(maxdist.value), ls='--', lw=1)
    ax.legend(loc='upper left')
    savefig(join(plot_path, 'hist_matches.png'), fig=fig)
    plot(cosmos, good, stars, galaxies, plot_path=plot_path)
    return


def cosmos_sources(seeing, with_cuts=True, path='../catalogs/COSMOS',
                   filename='COSMOS_wide_{0}_v3_withwlcuts.fits'):
    cat = fits.getdata(join(path, filename.format(seeing)))
    if with_cuts:
        sources = ones(cat['ira'].size, dtype=bool)
        print('Supplied catalog contains only {0} valid sources'.format(
                cat['ira'].size))
    else:
        mag = '{0}mag_forced_cmodel'
        err = '{0}mag_forced_cmodel_err'
        snr = npsum([(10**(0.4*(cat[mag.format(i)]-cat[err.format(i)])) > 5)
                     for i in 'grzy'], axis=0)
        sources = cat['weak_lensing_flag'] & (snr >= 2) & \
                  ~cat['iflags_pixel_bright_object_center'] & \
                  log10(cat['iblendedness_abs_flux']) < -0.375
        print('{0}/{1} objects pass'.format(flag[flag].size, flag.size))
    return cat, sources


def cosmos_objects(mu_class=2):
    filename = 'input/cosmos_sg_all_GOLD.fits'
    cat = fits.getdata(filename)
    ra, dec = cat['coord'].T
    obj = (cat['mu_class'] == mu_class)
    print('Using {0}/{1} true objects'.format(ra[obj].size, ra.size))
    return ra[obj], dec[obj]


def match(ra, dec, ra_stars, dec_stars, label='stars', seeing='median',
          maxdist=0.4*u.arcsec, fig=None, ax=None, save=False,
          plot_path='plots'):
    print('Matching {0} sources to {1} {2} with {3} seeing'.format(
            ra.size, ra_stars.size, label, seeing))
    sources = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    objects = SkyCoord(ra=ra_stars*u.degree, dec=dec_stars*u.degree)
    ref, d2d, d3d = sources.match_to_catalog_sky(objects)
    good = (d2d < maxdist)
    indices = numpy.arange(ra.size, dtype=int)[good]
    print('Found {0}/{1} successful matches (max={2})'.format(
            indices.size, ref.size, indices.max()))
    # plot minimum distances
    dist = d2d.to(u.arcsec).value
    if fig is None:
        fig, ax = pylab.subplots()
    hist, edges = ax.hist(
        log10(dist), bins=100, label='{0} {1}'.format(seeing, label),
        histtype='step', lw=2, log=True, bottom=1)[:2]
    ax.set_xlabel('log distance to closest match (arcsec)')
    ax.set_ylabel('1+N')
    if save:
        savefig(join(plot_path, 'hist_matches.png'), fig=fig)
    return fig, ax, indices


def plot(cosmos, good, stars, galaxies, rms=0.37, plot_path='plots'):
    """
    Plot the contamination as a function of magnitude.
    The first three arguments are dictionaries with seeing as keys

    """
    magbins = arange(15, 25.1, 0.5)
    mag = (magbins[:-1]+magbins[1:]) / 2
    keys = ('best', 'median', 'worst')
    # bin by magnitude
    fig, ax = pylab.subplots()
    for i, key in enumerate(keys):
        cat = cosmos[key]
        imag = cat['imag_forced_cmodel'][good[key]]
        # overall ratios
        ax.axhline(imag[stars[key]].size/imag.size, ls=':', lw=1,
                   color='C{0}'.format(i))
        # binned fractions
        ntot = histogram(imag, magbins)[0]
        nstars = histogram(imag[stars[key]], magbins)[0]
        ngals = histogram(imag[galaxies[key]], magbins)[0]
        ax.plot(mag, nstars/ntot, 'C{0}-'.format(i),
                label='{0} stars'.format(key))
        #ax.plot(mag, ngals/ntot, '--', label='{0} galaxies'.format(key))
        # now weighted fractions
        weight = 1 / (cat['ishape_hsm_regauss_sigma']**2 + rms**2)
        wtot = histogram(imag, magbins, weights=weight)[0]
        wstars = histogram(imag[stars[key]], magbins,
                           weights=weight[stars[key]])[0]
        wgals = histogram(imag[galaxies[key]], magbins,
                          weights=weight[galaxies[key]])[0]
        ax.plot(mag, wstars/wtot, 'C{0}--'.format(i),
                label='{0} w-stars'.format(key))
    ax.legend(loc='upper center')
    ax.set_xlabel('i-band magnitude')
    ax.set_ylabel('fraction of stars')
    savefig(join(plot_path, 'purity.png'), fig=fig)
    return


def savefig(output, fig=None, close=True, verbose=True, tight=True,
            tight_kwargs={'pad': 0.4}):
    """
    Wrapper to save figures, stolen from my own plottools
    (https://github.com/cristobal-sifon/plottools).

    Parameters
    ----------
        output  : str
                  Output file name (including extension)

    Optional parameters
    -------------------
        fig     : pyplot.figure object
                  figure containing the plot.
        close   : bool
                  Whether to close the figure after saving.
        verbose : bool
                  Whether to print the output filename on screen
        tight   : bool
                  Whether to call `tight_layout()`
        tight_kwargs : dict
                  keyword arguments to be passed to `tight_layout()`

    """
    if fig is None:
        fig = pylab
    if tight:
        fig.tight_layout(**tight_kwargs)
    fig.savefig(output)
    if close:
        pylab.close()
    if verbose:
        print('Saved to {0}'.format(output))
    return


main()




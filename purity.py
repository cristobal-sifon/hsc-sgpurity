#!/usr/bin/env python
from __future__ import division, print_function

import numpy
import pylab
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from itertools import izip
from matplotlib import ticker
from numpy import arange, linspace, log10, newaxis, ones, sort, sum as npsum
from os import makedirs
from os.path import isdir, join

# my code - just using my preferred settings
try:
    import plottools
    plottools.update_rcParams()
    do_hist = True
except ImportError:
    msg = 'WARNING: module plottools not available - histograms will not be' \
          ' generated. Please clone/download from' \
          ' https://github.com/cristobal-sifon/plottools'
    print(msg)
    do_hist = False


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
        print_columns = (seeing == 'best')
        cosmos[seeing], good[seeing] = \
            cosmos_sources(seeing, print_columns=print_columns)
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

    ## calculate and plot purity
    # these are masks to test the purity to different conditions (e.g.,
    # photo-z)
    #masks = [[],
             #(
    purity(cosmos, good, stars, galaxies, plot_path=plot_path)

    ## plot distributions of stars vs. galaxies
    if do_hist:
        keys = ('iblendedness_abs_flux', 'ishape_hsm_regauss_e1',
                'ishape_hsm_regauss_e2', 'ishape_hsm_regauss_resolution')
        keybins = (linspace(0, 0.45, 51), linspace(-2, 2, 51),
                   linspace(-2, 2, 51), linspace(0.3, 1, 51))
        for key, bins in izip(keys, keybins):
            histogram(cosmos, good, stars, galaxies, key, bins=bins,
                      plot_path=plot_path)
    return


def cosmos_objects(mu_class=2):
    filename = 'input/cosmos_sg_all_GOLD.fits'
    cat = fits.getdata(filename)
    ra, dec = cat['coord'].T
    obj = (cat['mu_class'] == mu_class)
    print('Using {0}/{1} true objects'.format(ra[obj].size, ra.size))
    return ra[obj], dec[obj]


def cosmos_sources(seeing, with_cuts=True, path='../catalogs/COSMOS',
                   filename='COSMOS_wide_{0}_v3_withwlcuts.fits',
                   print_columns=False):
    cat = fits.getdata(join(path, filename.format(seeing)))
    if print_columns:
        print('\ncolumns: {0}\n'.format(sort(cat.names)))
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


def histogram(cosmos, good, stars, galaxies, key, bins=50, plot_path='plots'):
    hist_path = join(plot_path, 'histograms')
    #cdf_path = join(cdf_path, 'cdf'
    if not isdir(hist_path):
        makedirs(hist_path)
        #makedirs(cdf_path)
    fig, ax = pylab.subplots()
    #cfig, cax = pylab.subplots()
    i = 0
    for seeing in ('best', 'median', 'worst'):
        color = 'C{0}'.format(i)
        s = cosmos[seeing][key][good[seeing]][stars[seeing]]
        ax.hist(s, bins, color='C{0}'.format(i), ls='-', lw=2,
                log=True, bottom=1,
                histtype='step', label='{0} stars'.format(seeing))
        g = cosmos[seeing][key][good[seeing]][galaxies[seeing]]
        ax.hist(g, bins, color='C{0}'.format(i+1), ls='-', lw=2,
                log=True, bottom=1,
                histtype='step', label='{0} galaxies'.format(seeing))
        i += 2
    ax.legend(loc='upper center')
    ax.set_xlabel(key.replace('_', '\_'))
    ax.set_ylabel('1+n')
    savefig(join(hist_path, 'hist-{0}.png'.format(key)), fig=fig)
    return


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


def purity(cosmos, good, stars, galaxies, rms=0.37, plot_path='plots',
           mask=[], show_weights=False, show_errors=True):
    """
    Calculate and plot the contamination as a function of magnitude.
    The first four arguments are dictionaries with seeing as keys

    """
    magbins = arange(16.6, 26, 0.6)
    mag = (magbins[:-1]+magbins[1:]) / 2
    keys = ('best', 'median', 'worst')
    overall = {key: 0 for key in keys}
    # bin by magnitude
    #fig, ax = pylab.subplots()
    fig = pylab.figure(figsize=(7,7))
    ax = pylab.subplot2grid((4,1), (0,0), rowspan=3)
    ax.set_xticklabels([])
    # axis for histograms
    hax = pylab.subplot2grid((4,1), (3,0))
    # main lines
    for i, key in enumerate(keys):
        imag = cosmos[key]['imag_forced_cmodel'][good[key]]
        color = 'C{0}'.format(i)
        # pass these weights to plot() if I want to show them
        weight = 1 / (cosmos[key]['ishape_hsm_regauss_sigma']**2 + rms**2)
        plot(ax, key, imag, stars[key], galaxies[key],
             mag, magbins, color=color, dx=0.1*(i-1))
        ntot = hax.hist(imag, magbins, histtype='step',
                        lw=2, color=color, log=True, bottom=1)[0]
    ax.legend(loc='upper center')
    # ticks and so on
    for i in (ax, hax):
        i.set_xlim(16.6, 25.4)
        i.xaxis.set_major_locator(ticker.MultipleLocator(2))
        i.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.001))
    hax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%d$'))
    ax.set_ylabel('stellar contamination')
    hax.set_xlabel('i-band magnitude')
    hax.set_ylabel('1+N')
    savefig(join(plot_path, 'purity.png'), fig=fig,
            tight_kwargs={'pad': 0.4, 'h_pad': 0.25})
    return


def plot(ax, key, data, stars, galaxies, mag, magbins, color='C0',
         dx=0, weight=[], show_errors=True):
    # overall ratios
    overall = data[stars].size/data.size
    ax.axhline(overall, ls=':', lw=1, color=color)
    print('Overall purity for {0} seeing is {1:.4f}'.format(
            key, overall))
    # binned fractions
    # uncomment these two (and comment fhe next two) instructions
    # to show the numerators instead of the denominators in the
    # bottom panel
    ntot = numpy.histogram(data, magbins)[0]
    #nstars = hax.hist(data[stars[key]], magbins, histtype='step',
                      #lw=2, color=color, log=True, bottom=1)[0]
    nstars = numpy.histogram(data[stars], magbins)[0]
    ngals = numpy.histogram(data[galaxies], magbins)[0]
    if len(weight) > 0:
        label = '{0}-n'.format(key.capitalize())
    else:
        label = key.capitalize()
    # Poisson errorbars
    err = 1 / nstars**0.5 / ntot
    if show_errors:
        ax.errorbar(dx+mag[nstars > 0], (nstars/ntot)[nstars > 0],
                    yerr=err[nstars > 0], fmt='-', color=color, mew=2,
                    capsize=1.5)
        ax.plot([], [], '-', color=color, label=label)
    else:
        ax.plot(mag[nstars > 0], (nstars/ntot)[nstars > 0],
                '-', color=color, label=label)
    #ax.plot(mag, ngals/ntot, '--', label='{0} galaxies'.format(key))
    # now weighted fractions
    if len(weight) > 0:
        wtot = numpy.histogram(data, magbins, weights=weight)[0]
        wstars = numpy.histogram(
            data[stars], magbins, weights=weight[stars])[0]
        wgals = numpy.histogram(
            data[galaxies], magbins, weights=weight[galaxies])[0]
        ax.plot(mag, wstars/wtot, '--', color=color,
                label=label.replace('-n', '-w'))
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




#!/usr/bin/env python
from __future__ import division, print_function

import numpy
import pylab
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from itertools import izip
from os import makedirs
from os.path import isdir, join

# my code - download from https://github.com/cristobal-sifon/plottools
import plottools
plottools.update_rcParams()

# local
import hst


def main():
    for ref in ('stars', 'galaxies', 'all'):
        second = {}
        for seeing in ('best', 'median', 'worst'):
            print('\n ** {0} seeing **'.format(seeing.capitalize()))
            second[seeing] = match_two(seeing, ref=ref)
            #break
        hist_second(second, ref=ref)
    return


def match_two(seeing, maxdist=2*u.arcsec, ref='stars'):
    """
    Find the next-nearest HST star to every star in the HSC source
    catalog. The goal here is to find plausible binaries so `maxdist`
    need not be very large - in fact, it should be smaller than the HSC
    PSF, but let's make it a little larger.

    """
    stars = ascii.read('output/stars_{0}.cat'.format(seeing))
    xystars = SkyCoord(ra=stars['ira']*u.degree, dec=stars['idec']*u.degree)
    matched = hst.match(xystars, ref=ref)
    if ref == 'stars':
        ra_hst, dec_hst = hst.stars['coord'].T
        hst_stars = hst.skycoord(hst.stars)
    elif ref == 'galaxies':
        ra_hst, dec_hst = hst.galaxies['coord'].T
        hst_stars = hst.skycoord(hst.galaxies)
    elif ref == 'all':
        ra_hst = hst.ra
        dec_hst = hst.dec
        hst_stars = hst.objects
    hst_matches = SkyCoord(ra=ra_hst[matched]*u.degree,
                           dec=dec_hst[matched]*u.degree)
    # find the next closest object in the HST catalog
    neighbors = []
    separation = []
    closest = numpy.zeros(ra_hst[matched].size)
    rng = numpy.arange(hst_stars.size, dtype=int)
    for i, xy in enumerate(hst_matches):
        sep = xy.separation(hst_stars)
        neighbors.append(rng[sep < maxdist])
        separation.append(sep[neighbors[i]])
        #print('closest = {0}'.format(numpy.sort(sep.to(u.arcsec).value)[:5]))
        closest[i] = numpy.sort(sep)[1].to(u.arcsec).value
    # number of neighbors
    nn = numpy.array([len(n) for n in neighbors])
    # candidates to be binary stars
    candidates = (nn > 1)
    print('{0}/{1} binary candidates'.format(nn[candidates].size, nn.size))
    return closest


def hist_second(dist, ref='stars', plot_path='plots'):
    if not isdir(plot_path):
        makedirs(plot_path)
    bins = numpy.logspace(-2, 2, 26)
    fig, ax = pylab.subplots()
    name = {'stars': 'star', 'galaxies': 'galaxy', 'all': 'object'}
    ax.annotate('Closest {0} (arcsec):'.format(name[ref]), xy=(0.03,0.6),
                xycoords='axes fraction', color='k', fontsize=16,
                ha='left', va='bottom')
    for i, seeing in enumerate(('best', 'median', 'worst')):
        color = 'C{0}'.format(i)
        ax.hist(dist[seeing], bins, histtype='step', lw=4-i, color=color,
                log=True, bottom=1, label=seeing.capitalize(), zorder=i)
        msg = '{0:.2f}'.format(dist[seeing].min())
        ax.annotate(msg, xy=(0.1,0.54-0.07*i), xycoords='axes fraction',
                    color=color, fontsize=16, ha='left', va='bottom')
    ax.legend(loc='upper left')
    ax.set_xscale('log')
    ax.set_xlabel('Distance to nearest {0} (arcsec)'.format(name[ref]))
    ax.set_ylabel('1+Nstars')
    plottools.savefig(join(plot_path, 'nearest_{0}.png'.format(ref)), fig=fig)
    return

main()




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
    second = {}
    for seeing in ('best', 'median', 'worst'):
        print('\n ** {0} seeing **'.format(seeing.capitalize()))
        second[seeing] = match_two(seeing)
        #break
    hist_second(second)
    return


def match_two(seeing, maxdist=2*u.arcsec):
    """
    Find the next-nearest HST star to every star in the HSC source
    catalog. The goal here is to find plausible binaries so `maxdist`
    need not be very large - in fact, it should be smaller than the HSC
    PSF, but let's make it a little larger.

    """
    stars = ascii.read('output/stars_{0}.cat'.format(seeing))
    xystars = SkyCoord(ra=stars['ira']*u.degree, dec=stars['idec']*u.degree)
    matched = hst.match(xystars, ref='stars')
    ra_hst, dec_hst = hst.stars['coord'].T
    hst_matches = SkyCoord(ra=ra_hst[matched]*u.degree,
                           dec=dec_hst[matched]*u.degree)
    # find the next closest object in the HST catalog
    neighbors = []
    separation = []
    closest = numpy.zeros(ra_hst[matched].size)
    hst_stars = hst.skycoord(hst.stars)
    rng = numpy.arange(hst_stars.size, dtype=int)
    for i, xy in enumerate(hst_matches):
        sep = xy.separation(hst_stars)
        neighbors.append(rng[sep < maxdist])
        separation.append(sep[neighbors[i]])
        closest[i] = numpy.sort(sep)[1].to(u.arcsec).value
    # number of neighbors
    nn = numpy.array([len(n) for n in neighbors])
    # candidates to be binary stars
    candidates = (nn > 1)
    print('{0}/{1} binary candidates'.format(nn[candidates].size, nn.size))
    return closest


def hist_second(dist, plot_path='plots'):
    if not isdir(plot_path):
        makedirs(plot_path)
    bins = numpy.logspace(-2, 2, 26)
    fig, ax = pylab.subplots()
    ax.annotate('Closest star (arcsec):', xy=(0.03,0.6),
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
    ax.set_xlabel('Distance to nearest star (arcsec)')
    ax.set_ylabel('1+Nstars')
    plottools.savefig(join(plot_path, 'distances_companion.png'), fig=fig)
    return

main()




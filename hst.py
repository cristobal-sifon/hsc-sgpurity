from __future__ import division, print_function

import numpy
from astropy import coordinates, units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits


# the entire catalog
filename = 'input/cosmos_sg_all_GOLD.fits'
catalog = fits.getdata(filename)
ra, dec = catalog['coord'].T
galaxies = (catalog['mu_class'] == 1)
stars = (catalog['mu_class'] == 2)
objects = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)


def closest(xy, ref='stars'):
    """
    Find the closest match in the HST catalog to every source

    """
    if type(xy) != coordinates.sky_coordinate.SkyCoord:
        xy = SkyCoord(ra=xy[0]*u.degree, dec=xy[1]*u.degree)
    if ref == 'stars':
        sample = stars
    elif ref == 'galaxies':
        sample = galaxies
    return xy.match_to_catalog_sky(objects[sample]) # = ref, d2d, d3d


def match(xy, ref='stars', maxdist=0.4*u.arcsec, verbose=True):
    """
    Find matching HST sources up to a given maximum matching distance

    """
    ref, d2d, d3d = closest(xy, ref=ref)
    good = (d2d < maxdist)
    indices = numpy.arange(ref.size, dtype=int)[good]
    if verbose:
        print('Found {0}/{1} successful matches'.format(
                indices.size, ref.size))
    return indices


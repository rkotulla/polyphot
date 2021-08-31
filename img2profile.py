#!/usr/bin/env python3

import os
import sys
import astropy.io.fits as pyfits
import astropy.wcs as astwcs
import numpy
import pandas
import scipy
import matplotlib.pyplot as plt
import ephem
import logging
import argparse


if __name__ == "__main__":

    cmdline = argparse.ArgumentParser()
    cmdline.add_argument("--dryrun", dest="dryrun", default=False, action='store_true',
                         help="dry-run only, no database ingestion")
    cmdline.add_argument("--debug", dest="debug", default=False, action='store_true',
                         help="output debug output")
    cmdline.add_argument("files", nargs="+",
                         help="list of input filenames")
    cmdline.add_argument("--mask", dest="mask_region_fn", default=None, type=str, nargs="+",
                         help='region filename for source masking')

    args = cmdline.parse_args()

    logging.basicConfig(format='%(name)s -- %(levelname)s: %(message)s', level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger("PolyFlux")
    logger.info("Sky parameters: %f // %f" % (args.deadspace, args.skywidth))


    masked_img_fn = sys.argv[1]
    img_hdu = pyfits.open(masked_img_fn)
    img = img_hdu[0].data
    wcs = astwcs.WCS(img_hdu[0].header)
    _pixelscale = astwcs.utils.proj_plane_pixel_scales(wcs)
    pixelscale = _pixelscale[0] * 3600.

    center_ra = sys.argv[2]
    center_dec = sys.argv[3]
    eq = ephem.Equatorial(center_ra, center_dec)

    distance = float(sys.argv[4])

    background = float(sys.argv[5])

    img -= background

    center_x, center_y = wcs.all_world2pix(
        numpy.rad2deg(eq.ra), numpy.rad2deg(eq.dec), 0)
    print("center at X/Y = %d / %d" % (center_x, center_y))

    iy,ix = numpy.indices(img.shape, dtype=numpy.float)
    ix -= center_x
    iy -= center_y

    radius_pixels = numpy.hypot(ix,iy)
    radius_arcsec = radius_pixels * pixelscale
    radius_kpc = numpy.tan(numpy.deg2rad(radius_arcsec/3600.)) * distance * 1000

    kpc2_per_pixel = (numpy.tan(numpy.deg2rad(pixelscale/3600.)) * distance * 1000)**2

    pyfits.PrimaryHDU(data=radius_kpc, header=img_hdu[0].header).writeto("radius_kpc.fits", overwrite=True)

    bins = numpy.arange(0,31,1)

    results = pandas.DataFrame(columns=['ri', 'ro', 'n_pixels', 'total_flux', 'avg', 'kpc2area'])
    for i in range(bins.shape[0]-1):
        ri = bins[i]
        ro = bins[i+1]
        print("Working on bin %d: %f -- %f" % (i, ri, ro))

        in_bin = (radius_kpc >= ri) & (radius_kpc < ro) & numpy.isfinite(img)
        # in_bin_img = img[in_bin_mask]

        n_pixels = numpy.sum(in_bin)
        total_flux = numpy.sum(img[in_bin])
        avg = total_flux / n_pixels

        results.loc[i, 'ri'] = ri
        results.loc[i, 'ro'] = ro
        results.loc[i, 'n_pixels'] = n_pixels
        results.loc[i, 'total_flux'] = total_flux
        results.loc[i, 'avg'] = avg
        results.loc[i, 'kpc2area'] = n_pixels * kpc2_per_pixel

    output_fn = sys.argv[6]
    results.to_csv(output_fn, index=False)


#!/usr/bin/env python3

import sys
import astropy.io.fits as pyfits
import astropy.wcs as astwcs
import numpy
import pandas
import matplotlib.pyplot as plt
import logging
import argparse
import astropy.coordinates
import astropy.units as u

import mask_image_regions

if __name__ == "__main__":

    cmdline = argparse.ArgumentParser()
    cmdline.add_argument("--dryrun", dest="dryrun", default=False, action='store_true',
                         help="dry-run only, no database ingestion")
    cmdline.add_argument("--debug", dest="debug", default=False, action='store_true',
                         help="output debug output")
    cmdline.add_argument("--mask", dest="mask_region_fn", default=None, type=str, nargs="*",
                         help='region filename for source masking')
    cmdline.add_argument("--response", dest="response", default=None, type=str,
                         help='filename for response/flatfield frame')
    cmdline.add_argument("--center", dest="center_coord", type=str, default=None,
                         help="if provided, calculate distance between source and center (format: HMS+dms, eg 14:23:45+23:45:56)")
    cmdline.add_argument("--distance", dest="distance", default=0, type=float,
                     help='distance to source in Mpc')
    cmdline.add_argument("--background", dest="background", default=0, type=float,
                     help='background level')
    cmdline.add_argument("--radii", dest="radii", default=0, type=float, nargs="+",
                     help='radii bins')
    cmdline.add_argument("--radius", dest="bin_unit", default="arcmin", type=str,
                     help='unit for radii bins')
    cmdline.add_argument("--autobg", dest="auto_background", default=0, type=int,
                     help='unit for radii bins')
    cmdline.add_argument("--output", dest="output_fn", default="output_profile.csv", type=str,
                     help='output filename')


    cmdline.add_argument("files", nargs="+",
                         help="list of input filenames")

    args = cmdline.parse_args()

    logging.basicConfig(format='%(name)s -- %(levelname)s: %(message)s', level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger("Img2Profile")
    # logger.info("Sky parameters: %f // %f" % (args.deadspace, args.skywidth))

    # print(args.center_coord)
    center_pos = astropy.coordinates.SkyCoord(args.center_coord, unit=(u.hourangle, u.deg))
    center_ra = center_pos.ra.deg  # sys.argv[2]
    center_dec = center_pos.dec.deg  # sys.argv[3]
    logger.info("Center coordinates: %f %+f (%s)" % (center_ra, center_dec, args.center_coord))

    # eq = ephem.Equatorial(center_ra, center_dec)

    for img_fn in args.files:

        logger.info("Reading data from %s" % (img_fn))

        img_hdu = pyfits.open(img_fn)
        img = img_hdu[0].data
        hdr = img_hdu[0].header
        wcs = astwcs.WCS(hdr)
        _pixelscale = astwcs.utils.proj_plane_pixel_scales(wcs)
        pixelscale = _pixelscale[0] * 3600.

        if (args.response is not None):
            try:
                response_hdu = pyfits.open(args.response)
                response = response_hdu[0].data
                max_response = numpy.percentile(response, 95)
                poor_response = response < 0.2 * max_response
                img[poor_response] = numpy.NaN
            except:
                logger.warning("Unable to handle response read from %s" % (args.response))
                pass



        logger.info("Applying all masks")
        total_mask = mask_image_regions.generate_mask(img=img, hdr=hdr, reg_fn_list=args.mask_region_fn)

        center_x, center_y = wcs.all_world2pix(center_ra, center_dec, 0)
            # numpy.rad2deg(eq.ra), numpy.rad2deg(eq.dec), 0)
        logger.info("center at X/Y = %d / %d" % (center_x, center_y))

        iy, ix = numpy.indices(img.shape, dtype=numpy.float)
        ix -= center_x
        iy -= center_y

        background_level = background_noise = 0.
        if (args.auto_background > 0):
            bg_img = numpy.array(img)
            bg_img[total_mask] = numpy.NaN

            bg_size = 20
            bg_xy = numpy.floor(numpy.random.random((args.auto_background,2)) * [img.shape[1]-bg_size, img.shape[0]-bg_size]).astype(numpy.int)
            print(bg_xy.shape)
            print(bg_xy[:20])
            bg_values = numpy.empty((args.auto_background))
            bg_values[:] = numpy.NaN
            for i in range(args.auto_background):
                _x,_y = bg_xy[i]
                bg_values[i] = numpy.median( bg_img[_y:_y+bg_size, _x:_x+bg_size] )
            print(bg_values)

            good_bg = numpy.isfinite(bg_values)
            for iter in range(3):
                _stats = numpy.nanpercentile(bg_values[good_bg], [16,50,84])
                _median = _stats[1]
                _sigma = 0.5*(_stats[2] - _stats[0])
                good_bg = (bg_values > _median - 3*_sigma) & (bg_values < _median + 3*_sigma)
            background_level = numpy.median(bg_values[good_bg])
            background_noise = numpy.std(bg_values[good_bg])
            logger.info("Background: %f +/- %f" % (background_level, background_noise))

        radius_pixels = numpy.hypot(ix, iy)
        radius_arcsec = radius_pixels * pixelscale

        bins_raw = numpy.array([0] + args.radii)
        if (args.bin_unit == "arcmin"):
            bins_pixels = bins_raw * 60 / pixelscale
        elif (args.bin_unit == "arcsec"):
            bins_pixels = bins_raw / pixelscale
        elif (args.bin_unit == "pixel"):
            bins_pixels = bins_raw

        # bins = numpy.arange(0, 31, 1)
        results = pandas.DataFrame(columns=['ri', 'ro', 'n_pixels', 'n_pixels_raw', 'total_flux', 'avg', 'kpc2area'])
        for i in range(bins_pixels.shape[0] - 1):

            ri = bins_pixels[i]
            ro = bins_pixels[i + 1]
            logger.debug("Working on bin %d: %f -- %f" % (i, ri, ro))

            raw_in_bin = (radius_pixels >= ri) & (radius_pixels < ro)
            in_bin = raw_in_bin & numpy.isfinite(img) & ~total_mask
            raw_in_bin_count = numpy.sum(raw_in_bin)
            # in_bin_img = img[in_bin_mask]

            n_pixels = numpy.sum(in_bin)
            total_flux = numpy.sum(img[in_bin])
            avg = total_flux / n_pixels

            results.loc[i, 'ri_px'] = ri
            results.loc[i, 'ro_px'] = ro
            results.loc[i, 'n_pixels'] = n_pixels
            results.loc[i, 'n_pixels_raw'] = raw_in_bin_count
            results.loc[i, 'total_flux'] = total_flux
            results.loc[i, 'avg_raw'] = avg


            # results.loc[i, 'kpc2area'] = n_pixels * kpc2_per_pixel
        results['avg'] = results['avg_raw'] - background_level
        results['ri_arcmin'] = results['ri_px'] * pixelscale / 60.
        results['ro_arcmin'] = results['ro_px'] * pixelscale / 60.
        results['ri_kpc'] = numpy.arctan(numpy.deg2rad(results['ri_arcmin'] / 60.)) * args.distance * 1000.
        results['ro_kpc'] = numpy.arctan(numpy.deg2rad(results['ro_arcmin'] / 60.)) * args.distance * 1000.

        # finally, make a quick plot of the data
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if (args.distance > 0 and False):
            bin_center = (results['ri_kpc'] + results['ro_kpc']) / 2.
            bin_width = results['ro_kpc'] - results['ri_kpc']
            ax.set_xlabel("Radius [kpc]")
            ax.set_xticks(results['ro_kpc'])
            ax.set_xlim((numpy.min(results['ri_kpc']), numpy.max(results['ro_kpc'])))
        else:
            bin_center = (results['ri_arcmin'] + results['ro_arcmin']) / 2.
            bin_width = results['ro_arcmin'] - results['ri_arcmin']
            ax.set_xlabel("Radius [arcmin]")
            ax.set_xticks(results['ro_arcmin'])
            ax.set_xlim((numpy.min(results['ri_arcmin']), numpy.max(results['ro_arcmin'])))

        #numpy.arange(bins_pixels[0], bins_pixels[-1] + 2, 2))
        ax.scatter(bin_center, results['avg'], c='red')
        ax.errorbar(x=bin_center, y=results['avg'],
                    xerr=bin_width / 2, yerr=background_noise,
                    linestyle=':', c='red',
                    label=r"H$\alpha$")
        # ax.scatter(bin_center, nuv_sfr_profile / bin_area, c='blue')
        # ax.errorbar(x=bin_center, y=nuv_sfr_profile / bin_area,
        #             xerr=bin_width / 2, yerr=nuv_sfr_profile_errors / bin_area,
        #             linestyle='--', c='blue',
        #             label='near-UV')
        ax.set_yscale('log')
        print("RESULTS:\n",numpy.log10(results['avg']))
        min_y = numpy.min(numpy.floor(numpy.log10(results['avg'])))
        max_y = numpy.max(numpy.ceil(numpy.log10(results['avg'])*5))/5.
        print(min_y, max_y, numpy.power(10.,min_y), numpy.power(10., max_y))
        ax.set_ylim((numpy.power(10.,min_y), numpy.power(10., max_y)))

        # ax.set_xlabel("distance [kpc]")
        ax.set_ylabel(r"surface brightness (counts/pixel)")
        fig.show()
        fig.savefig("img2profile.png")


        # masked_img_fn = sys.argv[1]
        # img_hdu = pyfits.open(masked_img_fn)
        # img = img_hdu[0].data
        #
        #

        output_fn = args.output_fn
        results.to_csv(output_fn, index=False)
        logger.info("Writing results to %s" % (output_fn))

    sys.exit(0)

    # distance = float(sys.argv[4])
    #
    # background = float(sys.argv[5])
    #
    # img -= background

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


#!/usr/bin/env python3

import os
import sys
import numpy
import matplotlib
import matplotlib.path as mpltPath
import scipy.ndimage

import astropy.io.fits as pyfits
import astropy.table

import astropy.wcs
import pandas
import argparse
import logging
import ephem
import astropy.coordinates
import astropy.units as u

# bufferzone = 3

def make_round_kernel(size):

    k = numpy.zeros((2*size+1, 2*size+1), dtype=float)

    _iy,_ix = numpy.indices(k.shape, dtype=float)
    _ix -= size
    _iy -= size
    _radius = numpy.hypot(_ix, _iy)
    k[_radius <= size] = 1.

    return k


def measure_polygons(polygon_list, image, wcs, edgewidth=1, deadspace=0, skysize=1.,
                     generate_check_images=False):

    logger = logging.getLogger("MeasurePolygon")
    bufferzone = edgewidth + 2

    pixelscale = astropy.wcs.utils.proj_plane_pixel_scales(wcs)
    print(pixelscale)

    dead_pixels = int(numpy.ceil(deadspace / (pixelscale[0] * 3600)))
    sky_pixels = int(numpy.ceil(skysize / (pixelscale[0] * 3600)))

    bufferzone = dead_pixels + sky_pixels + 2

    iy,ix = numpy.indices(image.shape)
    # print(iy)
    # print(ix)
    # print(ix.ravel())
    index_xy = numpy.hstack((ix.reshape((-1,1)), iy.reshape((-1,1))))
    # print(index_xy)
    # print(index_xy.shape)

    #edge_kernel = numpy.ones((2*edgewidth+1, 2*edgewidth+1))
    # dead_kernel = numpy.ones((2*dead_pixels+1, 2*dead_pixels+1))
    # sky_kernel = numpy.ones((2*sky_pixels+1, 2*sky_pixels+1))

    dead_kernel = make_round_kernel(dead_pixels)
    sky_kernel = make_round_kernel(sky_pixels)

    pyfits.PrimaryHDU(data=dead_kernel).writeto("poly2flux_kernel_dead.fits", overwrite=True)
    pyfits.PrimaryHDU(data=sky_kernel).writeto("poly2flux_kernel_sky.fits", overwrite=True)

    polygon_data = []

    check_sources = [pyfits.PrimaryHDU()]
    check_dead = [pyfits.PrimaryHDU()]
    check_sky = [pyfits.PrimaryHDU()]
    check_source_sky = [pyfits.PrimaryHDU()]

    polygon_catalog = pandas.DataFrame(
        columns=['center_x', 'center_y',
                 'src_flux', 'src_area',
                 'sky_median', 'sky_mean', 'sky_std', 'sky_area', 'sky_var',
                 ],
    )
    for ipoly, polygon in enumerate(polygon_list):

        # sys.stdout.write(".")
        # sys.stdout.flush()
        logger.debug("working on polygon %d of %d" % (ipoly+1, len(polygon_list)))

        # first, convert ra/dec to x/y
        xy = wcs.all_world2pix(polygon, 0)
        # print(xy)

        #
        # to speed things up, don't work on the whole image, but
        # rather only on the little area around and including the polygon
        #
        min_xy = numpy.floor(numpy.min(xy, axis=0)).astype(int) - [bufferzone,bufferzone]
        min_xy[min_xy < 0] = 0
        max_xy = numpy.ceil(numpy.max(xy, axis=0)).astype(int) + [bufferzone,bufferzone]
        # print(min_xy, max_xy)

        max_x, max_y = max_xy[0], max_xy[1]
        min_x, min_y = min_xy[0], min_xy[1]

        # cutout the area with points in the region
        poly_ix = ix[ min_y:max_y+1, min_x:max_x+1 ]
        poly_iy = iy[ min_y:max_y+1, min_x:max_x+1 ]
        poly_xy = numpy.hstack((poly_ix.reshape((-1,1)), poly_iy.reshape((-1,1))))
        # print(poly_xy.shape)
        # print(poly_xy)

        # use some matplotlib magic to figure out which points are inside the polygon
        path = mpltPath.Path(xy)
        inside2 = path.contains_points(poly_xy)
        inside2d = inside2.reshape(poly_ix.shape)
        # print(inside2d.shape)

        # to get at the border of the polygon, convolve the mask with a small filter
        dead_widened = scipy.ndimage.convolve(inside2d.astype(int), dead_kernel,
                               mode='constant', cval=0)
        sky_widened = scipy.ndimage.convolve(dead_widened.astype(int), sky_kernel,
                               mode='constant', cval=0)

        edge_only_pixels = (dead_widened > 0) & (~inside2d)
        sky_only_pixels = (sky_widened > 0) & ~(dead_widened > 0)
        dead_only_pixels = (dead_widened > 0) & (~inside2d)
        image_region = image[ min_y:max_y+1, min_x:max_x+1 ]

        # generate the check images
        # mask_image_region = mask_image[ min_y:max_y+1, min_x:max_x+1 ]
        # mask_image_region[inside2d] = image_region[inside2d]

        # edge_image_region = edge_image[ min_y:max_y+1, min_x:max_x+1 ]
        # edge_image_region[edge_only_pixels] += 1

        n_pixels = numpy.sum(inside2)

        # set some default values in case things go wrong down the line
        total_flux = -1
        center_x = -1
        center_y = -1
        edge_mean = edge_median = edge_area = -1
        sky_mean = sky_median = sky_area = sky_std = sky_var = -1


        if (n_pixels >= 1):
            total_flux = numpy.sum(image_region[inside2d])

            # calculate mean position of points inside polygon
            center_x = numpy.mean( poly_ix[inside2d] )
            center_y = numpy.mean( poly_iy[inside2d] )

            edge_mean = numpy.nanmean( image_region[edge_only_pixels] )
            edge_median = numpy.nanmedian( image_region[edge_only_pixels] )
            edge_area = numpy.sum( edge_only_pixels )

            edge_mean = numpy.nanmean( image_region[sky_only_pixels] )
            edge_median = numpy.nanmedian( image_region[sky_only_pixels] )
            edge_area = numpy.sum( sky_only_pixels )

            sky_pixels = image_region[sky_only_pixels]
            good = numpy.isfinite(sky_pixels)
            for iteration in range(3):
                _stats = numpy.nanpercentile(sky_pixels[good], [16,50,84])
                _median = _stats[1]
                _sigma = 0.5*(_stats[2]-_stats[0])
                outlier = (sky_pixels > (_median + 3*_sigma)) | (sky_pixels < (_median - 3*_sigma))
                good[outlier] = False

            sky_mean = numpy.nanmean(sky_pixels[good])
            sky_median = numpy.nanmedian(sky_pixels[good])
            sky_std = numpy.nanstd(sky_pixels[good])
            sky_var = numpy.nanvar(sky_pixels[good])
            sky_area = numpy.sum(sky_only_pixels)

            polygon_data.append([n_pixels, total_flux, center_x, center_y, edge_mean, edge_median, edge_area])

        polygon_catalog.loc[ipoly, 'center_x'] = center_x
        polygon_catalog.loc[ipoly, 'center_y'] = center_y
        polygon_catalog.loc[ipoly, 'src_flux'] = total_flux
        polygon_catalog.loc[ipoly, 'src_area'] = numpy.sum(inside2d)
        polygon_catalog.loc[ipoly, 'sky_median'] = sky_median
        polygon_catalog.loc[ipoly, 'sky_mean'] = sky_mean
        polygon_catalog.loc[ipoly, 'sky_std'] = sky_std
        polygon_catalog.loc[ipoly, 'sky_area'] = sky_area
        polygon_catalog.loc[ipoly, 'sky_var'] = sky_var

        # continue

        # do not use this doe, it's slow as hell
        # path = mpltPath.Path(xy)
        # inside2 = path.contains_points(index_xy)
        # inside2d = inside2.reshape(image.shape)

        # mask_image[inside2d] = 1

        if (generate_check_images):
            img = image_region.copy()
            img[~inside2d] = numpy.nan

            dead = image_region.copy()
            dead[~dead_only_pixels] = numpy.nan

            sky = image_region.copy()
            sky[~sky_only_pixels] = numpy.nan

            source_sky = image_region.copy()
            source_sky[~sky_only_pixels & ~inside2d] = numpy.nan

            check_sources.append(pyfits.ImageHDU(img))
            check_dead.append(pyfits.ImageHDU(dead))
            check_sky.append(pyfits.ImageHDU(sky))
            check_source_sky.append(pyfits.ImageHDU(source_sky))

    polygon_data = numpy.array(polygon_data)

    polygon_catalog['flux_bgsub'] = polygon_catalog['src_flux'] - polygon_catalog['src_area']*polygon_catalog['sky_median']

    if (generate_check_images):
        return polygon_catalog, (check_sources, check_dead, check_sky, check_source_sky)
    return polygon_catalog



def read_polygons_from_ds9_region_file(fn):

    logger = logging.getLogger("ReadPolyFromDS9RegionFile")

    #
    # Now read the region file
    #
    src_polygons = []
    lines = []
    with open(fn, "r") as regfile:
        lines = regfile.readlines()
        logger.info("Read %d lines" % (len(lines)))

    for line in lines:
        if (not line.startswith("polygon(")):
            # don't do anything
            continue

        coordinates_text = line.split("polygon(")[1].split(")")[0]
        coordinates = coordinates_text.split(",")
        # print(coordinates)

        try:
            coordinates2 = [float(c) for c in coordinates]
            coordinates_radec = numpy.array(coordinates2).reshape((-1, 2))
        except ValueError:
            # this most likely means that coordinates are in H:M:S format
            # easiest to use ephem to convert them to degrees
            logger.debug("Found coordinates in H:M:S system")
            coordinates_radec = []
            for c in range(0, len(coordinates), 2):
                # print(coordinates[c], coordinates[c+1])
                radec = ephem.Equatorial(coordinates[c], coordinates[c+1])
                coordinates_radec.append([radec.ra, radec.dec])
            coordinates_radec = numpy.rad2deg(numpy.array(coordinates_radec))

        # print(coordinates2)

        # print(coordinates_radec)

        src_polygons.append(coordinates_radec)

    return src_polygons



def main():

    cmdline = argparse.ArgumentParser()
    cmdline.add_argument("--dryrun", dest="dryrun", default=False, action='store_true',
                         help="dry-run only, no database ingestion")
    cmdline.add_argument("--debug", dest="debug", default=False, action='store_true',
                         help="output debug output")
    cmdline.add_argument("--checkimages", dest="checkimages", default=False, action='store_true',
                         help="generate check-images")

    cmdline.add_argument("--deadspace", dest="deadspace", type=float, default=0.0,
                         help='spacing between source aperture and sky perimeter [arcsec]')
    cmdline.add_argument("--skywidth", dest="skywidth", type=float, default=1.0,
                         help='size for sky perimeter [arcsec]')

    cmdline.add_argument("--merge", dest="merge", default=None, type=str,
                         help='filename for merged catalogs at the end')
    cmdline.add_argument("--nthreads", dest="n_threads", default=1, type=int,
                         help='number of parallel worker threads')

    cmdline.add_argument("--distance", dest="distance", default=0, type=float,
                         help='distance to source in Mpc')
    cmdline.add_argument("--calibrate", dest="calibrate", default=None, type=str, nargs="*",
                         help='calibration factor (format: filter:factor; e.g.: ha:1.e5e-9)')
    cmdline.add_argument("--gain", dest="gain", default=None, type=str, nargs="*",
                         help='gain (format: filter:gain; e.g.: ha:1.e5e-9; alternative: filter:!header_key)')

    cmdline.add_argument("--distance_to_center", dest="center_coord", type=str, default=None,
                         help="if provided, calculate distance between source and center (format: HMS+dms, eg 14:23:45+23:45:56)")

    cmdline.add_argument("--region", dest="region_fn", default=None, type=str,
                         help='region filename for source definition')
    cmdline.add_argument("--output", dest="output", default="polyflux.csv", type=str,
                         help='filename for output catalog')
    cmdline.add_argument("files", nargs="+",
                         help="list of input filenames")
    args = cmdline.parse_args()

    logging.basicConfig(format='%(name)s -- %(levelname)s: %(message)s',
                        level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger("PolyFlux")
    logger.info("Sky parameters: %f // %f" % (args.deadspace, args.skywidth))

    # find all filter names
    filter_names = []
    for image_fn in args.files:
        name, _ = os.path.splitext(image_fn)
        if (image_fn.find(":") > 0):
            items = image_fn.split(":")
            if (len(items) == 2):
                image_fn = items[0]
                name = items[1]
                filter_names.append(name)
    logger.info("Using these filters: %s" % ", ".join(filter_names))

    # parse the calibration constants
    calibration_factors = {name:1. for name in filter_names}
    print(args.calibrate)
    if (args.calibrate is not None):
        for calib in args.calibrate:  # .split(","):
            items = calib.split(":")
            if (len(items) == 2):
                filtername = items[0]
                factor = float(items[1])
                calibration_factors[filtername] = factor

    # parse the gain values
    gain_values = {name:(1.,None) for name in filter_names}
    print(args.gain)
    if (args.gain is not None):
        for calib in args.gain:  # .split(","):
            items = calib.split(":")
            if (len(items) == 2):
                filtername = items[0]
                value, key = None, None
                try:
                    value = float(items[1])
                except:
                    key = items[1]
                gain_values[filtername] = (value, key)

    distance_cm = 1.0
    if (args.distance > 0):
        distance_cm = args.distance * 3.0857e24  # cm/Mpc
        logger.info("Using distance of %.2f Mpc (%g cm)" % (args.distance, distance_cm))

    src_polygons = read_polygons_from_ds9_region_file(args.region_fn)
    logger.info("Found %d source polygons" % (len(src_polygons)))

    center_pos = None
    if (args.center_coord is not None):
        center_pos = astropy.coordinates.SkyCoord(args.center_coord, unit=(u.hourangle, u.deg))
    #
    # Let's run the integration code on all files, one after another
    #
    master_catalog = None
    for image_fn in args.files:

        name, _ = os.path.splitext(image_fn)
        if (image_fn.find(":") > 0):
            items = image_fn.split(":")
            if (len(items) == 2):
                image_fn = items[0]
                name = items[1]

        logger.info("Working on image file %s (regions: %s // name: %s)" % (image_fn, args.region_fn, name))
        named_logger = logging.getLogger(name)

        #
        # Now lets read the image
        #
        named_logger.debug("Reading %s" % (image_fn))
        image_hdu = pyfits.open(image_fn)
        # image_hdu.info()

        image_data = image_hdu[0].data
        wcs = astropy.wcs.WCS(image_hdu[0].header)

        if (name not in gain_values):
            gain = 1.
        else:
            (gain_value, gain_key) = gain_values[name]
            if (gain_key is not None):
                try:
                    gain = image_hdu[0].header['GAIN']
                    logger.info("Using GAIN = %.3f from header" % (gain))
                except:
                    gain = 1000.
                    logger.info("Using fall-back GAIN = %.3f" % (gain))
            else:
                gain = gain_value

        # print(wcs)

        # photflam = image_hdu['SCI'].header['PHOTFLAM']
        # photplam = image_hdu['SCI'].header['PHOTPLAM']
        # zp_ab = -2.5*numpy.log10(photflam) - 5*numpy.log10(photplam) - 2.408
        # print("ZP_AB = %f" % (zp_ab))
        # # see https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints

        # print("integrating sky polygons")
        # _, sky_data = measure_polygons(sky_polygons, image_data, wcs)
        named_logger.info("integrating source polygons")
        src_data, check_hdulists = measure_polygons(src_polygons, image_data, wcs,
                                                    deadspace=args.deadspace,
                                                    skysize=args.skywidth,
                                                    generate_check_images=True)

        (check_sources, check_dead, check_sky, check_source_sky) = check_hdulists

        if (args.checkimages):
            pyfits.HDUList(check_sources).writeto("check_sources.fits", overwrite=True)
            pyfits.HDUList(check_dead).writeto("check_dead.fits", overwrite=True)
            pyfits.HDUList(check_sky).writeto("check_sky.fits", overwrite=True)
            pyfits.HDUList(check_source_sky).writeto("check_source_sky.fits", overwrite=True)

        # src_data.info()
        # convert polygon center coordinates from native pixels to Ra/Dec
        # src_data.info()
        # print(src_data['center_x'].astype(float).to_numpy())
        _ra, _dec = wcs.all_pix2world(src_data['center_x'].astype(float).to_numpy(),
                                      src_data['center_y'].astype(float).to_numpy(), 1)
        src_data['center_ra'] = _ra
        src_data['center_dec'] = _dec
        # print(radec)

        # apply flux calibrations
        calib_factor = 1.0
        if (name in calibration_factors):
            calib_factor = calibration_factors[name]
            named_logger.info("Apply calibration factor: %g" % (calib_factor))

        sky_error = (src_data['src_area'] * src_data['sky_var']).astype(float).to_numpy()
        src_error = gain * src_data['src_flux'].astype(float).to_numpy()
        flx_error = numpy.fabs(src_error) + sky_error * gain ** 2
        flx_error[flx_error < 0] = 1e30
        # print(type(flx_error.to_numpy()))
        src_data['src_flux_error'] = numpy.power(flx_error, 0.5) / gain

        src_data['calib_flux'] = src_data['flux_bgsub'] * calib_factor
        src_data['calib_flux_error'] = src_data['src_flux_error'] * calib_factor

        if (args.debug):
            print(src_data[['flux_bgsub', 'src_flux_error', 'sky_std', 'sky_var']].to_markdown())

        # convert flux to luminosity (multiply with 4*pi*d^2)
        named_logger.info("calculating luminosity from flux and distance")
        src_data['calib_luminosity'] = src_data['calib_flux'] * 4 * numpy.pi * distance_cm ** 2
        src_data['calib_luminosity_error'] = src_data['calib_flux_error'] * 4 * numpy.pi * distance_cm ** 2

        if (center_pos is not None):
            src_skycoords = astropy.coordinates.SkyCoord(src_data['center_ra'], src_data['center_dec'], unit=u.deg)
            distances = src_skycoords.separation(center_pos)
            src_data['center_distance_deg'] = distances.deg
            # print(distances)

            if (args.distance is not None):
                src_data['center_distance_kpc'] = numpy.arctan(distances.rad) * args.distance * 1000.

        new_column_names = ["%s_%s" % (name, col) for col in src_data.columns]
        column_translate = dict(zip(src_data.columns, new_column_names))
        src_data.rename(columns=column_translate, inplace=True)
        # src_data.info()

        if (master_catalog is None):
            master_catalog = src_data
        else:
            master_catalog = master_catalog.merge(src_data, how='outer', left_index=True, right_index=True)

        named_logger.info("done with image %s" % (image_fn))

    master_catalog.info()
    logger.info("writing final catalog to %s" % (args.output))
    master_catalog.to_csv(args.output, index=False)
    logger.info("all done!")


if __name__ == "__main__":
    main()

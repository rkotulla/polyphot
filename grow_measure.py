#!/usr/bin/env python3

import os
import sys
import numpy
import argparse
import matplotlib
import matplotlib.path as mpltPath
import scipy.ndimage
import logging

import astropy.io.fits as pyfits
import astropy.table

import astropy.wcs
import pandas

# bufferzone = 3

def measure_polygons(polygon_list, image, wcs, edgewidth=1):

    bufferzone = edgewidth + 2

    iy,ix = numpy.indices(image.shape)
    # print(iy)
    # print(ix)
    # print(ix.ravel())
    index_xy = numpy.hstack((ix.reshape((-1,1)), iy.reshape((-1,1))))
    # print(index_xy)
    # print(index_xy.shape)

    edge_kernel = numpy.ones((2*edgewidth+1, 2*edgewidth+1))

    polygon_data = []
    mask_image = numpy.zeros_like(image)
    edge_image = numpy.zeros_like(image)

    for polygon in polygon_list:

        # sys.stdout.write(".")
        # sys.stdout.flush()

        # first, convert ra/dec to x/y
        xy = wcs.all_world2pix(polygon, 0)
        # print(xy)

        #
        # to speed things up, don't work on the whole image, but
        # rather only on the little area around and including the polygon
        #
        min_xy = numpy.floor(numpy.min(xy, axis=0)).astype(numpy.int) - [bufferzone,bufferzone]
        min_xy[min_xy < 0] = 0
        max_xy = numpy.ceil(numpy.max(xy, axis=0)).astype(numpy.int) + [bufferzone,bufferzone]
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
        widened = scipy.ndimage.convolve(inside2d.astype(numpy.int), edge_kernel,
                               mode='constant', cval=0)

        edge_only_pixels = (widened > 0) & (~inside2d)

        image_region = image[ min_y:max_y+1, min_x:max_x+1 ]

        # generate the check images
        mask_image_region = mask_image[ min_y:max_y+1, min_x:max_x+1 ]
        mask_image_region[inside2d] = image_region[inside2d]

        edge_image_region = edge_image[ min_y:max_y+1, min_x:max_x+1 ]
        edge_image_region[edge_only_pixels] += 1

        n_pixels = numpy.sum(inside2)

        # set some default values in case things go wrong down the line
        total_flux = -1
        center_x = -1
        center_y = -1
        edge_mean = edge_median = edge_area = -1

        if (n_pixels >= 1):
            total_flux = numpy.sum(image_region[inside2d])

            # calculate mean position of points inside polygon
            center_x = numpy.mean( poly_ix[inside2d] )
            center_y = numpy.mean( poly_iy[inside2d] )

            edge_mean = numpy.nanmean( image_region[edge_only_pixels] )
            edge_median = numpy.nanmedian( image_region[edge_only_pixels] )
            edge_area = numpy.sum( edge_only_pixels )

        polygon_data.append([n_pixels, total_flux, center_x, center_y, edge_mean, edge_median, edge_area])

        continue

        # do not use this doe, it's slow as hell
        # path = mpltPath.Path(xy)
        # inside2 = path.contains_points(index_xy)
        # inside2d = inside2.reshape(image.shape)

        # mask_image[inside2d] = 1

    polygon_data = numpy.array(polygon_data)

    return (mask_image, edge_image), polygon_data

if __name__ == "__main__":

    cmdline = argparse.ArgumentParser()
    cmdline.add_argument("--dryrun", dest="dryrun", default=False, action='store_true',
                         help="dry-run only, no database ingestion")
    cmdline.add_argument("--debug", dest="debug", default=False, action='store_true',
                         help="output debug output")
    cmdline.add_argument("--segmentation", dest='seg_fn', type=str,
                         help="segmentation filename defining regions")
    cmdline.add_argument("--grow", dest="grow_list", type=str, default="0",
                         help='list of grow steps')
    cmdline.add_argument("files", nargs="+",
                         help="list of input filenames")
    args = cmdline.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger("GrowMeasure")

    # read the segmentation filename
    logger.info("Reading segmentation from file: %s" % (args.seg_fn))
    segmentation_hdu = pyfits.open(args.seg_fn)
    # segmentation_hdu.info()
    segmentation = segmentation_hdu[0].data

    # find the number of apertures
    n_sources = numpy.max(segmentation)
    logger.info("Found %d sources" % (n_sources))
    #n_sources = 50

    # split and convert the list of radii
    grow_list = numpy.array([float(f) for f in args.grow_list.split(",")])
    grow_list.sort()
    logger.info("Using these growing radii: %s" % (grow_list))

    # open all image files
    img_data = []
    img_files = []
    for img_fn in args.files:
        try:
            img_hdu = pyfits.open(img_fn)
        except FileNotFoundError:
            logger.critical("input file not found: %s" % (img_fn))
            continue

        img = img_hdu[0].data
        # make sure the dimensions match the segmentation fn
        if (img.shape != segmentation.shape):
            logger.critical("Dimensions dont match: %s (%s vs %s)" % (img_fn, str(img.shape), str(segmentation.shape)))
            continue

        # mask out all illegal pixel values
        bad_data = (img < -1e25) | (img > 1e25)
        #img[bad_data] = numpy.NaN

        img_data.append(img) #[img_fn] = img
        img_files.append(img_fn)

    if (len(img_data) <= 0):
        logger.error("No image files to work on")
        sys.exit(0)
    else:
        logger.info("Starting to work on %d input images" % (len(img_data)))

    buffer_size = numpy.ceil(numpy.max(grow_list)).astype(numpy.int) + 1

    # generate growth kernels
    kernel1 = numpy.ones((2*buffer_size+1, 2*buffer_size+1))
    ky,kx = numpy.indices(kernel1.shape, dtype=numpy.float)
    ky -= buffer_size
    kx -= buffer_size
    kr = numpy.hypot(kx,ky)
    pyfits.PrimaryHDU(data=kr).writeto("kernel_radius.fits", overwrite=True)
    kernels = [None] * grow_list.shape[0]
    kernels_hdu = [pyfits.PrimaryHDU()]

    for i, grow_radius in enumerate(grow_list):
        logger.debug("Generating kernel for radius = %.1f" % (grow_radius))
        k = kr.copy()
        k[kr > grow_radius] = 0
        k[kr <= grow_radius] = 1

        gr = int(numpy.ceil(grow_radius))
        xy1 = buffer_size - gr - 1
        xy2 = buffer_size + gr + 1
        k_sized = k[xy1:xy2+1, xy1:xy2+1]

        kernels_hdu.append(pyfits.ImageHDU(data=k, name="RAW_%d" % (grow_radius)))
        kernels_hdu.append(pyfits.ImageHDU(data=k_sized, name="KERNEL_%d" % (grow_radius)))

        kernels[i] = k_sized

    kernels_hdu = pyfits.HDUList(kernels_hdu)
    kernels_hdu.info()
    kernels_hdu.writeto("kernels.fits", overwrite=True)

    # sys.exit(0)

    cols = ['id', 'x1', 'x2', 'y1', 'y2', 'npixraw']
    grow_cols = ['npix%d' % g for g in grow_list]
    flux_cols = ['flux%d' % g for g in grow_list]
    all_cols = cols + grow_cols + flux_cols
    df = pandas.DataFrame(numpy.zeros((n_sources, len(all_cols))), columns=all_cols)
    idx_y, idx_x = numpy.indices(segmentation.shape)
    df.info()

    img_phot = [pandas.DataFrame(numpy.zeros((n_sources, len(all_cols))), columns=all_cols) for i in img_files]

    dummy_before = [pyfits.PrimaryHDU()]
    dummy_after = [pyfits.PrimaryHDU()]

    img_chunks = [[[pyfits.PrimaryHDU()] for g in grow_list] for i in img_files]
    print(img_chunks)
    for source_id in range(1, n_sources): # start counting at 1, 0 is the background
        logger.debug("Working on source %d" % (source_id))

        # find maximum extent of source
        this_source = (segmentation == source_id)
        x1 = numpy.min(idx_x[this_source])
        x2 = numpy.max(idx_x[this_source])
        y1 = numpy.min(idx_y[this_source])
        y2 = numpy.max(idx_y[this_source])

        n_pixels = numpy.sum(this_source)
        df.loc[source_id, ['id', 'x1','x2', 'y1', 'y2', 'npixraw']] = [source_id, x1,x2,y1,y2, n_pixels]
        for i in range(len(img_files)):
            img_phot[i].loc[source_id, ['id', 'x1','x2', 'y1', 'y2', 'npixraw']] = [source_id, x1,x2,y1,y2, n_pixels]

        # now get a cutout from the segmentation mask
        _x1 = numpy.max([0, x1-buffer_size])
        _x2 = numpy.min([segmentation.shape[1], x2+buffer_size]) + 1
        _y1 = numpy.max([0, y1-buffer_size])
        _y2 = numpy.min([segmentation.shape[1], y2+buffer_size]) + 1
        seg_cutout = segmentation[_y1:_y2, _x1:_x2].copy()

        dummy_before.append(pyfits.ImageHDU(seg_cutout.copy()))

        # set all pixels outside the current source to 0
        logger.debug(seg_cutout.shape)
        bad = (seg_cutout != source_id)
        logger.debug(numpy.sum(bad))
        seg_cutout[bad] = 0 #seg_cutout != source_id] = 0
        dummy_after.append(pyfits.ImageHDU(data=seg_cutout.copy(),
                                           name='SEGM_%d' % (source_id)))


        # now grow the mask for each of the growing radii
        for i,grow_radius in enumerate(grow_list):
            grown_mask = scipy.ndimage.convolve(
                input=seg_cutout,
                weights=kernels[i],
                mode='constant', cval=0,
            )
            phot_mask = grown_mask >= 1
            grown_mask[phot_mask] = 1
            dummy_after.append(pyfits.ImageHDU(data=grown_mask.copy(),
                                               name='SEGM_%d++%d' % (source_id, grow_radius)))

            # get some info for the photometry masks
            mask_size = numpy.nansum(phot_mask)

            phot_cols = ['npix%d' % grow_radius, 'flux%d' % grow_radius]
            for f, img in enumerate(img_data):
                img_cutout = img[_y1:_y2, _x1:_x2]
                flux = numpy.sum(img_cutout[phot_mask])
                img_phot[f].loc[source_id, phot_cols] = [mask_size, flux]

                img_chunk = img_cutout.copy()
                #img_chunk[~phot_mask] = numpy.NaN
                img_chunks[f][i].append(pyfits.ImageHDU(data=img_chunk, name="PHOT_%d++%d" % (source_id, grow_radius)))

    pyfits.HDUList(dummy_before).writeto("dummy_before.fits", overwrite=True)
    pyfits.HDUList(dummy_after).writeto("dummy_after.fits", overwrite=True)

    # now save all the generated photometry results
    for i, img_fn in enumerate(img_files):
        dir,fn = os.path.split(img_fn)
        bn,ext = os.path.splitext(fn)
        csv_fn = "%s__grow.csv" % (bn)
        img_phot[i].to_csv(csv_fn, index=False)

        # also save all photometry chunks
        for g, grow_radius in enumerate(grow_list):
            hdulist = pyfits.HDUList(img_chunks[i][g])
            fits_fn = "%s__chunks_grow%d.fits" % (bn, grow_radius)
            hdulist.writeto(fits_fn, overwrite=True)


    df.to_csv("grow_measure.csv", index=False)




    #
    # region_fn = sys.argv[1]
    # # image_fn = sys.argv[2]
    #
    # #
    # # Now read the region file
    # #
    # src_polygons = []
    # sky_polygons = []
    # with open(region_fn, "r") as regfile:
    #     lines = regfile.readlines()
    #     print("Read %d lines" % (len(lines)))
    #
    # for line in lines:
    #     if (not line.startswith("polygon(")):
    #         # don't do anything
    #         continue
    #
    #     coordinates_text = line.split("polygon(")[1].split(")")[0]
    #     coordinates = coordinates_text.split(",")
    #     # print(coordinates)
    #
    #     coordinates2 = [float(c) for c in coordinates]
    #     # print(coordinates2)
    #
    #     coordinates_radec = numpy.array(coordinates2).reshape((-1,2))
    #     # print(coordinates_radec)
    #
    #     if (line.find("background") > 0):
    #         # this is a background lines
    #         # print("BACKGROUND:", line)
    #         sky_polygons.append(coordinates_radec)
    #     else:
    #         # print("not a background")
    #         src_polygons.append(coordinates_radec)
    #
    # print("Found %d source polygons and %d sky polygons" % (
    #     len(src_polygons), len(sky_polygons)
    # ))
    #
    #
    # #
    # # Let's run the integration code on all files, one after another
    # #
    # for image_fn in sys.argv[2:]:
    #
    #     print("Working on image file %s (regions: %s)" % (image_fn, region_fn))
    #
    #     #
    #     # Now lets read the image
    #     #
    #     image_hdu = pyfits.open(image_fn)
    #     # image_hdu.info()
    #
    #     image_data = image_hdu['SCI'].data
    #     wcs = astropy.wcs.WCS(image_hdu['SCI'].header)
    #     print(wcs)
    #
    #     photflam = image_hdu['SCI'].header['PHOTFLAM']
    #     photplam = image_hdu['SCI'].header['PHOTPLAM']
    #     zp_ab = -2.5*numpy.log10(photflam) - 5*numpy.log10(photplam) - 2.408
    #     print("ZP_AB = %f" % (zp_ab))
    #     # see https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
    #
    #
    #     print("integrating sky polygons")
    #     _, sky_data = measure_polygons(sky_polygons, image_data, wcs)
    #     print("integrating source polygons")
    #     src_images, src_data = measure_polygons(src_polygons, image_data, wcs)
    #
    #     src_mask, src_edges = src_images
    #     # pyfits.PrimaryHDU(data=sky_mask).writeto("sky_mask.fits", overwrite=True)
    #     pyfits.PrimaryHDU(data=src_mask).writeto("src_mask.fits", overwrite=True)
    #     pyfits.PrimaryHDU(data=src_edges).writeto("src_edges.fits", overwrite=True)
    #
    #     #
    #     # Figure out the average sky background level
    #     #
    #     if (len(sky_polygons) >= 1):
    #         median_sky_level = numpy.median( sky_data[:,1]/sky_data[:,0] )
    #         print("Median sky = %f" % (median_sky_level))
    #     else:
    #         median_sky_level = 0
    #         print("No Sky background polygon defined, resorting to using sky background = 0.")
    #
    #
    #     # now apply a sky background subtraction for all source polygons
    #     background_subtracted_src_data = src_data.copy()
    #     background_subtracted_src_data[:,1] -= background_subtracted_src_data[:,0] * median_sky_level
    #
    #     background_subtracted_src_data[:,4] -= median_sky_level
    #     background_subtracted_src_data[:,5] -= median_sky_level
    #
    #     df = pandas.DataFrame({
    #         "PolyArea": background_subtracted_src_data[:,0],
    #         "IntegratedFlux": background_subtracted_src_data[:,1],
    #         "Mean_X": background_subtracted_src_data[:,2] + 1, # add 1 since fits starts counting at 1
    #         "Mean_Y": background_subtracted_src_data[:,3] + 1,
    #         "Edge_Mean": background_subtracted_src_data[:,4],
    #         "Edge_Median": background_subtracted_src_data[:,5],
    #         "Edge_Area": background_subtracted_src_data[:,6],
    #         })
    #
    #     bad_photometry = df['IntegratedFlux'] <= 0
    #
    #     df['InstrumentalMagnitude'] = -2.5*numpy.log10(df['IntegratedFlux'])
    #     df['Magnitude_AB'] = df['InstrumentalMagnitude'] + zp_ab
    #
    #     df['InstrumentalMagnitude'][bad_photometry] = 99.999
    #     df['Magnitude_AB'][bad_photometry] = 99.999
    #
    #     # add your conversion here
    #     df['PolyMean'] = df['IntegratedFlux'] / df['PolyArea']
    #
    #     df['area_cm2'] = df['PolyArea'] * 1.6e39
    #     df['transmission'] = df['PolyMean'] / df['Edge_Median']
    #     df['optical_depth'] = -1 * numpy.log(df['transmission'])
    #
    #     transmission_constant_f435w = 1.4e21
    #     df['number_atoms'] = transmission_constant_f435w * df['transmission'] # for the F435W filter
    #     mass_per_atom = 2.2e-24 # that's in grams
    #     df['dustmass_grams'] = df['number_atoms'] * mass_per_atom * df['area_cm2']
    #     df['dustmass_solarmasses'] = df['dustmass_grams'] / 2.e33
    #
    #     print(df['dustmass_solarmasses'])
    #
    #     # convert mean_x/mean_y to ra/dec
    #     mean_ra_dec = wcs.all_pix2world(df['Mean_X'], df['Mean_Y'], 1.)
    #     # print(mean_ra_dec)
    #     df['RA'] = mean_ra_dec[0]
    #     df['DEC'] = mean_ra_dec[1]
    #
    #     df.info()
    #     df.to_csv(image_fn[:-5]+"_polygonflux.csv")
    #
    #     # also save as a votable for ds9
    #     table = astropy.table.Table.from_pandas(df)
    #     table.write(image_fn[:-5]+"_polygonflux.vot", format='votable', overwrite=True)
    #
    #     # print("\n\nSKY:")
    #     # print(sky_data)
    #     # print("\n\nSources:")
    #     # print(src_data)
    #
    #     print("done with image %s" % (image_fn))
    #
    # print("all done!")

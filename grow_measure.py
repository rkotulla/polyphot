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
    cmdline.add_argument("--merge", dest="merge", default=False, action='store_true',
                         help='merge all catalogs at the end')
    cmdline.add_argument("--rename", dest="rename", default=None, type=str,
                         help='rename catalogs instead of using filenames')
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
    n_sources = 50

    # split and convert the list of radii
    grow_list = numpy.array([float(f) for f in args.grow_list.split(",")])
    grow_list.sort()
    logger.info("Using these growing radii: %s" % (grow_list))

    # open all image files
    img_data = []
    img_files = []
    img_basenames = []
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
        img[bad_data] = numpy.NaN

        img_data.append(img) #[img_fn] = img
        img_files.append(img_fn)

        # get the basename (filename without directory and extension) se we can name the output files approprately
        dir,fn = os.path.split(img_fn)
        bn,ext = os.path.splitext(fn)
        img_basenames.append(bn)

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

    cols = ['id', 'x1', 'x2', 'y1', 'y2', 'npixraw', 'maxflux']
    grow_cols = ['npix%d' % g for g in grow_list]
    flux_cols = ['flux%d' % g for g in grow_list]
    overlap_cols = ['overlap%d' % g for g in grow_list]
    all_cols = cols + grow_cols + flux_cols + overlap_cols
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
        other_source = (seg_cutout != source_id) & (seg_cutout > 0)
        logger.debug(numpy.sum(bad))
        seg_cutout[bad] = 0 #seg_cutout != source_id] = 0
        dummy_after.append(pyfits.ImageHDU(data=seg_cutout.copy(),
                                           name='SEGM_%d' % (source_id)))


        # now grow the mask for each of the growing radii
        flux_vs_radius = numpy.zeros((len(img_data), len(grow_list)))
        flux_vs_radius[:,:] = numpy.NaN
        overlap_vs_radius = numpy.zeros((len(img_data), len(grow_list)))

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

            phot_cols = ['npix%d' % grow_radius, 'flux%d' % grow_radius, 'overlap%d' % grow_radius]
            for f, img in enumerate(img_data):
                img_cutout = img[_y1:_y2, _x1:_x2]
                flux = numpy.sum(img_cutout[phot_mask])

                # check if any of the new pixels now overlap other sources
                overlap_pixels = numpy.sum( other_source & phot_mask )

                img_phot[f].loc[source_id, phot_cols] = [mask_size, flux, overlap_pixels]
                img_chunk = img_cutout.copy()
                #img_chunk[~phot_mask] = numpy.NaN
                img_chunks[f][i].append(pyfits.ImageHDU(data=img_chunk, name="PHOT_%d++%d" % (source_id, grow_radius)))

                flux_vs_radius[f,i] = flux
                overlap_vs_radius[f,i] = overlap_pixels

        # now we have all the data we need, let's calculate the maximum flux before running into other sources
        overlapping = (overlap_vs_radius > 0)
        flux_vs_radius[overlapping] = numpy.NaN
        max_flux = numpy.nanmax(flux_vs_radius, axis=1)

        # and add this data to the output data
        for f in range(len(img_data)):
            img_phot[f].loc[source_id, 'maxflux'] = max_flux[f]

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


    if (args.merge):
        # check how to re-name the output columns
        have_good_names = False
        if (args.rename is not None):
            new_names = args.rename.split(",")
            if (len(new_names) == len(img_files)):
                have_good_names = True
        if (not have_good_names):
            new_names = img_basenames

        # now rename all columns in all the photometry catalogs
        for f in range(len(img_files)):
            rename_columns = {}
            for c in img_phot[f].columns:
                rename_columns[c] = '%s_%s' % (new_names[f], c)
            img_phot[f].rename(columns=rename_columns, inplace=True)
            img_phot[f].info()

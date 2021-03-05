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
import astropy.io.votable
import astropy.table

import astropy.wcs
import pandas
import threading
import queue
import multiprocessing

# bufferzone = 3

class thread_grow_and_measure (threading.Thread):

    def __init__(self, segmentation, buffer_size,
                 img_data, grow_list, source_queue, output_df,
                 kernels, img_chunks,
                 worker_id=0):

        threading.Thread.__init__(self)

        self.segmentation = segmentation
        self.buffer_size = buffer_size
        self.img_data = img_data
        self.grow_list = grow_list
        self.source_queue = source_queue
        self.output_df = output_df
        self.worker_id = worker_id
        self.kernels = kernels
        self.img_chunks = img_chunks

        self.logger = logging.getLogger("Worker%02d" % (self.worker_id))

        self.quit = False

    def run(self):

        # prepare what we need for all sources
        idx_y, idx_x = numpy.indices(self.segmentation.shape)
        buffer_size = self.buffer_size
        self.logger.info("Worker starting up")
        while (not self.quit):

            source_id = self.source_queue.get()
            if (source_id is None):
                self.source_queue.task_done()
                self.logger.info("Received termination signal, shutting down")
                break

            self.logger.debug("Working on source %d" % (source_id))

            # find maximum extent of source
            this_source = (self.segmentation == source_id)
            x1 = numpy.min(idx_x[this_source])
            x2 = numpy.max(idx_x[this_source])
            y1 = numpy.min(idx_y[this_source])
            y2 = numpy.max(idx_y[this_source])

            n_pixels = numpy.sum(this_source)
            # df.loc[source_id, ['id', 'x1', 'x2', 'y1', 'y2', 'npixraw']] = [source_id, x1, x2, y1, y2, n_pixels]
            for i in range(len(self.img_data)):
                self.output_df[i].loc[source_id, ['id', 'x1', 'x2', 'y1', 'y2', 'npixraw']] = [source_id, x1, x2, y1, y2,
                                                                                         n_pixels]

            # now get a cutout from the segmentation mask
            _x1 = numpy.max([0, x1 - buffer_size])
            _x2 = numpy.min([segmentation.shape[1], x2 + buffer_size]) + 1
            _y1 = numpy.max([0, y1 - buffer_size])
            _y2 = numpy.min([segmentation.shape[1], y2 + buffer_size]) + 1
            seg_cutout = segmentation[_y1:_y2, _x1:_x2].copy()

            # dummy_before.append(pyfits.ImageHDU(seg_cutout.copy()))

            # set all pixels outside the current source to 0
            self.logger.debug(seg_cutout.shape)
            bad = (seg_cutout != source_id)
            other_source = (seg_cutout != source_id) & (seg_cutout > 0)
            self.logger.debug(numpy.sum(bad))
            seg_cutout[bad] = 0  # seg_cutout != source_id] = 0
            # dummy_after.append(pyfits.ImageHDU(data=seg_cutout.copy(),
            #                                    name='SEGM_%d' % (source_id)))

            # now grow the mask for each of the growing radii
            flux_vs_radius = numpy.zeros((len(img_data), len(grow_list)))
            flux_vs_radius[:, :] = numpy.NaN
            overlap_vs_radius = numpy.zeros((len(img_data), len(grow_list)))

            for i, grow_radius in enumerate(grow_list):
                grown_mask = scipy.ndimage.convolve(
                    input=seg_cutout,
                    weights=self.kernels[i],
                    mode='constant', cval=0,
                )
                phot_mask = grown_mask >= 1
                grown_mask[phot_mask] = 1
                # dummy_after.append(pyfits.ImageHDU(data=grown_mask.copy(),
                #                                    name='SEGM_%d++%d' % (source_id, grow_radius)))

                # get some info for the photometry masks
                mask_size = numpy.nansum(phot_mask)

                phot_cols = ['npix%d' % grow_radius, 'flux%d' % grow_radius, 'overlap%d' % grow_radius]
                for f, img in enumerate(img_data):
                    img_cutout = img[_y1:_y2, _x1:_x2]
                    flux = numpy.sum(img_cutout[phot_mask])

                    # check if any of the new pixels now overlap other sources
                    overlap_pixels = numpy.sum(other_source & phot_mask)

                    img_phot[f].loc[source_id, phot_cols] = [mask_size, flux, overlap_pixels]
                    img_chunk = img_cutout.copy()
                    img_chunk[~phot_mask] = numpy.NaN
                    self.img_chunks[f][i].append(pyfits.ImageHDU(data=img_chunk, name="PHOT_%d++%d" % (source_id, grow_radius)))

                    flux_vs_radius[f, i] = flux
                    overlap_vs_radius[f, i] = overlap_pixels

            # now we have all the data we need, let's calculate the maximum flux before running into other sources
            overlapping = (overlap_vs_radius > 0)
            flux_vs_radius[overlapping] = numpy.NaN
            max_flux = numpy.nanmax(flux_vs_radius, axis=1)

            # and add this data to the output data
            for f in range(len(img_data)):
                self.output_df[f].loc[source_id, 'maxflux'] = max_flux[f]

            self.source_queue.task_done()

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
    cmdline.add_argument("--merge", dest="merge", default=None, type=str,
                         help='filename for merged catalogs at the end')
    cmdline.add_argument("--rename", dest="rename", default=None, type=str,
                         help='rename catalogs instead of using filenames')
    cmdline.add_argument("--refcat", dest="refcat", default=None, type=str,
                     help='source extractor reference catalog')
    cmdline.add_argument("--refname", dest="refname", default='ref', type=str,
                     help='how to rename the reference catalog in the final merged datafile')
    cmdline.add_argument("--refcol", dest="refcol", default='NUMBER', type=str,
                     help='column name in the reference catalog for source matching (default: NUMBER)')
    cmdline.add_argument("--nthreads", dest="n_threads", default=1, type=int,
                     help='number of parallel worker threads')

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
    # n_sources = 10

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
    df.info()

    img_phot = [pandas.DataFrame(numpy.zeros((n_sources, len(all_cols))), columns=all_cols) for i in img_files]
    img_chunks = [[[pyfits.PrimaryHDU()] for g in grow_list] for i in img_files]

    # start parallel workers
    workers = []
    source_queue = multiprocessing.JoinableQueue()
    for i in range(args.n_threads):
        t = thread_grow_and_measure(
            segmentation=segmentation,
            buffer_size=buffer_size,
            img_data=img_data,
            grow_list=grow_list,
            source_queue=source_queue,
            output_df=img_phot,
            kernels=kernels,
            img_chunks=img_chunks,
            worker_id=i+1,
        )
        t.daemon = True
        t.start()
        workers.append(t)

    # handout all jobs
    for source_id in range(1, n_sources):
        source_queue.put(source_id)

    # also queue up termination signals
    for w in workers:
        source_queue.put(None)

    # Now wait for all work to be done
    source_queue.join()


    dummy_before = [pyfits.PrimaryHDU()]
    dummy_after = [pyfits.PrimaryHDU()]

    # print(img_chunks)
    for source_id in range(1, n_sources): # start counting at 1, 0 is the background
        # now done in parallel
        pass

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
            logger.info("Writing grown chunks to %s" % (fits_fn))
            hdulist.writeto(fits_fn, overwrite=True)

    df.to_csv("grow_measure.csv", index=False)


    # open the reference catalog if requested
    ref_table = None
    if (args.refcat is not None):
        refcat = None
        logger.info("Importing additional info from reference catalog (%s)" % (args.refcat))
        try:
            tab = astropy.io.votable.parse_single_table(args.refcat).to_table()
            print(tab)
        except Exception as e:
            logger.error(e)
            tab = None
        if (tab is not None):
            ref_table = tab.to_pandas()
            # ref_table.info()

    if (args.merge is not None):
        logger.info("Generating merged catalog")
        # check how to re-name the output columns
        have_good_names = False
        if (args.rename is not None):
            new_names = args.rename.split(",")
            if (len(new_names) == len(img_files)):
                have_good_names = True
        if (not have_good_names):
            new_names = img_basenames

        # now rename all columns in all the photometry catalogs
        merged_df = None
        for f in range(len(img_files)):
            rename_columns = {}
            for c in img_phot[f].columns:
                rename_columns[c] = '%s_%s' % (new_names[f], c)
            img_phot[f].rename(columns=rename_columns, inplace=True)
            # img_phot[f].info()

            if (merged_df is None):
                merged_df = img_phot[f]
            else:
                merged_df = merged_df.merge(
                    right=img_phot[f],
                    how='outer',
                    left_index=True, right_index=True,
                    sort=False,
                )

        # also merge in the ref-table, if available
        if (ref_table is not None):
            for c in ref_table.columns:
                rename_columns[c] = '%s_%s' % (args.refname, c)
            merge_column = "%s_%s" % (args.refname, args.refcol)
            ref_table.rename(columns=rename_columns, inplace=True)
            merged_df = merged_df.merge(
                right=ref_table,
                how='outer',
                left_index=True, right_on=merge_column,
            )

        merged_df.info()
        merged_df.to_csv(args.merge, index=False)
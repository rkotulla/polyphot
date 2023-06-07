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



def mask_circles(img, circle_list):

    # intialize important variables
    iy,ix = numpy.indices(img.shape)
    mask = numpy.zeros_like(img, dtype=numpy.bool) #~numpy.isfinite(img)

    for circle in circle_list:
        # print(circle)
        cx,cy,r = circle

        _x1 = int(numpy.floor(cx - r))
        _x2 = int(numpy.ceil(cx + r))
        _y1 = int(numpy.floor(cy - r))
        _y2 = int(numpy.ceil(cy + r))

        cutout_x = ix[_y1:_y2 + 1, _x1:_x2 + 1] - cx
        cutout_y = iy[_y1:_y2 + 1, _x1:_x2 + 1] - cy
        cutout_radius = numpy.hypot(cutout_x, cutout_y)
        cutout_mask = (cutout_radius <= r)

        mask[_y1:_y2 + 1, _x1:_x2 + 1][cutout_mask] = True

    return mask

def mask_polygons(img, polygon_list):

    mask = ~numpy.isfinite(img)

    return mask



def generate_mask(img_fn=None, img=None, hdr=None, reg_fn_list=None, total_mask=None):

    logger = logging.getLogger("MaskGenerator")

    if (img_fn is not None):
        img_hdu = pyfits.open(img_fn)
        img = img_hdu[0].data
        hdr = img_hdu[0].header
    elif (img is not None and hdr is not None):
        # already got all info, nothing extra to do here
        pass
    else:
        print("No info provided, aborting")
        return None

    img_wcs = astwcs.WCS(hdr)
    _pixelscale = astwcs.utils.proj_plane_pixel_scales(img_wcs)
    # logger.debug(_pixelscale, _pixelscale*3600)
    pixelscale = _pixelscale[0] * 3600.

    if (total_mask is None):
        total_mask = numpy.zeros_like(img, dtype=numpy.bool)
        # print(total_mask[:10,:10])

    # region file

    for reg_fn in reg_fn_list:
        logger.info("Reading masks from %s" % (reg_fn))

        circle_list = []
        polygon_list = []
        with open(reg_fn, "r") as f:
            lines = f.readlines()
            for line in lines:
                if (line.startswith("circle(")):
                    # add to circle list
                    items = line.split("circle(")[1].split(")")[0].split(",")
                    eq = ephem.Equatorial(items[0], items[1])
                    radius_unit = items[2][-1]
                    if (radius_unit == '"'):
                        radius = float(items[2][:-1])
                    elif (radius_unit == "'"):
                        radius = float(items[2][:-1]) * 60.
                    # print(items, eq.ra, eq.dec, radius)
                    # circle_list.append((numpy.rad2deg(eq.ra), numpy.rad2deg(eq.dec), radius))
                    x,y = img_wcs.all_world2pix(numpy.rad2deg(eq.ra), numpy.rad2deg(eq.dec), 0)
                    circle_list.append((x, y, radius/pixelscale))
                    # pass
                elif (line.startswith("polygon(")):
                    # add to polygon list
                    pass
                else:
                    continue
        # print(circle_list)

        # now mask out all circles
        circle_mask = mask_circles(img, circle_list)
        polygon_mask = mask_polygons(img, polygon_list)

        # save masked image
        total_mask = total_mask | circle_mask | polygon_mask

    return total_mask





if __name__ == "__main__":

#    logging.basicConfig(filename='mask_image_regions.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)
    logging.info('Started')

    # read all image data
    img_fn = sys.argv[1]
    logging.info("Reading data from %s" % (img_fn))

    img_hdu = pyfits.open(img_fn)
    img = img_hdu[0].data
    hdr = img_hdu[0].header

    total_mask = generate_mask(img=img, hdr=hdr, reg_fn_list=sys.argv[2:])

    mask_fn = "mask.fits"
    masked_fn = "image_masked.fits"

    total_mask_img = total_mask.astype(numpy.int)
    mask_hdu = pyfits.PrimaryHDU(data=total_mask_img, header=hdr)
    mask_hdu.writeto(mask_fn, overwrite=True)
    print("Saving total mask to %s" % (mask_fn))

    img[total_mask] = numpy.NaN
    masked_img_hdu = pyfits.PrimaryHDU(data=img, header=hdr)
    masked_img_hdu.writeto(masked_fn, overwrite=True)
    print("Saving masked image to %s" % (masked_fn))
    # extract radial profiles etc


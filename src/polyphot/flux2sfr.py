#!/usr/bin/env python3

import os
import sys
import numpy
import pandas



if __name__ == "__main__":

    cat_fn = sys.argv[1]
    print("Reading input catalog")
    cat = pandas.read_csv(cat_fn)

    # apply dust corrections
    A_R = 1.15 #1.21
    A_NUV = 3.6 #
    cat['ha_dustcorr_lum'] = cat['ha_calib_luminosity'] * numpy.power(10., 0.4*A_R)
    cat['nuv_dustcorr_lum'] = cat['nuv_calib_luminosity'] * numpy.power(10., 0.4*A_NUV)

    cat['ha_dustcorr_lum_error'] = cat['ha_calib_luminosity_error'] * numpy.power(10., 0.4*A_R)
    cat['nuv_dustcorr_lum_error'] = cat['nuv_calib_luminosity_error'] * numpy.power(10., 0.4*A_NUV)

    # and convert to SFR
    cat['ha_sfr'] = cat['ha_dustcorr_lum'] * 7.9e-42
    cat['ha_sfr_error'] = cat['ha_dustcorr_lum_error'] * 7.9e-42

    cat['nuv_sfr'] = cat['nuv_dustcorr_lum'] * 1.4e-28
    cat['nuv_sfr_error'] = cat['nuv_dustcorr_lum_error'] * 1.4e-28

    print("Writing results")
    cat.to_csv(sys.argv[2], index=False)


    total_ha = numpy.sum(cat['ha_sfr'])
    total_nuv = numpy.sum(cat['nuv_sfr'])
    print("HA", total_ha)
    print("NUV", total_nuv)

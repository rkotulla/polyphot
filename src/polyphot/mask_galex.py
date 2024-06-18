#!/usr/bin/env python3


import os
import sys
import astropy.io.fits as pyfits
import numpy


if __name__ == "__main__":

    int_fn = sys.argv[1]
    rrhr_fn = sys.argv[2]

    int_hdu = pyfits.open(int_fn)
    rrhr_hdu = pyfits.open(rrhr_fn)

    int_hdu[0].data[ rrhr_hdu[0].data <= 0] = numpy.NaN

    int_hdu.writeto(sys.argv[3], overwrite=True)



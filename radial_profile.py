#!/usr/bin/env python3


import os
import sys
import astropy.io.fits as pyfits
import numpy
import pandas
import scipy
import matplotlib.pyplot as plt




if __name__ == "__main__":


    # read all data
    sfr_fn = sys.argv[1]
    data = pandas.read_csv(sfr_fn)
    data.info()

    nuv_profile = pandas.read_csv("nuv_profile.csv")
    ha_profile = pandas.read_csv("ha_profile.csv")
    nuv_profile.info()


    # data already contains all distances, so let's make a simple histogram

    radial_distance = data['ha_center_distance_kpc']
    print("distances [kpc]: %f -- %f" % (numpy.min(radial_distance), numpy.max(radial_distance)))

    bins = [0,1,2,3,4,5,6,7,8,9,10,12,14,17,20]
    hist,bins = numpy.histogram(radial_distance,
                                weights=data['ha_sfr'],
                                bins=bins,
                                #range=[0,30]
                                )

    bin_center = (bins[1:] + bins[:-1]) / 2.
    bin_width = numpy.diff(bins)
    bin_area = 4 * numpy.pi * (bins[1:]**2 - bins[:-1]**2) # that's in kpc^2

    ha_sfr_profile_errors, _ = numpy.histogram(radial_distance,
                                               weights=data['ha_sfr_error'], bins=bins)
    nuv_sfr_profile,_ = numpy.histogram(radial_distance,
                                    weights=data['nuv_sfr'], bins=bins)
    nuv_sfr_profile_errors,_ = numpy.histogram(radial_distance,
                                    weights=data['nuv_sfr_error'], bins=bins)

    # print(hist_errors)
    print(bin_area)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks(numpy.arange(bins[0], bins[-1]+2, 2))
    ax.scatter(bin_center, hist/bin_area, c='red')
    ax.errorbar(x=bin_center, y=hist/bin_area,
                xerr=bin_width/2, yerr=ha_sfr_profile_errors * 10/bin_area,
                linestyle=':', c='red',
                label=r"H$\alpha$")
    ax.scatter(bin_center, nuv_sfr_profile/bin_area, c='blue')
    ax.errorbar(x=bin_center, y=nuv_sfr_profile/bin_area,
                xerr=bin_width/2, yerr=nuv_sfr_profile_errors/bin_area,
                linestyle='--', c='blue',
                label='near-UV')
    ax.set_yscale('log')
    ax.set_ylim((3e-7,1e-2))
    ax.set_xlim((0,20.5))
    ax.set_xlabel("distance [kpc]")
    ax.set_ylabel(r"SFR surface density $\Sigma_{SFR} ~~ [$M$_{\odot}~ $yr$^{-1}~ $kpc$^{-2}]$")

    ax.scatter(0.5*(nuv_profile['ri']+nuv_profile['ro']),
               (((nuv_profile['total_flux'] / nuv_profile['kpc2area']) - 0.1) * 1.7e-3 * 0.2),
               label="NUV direct")

    ax.legend(loc='upper right')

    # ax.scatter(0.5*(ha_profile['ri']+ha_profile['ro']),
    #            ((ha_profile['total_flux'] / ha_profile['kpc2area'])+40000) * 9.9e-10,
    #            label="Ha direct")

    fig.tight_layout()
    fig.show()



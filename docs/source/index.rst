.. nirwals documentation master file, created by
   sphinx-quickstart on Thu Jan 18 16:42:09 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to polyphot's documentation!
===================================


What does it do?
================

in its simplest invocation, it takes user-defined regions in the form of a ds9 region file, and integrates the
flux within each polygon. It supports customizable local background subtraction
in an annulus of specified width, matching the overall shape of the chosen source
polygon, with a customizable buffer zone between source and sky areas. All this can
also be done consistently across images (without the need to match the images in
pixel-space; polygons should be defined by world-coordinates (Ra & Dec) that are
recomputed for each frame. It also supports a mechanism to apply additional
image-specific calibrations (e.g. to convert observed fluxes from counts into physical
units and, with a specified distance, to luminosities). Optional check-images allow to verify
proper functionality, and all results are written to a single, multi-band output file for ease of
follow-up processing.

Example polygon
================

![example polygon](docs/source/_static/demo_ic342.jpg)

In this example, the actual source region is selected by the green,
hashed polygon. The sky annulus, separated by some dead space, is shown in
the semi-transparent red region further out. Both width of the dead space
region and the sky annulus can be freely configured, and sky estimation also
includes some iterative sigma-clipping to avoid contamination due to nearby
source for more robust results.

Usage and options
===================

    polyphot --region=my_regions.reg --output=my_catalog.cat file1.fits:band1 file2.fits:band2

Important command line options
================================

`--region` region filename for source definition

`--output` filename for output catalog

`files` list of input filenames


Additional, optional command line options
==========================================


.. note::

   While much of the functionality already exists and is well tested, the
   project remains under active development.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

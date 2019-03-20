# Introduction
Python tools for structure function computation/analysis on 2d/3d images.

# Example Scripts
To see an example, run the repository as a Python package. Assuming the package is located in /path/to/topdir/image_structure, do the following:

cd /path/to/topdir

python3 -m image_structure.scripts.driver image_structure/scripts/inputs.dat

This example computes the following:

(1) Load the image image_structure/data/signal_gaussian_2d.npy

(2) Compute the 2d Fourier transform of the image

(3) Compute the radial average of the Fourier magnitude

(4) Fit a Gaussian about the peak magnitude

(5) Output the pair of numbers (mean, sigma) corresponding to the mean and standard deviation of the Gaussian fit

# Caveats

(1) This software has not yet been tested on a 3d image (although it should work on one)

(2) The Gaussian fit computed by the Fourier structure function fits a Gaussian centered about the peak Fourier magnitude. This is different than fitting a Gaussian to the entire spectrum, which might be important if there are nontrivial subharmonic present.

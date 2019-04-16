import argparse
import numpy as np
import matplotlib.pyplot as plt
from image_structure.src.ImageStructure import *
from image_structure.src.Fourier import *
from image_structure.src.InputFile import *

def main():
    """
    Example driver script for computing the structure function based on Fourier analysis.

    **Inputs**

    ----------
    args : command line arguments
        Command line arguments used in shell call for this main driver script. args must have a inputfilename member that specifies the desired inputfile name.

    **Outputs**

    -------
    inputs.outdir/results.txt : csv file
        output structure metrics. In the Fourier case, this is (mean, sigma) of the Gaussian fit to the average Fourier magnitude
    """

    # Problem setup: read input file options
    parser  = argparse.ArgumentParser(description='Input filename');
    parser.add_argument('inputfilename',\
                        metavar='inputfilename',type=str,\
                        help='Filename of the input file')
    args          = parser.parse_args()
    inputfilename = args.inputfilename
    inputs        = InputFile(inputfilename);

    # Compute structure function
    outdir             = '/'.join( inputs.output_file.split('/')[0:-1] ) + '/'
    str_figure         = '_'.join( ['m' + str(inputs.m) , 'sig' + str(inputs.sig)] )
    structure_analysis = ImageStructure(inputs)
    structure_metrics  = structure_analysis.compute_structure(plot_metrics=True, outdir=outdir, str_figure=str_figure)

    # Output and write to file
    try:
        results = np.array([inputs.m, inputs.sig, structure_metrics[0], structure_metrics[1]] )
    except:
        results = structure_metrics
    with open(inputs.output_file,'a') as f:
        np.savetxt(f,np.transpose(results))
    
if __name__ == '__main__':
    main()
    

import argparse, os
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from image_structure.src.ImageStructure import *
from image_structure.src.Fourier import *
from image_structure.src.InputFile import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_file' , type=str,   help='data file' , required=True)
    parser.add_argument('-m'         , type=float, help='m'          , required=True)
    parser.add_argument('-sig'       , type=float, help='sig'        , required=True)
    args      = vars(parser.parse_args())
    datafile  = args['input']
    m         = args['m']
    sig       = args['sig']
    try:
        dimensions         = args['dimensions']
        structure_function = args['structure_function']
    except:
        dimensions         = 2
        structure_function = 'fourier'
    return datafile,m,sig,dimensions,structure_function
    
def main( datafile=None , m=None , sig=None , dimensions=2 , structure_function='fourier'):
    """
    Example driver script for computing the structure function based on Fourier analysis.

    **Inputs**

    ----------
    args : string
        Command line arguments used in shell call for this main driver script.

    **Outputs**

    -------
    ./structure_vectors.dat : text file
        output structure metrics. In the Fourier case, this is (mean, sigma) of the Gaussian fit to the average Fourier magnitude
    """

    # HARD-CODE: outdir = current directory, outfilename = 'structure_vectors.dat'
    outdir             = os.getcwd() + '/'
    outfile            = outdir      + 'structure_vectors.dat'
    
    # Read from command-line if arguments are not passed directly to main()
    if ( (datafile is None) & (m is None) & (sig is None) ):
        datafile,m,sig,dimensions,structure_function = parse_args()

    # Named-tuple for handling input options
    Inputs             = namedtuple('Inputs' , 'data_file data_file_type m sig dimensions structure_function output_file')
    datafiletype       = datafile.split('.')[-1]
    inputs             = Inputs(datafile, datafiletype, m, sig , dimensions, structure_function, outfile)
    
    # Compute structure function
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
    

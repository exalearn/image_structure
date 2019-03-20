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
    inputs.printInputs();

    # Compute structure function
    structure_analysis = ImageStructure(inputs)
    structure_metrics  = structure_analysis.compute_structure(plot_metrics=True)

    # Output and write to file
    print("********************* OUTPUTS *********************")
    print('\n'.join("%s " % item for item in structure_metrics))
    print("**************************************************")
    np.save(inputs.output_file + '.npy',structure_metrics)
    
if __name__ == '__main__':
    main()
    

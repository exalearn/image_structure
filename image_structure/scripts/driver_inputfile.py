import argparse
import numpy as np
import matplotlib.pyplot as plt
from image_structure.src.ImageStructure import *
from image_structure.src.Fourier import *
from image_structure.src.InputFile import *

def main():

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
    str_figure         = 'script_output'
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
    

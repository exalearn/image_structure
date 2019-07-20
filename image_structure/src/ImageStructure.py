import numpy as np
from image_structure.src.Fourier import *
from image_structure.src.YagerFourier import *

class ImageStructure:

    def __init__(self,inputs):
        """
        Light wrapper class for image structure functions
        """
        self.dimensions = int(inputs.dimensions)
        assert ((self.dimensions == 2) | (self.dimensions == 3)) 
        self.set_input_data( self.read_input_data(inputs.data_file,inputs.data_file_type) )
        self.set_structure_function(inputs.structure_function)
        self.outdir    = '/'.join( inputs.output_file.split('/')[0:-1] ) + '/'

    def set_input_data(self,data):
        self.input_data = data
        
    def read_input_data(self,data_file,data_file_type):
        """
        Method to read input file data
        """
        if (data_file_type == 'npy'):
            input_data = np.load(data_file)
        elif (data_file_type == 'csv'):
            input_data = np.loadtxt(data_file,delimiter=',')
        else:
            sys.exit('Unsupported input data type. Supported types are: npy and csv.')
        return input_data

    def set_structure_function(self,structure_function_type):
        """
        Method to set the relevant structure function
        """
        if (structure_function_type == 'fourier'):
            self.structure_function = fit_gaussian_to_average_fourier_spectrum
        elif (structure_function_type == 'fourier_yager'):
            self.structure_function = structure_vector_yager_2d
        else:
            sys.exit('Unsupported structure function type. Supported types are: fourier , fourier_yager.')

    def compute_structure(self,plot_metrics=False,outdir=None,str_figure=None):
        """
        Main method to compute data structure
        """
        structure = self.structure_function(input_data   = self.input_data , \
                                            plot_metrics = plot_metrics , \
                                            output_dir   = outdir , \
                                            output_name  = str_figure )
        return structure
    

import numpy as np
import pickle

class ImageStructure:

    def __init__(self,inputs):
        """
        Base class for image structure
        """
        self.dimensions = inputs.dimensions
        self.input_data = self.read_input_data(inputs.data_file,inputs.data_file_type)
        
    def read_input_data(self,data_file,data_file_type):
        """
        Method to read input file data
        """
        if (data_file_type == 'npy'):
            input_data = np.load(data_file)
        elif (data_file_type == 'csv'):
            input_data = np.loadtxt(data_file,delimiter=',')
        else:
            sys.exit('Unsupported input data type: accepted types are npy and csv.')
        return input_data

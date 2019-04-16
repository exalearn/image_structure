import image_structure.scripts.driver
import os

def read_params_from_datafile(d):
    # This method assumes d = 'filename_something_m[mval]_sig[sigval]_something.npy'
    # where [mval] and [sigval] are the values of m,sig
    m   = float(d.split('_')[4][1:])
    sig = float(d.split('_')[3][3:])
    return m,sig

# Get list of all datafiles
datadir   = 'image_structure/data/'
datafiles = []
filenames = []
for file in os.listdir( datadir ):
    if file.endswith( '.npy' ):
        datafiles.append( os.path.join(datadir , file) )
        filenames.append( file )
        
for (datafile,filename) in zip(datafiles,filenames):
    try:
        m,sig = read_params_from_datafile(filename)
        print( datafile , m , sig )
        image_structure.scripts.driver.main( datafile , m , sig )
    except:
        print('Ignoring ' + datafile)
        pass

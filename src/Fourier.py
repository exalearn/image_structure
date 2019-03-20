import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def fit_gaussian_to_average_fourier_spectrum(data,plot_metrics=False):
    """
    Function to: 
    (1) compute the n-dimensional fourier transform of data
    (2) compute the 1-d average of this
    (3) fit gaussian to the peak
    return this gaussian fit
    """
    assert( (data.ndim == 2) | (data.ndim == 3) )
    data_hat    = np.fft.fftn(data)
    if (data.ndim == 2):
        nx,ny        = data.shape
        data_hat    *= (1./nx/ny)
        xhat         = np.fft.fftfreq(nx)*nx
        yhat         = np.fft.fftfreq(ny)*ny
        xxhat,yyhat  = np.meshgrid(xhat,yhat)
        freq_avg,data_hat_avg = compute_radial_average_2d(xxhat,yyhat,np.abs(data_hat))
    elif (data.ndim == 3):
        nx,ny,nz  = data.shape
        data_hat *= (1./nx/ny/nz)
        xhat      = np.fft.fftfreq(nx)*nx
        yhat      = np.fft.fftfreq(ny)*ny
        zhat      = np.fft.fftfreq(nz)*nz
        xxhat,yyhat,zzhat     = np.meshgrid(xhat,yhat,zhat)
        freq_avg,data_hat_avg = compute_radial_average_3d(xxhat,yyhat,zzhat,np.abs(data_hat))
    gauss_mean, gauss_sigma = fit_gaussian(freq_avg,data_hat_avg,plot_metrics)
    return gauss_mean,gauss_sigma

def compute_radial_average_2d(xx,yy,signal):
    assert( (xx.shape == yy.shape) & (xx.shape == signal.shape) )
    nx,ny               = xx.shape
    frequency_magnitude = np.sqrt( xx**2 + yy**2 ).ravel().astype(int)
    signal_avg          = np.zeros((frequency_magnitude.max()-frequency_magnitude.min())+1)
    signal              = signal.ravel()
    for i in range(len(frequency_magnitude)):
        signal_avg[frequency_magnitude[i]] += signal[i]
    return np.unique(frequency_magnitude) , signal_avg

def compute_radial_average_3d(xx,yy,zz,signal):
    assert( (xx.shape == yy.shape) & (xx.shape == zz.shape) & (xx.shape == signal.shape) )
    nx,ny,nz            = xx.shape
    frequency_magnitude = np.sqrt( xx**2 + yy**2 + zz**2 ).ravel().astype(int)
    signal              = signal.ravel()
    signal_avg          = np.zeros((frequency_magnitude.max()-frequency_magnitude.min())+1)
    for i in range(len(frequency_magnitude)):
        signal_avg[frequency_magnitude[i]] += signal[i]
    return np.unique(frequency_magnitude) , signal_avg

def fit_gaussian(x,y,plot_fit=False):
    interp_samps = 1000
    x_query      = np.linspace(np.min(x),np.max(x),interp_samps)
    y_interp     = np.interp(x_query,x,y)
    idx_peak     = np.argmax(y_interp)
    sigma        = np.sqrt( np.mean( y_interp*(x_query - x_query[idx_peak])**2 ) )
    if plot_fit:
        plt.figure()
        plt.plot(x_query,y_interp)
        plt.plot(x_query, np.max(y_interp)*np.exp(-0.5*(x_query-x_query[idx_peak])**2/sigma**2),'r')
        plt.legend(['Avg Fourier magnitude','Gaussian fit'])
        plt.show()
    return x_query[idx_peak] , sigma

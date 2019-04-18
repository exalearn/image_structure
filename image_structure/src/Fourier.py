import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.optimize as opt

def fit_gaussian_to_average_fourier_spectrum(data,plot_metrics=False,outdir=None,str_figure=None):
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
    gauss_mean, gauss_sigma = fit_gaussian(freq_avg,data_hat_avg,plot_metrics,outdir,str_figure)
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

def find_nontrivial_maximum(y):
    # Ignore the zero'th fourier coefficient in finding a maximum
    if (np.argmax(y) == 0):
        return np.argmax(y[1:])
    else:
        return np.argmax(y)

def return_peak_centered_spectrum(x,y,idx_peak):
    # Reflect (x,y) about idx_peak
    x_reflect  = np.hstack( [-np.flipud(x[idx_peak+1:]) + 2*x[idx_peak] , x[idx_peak:]] )
    y_reflect  = np.hstack( [ np.flipud(y[idx_peak+1:])                 , y[idx_peak:]] )
    return x_reflect,y_reflect

def restrict_x_data(x,y,decay):
    # Chop-off everything that has decayed past decay*max(y)
    return x[y >= decay*np.max(y)] , y[y >= decay*np.max(y)]
    
def fit_gaussian(x,y,plot_fit=False,outdir=None,str_figure=None):
    interp_samps = 1000  # Number of interpolation upsamples
    d            = 0.2   # Decay parameter for gaussian fitting    
    
    x_query      = np.linspace(np.min(x),np.max(x),interp_samps)
    y_interp     = np.interp(x_query,x,y)
    idx_peak     = find_nontrivial_maximum( y_interp )
    y_interp    /= np.trapz(y_interp,x_query)
    
    # Force mean of gaussian to be at peak and reflect x-axis about the peak for fitting
    x_reflect,y_reflect = return_peak_centered_spectrum(x_query, y_interp, idx_peak)
    x_fitting,y_fitting = restrict_x_data(x_reflect,y_reflect,d)
    guess               = [np.max(y_fitting), x_query[idx_peak] , (np.max(x_fitting)-np.min(x_fitting))/5.]
    try:
        params,uncert       = opt.curve_fit(gauss1d,x_fitting,y_fitting,p0=guess)
    except:
        x_fitting,y_fitting = restrict_x_data(x_reflect,y_reflect,d*0.5)
        guess               = [np.max(y_fitting), x_query[idx_peak] , (np.max(x_fitting)-np.min(x_fitting))/5.]
        params,uncert       = opt.curve_fit(gauss1d,x_reflect,y_reflect,p0=guess)
    params[1] = np.maximum( params[1] , 0 ) # Limiter on mean
    params[2] = np.maximum( params[2] , 0 ) # Limiter on std-dev
    
    if plot_fit:
        plt.figure()
        plt.plot(x_query,y_interp)
        plt.plot(x_query, np.max(y_interp)*np.exp(-0.5*(x_query-params[1])**2/params[2]**2),'r')
        plt.legend(['Avg Fourier magnitude','Gaussian fit'])
        if (outdir is not None):
            outfile = outdir + str_figure + ".png"
            plt.savefig(outfile, bbox_inches='tight')
    return params[1] , params[2]

def gauss1d(x,amp,mu,sigma):
    return amp*np.exp(-0.5*(x-mu)**2/sigma**2)

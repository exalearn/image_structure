import numpy as np
import matplotlib.pyplot as plt

nx = 256
ny = 256

x     = np.linspace(0,2*np.pi,nx)
y     = np.linspace(0,2*np.pi,ny)
xx,yy = np.meshgrid(x,y)
x_hat = np.fft.fftfreq(nx)*nx
y_hat = np.fft.fftfreq(ny)*ny
xx_hat,yy_hat  = np.meshgrid(x_hat,y_hat)

x0         = x_hat[32]
y0         = y_hat[16]
sigma      = float(nx)/128.
four_samp  = 1000
kx_samp    = np.random.normal(x0,sigma,four_samp)
ky_samp    = np.random.normal(y0,sigma,four_samp)
signal     = np.zeros([nx,ny])
for i in range(four_samp):
    signal += np.cos(kx_samp[i]*xx) + 0.3*np.cos(ky_samp[i]*yy)
signal /= four_samp

print( x0 , y0 , sigma )
signal_hat = np.fft.ifft2(signal)

plt.figure()
plt.subplot(121); plt.contourf(xx_hat,yy_hat,signal_hat); plt.title('Frequency domain')
plt.subplot(122); plt.contourf(xx,yy,signal); plt.title('Physical domain')
plt.show()

np.save('../data/signal_gaussian_2d.npy',signal)

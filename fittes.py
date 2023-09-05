import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.signal import bessel, sosfiltfilt


def smooth(x, dt, cutoff):
    #bessel filter
    cutoff = cutoff  # cutoff frequency in Hz
    order = 8  # order of the filter, for most general-purpose high- or low-pass filters, the terms pole and order may be used interchangeably and completely describe the rolloff rate
    # get the filter coefficients
    sos = bessel(order, cutoff, btype='lowpass', output='sos', fs=1.0/dt, norm='mag')  # mag: the filter is normalized such that the gain magnitude is -3dB at angular cutoff frequency
    # apply the filter with the filter coefficients
    return sosfiltfilt(sos, x)

def gauss_fct(x, A, mu, sigma):
    return np.abs(A) * np.exp(-np.square((x - mu))/np.square(sigma))

def gauss_dist_fit(t, xlims, bins):
    # dwell time fit
    h, dwt = np.histogram(t, range=xlims, bins=bins)
    #h = h / np.sum(h)
    dwt = 0.5*(dwt[:-1] + dwt[1:])

    i0, i1 = np.argmax(h), np.argmin(h)
    i1 = len(h)-1
    xv = dwt[i0:i1]
    yv = h[i0:i1] # / np.max(h[i0:i1])

    popt, pcov = curve_fit(gauss_fct, xv, yv, p0=(1.0,0.1,1.0))

    x = np.linspace(dwt[i0], dwt[i1])
    y = gauss_fct(x, popt[0], popt[1], popt[2])

    a = popt[1]
    err = np.sqrt(pcov[1,1])
    
    return x, y, np.power(10,a), np.log(10.0)*np.power(10,a)*err

def exp_fct(x, a, b):
    return a*np.exp(-b*x)

def exp_dist_fit(t, xlims, bins):
    # dwell time fit
    h, dwt = np.histogram(t, range=xlims, bins=bins)
    #h = h / np.sum(h)
    dwt = 0.5*(dwt[:-1] + dwt[1:])

    i0, i1 = np.argmax(h), np.argmin(h)
    i1 = len(h)-1
    xv = dwt[i0:i1]
    yv = h[i0:i1] # / np.max(h[i0:i1])

    popt, pcov = curve_fit(exp_fct, xv, yv, p0=(1.0,0.1))

    x = np.linspace(dwt[i0], dwt[i1])
    y = exp_fct(x, popt[0], popt[1])

    a = popt[1]
    err = np.sqrt(pcov[1][1])/np.square(popt[1])
    
    return x, y, a, err
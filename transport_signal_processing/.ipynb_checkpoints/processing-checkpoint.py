import numpy as np
from scipy.stats import skew, kurtosis


from .signals import barebone_cutoff, find_local_extrema


def extract_core(event):
    ids = barebone_cutoff(event[:,1])
    return event[ids[0]:ids[1]+1]


def extract_reduced(event, smoothing=False):
    # extract optimums
    idvar_min, idvar_max = find_local_extrema(event[:,1], smoothing=smoothing)
    idvar = np.sort(np.concatenate([idvar_min, idvar_max])).ravel()

    # compute
    # t = event[:,0]
    # x = event[:,1]
    # dx = (x[idvar+1]-x[idvar-1]) / (t[idvar+1]-t[idvar-1])
    # hx = (x[idvar+1]-2.0*x[idvar]+x[idvar-1]) / np.square(0.5*(t[idvar+1]-t[idvar-1]))

    return event[idvar]


def compute_stats(event):
    dwt = event[-1,0]
    mean_I = np.mean(event[:,1])
    std_I = np.std(event[:,1])
    skew_I = skew(event[:,1])
    kurt_I = kurtosis(event[:,1])

    return np.array([dwt, mean_I, std_I, skew_I, kurt_I])

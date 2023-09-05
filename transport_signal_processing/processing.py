import numpy as np
from scipy.stats import skew, kurtosis


from .signals import barebone_cutoff, find_local_extrema


def extract_heart(event, sigma_heart = 1, default_value = 1):
    """ extract only the datapoints that have reached a plateau 
    to avoid influence of rise and fall data points on current analysis
    
    event: current data points of event detected by threshold with same filter as during threshold
    sigma: threshold to cut the heart of the event ( x*std(gradient within detected event) ); default= 1
    default_value: threshold condition can not be fulfilled --> how many data points will be removed from each start and end of event; default = 1
    return: first (ev_st) and last (ev_en) index of event that is part of the heart
    """
    # get current values of event
    I_ev = event[:,1]
    #define the relative gradient such that all desired data points are <0
    rel_grad = abs(np.gradient(I_ev))-sigma_heart*np.std(abs(np.gradient(I_ev)))
    #find first desirable data_point in event
    ev_st = np.where(rel_grad == next((x for x in rel_grad if x <= 0), default_value))[0] 
    #find last desireble data point in event
    flp_ev_en = np.where(rel_grad[::-1] == next((x for x in rel_grad[::-1] if x <= 0), default_value))[0] +1
    ev_en = len(rel_grad)-flp_ev_en
    if np.shape(ev_st) == (0,):
        return 0, 0
    else:
        return ev_st[0], ev_en[0]


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


def compute_stats_flip(event, heart):
    dwt = event[-1,0]
    mean_I = np.mean(heart[:,1])
    std_I = np.std(heart[:,1])
    skew_I = skew(heart[:,1])
    kurt_I = kurtosis(heart[:,1])

    return np.array([dwt, mean_I, std_I, skew_I, kurt_I])



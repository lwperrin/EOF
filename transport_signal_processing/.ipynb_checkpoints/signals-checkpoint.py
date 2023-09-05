import numpy as np
from scipy import signal

from .fits import multi_gauss_dist_fit, exp_dist_fit
from .stats import zscore


def split_discont_ids(ids):
    ids = np.sort(ids).ravel()
    brk = np.where(np.diff(ids) > 1)[0]

    irng = np.concatenate([[-1], brk, [len(ids)-1]])

    ids_l = []
    for il, ih in zip(irng[:-1], irng[1:]):
        ids_l.append(ids[il+1:ih+1].ravel())

    return ids_l


def find_continuous_segments(ids):
    ids = np.sort(ids).ravel()
    brk = np.where(np.diff(ids) > 1)[0]
    ids_start = np.concatenate([[ids[0]], ids[brk+1]])
    ids_end = np.concatenate([ids[brk], [ids[-1]]])
    return np.stack([ids_start, ids_end], -1)


def barebone_cutoff(x):
    ids = np.argsort(x)
    il = 0
    ir = len(ids)-1
    vl = x[il]
    vr = x[ir]
    for i in ids[::-1]:
        if (il+1 == i == ir-1):
            return np.array([il+1, ir-1])
        elif (i == il+1):
            il = i
            if vl == x[i]:
                return np.array([il+1, ir-1])
            else:
                vl = x[i]
        elif (i == ir-1):
            ir = i
            if vr == x[i]:
                return np.array([il+1, ir-1])
            else:
                vr = x[i]
        elif (i == il) or (i == ir):
            pass
        else:
            return np.array([il+1, ir-1])


def gauss_filtering(x, weights=None, sigma_tol=2.0, res=1.0):
    # compute density
    nbins = int((np.max(x) - np.min(x)) / res)
    h, b = np.histogram(x, bins=nbins, weights=weights)
    # perform gaussian fit
    z, popt = multi_gauss_dist_fit(0.5*(b[:-1]+b[1:]), h / np.sum(h))
    # filter
    t0 = popt[1] - sigma_tol * np.abs(popt[2])
    t1 = popt[1] + sigma_tol * np.abs(popt[2])
    mask = ((t0 <= x) & (x <= t1))
    return np.where(mask)[0], popt[1], np.abs(popt[2])


def exp_filtering(x, sigma_tol=2.0, res=1.0):
    # compute density
    nbins = int((np.max(x) - np.min(x)) / res)
    h, b = np.histogram(x, bins=nbins)
    # perform exponential distribution fit
    z, popt = exp_dist_fit(0.5*(b[:-1]+b[1:]), h / np.sum(h))

    # filter
    t0 = 0.0
    t1 = sigma_tol / popt[0]
    mask = ((t0 <= x) & (x <= t1))
    return np.where(mask)[0], 1.0/popt[0]


def gauss_filt(N, xl=3.0):
    """Return a normal distribution filter of size N.

    Optional inputs:
        xl: normal distribution evaluation range from -xl to xl
    """
    x = np.linspace(-xl,xl,N)
    y = np.exp(-0.5*x*x)
    return y / np.sum(y)


def square_filt(N, m=10):
    """Return a square filter of size N.

    Optional inputs:
        m: number of points set to 0 at both ends of the filter
    """
    # create filter
    f = np.ones(N)
    # set ends to 0
    f[:m] = 0.0
    f[-m:] = 0.0
    # if only zeros, add a one in the center
    if np.allclose(f,0.0):
        f[int(np.round(0.5*N))] = 1.0

    return f / np.sum(f)


def envelope(x, r):
    X = np.stack([np.pad(x, (i,2*r-i), 'constant', constant_values=(np.nan,np.nan)) for i in range(2*r+1)])[:,r:-r]

    return np.nanmax(X,0), np.nanmin(X,0)


def downsample(x, N, dsf=np.mean):
    # if input is smaller than requrested length, then interpolate to have minimal length
    if len(x) < N:
        x_eff = np.interp(np.linspace(0.0,1.0,N), np.linspace(0.0,1.0,len(x)), x)
    else:
        x_eff = x.copy()

    # split into chuncks
    x_chks = np.array_split(x_eff, N)

    # get downsampling function of each chuncks
    xm = np.array([dsf(x_chk) for x_chk in x_chks])

    return xm


def group_list(l, n):
    N = len(l)
    M = int(np.ceil(N / n))

    l_l = []
    for k in range(n):
        l_l.append(l[k*M:(k+1)*M])

    return l_l


def duration_split(T_a, tol=0.1):
    """Return cluster labels for the input array of durations.

    Optional input:
        tol: maximum variation ratio between the shortest and longest event
             in a cluster
    """
    # sort all signals by durations
    Ts = np.sort(T_a)
    iTs = np.argsort(T_a)

    # set initial values for loop
    id_c, k_c = 0, 0
    T_clust_labels = np.zeros(len(T_a), dtype=int)
    while id_c is not None:
        # get events within the tolerance duration around id_c
        iT_c = iTs[(Ts < Ts[id_c]*(1.0 + tol)) & (Ts >= Ts[id_c])]

        # label events of this cluster
        for i in iT_c:
            T_clust_labels[i] = k_c

        # get next cluster center
        ids_sup = np.where(Ts >= Ts[id_c]*(1.0 + tol))[0]

        if len(ids_sup) > 0:
            id_c = ids_sup[0]
            k_c += 1
        else:
            id_c = None

    return T_clust_labels


def find_local_extrema(x, smoothing=False):
    # smooth signal
    if smoothing:
        x_smth = signal.wiener(x)
    else:
        x_smth = x

    # derivative sign changes
    dvar = np.diff(np.sign(np.diff(x_smth))).astype(int)

    # get local maxima and minima
    idvar_min = np.where(dvar > 0)[0] + 1
    idvar_max = np.where(dvar < 0)[0] + 1

    return idvar_min, idvar_max


def wavelet_transform(X, W, M):
    widths = np.arange(1, W+1)
    return signal.cwt(X, signal.ricker, widths)


def wt_features_extract(I_l, dt, wdt, mdt):
    W = int(wdt / dt)
    M = int(mdt / dt)
    N = len(I_l)

    C_l = []
    for k in range(N):
        # extract core current
        I_ = zscore(I_l[k])

        # compute wavelet transform
        cwtmatr = wavelet_transform(I_, W, M)

        C_l.append(cwtmatr)

    return C_l

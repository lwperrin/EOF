import numpy as np
from scipy.ndimage.filters import gaussian_filter


def zscore(x):
    return (x-np.mean(x))/np.std(x)


def probability_distribution_multidim(X, X_lims, N=100):
    # get limits
    X_min = X_lims[:,0]
    X_max = X_lims[:,1]
    # create bins
    t_bins = np.logspace(np.log(X_min[0]), np.log(X_max[0]), num=N, base=np.e)
    s_bins = np.linspace(X_min[1:], X_max[1:], N)
    X_bins = np.hstack((t_bins.reshape(-1,1),s_bins))
    # compute histogram
    H, _ = np.histogramdd(X, bins=X_bins.T)
    # gaussian smoothing
    H = gaussian_filter(H, 2.0)
    return H / (np.sum(H)+1e-6)


def prob_dist_1D(x, x_lim, N=20, log=False):
    # create bins
    if log:
        x_bins = np.logspace(np.log(x_lim[0]), np.log(x_lim[1]), num=N, base=np.e)
    else:
        x_bins = np.linspace(x_lim[0], x_lim[1], N)

    # compute histogram
    h, _ = np.histogram(x, bins=x_bins)

    return h / np.sum(h)


def probability_distribution(X, X_lims, N=50):
    D = X.shape[1]
    Pi = np.zeros((N-1,D), dtype=np.float64)
    for d in range(D):
        Pi[:,d] = prob_dist_1D(X[:,d], X_lims[d], log=(d==0), N=N)

    # return Pi / np.sum(Pi)
    return Pi


def DKL(p,q):
    p = (p + 1e-12)
    p /= np.sum(p)
    q = (q + 1e-12)
    q /= np.sum(q)
    return -np.sum(p * np.log(q / p))


def divergence_matrix(stats_l, eps=1e-3):
    # perform linkage
    N = len(stats_l)

    # get limits
    S = np.concatenate(stats_l)
    stats_lims = np.stack([np.quantile(S, eps, 0), np.quantile(S, 1.0-eps, 0)],1)
    # stats_lims = np.stack([np.min(np.concatenate(stats_l),0), np.max(np.concatenate(stats_l),0)],1)

    # metrics
    D = np.zeros((N,N), dtype=np.float64)
    for i in range(N):
        Pi = probability_distribution_multidim(stats_l[i], stats_lims)
        # Pi = probability_distribution(stats_l[i], stats_lims)
        for j in range(i, N):
            Pj = probability_distribution_multidim(stats_l[j], stats_lims)
            # Pj = probability_distribution(stats_l[j], stats_lims)
            if not (np.isnan(Pi).any() or np.isnan(Pj).any()):
                # D[i,j] = 1.0-np.sum(np.sqrt(Pi * Pj))
                z = 0.5 * (DKL(Pi,Pj) + DKL(Pj,Pi))
                D[i,j] = z
                D[j,i] = z
            else:
                D[i,j] = 1e8
                D[j,i] = 1e8

    return D

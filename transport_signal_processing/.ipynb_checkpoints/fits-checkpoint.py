import numpy as np
from scipy.optimize import curve_fit


def gauss(x, A, mu, sigma):
    return np.abs(A) * np.exp(-0.5*np.square((x - mu))/np.square(sigma))


def multi_gauss(x, *p):
    y = np.zeros(x.shape)
    for k in range(int(len(p) / 3)):
        y += gauss(x, p[3*k], p[3*k+1], p[3*k+2])

    return y


def multi_gauss_dist_fit(x, y, num_gauss=1, mu_guess=None):
    if mu_guess is None:
        #mu_guess = np.argmax(y)
        mu_guess = x[np.argmax(y)]

    # create guess for fit
    if num_gauss == 1:
        guess = np.array([np.max(y), mu_guess, np.std(x)])
    else:
        x_l = [np.max(y), mu_guess, 0.5*(np.max(x)-np.min(x))]
        x_l += [np.percentile(x, p) for p in np.linspace(40.0, 60.0, num_gauss-1)]
        guess = np.array([[np.max(y) / num_gauss, x_l[k], 1.0] for k in range(num_gauss)]).ravel()

    # perform multi gauss fit
    try:
        popt, pcov = curve_fit(multi_gauss, x.astype(np.float64), y.astype(np.float64), p0=guess.astype(np.float64))
    except RuntimeError:
        print("ERROR: gaussian fit did not converge")
        popt = np.zeros(guess.shape)
    except Exception as e:
        print("ERROR: {}".format(e))
        popt = np.zeros(guess.shape)

    # compute gaussian functions
    z = multi_gauss(x, *popt)

    return z, popt


def multi_gauss_decomposition(I, threshold=0.1, resolution=1.0, max_num_gauss=5, mu_guess=None):
    # determine the number of bins and make histogram
    nbins = max(int(np.floor((np.max(I) - np.min(I))/resolution)), 10)
    h = np.histogram(I, bins=nbins, density=True)

    # get histogram points
    y = h[0].copy()
    x = 0.5*(h[1][:-1] + h[1][1:])

    # compute integral
    s0 = np.sum(y)

    # perform interative gaussian fits until only "thr" of the distribution is unexplained
    best_popt_l = []
    best_r = 1e9
    for k in range(max_num_gauss):
        z, popt = multi_gauss_dist_fit(x, y, num_gauss=k+1, mu_guess=mu_guess)

        # compute residual error
        s = np.sum(np.abs(y - z))
        r = s/s0

        # evaluate result
        if r < threshold:
            return np.array([popt[3*k:3*(k+1)] for k in range(k+1)]), x, y
        elif best_r > r:
            best_popt_l = [popt[3*k:3*(k+1)] for k in range(k+1)]

    return np.array(best_popt_l), x, y


def exp_pdf(x, l):
    return l * np.exp(-l*x)


def exp_dist_fit(x, y):
    # create guess for fit
    guess = np.array([1.0])

    # perform multi poisson fit
    popt, pcov = curve_fit(exp_pdf, x.astype(np.float64), y.astype(np.float64), p0=guess.astype(np.float64))

    # compute poisson functions
    z = exp_pdf(x, popt[0])

    return z, popt

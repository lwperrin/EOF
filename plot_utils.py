import numpy as np
import matplotlib.pyplot as plt


import transport_signal_processing as tsp


def plot_segmented_signal(I, dt, segments_rng, n_skip=1, colors=('r', 'g')):
    # get time range
    t = np.arange(I.shape[0])*dt

    # plot
    plt.plot(t[::n_skip], I[::n_skip], 'k', alpha=0.3)
    for k in range(segments_rng.shape[0]):
        color = ['r', 'g'][k%2]
        rng = segments_rng[k]
        plt.plot(t[rng[0]:rng[1]+1][::n_skip], I[rng[0]:rng[1]+1][::n_skip], color=color)
    plt.xlim(t[0], t[-1])


def plot_open_pore_current_fit(fit_data, sigma_tol):
    # unpack data
    popt, x, y = fit_data
    mI_open = popt[1]
    sI_open = np.abs(popt[2])

    # plot
    plt.plot(x, y)
    plt.plot(x, tsp.fits.gauss(x, popt[0], popt[1], popt[2]))
    plt.plot([mI_open-sigma_tol*sI_open]*2,[0.0,np.max(y)], 'k--')


def plot_segment_with_detected_events(I_seg, dt, rng_events, mI_open, n_skip=1):
    # time step
    t = 1e3*np.arange(I_seg.shape[0])*dt
    # highlight events
    I_seg_evts = mI_open*np.ones(I_seg.shape[0])
    for i0,i1 in rng_events:
        I_seg_evts[i0:i1+1] = I_seg[i0:i1+1]

    # plot
    plt.plot(t[::n_skip], I_seg[::n_skip], lw=1.0, color='k', alpha=0.5)
    plt.plot(t[::n_skip], I_seg_evts[::n_skip], lw=2.0)
    plt.xlim(t[0], t[-1])


def plot_selected_segments(mI_open, sI_open, mI_open_lims, sI_open_lims, m):
    # define data and range
    x = mI_open
    y = sI_open
    xl = mI_open_lims
    yl = sI_open_lims

    # plot
    plt.plot(x[~m], y[~m], '.', ms=3)
    plt.plot(x[m], y[m], '.', ms=3)
    plt.plot([xl[0], xl[1]], [yl[0], yl[0]], 'k-', alpha=0.5)
    plt.plot([xl[0], xl[1]], [yl[1], yl[1]], 'k-', alpha=0.5)
    plt.plot([xl[0], xl[0]], [yl[0], yl[1]], 'k-', alpha=0.5)
    plt.plot([xl[1], xl[1]], [yl[0], yl[1]], 'k-', alpha=0.5)

def plot_selected_events(stats, I_lims, t_lims, mf, alpha=0.5):
    plt.semilogy(stats[~mf,1], 1e3*stats[~mf,0], '.', ms=1.0, alpha=alpha)
    plt.semilogy(stats[mf,1], 1e3*stats[mf,0], '.', ms=1.0, alpha=alpha)
    plt.plot([I_lims[0], I_lims[1]], [t_lims[0], t_lims[0]], 'k-', alpha=alpha)
    plt.plot([I_lims[0], I_lims[1]], [t_lims[1], t_lims[1]], 'k-', alpha=alpha)
    plt.plot([I_lims[0], I_lims[0]], [t_lims[0], t_lims[1]], 'k-', alpha=alpha)
    plt.plot([I_lims[1], I_lims[1]], [t_lims[0], t_lims[1]], 'k-', alpha=alpha)

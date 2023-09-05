import numpy as np
import pandas as pd
import transport_signal_processing as tsp


def segments_selection(sigman, sinfo_l, mI_open_lims, sI_open_lims):
    # pack info into dataframe
    df = pd.DataFrame(sinfo_l)

    # define data and range
    x = df['mI_open'].values
    y = df['sI_open'].values
    xl = mI_open_lims
    yl = sI_open_lims

    # count and range selection
    m = (((x >= xl[0]) & (x <= xl[1])) & ((y >= yl[0]) & (y <= yl[1])))

    # save selection
    for i in range(len(m)):
        sinfo = sinfo_l[i]
        bsel = m[i]
        sigman.insert_info(sinfo, 's{}'.format(sinfo['sid']), selected=int(bsel))

    return m


def events_selection(sigman, sinfo_l, I_lims, t_lims):
    # load statistics of all events
    s, ids = tsp.utils.load_segments_data(sigman, sinfo_l, "stats")
    s = np.array(s)

    # convert to dataframe
    df = pd.DataFrame(sinfo_l)

    # define selection filter
    mf = np.ones(s.shape[0], dtype=bool)

    # rescale with open pore current
    s_mI_open = df.iloc[ids]['mI_open'].values
    alpha = 100.0 / s_mI_open
    s[:,1] = s[:,1]*alpha

    # manual filter
    mf = mf & ((1e3*s[:,0] >= t_lims[0]) & (1e3*s[:,0] <= t_lims[1]))
    mf = mf & ((s[:,1] >= I_lims[0]) & (s[:,1] <= I_lims[1]))

    # one signal at a time for the current selected polymer name
    for i in np.unique(ids):
        mseg = (ids == i)
        sinfo = sinfo_l[i]

        # store info
        n_sel = np.sum(mf[mseg])
        n_tot = np.sum(mseg)
        r_sel = float(n_sel) / float(n_tot)
        sigman.insert_info(sinfo, 's{}'.format(sinfo['sid']), ratio_sel=r_sel)

        # store data
        m_seg_sel = mf[mseg].astype(int)
        sigman.insert_data(sinfo, 's{}-selected'.format(sinfo['sid']), m_seg_sel)

    return s, mf

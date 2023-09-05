import numpy as np

from .signals import split_discont_ids, find_continuous_segments
from .fits import multi_gauss_decomposition


def discontinuity_split(I, thr=50.0*1e9, L=20000, S=1000):
    k_splits = np.arange(0, I.shape[0]-L, S)
    Imax = np.zeros(k_splits.shape[0], dtype=np.float32)
    for j in range(k_splits.shape[0]):
        k = k_splits[j]
        I_slc = I[k:k+L]
        Imax[j] = np.max(I_slc) - np.mean(I_slc)

    dImax = np.diff(Imax)
    i_trs = np.where(np.abs(dImax) > thr)[0]

    mask_incr = (np.sign(dImax[i_trs]) > 0.0).astype(np.int64)

    ids_trs = S*i_trs + mask_incr*L

    ids_start = np.concatenate([[0], ids_trs+1])
    ids_end = np.concatenate([ids_trs, [I.shape[0]-1]])
    return np.stack([ids_start, ids_end], -1)


def signal_segmentation(I, V, dt, voltage=None, min_segment_duration=10.0, dt_stab=0.0, print=print):
    # debug print
    if voltage is not None:
        print("> Expected voltage mV: {:.0f}".format(voltage))

    # define minimum segment length
    Lmin = int(min_segment_duration/dt)

    # stabilization time
    n_stab = int(dt_stab / (dt*1e3))
    print("> Stabilization time {:.2f} ms".format(dt_stab))

    # segment by voltage discontinuity and filter
    if voltage is None:
        ids_V_split = split_discont_ids(np.arange(V.shape[0]))
    else:
        ids_V_split = split_discont_ids(np.where(np.isclose(V, voltage))[0])
    ids_V_split = [ids for ids in ids_V_split if ids.shape[0] > Lmin]

    # debug print
    print("> Segments count (voltage discontinuity): {}".format(len(ids_V_split)))

    # segment by large scale change
    rng_state_trans = []
    for ids in ids_V_split:
        rng_trs = discontinuity_split(I[ids])
        filter_trs = (np.diff(rng_trs,1) >= Lmin).ravel()
        rng_state_trans.append(rng_trs[filter_trs] + ids[0])

    if len(rng_state_trans) == 0:
        print("> No segments found")
        return np.array([])
    else:
        rng_state_trans = np.concatenate(rng_state_trans)
        # debug print
        print("> Segments count (voltage & current discontinuity): {}".format(rng_state_trans.shape[0]))

    # define number of segment and check that it is not zero
    N_segs = rng_state_trans.shape[0]
    if N_segs == 0:
        return np.array([])

    # check each segments
    kept_segments = []
    for k in range(N_segs):
        # extract signal and check that the signal is not too low
        rng = rng_state_trans[k]
        I_ = I[rng[0]:rng[1]+1]

        # perform checks
        if int(np.floor((np.max(I_) - np.min(I_))*2.0)) < 10:
            print("> Segment [{},{}] has too low amplitude: I_max - I_min = {:.3f}".format(rng[0], rng[1], np.max(I_) - np.min(I_)))
            continue

        if np.std(I_) < 0.2:
            print("> Segment [{},{}] has too low amplitude: sigma_I = {:.3f}".format(rng[0], rng[1], np.std(I_)))
            continue

        if np.quantile(I_, 0.9) - np.quantile(I_, 0.1) < 0.01:
            print("> Segment [{},{}] has too low quantile delta: delta_q = {:.4f}".format(rng[0], rng[1], np.quantile(I_, 0.9) - np.quantile(I_, 0.1)))
            continue

        # if checks successful, keep range
        kept_segments.append([rng[0]+n_stab, rng[1]])

    return np.array(kept_segments)


def detect_open_pore_level(I, expected_open_current=None, residual_thr=0.1, resolution=1.0):
    # perform multi gaussian distribution decomposition
    popts, x, y = multi_gauss_decomposition(I, threshold=residual_thr, resolution=resolution, mu_guess=expected_open_current)

    if expected_open_current is None:
        # get gaussian fit with highest current
        gid_sup = np.argmax(popts[:,0])
    else:
        # get gaussian fit closest to expected open current
        gid_sup = np.argmin(np.abs(popts[:,1] - expected_open_current))

    return popts[gid_sup], x, y


def extend_events(I, rng_events, I_thr):
    # mean event length
    mL = int(np.mean(np.array([rng[1]-rng[0]+1 for rng in rng_events])))+1
    
    # extend events forward
    rng_events_extended = []
    for i in range(len(rng_events)):
        k = rng_events[i,0]
        L = min(len(I), k+100*mL)
        for j in range(k,L):
            if I[j] > I_thr:
                break
        rng_events_extended.append([k,j])
            
    # update event ranges
    rng_events = np.array(rng_events_extended).copy()
    
    # extend events backward
    rng_events_extended = []
    for i in range(len(rng_events)):
        k = rng_events[i,1]
        L = max(0, k-100*mL)
        for j in range(k-1,L-1,-1):
            if I[j] > I_thr:
                break
        rng_events_extended.append([j,k])
            
    return np.unique(np.array(rng_events_extended), axis=0)


def parse_segment_legacy(I, mI_open, sI_open, dt, sigma_tol=3.0, sigma_tol_out=-1.0, min_duration=0.0002):
    # define threshold
    I_thr = mI_open - sigma_tol*np.abs(sI_open)

    # extract segments of open and closed pore current
    detect_mask = (I < I_thr)
    if np.sum(detect_mask) == 0:
        return [], []
    elif np.sum(~detect_mask) == 0:
        return [], []

    rng_events = find_continuous_segments(np.where(detect_mask)[0])
    rng_opens = find_continuous_segments(np.where(~detect_mask)[0])
    
    # extend events
    I_thr_out = mI_open - sigma_tol_out*np.abs(sI_open)
    rng_events = extend_events(I, rng_events, I_thr_out)

    # compute minimum event length and filter events
    min_length = max(int(min_duration / dt), 1)
    filter_mask = (np.diff(rng_events,1) + 1 >= min_length).ravel()
    rng_events = rng_events[filter_mask]

    return rng_events, rng_opens


def parse_segment(I, mI_open, sI_open, dt, sigma_tol=3.0, sigma_tol_out=-1.0, n_skip=10):
    # define threshold
    I_thr_out = mI_open - sigma_tol_out*np.abs(sI_open)
    I_thr = mI_open - sigma_tol*np.abs(sI_open)
    
    # detect events with multi-threshold
    rng_events = []
    i0, il, i1 = -1, -1, -1
    for i in range(0,len(I),n_skip):
        # in threshold reached
        if (I[i] > I_thr_out) and (il < 0):
            # backtrack to first hit from the right
            for j in range(min(i+n_skip,I.shape[0]-1), max(i-n_skip+1,0), -1):
                if (I[j] > I_thr_out) and (il < 0):
                    i0 = j
                    break

        # low threshold reached 
        if (I[i] < I_thr) and (i0 >= 0):
            # backtrack to first hit from the right
            for j in range(min(i+n_skip,I.shape[0]-1), max(i-n_skip+1,0), -1):
                if (I[j] < I_thr) and (i0 >= 0):
                    il = j
                    break

        # out threshold reached
        if (I[i] > I_thr_out) and (i0 >= 0) and (il >= 0):
            # backtrack to first hit from the left
            for j in range(max(i-n_skip,0), min(i+n_skip+1,I.shape[0])):
                if (I[j] > I_thr_out) and (i0 >= 0) and (il >= 0):
                    i1 = j
                    break
            rng_events.append(np.array([i0, i1]))
            i0, il, i1 = -1, -1, -1  # reset
            
    # get open events
    rng_opens = [np.array([r0[1]+1, r1[0]-1]) for r0, r1 in zip(rng_events[:-1], rng_events[1:])]

    return rng_events, rng_opens

def parse_segment_flip(I, mI_open, sI_open, dt, sigma_tol=3.0, sigma_tol_out=1.0, n_skip=10):
    """
    Parses segments from a time series data I based on multiple threshold criteria.
    
    Parameters:
        I (numpy.ndarray): Time series data.
        mI_open (float): Mean value of the open state.
        sI_open (float): Standard deviation of the open state.
        dt (float): Time step or time interval between data points.
        sigma_tol (float, optional): Threshold tolerance for low threshold. Default is 3.0.
        sigma_tol_out (float, optional): Threshold tolerance for outer threshold. Default is -1.0.
        n_skip (int, optional): Number of data points to skip during iteration. Default is 10.
        
    Returns:
        rng_opens (list of numpy.ndarray): List of arrays containing start and end positions of open events.
    """
    
    # Define threshold for the outer and inner thresholds
    I_thr_out = mI_open - sigma_tol_out * np.abs(sI_open)
    I_thr = mI_open - sigma_tol * np.abs(sI_open)
    
    # Initialize a list to store detected events
    rng_events = []
    i0, il, i1 = -1, -1, -1
    
    # Iterate through the time series data
    for i in range(0, len(I), n_skip):
        if (I[i] < I_thr) and (il < 0):
            for j in range(min(i + n_skip, I.shape[0] - 1), max(i - n_skip + 1, 0), -1):
                if (I[j] < I_thr) and (il < 0):
                    i0 = j #mark the start of an event
                    break

 
        if (I[i] < I_thr) and (i0 >= 0):
            # Backtrack to find the first hit from the right
            for j in range(min(i + n_skip, I.shape[0] - 1), max(i - n_skip + 1, 0), -1):
                if (I[j] < I_thr) and (i0 >= 0):
                    il = j  
                    break

    
        if (I[i] > I_thr_out) and (i0 >= 0) and (il >= 0):
            for j in range(max(i - n_skip, 0), min(i + n_skip + 1, I.shape[0])):
                if (I[j] > I_thr_out) and (i0 >= 0) and (il >= 0):
                    i1 = j  # Mark the end of the event
                    break
            rng_events.append(np.array([i0, il]))  # Store the event's start and end positions
            i0, il, i1 = -1, -1, -1  # Reset for the next event detection           
    # Create a list of arrays containing start and end positions of open events
    rng_opens = [np.array([r0[1] + 1, r1[0] - 1]) for r0, r1 in zip(rng_events[:-1], rng_events[1:])]
    
    return rng_events, rng_opens


def slice_events(I_seg, dt, rng_events, rng_opens):
    # extract and save events
    if len(rng_events) > 0:
        # extract events
        events = []
        for i0,i1 in rng_events:
            I_ = I_seg[i0:i1+1]
            t_ = np.arange(0, i1-i0+1) * dt

            events.append(np.stack([t_,I_], -1))

        # extract extended events
        ext_events = []
        for i0,i1 in rng_events:
            s = max(int((i1-i0)*1.0), 10)
            I_ = I_seg[max((i0-s),0):min((i1+s+1),len(I_seg))]
            t_ = np.arange(0, len(I_)) * dt

            ext_events.append(np.stack([t_,I_], -1))

        # extract open events
        open_events = []
        for i0,i1 in rng_opens:
            I_ = I_seg[i0:i1+1].astype(np.float32)
            t_ = (np.arange(0, i1-i0+1) * dt).astype(np.float32)

            open_events.append(np.stack([t_,I_], -1))

    return events, ext_events, open_events

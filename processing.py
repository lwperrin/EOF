from fittes import smooth
import numpy as np

import transport_signal_processing as tsp


def segment_signal(sigman, minfo, cutoff, swap_channel=False, filter_fct=None, min_segment_duration=0.0, dt_stab=0.0, voltage_segmentation=True, rescale_factor=1.0):
    # load measurement
    I, V, dt = tsp.utils.load_measurement(sigman, minfo, swap=swap_channel)

    # rescale current
    if max(I) < 3:
        I = I*1000

    # insert time step info into measurement info
    sigman.insert_info(minfo, 'm', dt=dt)

    # set voltage
    if voltage_segmentation:
        voltage = float(minfo['voltage'])
    else:
        voltage = None

    # filter function
    if filter_fct is None:
        Im = I
        Ic = I
    else:
        if cutoff[0] is None:
            Im = I
        else:
            Im = filter_fct(I, dt, cutoff[0])

        if cutoff[1] is None:
            Ic = I
        else:
            Ic = filter_fct(I, dt, cutoff[1])

    # update minimum segment duration
    min_segment_duration = max(min_segment_duration, 1e-3*dt_stab*2.0)

    # detect segments
    segments_rng = tsp.detection.signal_segmentation(Im, np.round(V), dt, voltage=voltage, min_segment_duration=min_segment_duration, dt_stab=dt_stab)

    # save ranges
    for k in range(segments_rng.shape[0]):
        rg = [int(s) for s in segments_rng[k]]
        sigman.insert_info(minfo, 's{}'.format(k), sid=k, segment_range=rg, segment_duration=(rg[1]-rg[0])*dt)

    return Im, Ic, I, dt, segments_rng


def detect_events_flip(sigman, sinfo, I_seg, Im_seg, dt, sigma_tol, sigma_tol_out, residual_thr, resolution, expected_open_current=None, n_skip=10):
    # find open pore current level on smoothed signal
    popt, x, y = tsp.detection.detect_open_pore_level(I_seg, expected_open_current=expected_open_current, residual_thr=residual_thr, resolution=resolution)
    mI_open = popt[1]
    sI_open = np.abs(popt[2])
    
    # parse segment for events
    rng_events, rng_opens = tsp.detection.parse_segment_flip(Im_seg, mI_open, sI_open, dt, sigma_tol=sigma_tol, sigma_tol_out=sigma_tol_out, n_skip=n_skip)
    rng_events = list(filter(lambda r: (r[1]-r[0])>1, rng_events))
    # store metadata
    sigman.update_info(sinfo, 's{}'.format(sinfo['sid']), mI_open=mI_open, sI_open=sI_open, N_events=len(rng_events))

    return rng_events, rng_opens, mI_open, sI_open, (popt, x, y)


def process_events_flip(sigman, sinfo, Im_seg, Ic_seg, dt, rng_events, rng_opens, sigma_heart, cutoff, min_length_core=2, min_num_extrema=3):
    # extract events from raw current
    events, ext_events, open_events = tsp.detection.slice_events(Im_seg, dt, rng_events, rng_opens)
    events_c, ext_events_c, open_events_c = tsp.detection.slice_events(Ic_seg, dt, rng_events, rng_opens)
    
    # extract events core and heart and reduced events
    cores = []
    reduced = []
    stats = []
    hearts =  []
    for w, evt in enumerate(events):
        if len(evt) > 5:         
            # extract core
            core = tsp.processing.extract_core(evt)
            # extract optimums
            optimums = tsp.processing.extract_reduced(core, smoothing=False)
            # extract indeces of heart
            ev_st, ev_en = tsp.processing.extract_heart(evt, sigma_heart)
            # define heart as the plateau part of the event with filter cutoff two if it is long enough            
            if (ev_en - ev_st) > 0:
                heart = events_c[w][ev_st:ev_en+1,:]
            else:
                heart = []           
            # check if core and optimums are valid
            if (len(heart) >= min_length_core) and (len(optimums) >= min_num_extrema):               
                cores.append(core)
                reduced.append(optimums)
                hearts.append(heart)
                # compute stats
                stats.append(tsp.processing.compute_stats_flip(core, heart))
                stats[-1][0] = evt[-1,0]  # correct dwell time

    # store info
    sid = sinfo['sid']
    print(sinfo)
    sigman.insert_info(sinfo, 's{}'.format(sid), N_cores=len(cores))
    sigman.insert_info(sinfo, 's{}'.format(sid), N_reduced=len(reduced))

    # store data
    sigman.insert_data(sinfo, 's{}-events-range'.format(sid), np.array(rng_events))
    sigman.insert_data(sinfo, 's{}-events'.format(sid), np.array(events, dtype=object))
    sigman.insert_data(sinfo, 's{}-open-events'.format(sid), np.array(open_events, dtype=object))
    sigman.insert_data(sinfo, 's{}-extended-events'.format(sid), np.array(ext_events, dtype=object))

    # store data
    if len(cores) > 0:
        sigman.insert_data(sinfo, 's{}-core-events'.format(sinfo['sid']), np.array(cores, dtype=object))
    if len(hearts) > 0:
        sigman.insert_data(sinfo, 's{}-heart-events'.format(sinfo['sid']), np.array(hearts, dtype=object))   
    if len(reduced) > 0:
        sigman.insert_data(sinfo, 's{}-reduced-events'.format(sinfo['sid']), np.array(reduced, dtype=object))
    if len(stats) > 0:
        sigman.insert_data(sinfo, 's{}-stats'.format(sinfo['sid']), np.array(stats))

    return events, cores, stats, hearts

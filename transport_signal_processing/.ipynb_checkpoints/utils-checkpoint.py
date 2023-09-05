import re
import os
import pyabf
import shutil
import numpy as np
import pandas as pd
from neo import AxonIO


def read_abf(filename):
    # I in [pA], dt in [s]
    abf = pyabf.ABF(filename)
    
    # sampling time
    x = abf.sweepX
    dt = np.mean(np.diff(x))
    
    # current
    abf.setSweep(sweepNumber=0, channel=0)
    I_l = [abf.sweepY]
    
    # voltage
    abf.setSweep(sweepNumber=0, channel=1)
    V_l = [abf.sweepY]

    return I_l, V_l, dt


def read_abf_python_neo(filename):
    # I in [pA], dt in [s]
    r = AxonIO(filename)
    bl = r.read()

    I_l, V_l = [], []
    for seg in bl[0].segments:
        I_l.append(np.array(seg.analogsignals[0]))
        V_l.append(np.array(seg.analogsignals[1]))

    dt = 1.0 / float(seg.analogsignals[0].sampling_rate)

    return I_l, V_l, dt


def read_dat_simon(filepath, verbose=False):
    # load file
    with open(filepath, 'rb') as fs:
        raw_data = fs.read()

    # read header
    header = raw_data[:1000].decode("utf-8").strip()

    # debug print
    if verbose:
        print(header)

    # get data starting byte location
    m = re.search(r"Binary data start at position \[Byte\]: ([0-9]*)", header)
    byte_start = int(m[1])
    # get aquisition rate [Hz]
    m = re.search(r"Acquisition Rate \[Hz\]: ([0-9]*\.[0-9]*)", header)
    acq_rate = float(m[1])

    # read data
    data = raw_data[byte_start:]
    buffer = np.frombuffer(data, dtype='>f4')
    # extract and convert relevant data
    I = buffer[::2]*1e12  # pA
    V = buffer[1::2]*1e3  # mV

    # clip
    I = np.clip(I, -300, 300)

    return [I], [V], 1.0 / acq_rate


def read_dat(filepath, verbose=False):
    # load file
    with open(filepath, 'rb') as fs:
        raw_data = fs.read()

    # read data
    buffer = np.frombuffer(raw_data, dtype='<f4')
    # extract and convert relevant data
    I = buffer[1::2]  # pA
    V = buffer[::2]  # mV

    # clip
    I = np.clip(I, -300, 300)

    return [I], [V], 1e-5


def load_measurement(sigman, minfo, swap=False):
    # extract useful info
    filepath = minfo['filepath']

    # read abf file and extract list of currents (I_l) [pA], voltages (V_l) [mV] and timestep (dt) [s]
    if filepath.split('.')[-1].lower() == "abf":
        I_l, V_l, dt = read_abf(filepath)
    elif filepath.split('.')[-1].lower() == "dat":
        I_l, V_l, dt = read_dat(filepath, verbose=True)
    else:
        assert False, f"File format not supported: {filepath}"

    if swap:
        I = V_l[0].ravel().astype(np.float32)  # in pA
        V = I_l[0].ravel().astype(np.float32)  # in mV
    else:
        I = I_l[0].ravel().astype(np.float32)  # in pA
        V = V_l[0].ravel().astype(np.float32)  # in mV

    print("mean voltage: {:.3f} mV".format(np.mean(V)))
    print("acquisition rate: {} Hz".format(1.0/dt))

    return I, V, dt


def check_db_clashes(sigman, df_metadata):
    paths = {}
    for i, msr in df_metadata.iterrows():
        path = sigman.define_db_path(msr)
        if path not in paths:
            paths[path] = msr
        else:
            assert False, f"Error: two measurements have the same database path: {path}\n{msr['filepath']}\n{paths[path]['filepath']}"


def insert_metadata_from_dataframe_(sigman, df_metadata, root="", overwrite=False):
    # insert without overwrite
    paths = []
    for i, msr in df_metadata.iterrows():
        path = os.path.join(sigman.db_root, sigman.define_db_path(msr))
        if os.path.exists(path):
            paths.append(path)
        else:
            # debug print
            print("inserting: {}".format(path))

            # insert data
            sigman.insert_info(dict(msr), 'm', filepath=os.path.join(root, msr['filepath']))

    # overwrite
    if overwrite:
        # clear database
        for path in paths:
            shutil.rmtree(path)

        # insert measurements
        for i, msr in df_metadata.iterrows():
            # insert data
            sigman.insert_info(dict(msr), 'm', filepath=os.path.join(root, msr['filepath']))

            # debug print
            print("overwriting: {}".format(os.path.join(sigman.db_root, sigman.define_db_path(msr))))


def insert_metadata_from_dataframe(sigman, df_metadata, root="", overwrite=False):
    # check for untracked
    db_set = set([os.path.join(sigman.db_root, sigman.define_db_path(minfo)) for minfo in sigman.load_info('*', 'm')])
    data_set = set([os.path.join(sigman.db_root, sigman.define_db_path(minfo)) for minfo in df_metadata.to_dict('records')])
    diff_set = db_set.difference(data_set)

    # clear untracked in database
    if len(diff_set) > 0:
        print("Removing untracked measures")
        for path in diff_set:
            print(path)
            shutil.rmtree(path)

    # check for overwrite
    paths = []
    inserts = []
    for i, msr in df_metadata.iterrows():
        path = os.path.join(sigman.db_root, sigman.define_db_path(msr))
        if os.path.exists(path):
            paths.append(path)
        else:
            inserts.append(i)

    # overwrite updated measurements
    if overwrite and (len(paths) > 0):
        # debug print
        print("Overwriting")
        print('\n'.join(paths)+'\n')

        # clear database
        for path in paths:
            shutil.rmtree(path)

        # insert measurements
        for i, msr in df_metadata.iterrows():
            # insert data
            msr['filepath'] = os.path.join(root, msr['filepath'])
            sigman.insert_info(dict(msr), 'm', **dict(msr))

    # insert new measurements
    for i, msr in df_metadata.loc[inserts].iterrows():
        # insert data
        msr['filepath'] = os.path.join(root, msr['filepath'])
        sigman.insert_info(dict(msr), 'm', **dict(msr))

    # debug print
    print(f"{len(inserts)} measurements inserted")


def load_segments_data(sigman, sinfo_l, mod, selected_only=False):
    ids = []
    data = []
    for i in range(len(sinfo_l)):
        sinfo = sinfo_l[i]

        if (selected_only and (sinfo['selected'] > 0)) or (not selected_only):
            curr_path = sigman.define_db_path(sinfo)

            data_l = sigman.load_data(curr_path, 's{}-{}'.format(sinfo['sid'], mod))
            assert len(data_l) < 2
            if len(data_l) == 1:
                if selected_only:
                    m = sigman.load_data(curr_path, 's{}-selected'.format(sinfo['sid']))[0].astype(bool)
                else:
                    m = np.ones(data_l[0].shape[0], dtype=bool)

                # assert len(m) == len(data_l[0])
                if len(m) != len(data_l[0]):
                    m = np.ones(data_l[0].shape[0], dtype=bool)

                if np.sum(m) > 0:
                    ids.append(i * np.ones(data_l[0].shape[0], dtype=int)[m])
                    data.extend(data_l[0][m])

    ids = np.concatenate(ids)

    return data, ids


def load_stats_for_key(sigman, sinfo_l, key_sel, selected_only=True):
    # construct dataframe
    df = pd.DataFrame(sinfo_l)

    # set unique instance id
    df['iid'] = np.unique(df[key_sel], return_inverse=True)[1]

    # debug print
    key_sels = np.unique(df[key_sel].values)

    # load statistics of all events
    ids = []
    stats = []
    for i in range(len(sinfo_l)):
        sinfo = sinfo_l[i]
        curr_path = sigman.define_db_path(sinfo)

        s_l = sigman.load_data(curr_path, 's{}-stats'.format(sinfo['sid']))
        if len(s_l) > 0:
            s = s_l[0]
            if selected_only:
                m = sigman.load_data(curr_path, 's{}-selected'.format(sinfo['sid']))[0].astype(bool)
                s = s[m]

            ids.append(df['iid'].values[i] * np.ones(s.shape[0], dtype=int))
            # rescale mean and standard deviation based on open pore stats
            s[:,1] = 100.0 * s[:,1] / sinfo['mI_open']
            #s[:,2] = s[:,2] / sinfo['sI_open']  # BAD
            stats.append(s)

    # pack stats
    ids_stats = np.concatenate(ids)
    stats = np.concatenate(stats)

    # rescale dwell time s -> ms
    stats[:,0] = stats[:,0] * 1e3

    # find reduced stats for each selected cases
    stats_dict = {}
    for key in key_sels:
        df_key = df[df[key_sel] == key]
        m = np.zeros(ids_stats.shape, dtype=bool)
        for i in np.unique(df_key['iid']):
            if i in ids_stats:
                m = (m | (ids_stats == i))

        stats_dict[key] = stats[m]

    return stats_dict


def load_core_events(sigman, sinfo_l, selected_only=True):
    # load core events
    cores_all = []
    for i in range(len(sinfo_l)):
        # get path within the database
        sinfo = sinfo_l[i]
        curr_path = sigman.define_db_path(sinfo)

        # load cores & reduced and filter keep only selected ones
        if selected_only:
            m = sigman.load_data(curr_path, 's{}-selected'.format(sinfo['sid']))[0].astype(bool)
            c = sigman.load_data(curr_path, 's{}-core-events'.format(sinfo['sid']))[0][m]
        else:
            c = sigman.load_data(curr_path, 's{}-core-events'.format(sinfo['sid']))[0]

        # process events
        for k in range(len(c)):
            # time [s] -> [ms]
            c[k][:,0] = 1e3 * c[k][:,0]
            # current -> relative current
            c[k][:,1] = 1e2 * c[k][:,1] / sinfo['mI_open']
        # append data
        cores_all += list(c)

    return cores_all


def load_reduced_events(sigman, sinfo_l, selected_only=True):
    # load core events
    reduced_all = []
    for i in range(len(sinfo_l)):
        # get path within the database
        sinfo = sinfo_l[i]
        curr_path = sigman.define_db_path(sinfo)

        # load cores & reduced and filter keep only selected ones
        if selected_only:
            m = sigman.load_data(curr_path, 's{}-selected'.format(sinfo['sid']))[0].astype(bool)
            r = sigman.load_data(curr_path, 's{}-reduced-events'.format(sinfo['sid']))[0][m]
        else:
            r = sigman.load_data(curr_path, 's{}-reduced-events'.format(sinfo['sid']))[0]

        # process events
        for k in range(len(r)):
            # time [s] -> [ms]
            r[k][:,0] = 1e3 * r[k][:,0]
            # current -> relative current
            r[k][:,1] = 1e2 * r[k][:,1] / sinfo['mI_open']
        # append data
        reduced_all += list(r)

    return reduced_all

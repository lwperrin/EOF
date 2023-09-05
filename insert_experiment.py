import os
import sys
import numpy as np
import pandas as pd
from glob import glob

def extract_date(filepaths):
    #return filepaths.str.extract(r'(K[0-9]{3}[A-Z]|[A-Z]{3})').fillna('')
    return filepaths.apply(lambda x: x.split('/')[1].split('-')[0].split('_')[0])

def extract_analyte(filepaths):
    #return filepaths.str.extract(r'(K[0-9]{3}[A-Z]|[A-Z]{3})').fillna('')
    return filepaths.apply(lambda x: x.split('/')[1].split('-')[0].split('_')[1])


def extract_pore(filepaths):
    #return filepaths.str.extract(r'(K[0-9]{3}[A-Z]|[A-Z]{3})').fillna('')
    return filepaths.apply(lambda x: x.split('/')[2].split('_')[1])


def extract_condition(filepaths):
    #return filepaths.str.extract(r'/.*([A-Z0-9].*?-[A-Z0-9].*?)/').fillna('')
    #return filepaths.str.extract(r'/([A-Z0-9].*?)/').fillna('')
    #return filepaths.str.extract(r'/(.*?)/').fillna('')
    return filepaths.apply(lambda x: '-'.join(x.split('/')[1].split('-')[1:]))

def extract_temperature(filepaths):
    return filepaths.str.lower().str.extract(r'_([0-9]+)(c|degree)_')[0].fillna('')

def extract_concentration(filepaths):
    return filepaths.str.lower().str.extract(r'_([0-9]+)um').fillna('')


def extract_voltage(filepaths):
    return filepaths.str.lower().str.extract(r'_([0-9]+)mv').fillna('')


def extract_channel(filepaths):
    return filepaths.str.extract(r'CH([0-9]+)').fillna('')


def extract_id(filepaths):
    return filepaths.str.extract(r'_([0-9]{1,2})_').fillna('0')


def gen_sub_ids(duplicated_keys):
    key_count = {}
    sub_ids = np.zeros(duplicated_keys.shape, dtype=int)
    for k in range(len(duplicated_keys)):
        key = duplicated_keys[k]
        if key in key_count:
            key_count[key] += 1
            sub_ids[k] = key_count[key]
        else:
            key_count[key] = 0
            sub_ids[k] = 0

    return sub_ids


def generate_metadata(root_dir):
    # locate all abf filepath in root directory
    sstr = os.path.join(root_dir, "**", "*.abf")
    abf_filepaths = glob(sstr, recursive=True) + glob(os.path.join(root_dir, "**", "*.dat"), recursive=True)

    # create dataframe for metadata
    df_metadata = pd.DataFrame({'filepath':abf_filepaths})

    # parse names
    df_metadata['date'] = extract_date(df_metadata['filepath'])
    df_metadata['pore'] = extract_pore(df_metadata['filepath'])
    df_metadata['analyte'] = extract_analyte(df_metadata['filepath'])
    df_metadata['condition'] = extract_condition(df_metadata['filepath'])
    df_metadata['concentration'] = extract_concentration(df_metadata['filepath'])
    df_metadata['temperature'] = extract_temperature(df_metadata['filepath'])
    df_metadata['voltage'] = extract_voltage(df_metadata['filepath'])
    df_metadata['channel'] = extract_channel(df_metadata['filepath'])
    df_metadata['id'] = extract_id(df_metadata['filepath'])

    # check for duplicated key
    keys = ['date','pore','analyte', 'condition', 'concentration', 'temperature', 'voltage', 'id', 'channel']
    df_tmp_keys = df_metadata[keys].astype(str).apply(lambda x: '_'.join(x), axis=1)
    df_duplicated_keys = df_tmp_keys[df_tmp_keys.duplicated(keep=False)]

    # generate sub ids for duplicated keys
    sub_ids = gen_sub_ids(df_duplicated_keys.values)
    for i, sub_id in zip(df_duplicated_keys.index, sub_ids):
        if len(df_metadata.loc[i,'id']) > 0:
            df_metadata.loc[i,'id'] = str(df_metadata.loc[i,'id'])+'-'+str(sub_id)
        else:
            df_metadata.loc[i,'id'] = str(df_metadata.loc[i,'id'])+'-'+str(sub_id)
   
        
    return df_metadata

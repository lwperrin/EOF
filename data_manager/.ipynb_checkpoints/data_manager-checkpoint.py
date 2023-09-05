import os
import re
from glob import glob

from .iomanip import save_json, load_json, save_arr, load_arr


class DataManager:
    def __init__(self, db_root, safe=True):
        # define root path for database
        self.db_root = db_root
        self.safe = safe

        # read gstr
        with open(os.path.join(db_root, 'meta'), 'r') as fs:
            self.gstr = fs.read().rstrip('\n')

        # precompile regex
        rstr = self.gstr.replace('<', '(?P<').replace('>', '>.*)')
        self.p = re.compile(rstr)

        # extract key values
        self.keys = [g.lstrip('<').rstrip('>') for g in self.p.match(self.gstr).groups()]

    def define_db_path(self, path_info):
        # copy generator string
        path = self.gstr
        # iteratively construct path from key values
        for key in self.keys:
            path = path.replace('<{}>'.format(key), str(path_info[key]))

        return path

    def parse_path(self, path):
        m = self.p.match(path)
        vals = list(m.groups())

        vals[0] = vals[0].split('/')[-1]
        vals[-1] = vals[-1].split('/')[0]

        return {k:v for k, v in zip(self.keys, vals)}

    def insert_info(self, path_info, mod, **kwargs):
        # define path
        db_path = self.define_db_path(path_info)
        path = os.path.join(self.db_root, db_path)
        # define filepath
        info_filepath = os.path.join(path, mod+'_info.json')
        # if file already exists insert / modify info
        if os.path.exists(info_filepath):
            info_dict = load_json(info_filepath)
            for key in kwargs:
                if key in info_dict:
                    if self.safe:
                        assert False, f"Error: overwriting {key} of {db_path}"
                info_dict[key] = kwargs[key]
            # save to json file
            save_json(info_filepath, info_dict)
        else:
            # save to json file
            save_json(info_filepath, kwargs)

    def load_info(self, db_path, mod):
        # add root
        path = os.path.join(self.db_root, db_path)

        # locate signals info filepaths
        info_filepaths = glob(os.path.join(path, '**', mod+'_info.json'), recursive=True)

        # load signals info
        info_l = []
        for info_filepath in info_filepaths:
            key_dict = self.parse_path(info_filepath)
            info_dict = load_json(info_filepath)

            info_l.append({**key_dict, **info_dict})

        return info_l

    def update_info(self, path_info, mod, **kwargs):
        # define path
        db_path = self.define_db_path(path_info)
        path = os.path.join(self.db_root, db_path)
        # locate signals info filepaths
        info_filepaths = glob(os.path.join(path, '**', mod+'_info.json'), recursive=True)

        # update each info files
        for info_filepath in info_filepaths:
            info_dict = load_json(info_filepath)
            for key in kwargs:
                info_dict[key] = kwargs[key]

            save_json(info_filepath, info_dict)

    def insert_data(self, path_info, mod, data):
        # define path
        db_path = self.define_db_path(path_info)
        path = os.path.join(self.db_root, db_path)
        # define filepath
        data_filepath = os.path.join(path, mod+'_data.npy')
        if os.path.exists(data_filepath):
            if self.safe:
                print("Error: {} already exists".format(data_filepath))
                assert False
        # save arr
        save_arr(data_filepath, data)

    def load_data(self, db_path, mod):
        # add root
        path = os.path.join(self.db_root, db_path)

        # locate signals data filepaths
        data_filepaths = glob(os.path.join(path, '**', mod+'_data.npy'), recursive=True)

        # load all selected data
        data = []
        for data_filepath in data_filepaths:
            data.append(load_arr(data_filepath))

        return data

    def remove(self, db_path, sel, verbose=False):
        # check safe mode
        assert not self.safe, "ERROR: trying to remove entry in safe mode"

        # add root
        path = os.path.join(self.db_root, db_path)

        # locate filepaths
        filepaths = glob(os.path.join(path, '**', sel), recursive=True)

        # remove found files
        for filepath in filepaths:
            if verbose:
                print(f"deleting: {filepath}")

            # remove file
            os.remove(filepath)

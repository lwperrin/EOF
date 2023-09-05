# meta example: <date>-<analyte>-<buffer>/<date>_<pore>_<analyte>_<voltage>mV_<temperature>_<pore_nr>/CH<channel>/

metadata_rules = {
    'pore': [r'_(K[0-9]{3}[A-Z])_', '', str],
    'polymer_name': [r'-(.*?)(P[0-9a-z]{1,3})?-', '', str],
   # 'buffer': [r'-(.*?)(P[0-9a-z]{1,3})/','', str],
    'temperature': [r'_([0-9]+)(C|c|degree)_', -1, int],
    'voltage': [r'_([0-9]+)(mv|mV)', -1, int],
    'channel': [r'CH([0-9]+)', '', str],
    'id': [r'_([0-9a-z]{1,3})/', '', str],
    'part': [r'CH[0-9]+_([0-9]{3})', '', str],
    'date': [r'/(.*?)(P[0-9a-z]{1,3})?-','', str],
}


# meta example: <pore>_<analyte>_<voltage>mV_<temperature>C/CH<channel>_<replica>_<part>/

"""metadata_rules = {
    'pore': [r'_(K[0-9]{3}[A-Z])_', '', str],
    'analyte': [r'/(.*?)(P[0-9a-z]{1,3})?/', '', str],
    'temperature': [r'_([0-9]+)(C|c|degree)_', -1, int],
    'voltage': [r'_([0-9]+)(mv|mV)', -1, int],
    'channel': [r'CH([0-9]+)', '', str],
    'pore_nr': [r'_([0-9a-z]{1,3})/', '', str],
    'part': [r'CH[0-9]+_([0-9]{3})', '', str],
    'date': [r'/([0-9]{6})_', '', str],
    'measure': [r'(.*)', '', str],
}"""

# R21 221110 P_K238A A_MP446P1Cy5 T_25 V_130 Pq_1p0uL Aq_5uL S_190KCl1M _CH002_000.abf
"""metadata_rules = {
    'pore': [r'P_(.*?) ', '', str],
    'pore_quantity': [r'Pc_(.*?) ', '', str],
    'analyte': [r'A_(.*?) ', '', str],
    'analyte_quantity': [r'Ac_(.*?) ', '', str],
    'temperature': [r'T_(.*?) ', -1, int],
    'voltage': [r'V_(.*?) ', -1, int],
    'channel': [r'CH([0-9]+)', '', str],
    'count': [r'_CH[0-9]{3}_([0-9]{3})', '', str],
    'date': [r'/([0-9]{6})', '', str],
    'buffer': [r'B_(.*?) ', '', str],
    'buffer_quantity': [r'Bc_(.*?) ', '', str],
    'recording': [r'R([0-9]{2})', '', str],
}"""
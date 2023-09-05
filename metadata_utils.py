import re


def extract_metadata(abf_filepath, rules):
    metadata = {'filepath': abf_filepath}
    for key in rules:
        expr_str, default, vtype = rules[key]
        m = re.search(expr_str, abf_filepath)
        if m is None:
            metadata[key] = default
        else:
            metadata[key] = vtype(m[1])

    return metadata

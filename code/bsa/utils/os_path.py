"""
File path manipulation.
"""
__authors__ = "Carlos A. Silva"
__copyright__ = "Copyright 2013-2014, University of Minho"
__credits__ = ["Carlos A. Silva"]
__license__ = "Any use is forbidden"
__maintainer__ = "Carlos A. Silva"

import os, re

from glob import glob

__all__ = [
    "change_base_dir",
    "create_dir_recursively",
    "expand_path_regex",
    "get_name",
    "relative_path"
]


#
# ===== --------------------------
def relative_path(full_path, from_base_dir):
    parts_a = list(full_path.split(os.sep))
    parts_b = list(from_base_dir.split(os.sep))

    for name in parts_b:
        parts_a.remove(name)
    return os.path.join(*parts_a)


#
# ===== --------------------------
def change_base_dir(full_path, from_base_dir, to_base_dir):
    rel_path = relative_path(full_path, from_base_dir)
    return os.path.join(to_base_dir, rel_path)


#
# ===== --------------------------
def expand_path_regex(rx_path, base_dir):
    parts = rx_path.split(os.sep)
    path = base_dir
    regexp = re.compile('\*')
    for part in parts:
        if part:
            m = regexp.search(re.escape(part))
            if m is None:
                path = os.path.join(path, part)
            else:
                tp = os.path.join(path, part)
                tp = glob(tp)[0].split(os.sep)[-1]
                path = os.path.join(path, tp)
    return path


#
# ===== --------------------------
def create_dir_recursively(full_path):
    parts = full_path.split(os.sep)

    # -- create project base directory
    path = os.sep
    for name in parts:
        path = os.path.join(path, name)
        if not os.path.exists(path):
            os.mkdir(path)


#
# ===== --------------------------
def get_name(full_path):
    return full_path.split(os.sep)[-1]

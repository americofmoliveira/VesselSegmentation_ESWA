__authors__ = "Carlos A. Silva"
__copyright__ = "Copyright 2013-2014, University of Minho"
__credits__ = ["Carlos A. Silva"]
__license__ = "Any use is forbidden"
__maintainer__ = "Carlos A. Silva"

from glob import glob
from .utils import expand_path_regex, relative_path

import os.path

__all__ = [
    "FilePathResource",
]


#
# ===== --------------------------
class FilePathResource(object):
    def __init__(self, name_functor, base_dir, sub_dir, sub_dir_functor,
                 regex_name, regex_subdir, valid_extensions=None):
        self._sub_dir = sub_dir
        self.name_functor = name_functor
        self.name = None
        self.base_dir = base_dir
        self.sub_dir_functor = sub_dir_functor
        self.regex_name = regex_name
        self.regex_subdir = regex_subdir
        self._extensions = valid_extensions

    @staticmethod
    def create(name_functor=None, base_dir=None, sub_dir=None,
               sub_dir_functor=None, regex_name=False, regex_subdir=False,
               valid_extensions=None):
        return FilePathResource(name_functor, base_dir, sub_dir,
                                sub_dir_functor, regex_name, regex_subdir,
                                valid_extensions)

    def fname(self, **pars):
        def glob_name(f_name):
            path = os.sep.join((self.base_dir, self.sub_dir(**pars), f_name))
            path = glob(path)
            if len(path) > 0:
                path = path[0]
                name = os.path.split(path)[-1]
            else:
                name = None
            return name

        if self.name is None:
            if self.regex_name:
                if self._extensions:
                    for ext in self._extensions:
                        name = glob_name('*.{0}'.format(ext))
                        if name is not None:
                            break
                else:
                    name = glob_name('*')
            else:
                name = self.name_functor(**pars)
        else:
            name = self.name
        return name

    def sub_dir(self, **pars):
        if self._sub_dir is None:
            name_dir = self.sub_dir_functor(**pars)
            if self.regex_subdir:
                name_dir = expand_path_regex(name_dir, self.base_dir)
                name_dir = relative_path(name_dir, self.base_dir)
        else:
            name_dir = self._sub_dir
        return name_dir

    def path(self, **pars):
        if self.sub_dir_functor is None:
            path = os.path.join(self.base_dir, self._sub_dir)
        else:
            path = os.path.join(self.base_dir, self.sub_dir(**pars))
        return path

    def full_path(self, **pars):
        path = os.path.join(self.base_dir, self.sub_dir(**pars),
                            self.fname(**pars))

        return path

    # = ------------------------------
    def mk_dir(self, **pars):
        parts = self.path(**pars).split(os.sep)

        # -- create project base directory
        path = os.sep
        for name in parts:
            path = os.path.join(path, name)
            if not os.path.exists(path):
                os.mkdir(path)

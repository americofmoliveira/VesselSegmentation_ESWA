__authors__ = "Carlos A. Silva"
__copyright__ = "Copyright 2013-2014, University of Minho"
__credits__ = ["Carlos A. Silva"]
__license__ = "Any use is forbidden"
__maintainer__ = "Carlos A. Silva"


#
# ===== ------------------------------
class ParameterInputMixin(object):
    def _validate_input(self, kwargs):
        if kwargs:
            raise ValueError('Undefined parameter \'{0}\''.format(kwargs))

    def _get_param(self, kwargs, name, optional=False, value=None):
        try:
            if optional:
                return kwargs.pop(name, value)
            return kwargs.pop(name)
        except Exception:
            raise ValueError('Input parameter not defined \'{0}\''.
                             format(name))


#
# ===== ------------------------------
class DataFileIterator(ParameterInputMixin):
    def __init__(self, obj, rtype, itype, cases, **kwargs):
        """
        Creates an instance of class DataFileIterator.
        """
        if rtype == 'image':
            resource = obj._image_resource
        elif rtype == 'gt':
            gt_resource = self._get_param(kwargs, 'gt_resource', optional=True)
            if gt_resource:
                resource = gt_resource
            else:
                resource = obj._gt_resource
        elif rtype == 'mask':
            mask_resource = self._get_param(kwargs, 'mask_resource',
                                            optional=True)
            if mask_resource:
                resource = mask_resource
            else:
                resource = obj._mask_resource
        else:
            raise ValueError("Unknown value for 'rtype' (=\'{0}\') in "
                             "'MRIDataFileIterator'".format(rtype))
        if itype == 'full_path':
            self._path = resource.full_path
        elif itype == 'path':
            self._path = resource.path
        elif itype == 'file_name':
            self._path = resource.fname
        else:
            raise ValueError("Unknown value for 'itype' (=\'{0}\') in "
                             "'MRIDataFileIterator'".format(itype))
        self._full_name = resource.full_path
        self._name = resource.fname
        self._current_idx = 0
        self._cases = cases

        self._idx = 0
        self._obj = obj

        self._kwargs = kwargs

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def begin(self):
        self._idx, self._current_idx = 0, 0

    def name(self):
        return self._name(dataset=self._obj.dataset,
                          case=self._cases[self._current_idx], **self._kwrds)

    def full_path(self):
        return self._full_name(dataset=self._obj.dataset,
                               case=self._cases[self._current_idx],
                               **self._kwrds)

    def full_path_for(self, case):
        return self._full_name(dataset=self._obj.dataset, case=case,
                               **self._kwrds)

    def full_path_mask_for_idx(self, idx):
        return self._obj._mask_resource.path(dataset=self._obj.dataset,
                                             case=self._cases[idx],
                                             **self._kwrds)

    def case_for_idx(self, idx):
        return self._cases[idx]


#
# ===== ------------------------------
class DataBase(ParameterInputMixin):
    def __init__(self, base_dir, dataset):
        self._base_dir = base_dir
        self._dataset = dataset
        self._db_name = None

    @property
    def base_dir(self):
        return self._base_dir

    @property
    def dataset(self):
        return self._dataset

    @property
    def db_name(self):
        return self._db_name

    def list_cases(self, **kwargs):
        raise NotImplementedError

    def list_datasets(self):
        raise NotImplementedError

    def list_groups(self):
        return ('All',)

    def full_path(self, **kwargs):
        raise NotImplementedError

    def path(self, **kwargs):
        raise NotImplementedError


#
# ===== ------------------------------
class MRIDataBase(DataBase):
    def __init__(self, base_dir, dataset):
        super().__init__(base_dir, dataset)

    def list_sequences(self):
        raise NotImplementedError

"""
Access to DRIVE files.
"""
__authors__ = "Carlos A. Silva"
__copyright__ = "Copyright 2015, University of Minho"
__credits__ = ["Carlos A. Silva"]
__license__ = "Any use is forbidden"
__maintainer__ = "Carlos A. Silva"

from functools import partial

import os

from .data_files import DataBase
from .file_resource import FilePathResource

__all__ = [
    "STAREFiles",
]


#
# ===== ------------------------------
class STAREFiles(DataBase):
    def __init__(self, base_dir, dataset=None):
        super().__init__(base_dir, 'Training'.lower())
        self._db_name = 'STARE'

        # -- C R E A T I N G   I M A G E   R E S O U R C E
        # -- sub-directory name functor.
        def sub_dir(**kwargs):
            return 'images'

        # -- file name functor.
        def name_functor(**kwargs):
            return 'im{code:04d}.ppm.gz'.format(code=int(kwargs['case']))

        # -- Create file resource.
        self._image_resource = \
            FilePathResource.create(base_dir=base_dir,
                                    name_functor=name_functor,
                                    sub_dir_functor=partial(sub_dir))

        # -- C R E A T I N G   G T   R E S O U R C E
        # -- sub-directory name functor.
        def gt_sub_dir(**kwargs):
            gt_type = kwargs.pop('gt_type', 'gt1st')
            if gt_type == 'gt2nd':
                return 'gt/labels-vk'
            return 'gt/labels-ah'

        # -- file name functor.
        def gt_name_functor(**kwargs):
            gt_type = kwargs.pop('gt_type', 'gt1st')
            if gt_type == 'gt2nd':
                return 'im{case:04d}.vk.ppm.gz'. \
                    format(case=int(kwargs['case']))
            return 'im{case:04d}.ah.ppm.gz'.format(case=int(kwargs['case']))

        # -- Create file resource.
        self._gt_resource = \
            FilePathResource.create(base_dir=base_dir,
                                    name_functor=gt_name_functor,
                                    sub_dir_functor=gt_sub_dir)

        # -- C R E A T I N G   O R I G I N A L   M A S K   R E S O U R C E
        # -- sub-directory name functor.
        def m_sub_dir(**kwargs):
            return 'masks'

        # -- file name functor.
        def m_name_functor(**kwargs):
            return 'im{case:04d}-msf.png'.format(case=int(kwargs['case']))

        # -- Create file resource.
        self._omask_resource = \
            FilePathResource.create(base_dir=base_dir,
                                    name_functor=m_name_functor,
                                    sub_dir_functor=m_sub_dir)

    @staticmethod
    def create(dbase_dir, dataset=None):
        return STAREFiles(dbase_dir, dataset=dataset)

    def list_datasets(self):
        return ['Training']

    def list_cases(self, **kwargs):
        cases = []
        r = [1, 2, 3, 4, 5, 44, 77, 81, 82, 139, 162, 163, 235, 236, 239, 240,
             255, 291, 319, 324]
        for n in r:
            cases.append('{0:04d}'.format(n))

        return cases

    def full_path(self, **kwargs):
        resource, case = self._get_resource(kwargs)
        return resource.full_path(case=case, **kwargs)

    def path(self, **kwargs):
        resource, case = self._get_resource(kwargs)
        return resource.path(dataset=self._dataset, case=case, **kwargs)

    def name(self, **kwargs):
        return os.path.basename(self.full_path(**kwargs))

    def _get_resource(self, kwargs):
        rtype = self._get_param(kwargs, 'rtype')
        case = self._get_param(kwargs, 'case')
        gt_type = self._get_param(kwargs, 'gt_type', optional=True,
                                  value='gt1st')
        self._validate_input(kwargs)

        if rtype == 'image':
            resource = self._image_resource
        elif rtype == 'gt':
            resource = self._gt_resource
            kwargs['gt_type'] = gt_type
        elif rtype == 'mask':
            resource = self._omask_resource
        else:
            raise ValueError("Unknown type of resource: "
                             "'{0}'".format(rtype))
        return resource, case

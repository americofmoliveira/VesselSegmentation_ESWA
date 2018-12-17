__authors__ = "Carlos A. Silva"
__copyright__ = "Copyright 2015-2018, University of Minho"
__credits__ = ["Carlos A. Silva"]
__license__ = "Any use is forbidden"
__maintainer__ = "Carlos A. Silva"

from functools import partial

import os

from .data_files import DataBase

from .file_resource import FilePathResource

__all__ = [
    "DRIVEFiles",
]


#
# ===== ------------------------------
class DRIVEFiles(DataBase):
    def __init__(self, dataset, base_dir):
        super().__init__(base_dir, dataset.lower())
        self._db_name = 'DRIVE'

        # -- C R E A T I N G   I M A G E   R E S O U R C E
        # -- sub-directory name functor.
        def sub_dir(**kwargs):
            return '{grp}/images'.format(grp=kwargs['dataset'])

        # -- file name functor.
        def name_functor(**kwargs):
            return '{code:02d}_{grp}.tif'.format(code=int(kwargs['case']),
                                                 grp=kwargs['dataset'])

        # -- Create file resource.
        self._image_resource = \
            FilePathResource.create(base_dir=base_dir,
                                    name_functor=name_functor,
                                    sub_dir_functor=partial(sub_dir,
                                                            dataset=dataset))

        # -- C R E A T I N G   G T   R E S O U R C E
        # -- sub-directory name functor.
        def gt_sub_dir(**kwargs):
            gt_type = kwargs.pop('gt_type', 'gt1st')
            if gt_type == 'gt2nd':
                return '{grp}/2nd_manual'.format(grp=kwargs['dataset'])
            return '{grp}/1st_manual'.format(grp=kwargs['dataset'])

        # -- file name functor.
        def gt_name_functor(**kwargs):
            gt_type = kwargs.pop('gt_type', 'gt1st')
            if gt_type == 'gt2nd':
                return '{case:02d}_manual2.gif'. \
                    format(case=int(kwargs['case']))
            return '{case:02d}_manual1.gif'.format(case=int(kwargs['case']))

        # -- Create file resource.
        self._gt_resource = \
            FilePathResource.create(base_dir=base_dir,
                                    name_functor=gt_name_functor,
                                    sub_dir_functor=gt_sub_dir)

        # -- C R E A T I N G   O R I G I N A L   M A S K   R E S O U R C E
        # -- sub-directory name functor.
        def m_sub_dir(**kwargs):
            return '{grp}/mask'.format(grp=kwargs['dataset'])

        # -- file name functor.
        def m_name_functor(**kwargs):
            return '{case:02d}_{grp}_mask.gif'.format(case=int(kwargs['case']),
                                                      grp=kwargs['dataset'])

        # -- Create file resource.
        self._omask_resource = \
            FilePathResource.create(base_dir=base_dir,
                                    name_functor=m_name_functor,
                                    sub_dir_functor=m_sub_dir)

        # -- C R E A T I N G   N E W   M A S K   R E S O U R C E
        # -- sub-directory name functor.
        def new_m_sub_dir(**kwargs):
            return '{grp}/new_mask'.format(grp=kwargs['dataset'])

        # -- Create file resource.
        self._new_mask_resource = \
            FilePathResource.create(base_dir=base_dir,
                                    name_functor=m_name_functor,
                                    sub_dir_functor=new_m_sub_dir)

    @staticmethod
    def create(dataset, base_dir):
        return DRIVEFiles(dataset, base_dir)

    def list_datasets(self):
        return ['Training', 'Test']

    def list_cases(self, **kwargs):
        dataset = self._get_param(kwargs, 'dataset', optional=True,
                                  value=self._dataset)
        group = self._get_param(kwargs, 'group', optional=True, value='All')
        if group != 'All':
            raise ValueError('Unknown group \'{0}\''.format(group))
        self._validate_input(kwargs)

        if dataset.lower() == 'test':
            r = list(range(1, 21))
        elif dataset.lower() == 'training':
            r = list(range(21, 41))
        else:
            raise ValueError('Undefined dataset \'{0}\''.format(dataset))
        cases = []
        for n in r:
            cases.append('{0:02d}'.format(n))

        return cases

    def full_path(self, **kwargs):
        resource, case = self._get_resource(kwargs)
        return resource.full_path(dataset=self.dataset, case=case, **kwargs)

    def path(self, **kwargs):
        resource, case = self._get_resource(kwargs)
        return resource.path(dataset=self._dataset, case=case, **kwargs)

    def name(self, **kwargs):
        return os.path.basename(self.full_path(**kwargs))

    def _get_resource(self, kwargs):
        rtype = self._get_param(kwargs, 'rtype')
        case = self._get_param(kwargs, 'case')
        use_new_mask = self._get_param(kwargs, 'use_new_mask', optional=True)
        gt_type = self._get_param(kwargs, 'gt_type', optional=True,
                                  value='gt1st')
        self._validate_input(kwargs)

        if rtype == 'image':
            resource = self._image_resource
        elif rtype == 'gt':
            resource = self._gt_resource
            kwargs['gt_type'] = gt_type
        elif rtype == 'mask' and not use_new_mask:
            resource = self._omask_resource
        elif rtype == 'mask' and use_new_mask:
            resource = self._new_mask_resource
        else:
            raise ValueError("Unknown type of resource: "
                             "'{0}'".format(rtype))
        return resource, case

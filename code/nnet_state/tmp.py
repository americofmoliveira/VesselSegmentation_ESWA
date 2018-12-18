#!/usr/bin/env python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path

from bsa.prediction import predict
from bsa.cases import list_cases
from bsa.util import load

import argparse
import os


#
# ===== ------------------------------
def configure_device(device):
    import os
    import json

    def backend():
        fpath = os.path.join(os.environ['HOME'], '.keras', 'keras.json')
        if not os.path.isfile(fpath):
            raise RuntimeError('No keras environment.')
        handle = open(fpath)
        keras_def = json.load(handle)
        return keras_def['backend']

    if backend() == 'tensorflow':
        c = {'gpu0': '0', 'gpu1': '1', 'gpu2': '2', 'gpu3': '3', 'cpu': '0'}
        if device == 'cpu' or device is None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = c[device]
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        print('Using device: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    else:
        c = {'gpu0': 'cuda0', 'gpu1': 'cuda1', 'gpu2': 'cuda2', 'gpu3': 'cuda3', 'cpu': 'cpu'}
        env = 'device={},floatX=float32,gpuarray.preallocate=0.9'.format(c[device])
        os.environ['THEANO_FLAGS'] = env


#
# ===== ------------------------------
def validate_dir_structure(dataset):
    path = Path(os.path.join(os.getcwd(), 'datasets'))
    if not path.exists():
        print('The image datasets (Drive, Stare, chase) should be in a subdirectory \'datasets\'')
        raise RuntimeError

    path = Path(os.path.join(os.getcwd(), 'datasets', dataset.upper()))
    if not path.exists():
        print('The image datasets (Drive, Stare, chase) should be in a subdirectory \'datasets\'')
        raise RuntimeError

    path = Path(os.path.join(os.getcwd(), 'segs'))
    if not path.exists():
        path.mkdir()

    path = Path(os.path.join(os.getcwd(), 'segs', dataset))
    if not path.exists():
        path.mkdir()


#
# ===== ------------------------------
def load_cases(dataset, fold):
    cases, db = list_cases(dataset, fold)
    images, masks, img_filenames = [], [], []
    for case in cases:
        fname = db.full_path(rtype='image', case=case)
        images.append(load(fname))
        img_filenames.append(os.path.basename(fname))

        fname = db.full_path(rtype='mask', case=case)
        masks.append(load(fname))

    return images, masks, img_filenames


#
# ===== ------------------------------
def nnet_state_filename(dataset, fold):
    fname = '.'.join(('-'.join(('nnet-state', dataset, str(fold))), 'h5'))
    fname = os.path.join(os.getcwd(), 'nnet_state', fname)
    if not os.path.exists(fname):
        print('Error: No state file found -', fname)
        raise RuntimeError

    return fname


#
# ===== ------------------------------
def load_calibrations(dataset):
    f_name = os.path.join(os.getcwd(), 'calibrations', '-'.join((dataset, 'cal.npy')))
    return load(f_name)


#
# ===== ------------------------------
def main(args):
    dataset = args.dataset.upper()
    validate_dir_structure(dataset)
    configure_device(args.device)

    patch_shape = (98, 98)
    inner_patch_shape = (32, 32)
    images, masks, img_filenames = load_cases(dataset, args.fold)
    calibrations = load_calibrations(dataset)
    state_file = nnet_state_filename(dataset, args.fold)

    predict(state_file, dataset, img_filenames, images, masks, patch_shape,
            inner_patch_shape, calibrations)


#
# ===== ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command-line interface to '
                                     'nnet.')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['gpu0', 'gpu1', 'gpu2', 'gpu3'], help='Device selection.')
    parser.add_argument('--dataset', type=str, help='Dataset name', default='drive',
                        choices=['drive', 'chase', 'stare'])
    parser.add_argument('--fold', type=int, help='Fold number', default=1)

    args = parser.parse_args()
    main(args)

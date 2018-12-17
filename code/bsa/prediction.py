from keras.models import load_model
from scipy.misc import imsave
from sklearn.externals.joblib import dump

from .unet import get_nnet
from .batch import MiniBatchPredict2D
from .transform_image import AutoPadUnpad2D, Sampler, Wavelet2D

import numpy as np
import os
import sys


#
# ===== ------------------------------
def predict(state_file, dataset, img_filenames, images, masks, patch_shape,
            inner_patch_shape, calibrations):
    batch_size = 4
    nnet = load_module(state_file, 2, patch_shape, inner_patch_shape)
    batch_provider = MiniBatchPredict2D(batch_size, patch_shape, inner_patch_shape)
    batch_provider.set_calibration(calibrations[1], calibrations[2])
    sampler = Sampler(patch_shape, inner_patch_shape)
    img_transformation = AutoPadUnpad2D(patch_shape, inner_patch_shape)
    wavelet_transform = Wavelet2D(mode='features', wavelet='haar', channels=['r'])
    w = np.prod(inner_patch_shape)
    buffer = np.zeros((batch_provider.tot_samples_in_container, w, 2), dtype=np.float32)

    for image, mask, f_name in zip(images, masks, img_filenames):
        image, mask = img_transformation.pad(image, mask)
        image, mask = wavelet_transform(image, mask)

        batch_provider.set_image(sampler, image, mask)
        data = []

        for containers, n_batches in batch_provider:
            for n in range(n_batches):
                ni = n * batch_size
                nf = (n + 1) * batch_size

                buffer[ni:nf, :] = \
                    nnet.predict_on_batch(containers[ni:nf])

            if isinstance(data, list):
                data = buffer.copy()
            else:
                data = np.vstack([data, buffer])

        data = batch_provider.post_patch_transform(data)

        seg, probs = sampler.rebuild(data)
        seg, probs = img_transformation.unpad(seg, probs)
        save_data(dataset, f_name, seg, probs)
        print('.', end='')
        sys.stdout.flush()
    print()


#
# ===== ------------------------------
def load_module(state_file, nb_labels, patch_shape, inner_patch_shape):
    nnet = get_nnet(nb_labels, patch_shape, inner_patch_shape)

    if hasattr(nnet, 'load_model'):
        nnet = nnet.load_model(str(state_file))
    else:
        nnet = load_model(str(state_file))

    return nnet


#
# ===== ------------------------------
def save_data(dataset, f_name, seg, probs):
    f_name = f_name.split('.')[0]

    # -- save as png
    seg_name = '.'.join(('-'.join((f_name, 'seg')), 'png'))
    seg_name = os.path.join(os.getcwd(), 'segs', dataset, seg_name)
    imsave(seg_name, seg)

    # -- save as npy
    seg_name = '.'.join(('-'.join((f_name, 'seg')), 'npy'))
    seg_name = os.path.join(os.getcwd(), 'segs', dataset, seg_name)
    dump(seg, seg_name, compress=3)

    # -- save as png
    prob_name = '.'.join(('-'.join((f_name, 'prob')), 'png'))
    prob_name = os.path.join(os.getcwd(), 'segs', dataset, prob_name)
    imsave(prob_name, probs[-1])

    # -- save as npy
    prob_name = '.'.join(('-'.join((f_name, 'prob')), 'npy'))
    prob_name = os.path.join(os.getcwd(), 'segs', dataset, prob_name)
    dump(probs, prob_name, compress=3)

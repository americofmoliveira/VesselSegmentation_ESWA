from .patch import Patch
from .exceptions import StopPatchIteration
from .transform_patch import BlockRotateTransform

import numpy as np


#
# ===== ------------------------------
class MiniBatchPredict2D(object):
    def __init__(self, batch_size, patch_shape, inner_patch_shape,
                 container_size=80):
        super().__init__()
        self._container_size = container_size
        self._batch_size = batch_size
        self._nb_samples_in_container = self._container_size * self._batch_size
        self._patch_shape = patch_shape
        self._inner_patch_shape = inner_patch_shape
        self._transform = BlockRotateTransform(n_directions=3, inner_patch_shape=inner_patch_shape)
        self._mean = np.array([0.0], dtype=np.float)
        self._std = np.array([1.0], dtype=np.float)

        self._patch_provider = Patch(patch_shape, inner_patch_shape)
        self._nb_output_channels = self._patch_provider.nb_input_channels

        w, h = self._patch_shape
        self._shape = (self._nb_samples_in_container, w, h, self._nb_output_channels)

    def _init(self):
        self._signal_end = False

    @property
    def tot_samples_in_container(self):
        return self._nb_samples_in_container

    def set_calibration(self, data_mean, data_std):
        if isinstance(data_mean, (list, np.ndarray)):
            self._mean = data_mean
        else:
            self._mean = np.array([data_mean], dtype=np.float)

        if isinstance(data_std, (list, np.ndarray)):
            self._std = data_std
        else:
            self._std = np.array([data_std], dtype=np.float)

    def set_image(self, sampler, image, mask):
        self._patch_provider.set_image(sampler, image, mask)

    def post_patch_transform(self, data):
        return self._transform.post(data)

    def __iter__(self):
        self._init()
        return self

    def __next__(self):
        if self._signal_end:
            raise StopIteration

        sample_container = np.zeros(self._shape, dtype=np.float32)
        count, n_batches = 0, 0

        try:
            nb_channels = self._nb_output_channels
            while count < self._nb_samples_in_container:
                patch, masks = self._patch_provider.get()
              
                patches, masks = self._transform(patch, masks)
                
                for patch, mask in zip(patches, masks):
                    for ch in range(nb_channels):
                        sample_container[count, :, :, ch] = \
                            ((patch[ch, :, :] - self._mean[ch]) / self._std[ch])
                    count += 1
                
        except StopPatchIteration:
            if not count:
                
                raise StopIteration
            else:
                
                self._signal_end = True

        if count % self._batch_size:
            n_batches = count // self._batch_size + 1
            end = n_batches * self._batch_size
            sample_container[count:end, :, :, :] = \
                sample_container[count - 1, :, :, :]
        else:
            if count == self._nb_samples_in_container:
                n_batches = self._container_size
            else:
                n_batches = count // self._batch_size

        return sample_container, n_batches

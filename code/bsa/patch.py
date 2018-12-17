from .exceptions import StopPatchIteration

import numpy as np


#
# ===== ------------------------------
class Patch(object):
    def __init__(self, patch_shape, inner_patch_shape):
        super().__init__()
        self._patch_shape = patch_shape
        self._inner_patch_shape = inner_patch_shape
        self._batch_size = None
        self._nb_input_channels = 4

        self._init_patch_offset()

    @property
    def nb_input_channels(self):
        return self._nb_input_channels

    @nb_input_channels.setter
    def nb_input_channels(self, value):
        self._nb_input_channels = value

    @property
    def total_nb_patches(self):
        return self._total_nb_patches

    def _get_block_patch_shape(self):
        return (self.nb_input_channels, self._patch_shape[0],
                self._patch_shape[1])

    def _get_block_mask_shape(self):
        return (self._patch_shape[0], self._patch_shape[1])

    def _init_patch_offset(self):
        wx, wy = self._patch_shape
        self._wx = wx // 2
        self._wy = wy // 2

        # -- The patches may have sides that are even or odd. We have to treat
        #    these two cases differently.
        self._ox = 0
        if wx > 2 * self._wx:
            # -- It is odd.
            self._ox = 1

        self._oy = 0
        if wy > 2 * self._wy:
            # -- It is odd.
            self._oy = 1

    def set_image(self, sampler, image, mask):
        self._seqs = image
        self._mask = mask
        self._shape = self._get_block_patch_shape()
        self._patch = np.zeros(self._shape)
        self._patch_mask = np.zeros(self._get_block_mask_shape())

        self._sample_idx = 0
        self._nb_patches = 0

        self._coords, self._samples_in_img = sampler.coordinates(self._mask)
        self._total_nb_patches = self._samples_in_img
        assert self._samples_in_img is not None

    def _read_patch(self):
        ii, jj = self._coords[self._sample_idx].tolist()

        # -- Extract image patch and labels.
        for n, seq_img in enumerate(self._seqs):
            self._patch[n, :, :] = \
                seq_img[ii - self._wx:ii + self._wx + self._ox, jj -
                        self._wy:jj + self._wy + self._oy].copy()

        self._patch_mask[:, :] = self._mask[ii - self._wx:ii + self._wx + self._ox, jj -
                                            self._wy:jj + self._wy + self._oy].copy()

        return self._patch, self._patch_mask

    def get(self):
        # -- Have we read all patches from all images ???
        if self._total_nb_patches:
            if self._nb_patches >= self._total_nb_patches:
                raise StopPatchIteration

        # -- Extract a patch (or patches from the sequences).
        patch, mask = self._read_patch()

        self._samples_in_img -= 1
        self._sample_idx += 1
        self._nb_patches += 1

        return patch, mask

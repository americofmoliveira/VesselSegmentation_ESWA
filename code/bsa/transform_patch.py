from .util import array_tf_0, array_tf_90, array_tf_180, array_tf_270

import numpy as np


#
# ===== ------------------------------
class BlockRotateTransform(object):
    def __init__(self, n_directions, inner_patch_shape):
        self._n_directions = n_directions
        self._rotate = [array_tf_90, array_tf_180, array_tf_270]
        self._post_rotate = [array_tf_0, array_tf_270, array_tf_180, array_tf_90]
        self._inner_ps = inner_patch_shape
        self._size_reduction = 4
        self._n_labels = 2

    def __call__(self, patch, mask):
        if type(patch) not in (list, tuple):
            patch = [patch]
        if type(mask) not in (list, tuple):
            mask = [mask]

        t_patch = []
        t_mask = []
        for p, m in zip(patch, mask):
            t_patch.append(p)
            t_mask.append(m)
            for n in range(self._n_directions):
                t_patch.append(self._rotate[n](p))
                t_mask.append(self._rotate[n](m))

        return t_patch, t_mask

    @property
    def size_reduction(self):
        return self._size_reduction

    def post(self, patch):
        n_patches, n_pixels, n_labels = patch.shape
        self._i_shape = (n_patches, self._inner_ps[0], self._inner_ps[1], n_labels)
        self._new_shape = (n_patches // self._size_reduction,
                           self._inner_ps[0], self._inner_ps[1], n_labels)
        self._o_shape = (n_patches // self._size_reduction,
                         self._inner_ps[0] * self._inner_ps[1], n_labels)
        self._reduced_n_patches = n_patches // self._size_reduction

        shape = (self._size_reduction, self._inner_ps[0], self._inner_ps[1], n_labels)
        self._patches = np.zeros(shape)

        patch = np.reshape(patch, self._i_shape)
        new_patch = np.zeros(self._new_shape)

        for n in range(self._reduced_n_patches):
            ni = n * self.size_reduction
            nf = (n + 1) * self.size_reduction

            # -- Rotate patches to the original direction.
            for j, nn in enumerate(range(ni, nf)):
                for i in range(self._n_labels):
                    self._patches[j, :, :, i] = \
                        self._post_rotate[j](patch[nn, :, :, i])

            # -- Find the mean prob for each label.
            for i in range(self._n_labels):
                new_patch[n, :, :, i] = \
                    self._patches[:, :, :, i].mean(axis=0)

        new_patch = np.reshape(new_patch, self._o_shape)

        return new_patch

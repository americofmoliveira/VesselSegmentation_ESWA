from collections import OrderedDict

import math
import numpy as np
import pywt


#
# ===== ------------------------------
class AutoPadUnpad2D(object):
    def __init__(self, patch_shape, inner_patch_shape):
        self._data = []
        self._patch_shape = patch_shape
        self._inner_patch_shape = inner_patch_shape

    def _find_2D_bound_box(self, mask):
        i_min = mask.shape[0] - 1
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] > 0:
                    if i_min > i:
                        i_min = i

        i_max = 0
        for i in range(mask.shape[0] - 1, -1, -1):
            for j in range(mask.shape[1]):
                if mask[i, j] > 0:
                    if i_max < i:
                        i_max = i

        j_min = mask.shape[1] - 1
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] > 0:
                    if j_min > j:
                        j_min = j

        j_max = 0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1] - 1, -1, -1):
                if mask[i, j] > 0:
                    if j_max < j:
                        j_max = j

        return i_min, i_max, j_min, j_max

    def _find_new_2D_image_shape(self, mask, bbox):
        margin = []
        for ps, pi in zip(self._patch_shape, self._inner_patch_shape):
            margin.append(int(math.ceil(0.5 * (ps - pi))))

        s0, s1 = mask.shape

        m00 = bbox[0] if bbox[0] > margin[0] else margin[0]
        m01 = s0 - bbox[1]
        m01 = m01 if m01 > margin[0] else margin[0]

        m10 = bbox[2] if bbox[2] > margin[1] else margin[1]
        m11 = s1 - bbox[3]
        m11 = m01 if m01 > margin[0] else margin[0]

        w0 = bbox[1] - bbox[0] + 1
        w1 = bbox[3] - bbox[2] + 1

        bbox = [m00, m00 + w0, m10, m10 + w1]
        ns0, ns1 = m00 + w0 + m01, m10 + w1 + m11

        side = ns0 - 2 * margin[0]
        r_box = side / float(self._inner_patch_shape[0])
        n_box = int(r_box)
        if math.fabs(n_box - r_box) > 0:
            n_box = int(math.ceil(r_box))
        ns0 = 2 * margin[0] + n_box * self._inner_patch_shape[0]

        side = ns1 - 2 * margin[1]
        r_box = side / float(self._inner_patch_shape[1])
        n_box = int(r_box)
        if math.fabs(n_box - r_box) > 0:
            n_box = int(math.ceil(r_box))
        ns1 = 2 * margin[1] + n_box * self._inner_patch_shape[1]

        self._new_shape = ns0, ns1
        self._old_shape = mask.shape

        return bbox

    def _compute_transform(self, mask):
        o_bbox = self._find_2D_bound_box(mask)

        n_bbox = self._find_new_2D_image_shape(mask, o_bbox)

        w0 = o_bbox[1] - o_bbox[0] + 1
        w1 = o_bbox[3] - o_bbox[2] + 1

        self.ni0, self.ni1 = n_bbox[0], n_bbox[0] + w0
        self.nj0, self.nj1 = n_bbox[2], n_bbox[2] + w1

        self.oi0, self.oi1 = o_bbox[0], o_bbox[0] + w0
        self.oj0, self.oj1 = o_bbox[2], o_bbox[2] + w1

    def _pad_transform(self, array):
        new_array = np.zeros(self._new_shape, dtype=array.dtype)
        new_array[self.ni0:self.ni1, self.nj0:self.nj1] = \
            array[self.oi0:self.oi1, self.oj0:self.oj1]
        return new_array

    def _set_unpadding_state(self):
        self._data = []
        self._data.append((self._new_shape, self._old_shape))
        self._data.append((self.ni0, self.ni1))
        self._data.append((self.nj0, self.nj1))
        self._data.append((self.oi0, self.oi1))
        self._data.append((self.oj0, self.oj1))

    def pad(self, images, mask):
        self._compute_transform(mask)
        self._set_unpadding_state()

        if type(images) not in (list, tuple):
            images = [images]
        t_images = []
        for img in images:
            e = self._pad_transform(img)
            t_images.append(e)

        e_mask = mask if mask is None else self._pad_transform(mask)
        return t_images, e_mask

    def _unpad_transform(self, array):
        old_array = np.zeros(self._old_shape, dtype=array.dtype)
        old_array[self.oi0:self.oi1, self.oj0:self.oj1] = \
            array[self.ni0:self.ni1, self.nj0:self.nj1]
        return old_array

    def _get_unpadding_state(self):
        self._new_shape, self._old_shape = self._data[0]
        self.ni0, self.ni1 = self._data[1]
        self.nj0, self.nj1 = self._data[2]
        self.oi0, self.oi1 = self._data[3]
        self.oj0, self.oj1 = self._data[4]

    def unpad(self, segs, prob):
        self._get_unpadding_state()

        segs = self._unpad_transform(segs)

        e_prob = []
        if prob is not None:
            for n in range(prob.shape[0]):
                e_prob.append(self._unpad_transform(prob[n, :, :]))

        return segs, e_prob


#
# ===== ------------------------------
def extract_all_patches(mask, patch_shape, inner_patch_shape):
    height, width = patch_shape
    inner_height, inner_width = inner_patch_shape

    margin_h = (height - inner_height) // 2
    margin_w = (width - inner_width) // 2

    inner_margin_h = inner_height // 2
    inner_margin_w = inner_width // 2

    img_h, img_w = mask.shape
    nb_patches_h = (img_h - 2 * margin_h) // inner_height
    nb_patches_w = (img_w - 2 * margin_w) // inner_width

    coordinates = np.empty((nb_patches_h * nb_patches_w, 2), dtype=np.int)

    iter_tot = 0
    for i in range(nb_patches_h):
        for j in range(nb_patches_w):
            ii = margin_h + inner_margin_h + i * inner_height
            jj = margin_w + inner_margin_w + j * inner_width
            if np.count_nonzero(mask[ii - inner_margin_h:
                                     ii + inner_margin_h,
                                     jj - inner_margin_w:
                                     jj + inner_margin_w]) > 0:
                coordinates[iter_tot, 0] = ii
                coordinates[iter_tot, 1] = jj
                iter_tot += 1

    return coordinates[0: iter_tot, :]


#
# ===== ------------------------------
class Sampler(object):

    def __init__(self, patch_shape, inner_patch_shape):
        self._patch_shape = patch_shape
        self._inner_patch_shape = inner_patch_shape

        self._patch_h, self._patch_w = \
            self._patch_shape[0] // 2, self._patch_shape[1] // 2

        self._inner_patch_h, self._inner_patch_w = \
            self._inner_patch_shape[0] // 2, self._inner_patch_shape[1] // 2

    def coordinates(self, mask):
        self._mask = mask
        self._coordinates = extract_all_patches(mask, self._patch_shape,
                                                self._inner_patch_shape)
        self._image_shape = mask.shape

        return self._coordinates, self._coordinates.shape[0]

    def _rebuild_image(self, patches_prediction, coordinates,
                       wanted_min_label=0):
        ii, jj = coordinates[0], coordinates[1]
        h, w = self._inner_patch_h, self._inner_patch_w

        for i in range(0, patches_prediction.shape[0]):
            self._predicted_prob_images[i, ii - h: ii + h,
                                        jj - w: jj + w] = \
                patches_prediction[i, :, :]

        pred = patches_prediction.argmax(axis=0)
        self._predicted_label_image[ii - h: ii + h, jj - w: jj + w] = \
            pred + wanted_min_label

    def rebuild(self, classification_vector, wanted_min_label=0):
        '''
        classification_vector: matriz n_pixels X n_classes
        coordinates: array [n_pixels, x, y]
        '''
        classification_vector = classification_vector[
            0: self._coordinates.shape[0], :, :]
        c_shape = classification_vector.shape
        c = np.zeros((c_shape[0] * c_shape[1], c_shape[2]))
        idx = 0

        for i in range(c_shape[0]):
            for j in range(c_shape[1]):
                for k in range(c_shape[2]):
                    c[idx, k] = classification_vector[i, j, k]
                idx += 1

        classification_vector = c

        prob_image_shape = np.insert(self._image_shape,
                                     0, classification_vector.shape[1])
        self._predicted_prob_images = np.zeros(prob_image_shape)
        self._predicted_label_image = np.zeros(self._image_shape)

        # We get an array with shape [n_classes, patch_dim1, patch_dim2]
        patch_probs_shape = np.insert(self._inner_patch_shape, 0,
                                      classification_vector.shape[1])
        patch_probs = np.zeros(patch_probs_shape)

        n_patch_elements = np.prod(self._inner_patch_shape)

        for i_coor, i_patch in \
                enumerate(range(0, classification_vector.shape[0],
                                n_patch_elements)):

            for j in range(0, classification_vector.shape[1]):
                tmp = classification_vector[
                    i_patch: i_patch + n_patch_elements, j]
                tmp = tmp.reshape(self._inner_patch_shape)
                patch_probs[j, :, :] = tmp

            patch_coordinates = self._coordinates[i_coor, :]
            self._rebuild_image(patches_prediction=patch_probs,
                                coordinates=patch_coordinates,
                                wanted_min_label=wanted_min_label)

        return self._predicted_label_image, self._predicted_prob_images


#
# ===== ------------------------------
class Wavelet2D(object):
    def __init__(self, mode, wavelet, channels):
        self._channels = channels
        self._wavelet = wavelet
        self._mode = mode

    def _compute_shape(self, image):
        is_padded = False
        h, w = image.shape
        nh = 4 * (h // 4)
        if h != nh:
            is_padded = True
            nh = 4 * (h // 4 + 1)

        nw = 4 * (w // 4)
        if w != nw:
            is_padded = True
            nw = 4 * (w // 4 + 1)

        if nh > nw:
            nw = nh
        else:
            nh = nw

        self._orig_shape = image.shape
        self._new_shape = (nh, nw)

        self._offset = (nh - h) // 2, (nw - w) // 2

        return is_padded

    def _pad(self, image):
        self._is_padded = self._compute_shape(image)
        if not self._is_padded:
            return image

        h, w = self._orig_shape
        new_image = np.zeros(self._new_shape)
        oh, ow = self._offset
        new_image[oh:oh + h, ow:ow + w] = image

        return new_image

    def _unpad(self, image):
        if not self._is_padded:
            return image

        h, w = self._orig_shape
        new_image = np.zeros(self._orig_shape)
        oh, ow = self._offset
        new_image[:, :] = image[oh:oh + h, ow:ow + w]

        return new_image

    def _transform(self, image, channel=['r']):
        new_image = self._pad(image)
        wave = pywt.Wavelet(self._wavelet)
        cA_1, (cH_1, cV_1, cD_1) = pywt.swt2(new_image, wave, level=1,
                                             start_level=1)[0]
        if self._mode == 'rebuild':
            cA = np.zeros_like(cA_1)
            coeffs = ((cA, (cH_1, cV_1, cD_1)),)
            reb_image = pywt.iswt2(coeffs, wave)
            channels = (channel, 'high_pass')
            return channels, (image, self._unpad(reb_image))
        elif self._mode == 'features':
            channels = (channel, 'cH', 'cV', 'cD')
            return channels, (image, self._unpad(cH_1), self._unpad(cV_1),
                              self._unpad(cD_1))
        elif self._mode == 'features-all':
            channels = (channel, 'cA', 'cH', 'cV', 'cD')
            return channels, (image, self._unpad(cA_1), self._unpad(cH_1),
                              self._unpad(cV_1), self._unpad(cD_1))
        elif self._mode == 'features-only':
            channels = ('cA', 'cH', 'cV', 'cD')
            return channels, (self._unpad(cA_1), self._unpad(cH_1),
                              self._unpad(cV_1), self._unpad(cD_1))

    def transformed_channels(self, channels):
        c = list(channels)
        if self._mode == 'rebuild':
            return c + ['high_pass']
        elif self._mode == 'features':
            return c + ['cH', 'cV', 'cD']
        elif self._mode == 'features-all':
            return c + ['cA', 'cH', 'cV', 'cD']
        elif self._mode == 'features-only':
            return ['cA', 'cH', 'cV', 'cD']

    def __call__(self, images, mask):
        if isinstance(images, (dict, OrderedDict)):
            t_images = OrderedDict()
            for channel in images:
                img = images[channel]
                if channel in self._channels:
                    channel_names, imgs = self._transform(img, channel)
                    for name, img in zip(channel_names, imgs):
                        t_images[name] = img
        else:
            if type(images) not in (list, tuple):
                images = [images]
            t_images = []
            for img in images:
                _, e = self._transform(img)
                for img in e:
                    t_images.append(img)
            # assert False

        return t_images, mask

from PIL import Image

import gzip
import scipy.ndimage, scipy.misc, joblib

import numpy as np


#
# ===== ------------------------------
def array_tf_0(arr):
    return arr


#
# ===== ------------------------------
def array_tf_90(arr):
    axes_order = list(range(arr.ndim - 2)) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + \
        [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)


#
# ===== ------------------------------
def array_tf_180(arr):
    slices = [slice(None) for _ in range(arr.ndim - 2)] + \
        [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)]


#
# ===== ------------------------------
def array_tf_270(arr):
    axes_order = list(range(arr.ndim - 2)) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + \
        [slice(None, None, -1), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)


#
# ===== ------------------------------
def load(fname, as_npy=True, channel=1):
    """
    Load image file. Analyse filename extension to load the image. Current
    supported types are {'gif', 'nii', 'nii.gz', 'npy', 'png', 'tif'}

    Parameters
    ----------
    fname: python string.
        full filename.
    as_npy: bool.
        Convert the object to a numpy array.
    channel: int
        Primary use in color images.
    normalize: Bool
        Normalize by the maximum value in the array.

    Returns
    -------
    image: numpy array.
    """
    split = fname.split('.')
    ext = split[-1]
    ext2 = '.'.join((split[-2], ext))

    if ext in {'gif', 'png', 'tif', 'ppm', 'jpg'}:
        data = scipy.ndimage.imread(fname)
    elif ext2 in {'ppm.gz', 'gif.gz', 'png.gz', 'tif.gz'}:
        gz_file = gzip.open(fname, 'rb')
        if ext2 == 'ppm.gz':
            data = np.asarray(Image.open(gz_file))
        else:
            data = scipy.ndimage.imread(gz_file)
    elif ext in {'npy', 'pkl'}:
        data = joblib.load(fname)
    else:
        raise ValueError("Unsupported image file extension '{ext}'"
                         .format(ext=ext))
    if isinstance(data, (np.ndarray, np.generic)) and len(data.shape) == 3:
        if channel < data.shape[2]:
            data = data[:, :, channel]
        else:
            raise ValueError("Channel should be only used for color images")

    return data

"""
Global initialization of io package.
"""
__authors__ = "Carlos A. Silva"
__copyright__ = "Copyright 2013-2014, University of Minho"
__credits__ = ["Carlos A. Silva"]
__license__ = "Any use is forbidden"
__maintainer__ = "Carlos A. Silva"

#
# ===== --------------------------
import gzip
import nibabel, scipy.ndimage, scipy.misc, sklearn.externals.joblib
import numpy as np

__all__ = [
    "load",
    "dump",
]


#
# === ---------------------------------
def load(fname, as_npy=True, affine=False, channel=None, normalize=False):
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

    if ext in {'gif', 'png', 'tif', 'ppm'}:
        data = scipy.ndimage.imread(fname)
    elif ext2 in {'ppm.gz', 'gif.gz', 'png.gz', 'tif.gz'}:
        gz_file = gzip.open(fname, 'rb')
        data = scipy.ndimage.imread(gz_file)
    elif ext == 'nii' or ext2 == 'nii.gz':
        data = nibabel.load(fname)
        if affine:
            affine_info = data.get_affine()
        if as_npy:
            data = np.asarray(data.get_data(), dtype=np.float32, order='C')
    elif ext in {'npy', 'pkl'}:
        data = sklearn.externals.joblib.load(fname)
    else:
        raise ValueError("Unsupported image file extension '{ext}'"
                         .format(ext=ext))
    if channel is not None:
        if len(data.shape) == 3:
            data = data[:, :, channel]
        else:
            raise ValueError("Channel should be only used for color images")

    if normalize:
        data = data / data.max()

    if affine:
        return data, affine_info
    return data


#
# === ---------------------------------
def dump(obj, full_name, affine=None, compress=3):
    """
    Load image file. Analyse filename extension to load the image. Current
    supported types are {'gif', 'nii', 'nii.gz', 'npy', 'png', 'tif'}

    Parameters
    ----------
    fname: python string.
        full filename.
    convert_to_numpy: bool.
        Convert the object to a numpy array.
    channel: int
        Primary use in color images.

    Returns
    -------
    image: numpy array.
    """
    ext = full_name.split('.')[-1]

    if ext in {'gif', 'png', 'tif', 'ppm'}:
        scipy.misc.imsave(full_name, obj)
    elif ext in {'nii', 'gz'}:
        if affine is None:
            nibabel.save(nibabel.Nifti1Image(obj), full_name)
        else:
            nibabel.save(nibabel.Nifti1Image(obj, affine), full_name)
    elif ext in {'npy', 'pkl'}:
        sklearn.externals.joblib.dump(obj, full_name, compress=compress)
    else:
        raise ValueError("Unsupported image file extension '{ext}'"
                         .format(ext=ext))

import numpy as np
import tensorflow as tf
from skimage.transform import rescale


def getfun_match_scaling_and_normalize(src_spacing, dst_spacing):
    """
    Constructs the pre-processing function to rescale the image such that
    the image spacing matches the model spacing. In a second step the data
    is normalized to have zero mean and a standard deviation of one.
    Finally a singleton channel dimension is appended.

    The preprocessing function returns:
    tuple(img, file, img_shape)

    Parameters:
        src_spacing: tuple
            The spacing of the input images
        dst_spacing: tuple
            The spacing of the model

    Returns:
        match_scaling_and_normalize: function
            The pre-processing function
    """
    scale = np.array(src_spacing) / np.array(dst_spacing)

    def match_scaling_and_normalize(img, file, convert_shape = True):
        if convert_shape:
            img_shape = tf.convert_to_tensor(img.shape.as_list())
        else:
            img_shape = img.shape
        img = rescale(img,
                       scale=scale,
                       anti_aliasing=True,
                       order=1)
        img = (img - img.mean())/img.std()
        return img[..., None], file, img_shape

    return match_scaling_and_normalize

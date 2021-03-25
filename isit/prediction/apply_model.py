import numpy as np
import tensorflow as tf

from improc.resample import match_spacing
from skimage.transform import resize
from improc.label import smooth_label_edges

from ..luigi_utils import thread_lock_wrapper


def apply_instance_seg_model(inference_model, img, img_spacing, lock=None):

    model_spacing = inference_model.signatures['get_model_spacing'](
    )['spacing'].numpy()
    predict_fn = inference_model.signatures['pred_instance']

    if lock is not None:
        predict_fn = thread_lock_wrapper(predict_fn, lock)

    if (model_spacing != img_spacing).any():
        img_shape = img.shape
        img = match_spacing(img,
                            img_spacing,
                            model_spacing,
                            image_type='greyscale')

    pred = predict_fn(tf.convert_to_tensor(img))['labels'].numpy()

    if (model_spacing != img_spacing).any():

        smooth_sigma = np.max(
            np.array(img_shape) / np.array(pred.shape)) * np.sqrt(2)
        pred = resize(pred.astype(np.float32),
                      output_shape=img_shape,
                      order=0,
                      preserve_range=True).astype(np.uint16)
        pred = smooth_label_edges(pred, smooth_sigma)

    return pred.astype(np.uint16)

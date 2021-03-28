import numpy as np
import tensorflow as tf

from improc.resample import match_spacing
from skimage.transform import resize
from improc.label import smooth_label_edges

from dlutils.postprocessing.voting import embeddings_to_labels
from dlutils.layers.semi_conv import generate_coordinate_grid
from dlutils.preprocessing.normalization import standardize

from ..luigi_utils import thread_lock_wrapper
from .tiling import TileGenerator


def apply_instance_seg_model(inference_model, img, img_spacing, lock=None):

    model_spacing = inference_model.signatures['get_config'](
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


def tiled_apply_instance_seg_model(inference_model,
                                   img,
                                   img_spacing,
                                   patch_size,
                                   border,
                                   gpu_postproc=False,
                                   lock=None):

    config = inference_model.signatures['get_config']()
    model_spacing = config['spacing'].numpy()
    peak_min_distance = config['peak_min_distance'].numpy()
    model_fn = inference_model.signatures['model_fun']

    if lock is not None:
        model_fn = thread_lock_wrapper(model_fn, lock)

    if (model_spacing != img_spacing).any():
        img_shape = img.shape
        img = match_spacing(img,
                            img_spacing,
                            model_spacing,
                            image_type='greyscale')

    img = img.astype(np.float32)
    img = standardize(img)[..., None]
    tile_generator = TileGenerator(img.shape,
                                   patch_size=patch_size,
                                   border=border)

    fg_mask = np.zeros(img.shape[:-1] + (1, ), dtype=bool)
    vfield = np.zeros(img.shape[:-1] + (img.ndim - 1, ), dtype=np.float32)
    img = np.pad(img, pad_width=tile_generator.pad_width, mode='symmetric')

    for loc, padded_loc, tile_loc in tile_generator:
        tile_img = img[padded_loc]

        nopostproc = model_fn(tf.convert_to_tensor(tile_img[None]))
        tile_embeddings = nopostproc['embeddings'][0]
        tile_fg_mask = nopostproc['classes'][0, ..., 1:] > 0.5

        # subtrack the tile coordinates (and add stitch image coordiantes later)
        tile_coords = generate_coordinate_grid(
            (1, ) + tile_embeddings.shape,
            tile_embeddings.ndim - 1) * model_spacing
        tile_vfield = tile_embeddings - tile_coords.numpy()

        fg_mask[loc] = tile_fg_mask[tile_loc]
        vfield[loc[:-1]] = tile_vfield[tile_loc[:-1]]

    img_coords = generate_coordinate_grid(
        (1, ) + vfield.shape, vfield.ndim - 1) * model_spacing
    embeddings = vfield + img_coords
    fg_mask = fg_mask[..., 0]

    device = '/GPU' if gpu_postproc else '/CPU'
    with tf.device(device):
        pred = embeddings_to_labels(embeddings,
                                    fg_mask,
                                    peak_min_distance=peak_min_distance,
                                    spacing=model_spacing,
                                    min_count=2).numpy()

    if (model_spacing != img_spacing).any():
        smooth_sigma = np.max(
            np.array(img_shape) / np.array(pred.shape)) * np.sqrt(2)
        pred = resize(pred.astype(np.float32),
                      output_shape=img_shape,
                      order=0,
                      preserve_range=True).astype(np.uint16)
        pred = smooth_label_edges(pred, smooth_sigma)

    return pred.astype(np.uint16)

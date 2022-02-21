import numpy as np
import tensorflow as tf

from improc.resample import match_spacing
from skimage.transform import resize
from improc.label import smooth_label_edges

from dlutils.postprocessing.voting import embeddings_to_labels
from dlutils.layers.semi_conv import generate_coordinate_grid
from dlutils.preprocessing.normalization import standardize

from .monkeypatch import remove_shm_from_resource_tracker
from ..luigi_utils import thread_lock_wrapper
from .tiling import TileGenerator

from multiprocessing.shared_memory import SharedMemory


def allocate_shared_memory(img_shape, labels, out_file):
    shm = SharedMemory(create=True, size=labels.nbytes)
    shm_labels = np.ndarray(labels.shape, dtype=labels.dtype, buffer=shm.buf)
    shm_labels[:] = labels[:]
    result = tuple([shm.name, labels.shape, labels.dtype, out_file.numpy().decode("utf-8"), img_shape.numpy()])
    shm.close()
    return result


def dynamically_apply_instance_seg_model(model_fun,
                                         img,
                                         out_file,
                                         img_shape,
                                         patch_size,
                                         border,
                                         peak_min_distance,
                                         model_spacing):
    """
    Dynamically apply the model to the whole or tiled image based on `patch_size`.
    If the image exceeds 'patch_size' and is predicted in a tiled manner, the post-processing
    i.e. the label computation from the embeddings is performed on CPU.

    Parameters:
        model_fun: function
            :func:'~isit.training.training.InstanceModelExportTask.model_fun'
        img: tf.Tensor(tf.float32)
            Input image
        out_file: String,
            Output file path
        img_shape: tuple
            The original (not rescale) image shape without channel dimension
        patch_size: tuple
            Maximum patch size that fits on the GPU.
        border: tuple
            Padding used in tiled prediction.
        peak_min_distance: int
            minimum distance between instance centers
        model_spacing: tuple
            Voxel size on which this model was trained.

    Returns:
        labels: np.array
            The predicted instance segmentations
        path: String
            Path of the input file
        img_shape: tuple
            The original image shape
    """
    remove_shm_from_resource_tracker()
    if patch_size is None or all([s <= p for s, p in zip(img.shape, patch_size)]):
        prediction = model_fun(img[None])
        embeddings = prediction['embeddings'][0]
        fg_mask = prediction['classes'][0, ..., 1] > 0.5

        labels = embeddings_to_labels(embeddings,
                                      fg_mask,
                                      peak_min_distance=peak_min_distance,
                                      spacing=model_spacing,
                                      min_count=2)

        labels = tf.cast(labels, tf.uint16).cpu().numpy()

        return allocate_shared_memory(img_shape, labels, out_file)
    else:
        tile_generator = TileGenerator(tuple(img.shape),
                                       patch_size=patch_size,
                                       border=border)

        fg_mask = np.zeros(img.shape[:-1] + (1,), dtype=bool)
        vfield = np.zeros(img.shape[:-1] + (img.ndim - 1,), dtype=np.float32)
        img = tf.pad(img, paddings=tf.convert_to_tensor(np.array(tile_generator.pad_width)), mode='CONSTANT')

        for loc, padded_loc, tile_loc in tile_generator:
            tile_img = img[padded_loc]

            prediction = model_fun(tile_img[None])
            tile_embeddings = prediction['embeddings'][0]
            tile_fg_mask = prediction['classes'][0, ..., 1:] > 0.5

            # subtrack the tile coordinates (and add stitch image coordiantes later)
            tile_coords = generate_coordinate_grid(
                (1,) + tile_embeddings.shape,
                tile_embeddings.ndim - 1) * model_spacing
            tile_vfield = tile_embeddings - tile_coords

            fg_mask[loc] = tile_fg_mask[tile_loc]
            vfield[loc[:-1]] = tile_vfield[tile_loc[:-1]]

        img_coords = generate_coordinate_grid(
            (1,) + vfield.shape, vfield.ndim - 1) * model_spacing
        embeddings = vfield + img_coords

        with tf.device('/CPU'):
            labels = embeddings_to_labels(embeddings,
                                          fg_mask[..., 0],
                                          peak_min_distance=peak_min_distance,
                                          spacing=model_spacing,
                                          min_count=2)

        labels = tf.cast(labels, tf.uint16).cpu().numpy()

        return allocate_shared_memory(img_shape, labels, out_file)


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

    fg_mask = np.zeros(img.shape[:-1] + (1,), dtype=bool)
    vfield = np.zeros(img.shape[:-1] + (img.ndim - 1,), dtype=np.float32)
    img = np.pad(img, pad_width=tile_generator.pad_width, mode='symmetric')

    for loc, padded_loc, tile_loc in tile_generator:
        tile_img = img[padded_loc]

        nopostproc = model_fn(tf.convert_to_tensor(tile_img[None]))
        tile_embeddings = nopostproc['embeddings'][0]
        tile_fg_mask = nopostproc['classes'][0, ..., 1:] > 0.5

        # subtrack the tile coordinates (and add stitch image coordiantes later)
        tile_coords = generate_coordinate_grid(
            (1,) + tile_embeddings.shape,
            tile_embeddings.ndim - 1) * model_spacing
        tile_vfield = tile_embeddings - tile_coords.numpy()

        fg_mask[loc] = tile_fg_mask[tile_loc]
        vfield[loc[:-1]] = tile_vfield[tile_loc[:-1]]

    img_coords = generate_coordinate_grid(
        (1,) + vfield.shape, vfield.ndim - 1) * model_spacing
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

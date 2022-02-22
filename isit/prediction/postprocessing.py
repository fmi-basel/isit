import os
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from skimage.transform import resize

from improc.label import smooth_label_edges
from improc.morphology import clean_up_labels

from skimage.io import imsave

from .monkeypatch import remove_shm_from_resource_tracker


def queue_worker(queue, fun):
    remove_shm_from_resource_tracker()
    element = queue.get()
    while not element is None:
        fun(element)
        element = queue.get()


def postprocess(labeling, img_shape, labels_fill_holes, labels_size_threshold, labels_keep_largest,
                labels_sequential, image_spacing):
    smooth_sigma = np.max(np.array(img_shape) / np.array(labeling.shape)) * np.sqrt(2)
    labeling = resize(labeling.astype(np.float32),
                      output_shape=img_shape,
                      order=0,
                      preserve_range=True).astype(np.uint16)
    labeling = smooth_label_edges(labeling, smooth_sigma)
    labeling = labeling.astype(np.uint16)

    labeling = clean_up_labels(labeling,
                               fill_holes=labels_fill_holes,
                               radius=None,
                               size_threshold=labels_size_threshold,
                               keep_largest=labels_keep_largest,
                               sequential=labels_sequential,
                               spacing=image_spacing)
    return labeling


def get_postprocess_and_save_fun(labels_fill_holes, labels_size_threshold,
                                 labels_keep_largest, labels_sequential,
                                 image_spacing):
    def post_process_and_save(element):
        shm_name, shape, dtype, outpath, img_shape = element

        shm = SharedMemory(shm_name)
        labeling = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        labeling = postprocess(labeling, img_shape, labels_fill_holes,
                               labels_size_threshold, labels_keep_largest,
                               labels_sequential, image_spacing)

        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        imsave(outpath, labeling.astype(np.uint16), compress=9, check_contrast=False)

        shm.close()
        shm.unlink()

    return post_process_and_save

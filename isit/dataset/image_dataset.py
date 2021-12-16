import tensorflow as tf
from tensorflow.data import Dataset

from tifffile import imread
import numpy as np


class ImageDataset(Dataset):
    """
    Loades images from a list of files.

    The generator yields tuples of image and out_file.
    """

    def _img_reader(in_files, out_files, start, step):
        in_files_ = in_files[start::step]
        out_files_ = out_files[start::step]
        for in_file, out_file in zip(in_files_, out_files_):
            img = imread(in_file.decode("utf-8")).astype(np.float32)
            yield (img, out_file)

    def __new__(cls, files, out_shape=(None, None, None), start=0, step=1):
        """
        Create a new Image Dataset which loads images from `files` and
        yields tuple(image, file).

        Parameters:
            files: List
                List of file paths.
            out_shape: tuple
                Shape of the images.
            start: int
                Index from which to start reading the files list.
            step: int
                Steps size used to walk the files list.
        """
        return Dataset.from_generator(
            cls._img_reader,
            output_shapes=(out_shape, ()),
            output_types=(tf.float32, tf.string),
            args=(files, start, step)
        )


class ParallelImageDatasetFactory():
    """
    Factory to create interleaved, prefetched tensorflow datasets.
    A pre-processing function which is applied to each loaded image can be provided.

    We use the `tf.data.experimental.AUTOTUNE` option for the interleave and prefetch
    calls, which allows tensorflow to optimize parallelization.

    Parameters:
        pre_proc_fun: Function
            Applied to each loaded image.
        num_interleaved_ds: int
            Number of interleaved datasets.
            Note: 1 is usually sufficient.

    """

    def _create(files, pre_proc_fun=None, num_interleaved_ds=1):
        if pre_proc_fun:
            return tf.data.Dataset.range(num_interleaved_ds) \
                .interleave(
                lambda i: ImageDataset(files, start=i, step=num_interleaved_ds),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            ) \
                .map(pre_proc_fun) \
                .prefetch(tf.data.experimental.AUTOTUNE)
        else:
            return tf.data.Dataset.range(num_interleaved_ds) \
                .interleave(
                lambda i: ImageDataset(files, start=i, step=num_interleaved_ds),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            ) \
                .prefetch(tf.data.experimental.AUTOTUNE)

    def __new__(cls, files, pre_proc_fun=None, num_interleaved_ds=1):
        return cls._create(files, pre_proc_fun, num_interleaved_ds)

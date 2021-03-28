import luigi
import logging
import numpy as np
import os
import threading
from luigi.util import requires
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from skimage.io import imread, imsave
from improc.morphology import clean_up_labels

from ..data_collection import ParseCollectionTask, CollectionTarget
from ..luigi_utils import GPUTask, task_to_hash, monitor_futures


class LabelCleaningParams(luigi.Config):

    # yapf: disable
    labels_fill_holes = luigi.BoolParameter(False, description='whether to fill holes in labels. (done per slice in 3D case)', parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    labels_size_threshold = luigi.IntParameter(None, description="minimum label area in px")
    labels_keep_largest = luigi.BoolParameter(False, description='when true, in case of disjoint regions with the same label, only keeps the largest one', parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    labels_sequential = luigi.BoolParameter(True, description='Rename labels sequentialy', parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    # yapf: enable


class TilingParams(luigi.Config):

    # yapf: disable
    tiling_patch_size = luigi.TupleParameter(None, description='patch size for tile processing. Use if processing the full image does not fit on the GPU')
    tiling_border = luigi.TupleParameter(None, description='border around each patch when using tile processing. set to ~object size')
    gpu_postproc = luigi.BoolParameter(False, description='whether to perform postprocessing on gpu (on full image)', parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    # yapf: enable


@requires(ParseCollectionTask)
class SegmentInstancesTask(TilingParams, LabelCleaningParams, GPUTask):

    # yapf: disable
    model_path = luigi.Parameter(description="path to inference model")
    image_spacing = luigi.TupleParameter(description='pixel/voxel size of the input image')
    input_layer = luigi.Parameter(description='name of input sub-directory/layer, "layer" field must exist in the input collection')
    output_layer = luigi.Parameter(description='name of output sub-directory/layer, "layer" field must exist in the input collection')
    input_filters = luigi.parameter.OptionalParameter(None, description="""pandas query string specifying elements to process.e.g. 'org_id==3 & plate_row in ["B","D"]' """)
    n_threads = luigi.IntParameter(4, description='max number of threads for pre/post processing')
    # yapf: enable

    @property
    def resources(self):
        # gpu bound, estimated effective load
        resources = super().resources
        resources.update({'pool_workers': self.n_threads})

        if not self.gpu_postproc and self.tiling_patch_size is not None and self.tiling_border is not None:
            # tensorflow nearest neighbhor (embedding clustering) uses all available cores whe on CPU...
            resources.update({'pool_workers': multiprocessing.cpu_count()})

        return resources

    def output(self):
        if self.outdir is None:
            self.outdir = self.datadir

        return CollectionTarget(
            self.outdir, 'instance_seg_out_collection_' + task_to_hash(self))

    def build_output_collection(self, df):
        df_out = df.query('layer=="{}"'.format(
            self.input_layer)).copy().reset_index()
        df_out['layer'] = self.output_layer
        return df_out

    def _predict_single_image(self, in_path, out_path, model, lock):
        from .apply_model import apply_instance_seg_model
        from .apply_model import tiled_apply_instance_seg_model

        logger = logging.getLogger('luigi-interface')
        logger.info('instance segmentation of: {}'.format(in_path))

        img = imread(in_path)
        if len(self.image_spacing) == 2 and img.ndim == 3:
            # assume yokogawa extra channel
            img = img[0]

        if self.tiling_patch_size is None or all(
            [s <= p for s, p in zip(img.shape, self.tiling_patch_size)]):
            pred = apply_instance_seg_model(model, img, self.image_spacing,
                                            lock)
        else:
            if self.tiling_border is None:
                raise ValueError(
                    'tiling_border option must be set. Alternatively set tiling_patch_size to None to disable tile processing'
                )

            pred = tiled_apply_instance_seg_model(model, img,
                                                  self.image_spacing,
                                                  self.tiling_patch_size,
                                                  self.tiling_border,
                                                  self.gpu_postproc, lock)

        pred = clean_up_labels(pred,
                               fill_holes=self.labels_fill_holes,
                               radius=None,
                               size_threshold=self.labels_size_threshold,
                               keep_largest=self.labels_keep_largest,
                               sequential=self.labels_sequential,
                               spacing=self.image_spacing)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        imsave(out_path,
               pred.astype(np.uint16),
               compress=9,
               check_contrast=False)

    def gpu_run(self):
        logger = logging.getLogger('luigi-interface')

        import tensorflow as tf
        model = tf.saved_model.load(self.model_path)
        lock = threading.Lock()

        df = self.input().load()
        if self.input_filters is not None:
            df = df.query(self.input_filters)
        df.sort_index(inplace=True)

        df_in = df.query('layer=="{}"'.format(self.input_layer))
        df_out = self.build_output_collection(df)

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for in_path, out_path in zip(df_in.dc.path, df_out.dc.path):

                if os.path.isfile(out_path):
                    logger.info(
                        'File(s) already exist, skipping: {}'.format(out_path))
                    continue

                futures.append(
                    executor.submit(self._predict_single_image, in_path,
                                    out_path, model, lock))

            monitor_futures(futures)

        # double check that individuals output files have been written
        missing_output_files = [
            p for p in df_out.dc.path if not os.path.isfile(p)
        ]
        if len(missing_output_files) > 0:
            raise ValueError('Some files were not processed: {}'.format(
                missing_output_files))

        self.output().save(df_out, self.param_kwargs)


class BatchedSegmentInstancesTask(SegmentInstancesTask):
    '''Segments planes of 3D stacks as independent 2D images'''
    def _predict_single_image(self, in_path, out_path, model, lock):
        from .apply_model import apply_instance_seg_model

        logger = logging.getLogger('luigi-interface')
        logger.info('instance segmentation of: {}'.format(in_path))

        imgs = imread(in_path)
        preds = []
        last_highest_label = 0
        for img in imgs:
            pred = apply_instance_seg_model(model, img, self.image_spacing,
                                            lock)
            pred = clean_up_labels(pred,
                                   fill_holes=self.labels_fill_holes,
                                   radius=None,
                                   size_threshold=self.labels_size_threshold,
                                   keep_largest=self.labels_keep_largest,
                                   sequential=self.labels_sequential,
                                   spacing=self.image_spacing)

            pred[pred > 0] += last_highest_label
            last_highest_label = pred.max()
            preds.append(pred)

        preds = np.stack(preds)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        imsave(out_path,
               preds.astype(np.uint16),
               compress=9,
               check_contrast=False)


def main():
    luigi.run(main_task_cls=SegmentInstancesTask, local_scheduler=True)


if __name__ == "__main__":
    main()

import luigi
import logging
import abc
import os
import random
import numpy as np
from luigi.util import requires

from improc.resample import match_spacing
from improc.roi import crop_object

from ..luigi_utils import stable_hash
from ..data_collection import ParseConfigCollectionTask


@requires(ParseConfigCollectionTask)
class BuildTrainingRecordBaseTask(luigi.Task):
    '''Base task to build train/valid/test records
    '''

    # yapf: disable
    sample_group_keys = luigi.ListParameter(['fname'], description="pandas grouping keys to get one sample per group from the input collection")
    input_filters = luigi.parameter.OptionalParameter(None, description="""pandas query string specifying elements to process.e.g. 'mag in [20]'""")
    raw_filter = luigi.Parameter('layer=="raw"', description="pandas query string to get the input image")
    annot_filter = luigi.Parameter('layer=="annot"', description="pandas query string to get the annot image")
    min_patch_size = luigi.TupleParameter(description='minimum patch size after resampling to model_spacing. Pads with reflection if smaller')
    patch_margins = luigi.TupleParameter(description='cropping margins around labeled region')
    model_spacing = luigi.TupleParameter(description='tuple indicating the pixel/voxel size on which the model is trained')
    training_dir = luigi.Parameter()
    record_name = luigi.Parameter()

    train_fraction = luigi.FloatParameter(default=0.9)
    valid_fraction = luigi.FloatParameter(default=0.1)
    seed = luigi.IntParameter()
    # yapf: enable

    def output(self):
        targets = {}
        for split_name in ['train', 'valid', 'test']:
            outpath = os.path.join(
                self.training_dir,
                '{}_{}.tfrec'.format(self.record_name, split_name))
            targets[split_name] = luigi.LocalTarget(outpath)

        return targets

    @abc.abstractmethod
    def _data_gen(self, inputs):
        '''generator to build the training record'''
        pass

    @abc.abstractmethod
    def _get_serialization_fun(self):
        '''Returns a record parser serialization function'''
        pass

    def _split_data(self):
        '''split annot/raw input pairs into train|valid|test sets'''

        if not 0. <= self.train_fraction <= 1.:
            raise ValueError(
                'training fraction should be within [0.,1.]. got {}'.format(
                    self.train_fraction))
        if not 0. <= self.valid_fraction <= 1.:
            raise ValueError(
                'training fraction should be within [0.,1.]. got {}'.format(
                    self.valid_fraction))

        df = self.input().load()

        if self.input_filters is not None:
            df = df.query(self.input_filters)

        # randomize sample order, e.g mixes timepoints
        record_input = list(df.groupby(list(self.sample_group_keys)))
        random.seed(self.seed)
        random.shuffle(record_input)

        test_fraction = max(0., 1. - self.train_fraction - self.valid_fraction)
        n_samples = len(record_input)
        probs = [self.train_fraction, self.valid_fraction, test_fraction]
        split = np.random.choice([0, 1, 2],
                                 size=n_samples,
                                 replace=True,
                                 p=probs)
        sets = {}
        for idx, split_name in enumerate(['train', 'valid', 'test']):
            inputs = [p for s, p in zip(split, record_input) if s == idx]
            sets[split_name] = inputs

        return sets

    def run(self):
        from dlutils.dataset.tfrecords import tfrecord_from_iterable

        np.random.seed(self.seed)

        for split_name, inputs in self._split_data().items():

            with self.output()[split_name].temporary_path(
            ) as temp_output_path:
                tfrecord_from_iterable(temp_output_path,
                                       self._data_gen(inputs),
                                       self._get_serialization_fun(),
                                       verbose=False)


class BuildInstanceRecordTask(BuildTrainingRecordBaseTask):
    def _data_gen(self, inputs):
        '''generator to build the training record'''

        from dlutils.preprocessing.normalization import standardize

        logger = logging.getLogger('luigi-interface')

        for idx, subdf in inputs:

            raw_row = subdf.query(self.raw_filter)
            annot_row = subdf.query(self.annot_filter)

            if not (len(raw_row) == 1 and len(annot_row) == 1):
                continue

            logger.info(
                'adding raw|annot pair to training record: {} - {} to {}'.
                format(raw_row.dc.path[0], annot_row.dc.path[0],
                       self.record_name))

            raw = raw_row.dc.read()[0]
            annot = annot_row.dc.read()[0]

            sample_spacing = tuple(raw_row.iloc[0].spacing)
            raw = match_spacing(raw,
                                sample_spacing,
                                self.model_spacing,
                                image_type='greyscale')
            annot = match_spacing(annot,
                                  sample_spacing,
                                  self.model_spacing,
                                  image_type='label_nearest')

            #normalize input image
            raw = standardize(raw)

            # tighter crop based on annotations/pad to minsize
            crop_mask = annot > 0
            if crop_mask.max() < 1:
                crop_mask[:] = True

            raw = crop_object([raw],
                              crop_mask,
                              min_shape=self.min_patch_size,
                              margins=self.patch_margins,
                              mode='reflect')[0]

            annot = crop_object([annot],
                                crop_mask,
                                min_shape=self.min_patch_size,
                                margins=self.patch_margins,
                                constant_values=-1)[0]

            raw = raw.astype(np.float32)
            annot = annot.astype(np.int16)

            # add channel dim
            yield raw[..., None], annot[..., None]

    def _get_serialization_fun(self):
        '''Returns a record parser serialization function'''
        import tensorflow as tf
        from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser

        return ImageToSegmentationRecordParser(
            tf.float32, tf.int16, fixed_ndim=len(self.model_spacing)).serialize


@requires(ParseConfigCollectionTask)
class MultiInstanceRecordTask(luigi.Task):
    '''~wrapper task to split a training data collection into multiple training records'''

    multi_records_group_keys = luigi.ListParameter(
        None,
        description=
        "pandas grouping keys to get subframe per training record from the input collection"
    )
    input_filters = luigi.parameter.OptionalParameter(
        None,
        description=
        """pandas query string specifying elements to process.e.g. 'mag in [20]'"""
    )

    def _get_record_tasks(self):
        record_tasks = []
        if self.multi_records_group_keys is None:
            record_tasks.append(
                BuildInstanceRecordTask(seed=57,
                                        record_name='complete_record'))

        else:
            df = self.input().load()
            if self.input_filters is not None:
                df = df.query(self.input_filters)

            group_ids = [
                g[0] for g in df.groupby(list(self.multi_records_group_keys))
            ]
            if len(self.multi_records_group_keys) == 1:
                group_ids = [(g, ) for g in group_ids]

            for group_id in group_ids:
                group_query = []
                for key, val in zip(self.multi_records_group_keys, group_id):
                    if isinstance(val, str):
                        group_query.append('{}=="{}"'.format(key, val))
                    else:
                        group_query.append('{}=={}'.format(key, val))

                group_query = ' & '.join(group_query)

                if self.input_filters is not None:
                    group_query = group_query + ' & ' + self.input_filters

                record_name = "-".join([str(i) for i in group_id])
                seed = int.from_bytes(stable_hash(record_name).encode(),
                                      byteorder='big',
                                      signed=False) % 256
                record_tasks.append(
                    BuildInstanceRecordTask(seed=seed,
                                            input_filters=group_query,
                                            record_name=record_name))
        return record_tasks

    def output(self):

        if not self.input().exists():
            return self.input()
        else:
            tasks = self._get_record_tasks()
            targets = {}
            for split_name in ['train', 'valid', 'test']:
                paths = [t.output()[split_name].path for t in tasks]
                targets[split_name] = [luigi.LocalTarget(p) for p in paths]

            return targets

    def run(self):
        yield self._get_record_tasks()

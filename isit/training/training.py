import luigi
import os
import logging
import abc
import numpy as np
from multiprocessing import Lock, Manager
from luigi.util import requires

from ..luigi_utils import GPUTask
from .augmentations import AugmentationMixin
from .model import BuildInstanceModelMixin
from .record import MultiInstanceRecordTask


def plot_instance_dataset(path, tf_dataset, n_samples=10):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    import inter_view.color

    def create_single_page(raw, y_true):
        raw = raw.numpy().squeeze()
        labels = y_true['embeddings'].numpy().squeeze().astype(np.int16)
        one_hot_classes = y_true['semantic_class'].numpy().squeeze()
        classes = np.argmax(one_hot_classes,
                            axis=-1) + one_hot_classes.sum(axis=-1) - 1
        classes = classes.astype(np.int16)

        if raw.ndim == 2:
            raw = raw[None]
            labels = labels[None]
            classes = classes[None]

        zslice = raw.shape[0] // 2
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].imshow(raw[zslice], cmap='Greys_r')
        axs[1].imshow(labels[zslice],
                      cmap='blk_glasbey_hv',
                      interpolation='nearest',
                      vmin=-1,
                      vmax=254)
        axs[2].imshow(classes[zslice],
                      cmap='blk_glasbey_hv',
                      interpolation='nearest',
                      vmin=-1,
                      vmax=254)

        axs[0].set_xlabel('image')
        axs[1].set_xlabel('embeddings labels')
        axs[2].set_xlabel('classes labels')

        axs[0].set_title('Min: {:4.1f}, Max: {:4.1f}'.format(
            raw.min(), raw.max()))
        axs[1].set_title('Min: {:4.1f}, Max: {:4.1f}'.format(
            labels.min(), labels.max()))
        axs[2].set_title('Min: {:4.1f}, Max: {:4.1f}'.format(
            classes.min(), classes.max()))
        plt.tight_layout()

    with PdfPages(path) as pdf:
        for raw, y_true in tf_dataset.unbatch().take(n_samples):
            create_single_page(raw, y_true)
            pdf.savefig(bbox_inches='tight')
            plt.close()


class InstanceTrainingParams(luigi.Config):
    intra_margin = luigi.FloatParameter(description='intra embedding margin')
    inter_margin = luigi.FloatParameter(description='inter embedding margin')
    jaccard_hinge = luigi.FloatParameter(
        0.3, description='lower hinge for binary Jaccard loss')
    jaccard_eps = luigi.FloatParameter(
        1., description='epsilon/smoothing parameter for binary Jaccard loss')


class TrainingMixin:
    '''Adds default training parameters as luigi task parameters
    and provides the common_callbacks() method.

    '''

    # yapf: disable
    training_dir = luigi.Parameter(description='training base directory.')
    train_batch_size = luigi.IntParameter(description='training batch size.')
    valid_batch_size = luigi.IntParameter(description='validation batch size.')
    train_batches_per_epoch = luigi.IntParameter(description='Number of training batches per epoch (infinitely repeating dataset)')
    valid_batches = luigi.IntParameter(description='Number of validation batches per epoch (infinitely repeating dataset)')
    balance_records = luigi.BoolParameter(False, description='whether to balances training datasets by repeating smaller subsets more frequently', parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    epochs = luigi.IntParameter(description='number of training epochs.')
    lr_min = luigi.FloatParameter(description='minimum learning rate.')
    lr_max = luigi.FloatParameter(description='maximum learning rate.')
    epoch_to_restart_growth = luigi.FloatParameter(2., description='growth factor of the number of epochs after each restart')
    n_restarts = luigi.IntParameter(1, description='number of restarts for the cosine annealing scheduler')
    patience = luigi.IntParameter(200, description='number of epochs without improvment to wait before stopping')
    patch_size = luigi.TupleParameter(description='training patch size')
    resume_weights = luigi.OptionalParameter(None, description='path to weigths used to resume training')
    plot_dataset = luigi.BoolParameter(True, description='plot samples from from the training set to pdf at the beginning of training', parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    # yapf: enable

    def common_callbacks(self, output_folder):
        '''creates several keras callbacks to be used in model.fit or
        model.fit_generator.
    
        '''

        from dlutils.training.callbacks import ModelConfigSaver
        from tensorflow.keras.callbacks import ModelCheckpoint
        from tensorflow.keras.callbacks import TensorBoard
        from tensorflow.keras.callbacks import LearningRateScheduler
        from tensorflow.keras.callbacks import TerminateOnNaN
        from tensorflow.keras.callbacks import EarlyStopping

        from dlutils.training.scheduler import CosineAnnealingSchedule

        n_restarts_factor = sum(self.epoch_to_restart_growth**x
                                for x in range(self.n_restarts))

        epochs_to_restart = (self.epochs + 1) / n_restarts_factor
        if epochs_to_restart < 1:
            raise ValueError(
                'Initial epoch_to_restart ({}) < 1. Decrease n_restarts ({}) or epoch_to_restart_growth ({})'
                .format(epochs_to_restart, self.n_restarts,
                        self.epoch_to_restart_growth))

        epochs_to_restart = int(np.ceil(epochs_to_restart))

        callbacks = []

        if self.lr_max != self.lr_min:
            callbacks.append(
                LearningRateScheduler(
                    CosineAnnealingSchedule(
                        lr_max=self.lr_max,
                        lr_min=self.lr_min,
                        epoch_max=epochs_to_restart,
                        epoch_max_growth=self.epoch_to_restart_growth,
                        reset_decay=1.)))

        callbacks.extend([
            TerminateOnNaN(),
            TensorBoard(os.path.join(output_folder, 'tensorboard-logs'),
                        write_graph=True,
                        write_grads=False,
                        write_images=False,
                        histogram_freq=0),
            ModelCheckpoint(os.path.join(output_folder, 'weights_best.h5'),
                            save_best_only=True,
                            save_weights_only=True),
            ModelCheckpoint(os.path.join(output_folder, 'weights_latest.h5'),
                            save_best_only=False,
                            save_weights_only=True),
        ])

        if self.patience is not None and self.patience >= 1:
            callbacks.append(EarlyStopping(patience=self.patience))

        return callbacks

    @abc.abstractmethod
    def split_samples(self, data):
        '''Function to apply after data augmentation to get training input/output pair'''

        pass


class ModelFittingBaseTask(GPUTask, BuildInstanceModelMixin, AugmentationMixin,
                           TrainingMixin):
    '''

    '''
    def output(self):
        return luigi.LocalTarget(
            os.path.join(self.training_dir, self.model_name))

    @abc.abstractmethod
    def get_training_losses(self):
        '''Function to apply after data augmentation to get training input/output pair'''

        pass

    @abc.abstractmethod
    def _get_parser_fun(self):
        '''Returns a record parser function'''
        pass

    def gpu_run(self):

        import tensorflow as tf
        from dlutils.dataset.dataset import create_dataset

        with self.output().temporary_path() as model_dir:
            logger = logging.getLogger('luigi-interface')
            logger.info('Starting training model: {}'.format(model_dir))

            augmentations = self.get_augmentations() + [self.split_samples]

            trainset = create_dataset([t.path for t in self.input()['train']],
                                      batch_size=self.train_batch_size,
                                      parser_fn=self._get_parser_fun(),
                                      transforms=augmentations,
                                      shuffle_buffer=500,
                                      shuffle=True,
                                      drop_remainder=False,
                                      cache_after_parse=False,
                                      patch_size=self.patch_size,
                                      balance_records=self.balance_records)

            validset = create_dataset([t.path for t in self.input()['valid']],
                                      batch_size=self.valid_batch_size,
                                      parser_fn=self._get_parser_fun(),
                                      transforms=[self.split_samples],
                                      drop_remainder=False,
                                      cache_after_parse=False,
                                      patch_size=self.patch_size,
                                      balance_records=self.balance_records)

            trainset = trainset.repeat()
            validset = validset.repeat()

            model = self.construct_model()
            model.save(model_dir)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr_max),
                loss=self.get_training_losses(),
                metrics=None)

            if self.resume_weights:
                model.load_weights(self.resume_weights)

            if self.plot_dataset:
                logger = logging.getLogger('luigi-interface')
                logger.info('plotting training examples to pdf')
                plot_instance_dataset(
                    os.path.join(model_dir, 'training_samples.pdf'), trainset,
                    100)

            history = model.fit(trainset,
                                validation_data=validset,
                                epochs=self.epochs,
                                steps_per_epoch=self.train_batches_per_epoch,
                                validation_steps=self.valid_batches,
                                callbacks=self.common_callbacks(model_dir))


class InferenceModelExportBaseTask(luigi.Task):
    def run(self):
        import tensorflow as tf

        model = tf.keras.models.load_model(self.input().path, compile=False)
        model.load_weights(os.path.join(self.input().path,
                                        'weights_latest.h5'))

        with self.output().temporary_path() as temp_output_path:
            tf.saved_model.save(model,
                                export_dir=temp_output_path,
                                signatures=self.get_serve_signatures(model))

    def get_serve_signatures(self, model):
        return {}

    def output(self):
        return luigi.LocalTarget(self.input().path + '_inference')


@requires(MultiInstanceRecordTask)
class InstanceModelFittingTask(InstanceTrainingParams, ModelFittingBaseTask):
    def split_samples(self, data):
        import tensorflow as tf

        fg = tf.minimum(tf.cast(data['segm'], tf.int32), 1)
        fg = tf.squeeze(fg, axis=-1)
        fg_hot = tf.one_hot(fg, depth=2)

        return data['image'], {
            'embeddings': data['segm'],
            'semantic_class': fg_hot
        }

    def _get_parser_fun(self):
        import tensorflow as tf
        from dlutils.dataset.tfrecords import ImageToSegmentationRecordParser

        fixed_ndim = len(self.input_shape)
        return ImageToSegmentationRecordParser(tf.float32,
                                               tf.int16,
                                               fixed_ndim=fixed_ndim).parse

    def get_training_losses(self):
        from dlutils.losses.jaccard_loss import HingedJaccardLoss
        from dlutils.losses.embedding.embedding_loss import MarginInstanceEmbeddingLoss

        return {
            'embeddings':
            MarginInstanceEmbeddingLoss(intra_margin=self.intra_margin,
                                        inter_margin=self.inter_margin,
                                        parallel_iterations=4),
            'semantic_class':
            HingedJaccardLoss(hinge_thresh=self.jaccard_hinge,
                              eps=self.jaccard_eps),
        }


@requires(InstanceModelFittingTask)
class InstanceModelExportTask(InferenceModelExportBaseTask):
    def get_serve_signatures(self, model):
        import tensorflow as tf
        from dlutils.postprocessing.voting import embeddings_to_labels

        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, ) + self.input_shape, dtype=tf.float32)
        ])
        def model_fun(img):
            '''applies the trained model without pre/post proc. Useful for tile processing
            '''
            embeddings, classes = model(img, training=False)

            return {'embeddings': embeddings, 'classes': classes}

        @tf.function(input_signature=[
            tf.TensorSpec(shape=self.input_shape[:-1], dtype=tf.uint16)
        ])
        def instance_seg_inference_fun(img):
            '''preprocesses image, runs model on it, and postprocesses the models output ... 
            '''
            #preprocess
            img = tf.cast(img, tf.float32)
            img = img - tf.reduce_mean(img)
            img = img / tf.math.reduce_std(img)
            img = img[None, ..., None]

            # apply the actual model.
            embeddings, classes = model(img, training=False)

            # postprocess
            embeddings = embeddings[0]
            fg_mask = classes[0, ..., 1] > 0.5

            # ~with tf.device('/CPU'):
            labels = embeddings_to_labels(embeddings,
                                          fg_mask,
                                          peak_min_distance=2 *
                                          self.intra_margin,
                                          spacing=self.model_spacing,
                                          min_count=2)

            labels = tf.cast(labels, tf.uint16)
            return {'labels': labels}

        @tf.function(input_signature=[])
        def get_config():
            return {
                'spacing': tf.constant(self.model_spacing),
                'peak_min_distance': tf.constant(2 * self.intra_margin)
            }

        return {
            'pred_instance': instance_seg_inference_fun,
            'get_config': get_config,
            'model_fun': model_fun,
        }


def main():
    luigi.run(main_task_cls=InstanceModelExportTask, local_scheduler=True)


if __name__ == "__main__":
    main()

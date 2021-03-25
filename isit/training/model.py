import luigi


def _format_tuple(val):
    '''Format tuple param to write in model name'''

    unique_val = tuple(set(val))

    if len(unique_val) == 1:
        return str(unique_val[0])
    else:
        return str(val).replace(', ', '-').replace('(', '').replace(')', '')


class BuildInstanceModelMixin:

    # yapf: disable
    input_shape = luigi.TupleParameter((None, None, 1), description='model input shape')
    downsampling_factor = luigi.TupleParameter(description='Downsampling factor, can be specified for each dimension separately.')
    n_downsampling_channels = luigi.IntParameter(description='number of channels after downsampling (strided conv).')
    n_groups = luigi.IntParameter(description='number of groups in group conv.')
    channels_per_group = luigi.IntParameter(description='number of channels per group.')
    dilation_rates = luigi.TupleParameter(description='Dilation rates used in dilated conv pyramid')
    dropout = luigi.FloatParameter(0.1, description='spatial dropout rate.')
    n_steps = luigi.IntParameter(5, description='number of steps of delta loop')
    suffix = luigi.Parameter('', description='suffix appended to model name')
    n_classes = luigi.IntParameter(2, description='number of semantic classes')
    model_spacing = luigi.TupleParameter((1., ), description='scaling of embedding coordinates, e.g. voxel size.')
    # yapf: enable

    @property
    def model_name(self):
        '''
        '''

        model_name = 'RDCNet-F{}-DC{}-G{}-DR{}-GC{}-S{}-D{}'.format(
            _format_tuple(self.downsampling_factor),
            self.n_downsampling_channels, self.n_groups,
            _format_tuple(self.dilation_rates), self.channels_per_group,
            self.n_steps, self.dropout)

        if len(self.suffix) > 0:
            model_name += '_' + self.suffix

        return model_name

    def construct_model(self):
        '''composes model from backbone and head.
        '''

        from dlutils.models.rdcnet import GenericRDCnetBase
        from dlutils.models.heads import split_output_into_instance_seg

        ndim = len(self.input_shape) - 1
        n_output_channels = self.n_classes + ndim
        if ndim == 3:
            up_method = 'transposed_conv'
        else:
            up_method = 'upsample'

        model = GenericRDCnetBase(
            input_shape=self.input_shape,
            downsampling_factor=self.downsampling_factor,
            n_downsampling_channels=self.n_downsampling_channels,
            n_output_channels=n_output_channels,
            n_groups=self.n_groups,
            dilation_rates=self.dilation_rates,
            channels_per_group=self.channels_per_group,
            n_steps=self.n_steps,
            dropout=self.dropout,
            up_method=up_method)

        return split_output_into_instance_seg(model,
                                              n_classes=self.n_classes,
                                              spacing=self.model_spacing,
                                              class_activation=False)

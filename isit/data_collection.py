import os
import pandas as pd
import luigi
import logging
from functools import lru_cache
import json

from improc.io import parse_collection, DCAccessor

DCAccessor.register()


class CollectionTarget(luigi.LocalTarget):
    extension = 'h5'

    def __init__(self, outdir, filename, *args, **kwargs):
        '''
        '''
        path = os.path.join(outdir, '{}.{}'.format(filename, self.extension))
        super().__init__(path, *args, **kwargs)

    def load(self):
        '''
        '''
        return pd.read_hdf(self.path, 'dc')

    def save(self, df, task_config, **kwargs):
        '''
        '''

        config = pd.Series(task_config)
        self.makedirs()

        with self.temporary_path() as path:
            df.to_hdf(path, key='dc', **kwargs)
            config.to_hdf(path, key='config', **kwargs)


class ParseCollectionTask(luigi.Task):
    ''''''

    datadir = luigi.Parameter()
    pattern = luigi.Parameter()
    index = luigi.ListParameter()

    collection_name = luigi.Parameter(
        'collection_input', description='filename of exported data collection')
    outdir = luigi.OptionalParameter(None)

    def requires(self):
        return None

    def output(self):
        if self.outdir is None:
            self.outdir = self.datadir

        return CollectionTarget(self.outdir, self.collection_name)

    def run(self):
        df = parse_collection(os.path.join(self.datadir, self.pattern),
                              list(self.index))

        self.output().save(df, self.param_kwargs)


@lru_cache(maxsize=128)
def _get_config(path):
    with open(path, 'r') as f:
        return pd.Series(json.load(f))


class ParseConfigCollectionTask(ParseCollectionTask):
    '''Parses a data collection and append extra json config as new columns'''

    json_relative_path = luigi.Parameter(
        description='path json config relative to the files in collection')

    def add_json_config(self, row):

        config_path = os.path.normpath(
            os.path.join(row.dc.path[0], self.json_relative_path))
        config = _get_config(config_path)
        new_cols = list(set(config.index) - set(row.index))
        return row.append(config[new_cols])

    def run(self):
        df = parse_collection(os.path.join(self.datadir, self.pattern),
                              list(self.index))

        df = df.reset_index().apply(self.add_json_config,
                                    axis=1).set_index(list(self.index))

        print(df.columns)

        self.output().save(df, self.param_kwargs)

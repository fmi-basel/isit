import logging
import luigi
import os

from .training.training import InstanceModelExportTask
from .prediction.predict_tasks import SegmentInstancesTask, BatchedSegmentInstancesTask

config = luigi.configuration.get_config()

logdir = config.get('core', 'outdir', fallback=os.getcwd())
log_level = config.get('core', 'log_level', fallback=logging.INFO)
os.makedirs(logdir, exist_ok=True)

# setup logging to file. better way to do it with luigi?
logger = logging.getLogger('luigi-interface')
formatter = logging.Formatter(
    '[%(asctime)s] (%(name)s) [%(levelname)s]: %(message)s',
    '%d.%m.%Y %I:%M:%S')
fh = logging.FileHandler(os.path.join(logdir, 'luigi.log'))
fh.setLevel(log_level)
fh.setFormatter(formatter)
logger.addHandler(fh)

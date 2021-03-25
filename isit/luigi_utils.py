import luigi
import abc
import json
import hashlib
import concurrent
from tqdm import tqdm
from multiprocessing import Lock, Manager


def thread_lock_wrapper(fun, lock):
    def _wrapped_fun(*args, **kargs):
        with lock:
            return fun(*args, **kargs)

    return _wrapped_fun


def monitor_futures(futures, pbar=False):
    '''Collect futures as completed and cancel on first encountered error'''

    #NOTE if progress bar was no needed, one could use:
    # done_futures, not_done_futures = concurrent.futures.wait(futures, return_when='FIRST_EXCEPTION')
    # ...

    iterator = concurrent.futures.as_completed(futures)
    if pbar:
        iterator = tqdm(iterator, total=len(futures))

    for f in iterator:
        if f.exception() is not None:
            [f.cancel() for f in futures]
            break

    return [f.result() for f in futures]


def stable_hash(a_string, hash_length=8):
    '''Returns a hash that does not change with python instance (i.e. unlike hash()) '''

    str_hash = hashlib.md5(a_string.encode('utf-8')).hexdigest()

    if len(str_hash) > hash_length:
        return str_hash[:hash_length]
    if len(str_hash) < hash_length:
        return '_' * (hash_length - len(str_hash)) + str_hash
    return str_hash


def task_to_hash(task):
    '''snippet from luigi.task.task_id_str

    '''
    params = task.to_str_params(only_significant=True, only_public=True)
    param_str = json.dumps(params, separators=(',', ':'), sort_keys=True)
    return stable_hash(param_str)


# NOTE alternatively check actual usage with nvidia-smi, (e.g. nvgpu python wrapper)
# (tensorflow would have to be initialized and allocate memory before releasing lock)
class GPUTask(luigi.Task):

    _lock = Lock()
    _used_gpus = Manager().dict()

    resources = {'gpu': 1}

    def _acquire_gpu(self):

        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')

        self.gpu_idx = -1
        gpu_device = []
        with GPUTask._lock:
            for idx, device in enumerate(physical_devices):
                if not GPUTask._used_gpus.get(idx, False):
                    GPUTask._used_gpus[idx] = True
                    self.gpu_idx = idx
                    gpu_device = [device]
                    break

        if self.gpu_idx < 0:
            raise RuntimeError(
                'no available GPU found. Check that luigi resources "gpu_task" matches the number of physical GPU'
            )
            # TODO try get "gpu_task" from luigi config and compare to number of available gpu
            # log warning instead and attempt to run on cpu?

        # print('Placing on GPU {}'.format(self.gpu_idx))
        tf.config.set_visible_devices(gpu_device, 'GPU')

        # to be able to estimate VRAM usage with nvidia-smi
        tf.config.experimental.set_memory_growth(
            physical_devices[self.gpu_idx], True)

    def _release_gpu(self):
        if hasattr(self, 'gpu_idx'):
            with GPUTask._lock:
                GPUTask._used_gpus[self.gpu_idx] = False

    def run(self):
        self._acquire_gpu()
        self.gpu_run()
        self._release_gpu()

    def on_failure(self, exception):
        self._release_gpu()
        return super().on_failure(exception)

    @abc.abstractmethod
    def gpu_run(self):
        pass

import logging
import luigi
import numpy as np
import os
import tifffile

from multiprocessing import Queue, Process
from queue import Empty
from traceback import format_exc

from isit.prediction.apply_model import dynamically_apply_instance_seg_model
from isit.prediction.postprocessing import get_postprocess_and_save_fun, queue_worker
from isit.prediction.preprocessing import getfun_match_scaling_and_normalize

def load_image(input_image):
	img = tifffile.imread(input_image).astype(np.float32)
	return img

def close_postprocessing(out_queue, post_processors, force = False):
	join_timeout = None
	if force:
		# some libraries (e.g. TF) can be super slow to terminate, let's be generous
		join_timeout = 5
		# we might need to kill some worker processes if they are stuck
		# if they can't get their item from the out_queue the queue will fail to properly close
		# as by default Queues will wait until the entire buffer is flushed before actually
		# closing when garbage collected. This can be prevented with cancel_join_thread()
		out_queue.cancel_join_thread()
	logger = logging.getLogger('luigi-interface')
	logger.info('Closing processing, sending termination item to the queue')
	for _ in range(len(post_processors)):
		out_queue.put(None)
	logger.info('Waiting for children processes to terminate')
	for process in post_processors:
		process.join(join_timeout)
	# check if any process is still alive and force the termination if so
	if force:
		logger.info('Checking for stuck worker processes')
		for process in post_processors:
			if process.is_alive():
				logger.info('worker process %d not terminating after timeout, sending SIGKILL' % process.pid)
				os.kill(process.pid, signal.SIGKILL)
	logger.info('Closing the queue')
	out_queue.close()

def gpu_run(input_image_paths, model_path, labels_fill_holes, labels_size_threshold, labels_keep_largest, labels_sequential, image_spacing, tiling_patch_size, tiling_border, num_post_processes = 4):
	logger = logging.getLogger('luigi-interface')

	import tensorflow as tf

	model = tf.saved_model.load(model_path)
	model_spacing = model.signatures['get_config']()['spacing'].numpy()

	peak_min_distance = model.signatures['get_config']()['peak_min_distance'].numpy()
	model_fun = model.signatures['model_fun']

	logger.info('Start {} post-processing processes.'.format(num_post_processes))
	out_queue = Queue()
	post_processors = []
	for _ in range(num_post_processes):
		postprocess_fun = get_postprocess_and_save_fun(labels_fill_holes,
								labels_size_threshold,
								labels_keep_largest,
								labels_sequential,
								image_spacing)
		t = Process(target=queue_worker,
			args=(out_queue,
				postprocess_fun))
		t.start()
		post_processors.append(t)

	try:
		fun_preprocessing = getfun_match_scaling_and_normalize(image_spacing, model_spacing)
		# def mapped_function(img, file):
		# 	return tf.py_function(fun_preprocessing, inp=[img, file], Tout=[tf.float32, tf.string, tf.int32])

		# logger.info('Create ParallelImageDataset.')
		# ds_pre = ParallelImageDatasetFactory(df_in.dc.path, df_out.dc.path, out_shape=(None,) * len(model_spacing), pre_proc_fun=mapped_function, num_interleaved_ds=1)
		
		for input_image in input_image_paths:
			out_file = os.path.join(os.path.dirname(input_image), 'enricos.tif')
			img = load_image(input_image)
			img, _, img_shape = fun_preprocessing(img, input_image, False)
			img = tf.convert_to_tensor(img)
			out_queue.put(
				dynamically_apply_instance_seg_model(
					model_fun,
					img,
					tf.convert_to_tensor(out_file),
					tf.convert_to_tensor(img_shape),
					tiling_patch_size, tiling_border,
					peak_min_distance,
					model_spacing))
			logger.info('Queued prediction for saving at {}.'.format(out_file))
		
		# for img, out_file, img_shape in ds_pre:
		# 	out_queue.put(
		# 	dynamically_apply_instance_seg_model(model_fun, img, out_file, img_shape,
		# 						tiling_patch_size, tiling_border,
		# 						peak_min_distance,
		# 						model_spacing))
		# 	logger.info('Queued prediction for saving at {}.'.format(out_file))

	# KeyboardInterrupt inherits from BaseException and so catching for Exception only will not
	# run the following piece of code to cleanup
	except (Exception, KeyboardInterrupt) as exc:
		logger.error('Got exception: %s' % str(exc))
		logger.error('%s' % format_exc())
		logger.error('Flushing the queue')
		while not out_queue.empty():
			try:
				out_queue.get(False)
			except Empty:
				break

		close_postprocessing(out_queue, post_processors, True)

		raise exc

	close_postprocessing(out_queue, post_processors)
	# self.check_output(df_out)
	# self.output().save(df_out, self.param_kwargs)

def standalone_run(input_image_paths, model_path, labels_fill_holes, labels_size_threshold, labels_keep_largest, labels_sequential, image_spacing, tiling_patch_size, tiling_border):
	gpu_run(input_image_paths, model_path, labels_fill_holes, labels_size_threshold, labels_keep_largest, labels_sequential, image_spacing, tiling_patch_size, tiling_border)

if __name__ == '__main__':
	import sys
	
	logger = logging.getLogger('luigi-interface')
	format_without_date = logging.Formatter('%(levelname)s: %(message)s')
	logger.setLevel(logging.INFO)
	stdout_handler = logging.StreamHandler(sys.stdout)
	stdout_handler.setFormatter(format_without_date)
	logger.addHandler(stdout_handler)
	
	model_path = r'/tungstenfs/scratch/gliberal/Users/repinico/Training/nuc_training/lapisFOCM/training_output/RDCNet-F1-3-3-DC16-G4-DR1-2-4-GC32-S5-D0.1_lapisFOCM_inference'
	tiling_patch_size = (100, 400, 400, 1)
	tiling_border = (25, 50, 50, 0)
	image_spacing = (0.6, 0.216, 0.216)
	labels_keep_largest = True
	labels_fill_holes = True
	labels_size_threshold = 0
	labels_sequential = True
	input_image_paths = [r'/tungstenfs/nobackup/gliberal/taglenri/nicole/isit-test/orig/day1p5/obj_v0.3_ROI/C02/object_38/220605_151046_C02_T0001F001L01A04Z01C01O00038_TIF-OVR.tif']

	standalone_run(input_image_paths, model_path, labels_fill_holes, labels_size_threshold, labels_keep_largest, labels_sequential, image_spacing, tiling_patch_size, tiling_border)

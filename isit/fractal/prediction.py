import logging
import luigi
import numpy as np
import os
import tensorflow as tf
import tifffile

from traceback import format_exc

from isit.prediction.apply_model import dynamically_apply_instance_seg_model
from isit.prediction.postprocessing import postprocess
from isit.prediction.preprocessing import getfun_match_scaling_and_normalize

def load_image(input_image):
	img = tifffile.imread(input_image).astype(np.float32)
	return img

def save_image(output_image, img):
	return tifffile.imwrite(output_image, img)

def segment_ROI(img, model_fun, tiling_patch_size, tiling_border, peak_min_distance, model_spacing, labels_fill_holes, labels_size_threshold, labels_keep_largest, labels_sequential, image_spacing, num_post_processes):
	img, _, img_shape = preprocessing_fun(img, '', False)
	img_shape = tf.convert_to_tensor(img_shape)
	img = tf.convert_to_tensor(img)
	labels, _, _ = dynamically_apply_instance_seg_model(
			model_fun,
			img,
			tf.convert_to_tensor(''), # no image output required
			img_shape,
			tiling_patch_size, tiling_border,
			peak_min_distance,
			model_spacing)
	labels = postprocess(
			labels,
			img_shape,
			labels_fill_holes,
			labels_size_threshold,
			labels_keep_largest,
			labels_sequential,
			image_spacing,
			num_post_processes)
	return labels

def isit_segmentation(io_image_paths, model_path, labels_fill_holes, labels_size_threshold, labels_keep_largest, labels_sequential, image_spacing, tiling_patch_size, tiling_border, num_post_processes = 4):
	logger = logging.getLogger('luigi-interface')
	
	model = tf.saved_model.load(model_path)
	model_spacing = model.signatures['get_config']()['spacing'].numpy()

	peak_min_distance = model.signatures['get_config']()['peak_min_distance'].numpy()
	model_fun = model.signatures['model_fun']

	global preprocessing_fun
	preprocessing_fun = getfun_match_scaling_and_normalize(image_spacing, model_spacing)

	for input_image, out_file in io_image_paths:
		logger.info('Loading and processing image: %s' % input_image)
		img = load_image(input_image)
		labels = segment_ROI(img, model_fun, tiling_patch_size, tiling_border, peak_min_distance, model_spacing, labels_fill_holes, labels_size_threshold, labels_keep_largest, labels_sequential, image_spacing, num_post_processes)
		logger.info('Saving labels to: %s' % out_file)
		save_image(out_file, labels)
	return

if __name__ == '__main__':
	import sys
	
	from glob import glob
	
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
	input_image_paths = glob(r'obj_v0.3_ROI/*/object_*/220605_151046*C01*_TIF-OVR.tif')
	output_image_paths = []
	for image_path in input_image_paths:
		filename, ext = os.path.splitext(image_path)
		output_image_paths.append('%s%s%s' % (filename.replace('TIF-OVR', 'NUC_SEG3D'), '_enrico', ext))
	
	isit_segmentation(zip(input_image_paths, output_image_paths), model_path, labels_fill_holes, labels_size_threshold, labels_keep_largest, labels_sequential, image_spacing, tiling_patch_size, tiling_border, 1)

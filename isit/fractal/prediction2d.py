import logging
import numpy as np
import os
import tensorflow as tf
import tifffile

from scipy.ndimage import find_objects
from dlutils.models import load_model as dlutils_load_model
from improc.morphology import clean_up_labels
from isit.prediction.apply_model import apply_instance_seg_model, tiled_apply_instance_seg_model

def load_image(input_image):
	img = tifffile.imread(input_image).astype(np.float32)
	if len(img.shape) == 3:
		img = img[0]
	return img

def save_image(output_image, img):
	return tifffile.imwrite(output_image, img)

# the following is a rework from the drogon version, see
# https://github.com/fmi-basel/job-system-workflows/blob/585cfb270a01fda04c50cac4f37857d9be1eb69e/src/gliberal_workflows/tasks/cnn_segmentation_task/versions/version_0_3.py#L178

def segment_ROI2d(img, model, physical_pixel_size, magnification, patch_size, border, gpu_postproc, lock):
	image_spacing = (
		physical_pixel_size / magnification,
		physical_pixel_size / magnification
	)
	if len(img.shape) != 2:
		raise ValueError('Input image must be exactly 2D')
	pred = tiled_apply_instance_seg_model(
		model, img, image_spacing,
		patch_size = patch_size,
		border = border,
		gpu_postproc = gpu_postproc,
		lock = lock
	)
	locs = find_objects(pred)
	locs = [
		tuple(slice(max(0, s.start - 1), s.stop + 1)
			for s in loc) if loc else None for loc in locs
	]
	label_image = np.zeros_like(pred)
	for l, loc in enumerate(locs, start=1):
		mask = pred[loc] == l
		if loc:
			label_image[loc][mask] = l
	
	return label_image

def isit_segmentation2d(io_image_paths, model_path, magnification, core_count = 4):
	logger = logging.getLogger('luigi-interface')
	
	tf.config.threading.set_intra_op_parallelism_threads(core_count)
	tf.config.threading.set_inter_op_parallelism_threads(core_count)
	
	model = dlutils_load_model(model_path)
	for input_image, out_file in io_image_paths:
		logger.info('Loading and processing image: %s' % input_image)
		img = load_image(input_image)
		# the hardcoded values in the following were hardcoded in drogon as well
		labels = segment_ROI2d(img, model, physical_pixel_size = 6.45, magnification = magnification, patch_size = (1300, 1300, 1), border = (300, 300, 0), gpu_postproc = False, lock = None)
		logger.info('Saving labels to: %s' % out_file)
		save_image(out_file, labels)
	return


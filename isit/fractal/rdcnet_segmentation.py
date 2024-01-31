 # Copyright 2024 (C) Friedrich Miescher Institute for Biomedical Research
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Marco Franzon <marco.franzon@exact-lab.it>
# Joel Lüthi  <joel.luethi@fmi.ch>
# Franziska Moos <franziska.moos@fmi.ch>
# Enrico Tagliavini <enrico.tagliavini@fmi.ch>

import logging
import numpy as np
import os
import tensorflow as tf

from traceback import format_exc

from pydantic.decorator import validate_arguments

import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata._io.specs import write_elem
from dlutils.models import load_model as dlutils_load_model
from isit.prediction.apply_model import apply_instance_seg_model, tiled_apply_instance_seg_model
from scipy.ndimage import find_objects

import fractal_tasks_core
from fractal_tasks_core.lib_channels import ChannelNotFoundError
from fractal_tasks_core.lib_channels import get_channel_from_image_zarr
from fractal_tasks_core.lib_channels import OmeroChannel
from fractal_tasks_core.lib_input_models import Channel
from fractal_tasks_core.lib_ngff import load_NgffImageMeta
from fractal_tasks_core.lib_masked_loading import masked_loading_wrapper
from fractal_tasks_core.lib_pyramid_creation import build_pyramid
from fractal_tasks_core.lib_regions_of_interest import (
    array_to_bounding_box_table,
)
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.lib_regions_of_interest import check_valid_ROI_indices
from fractal_tasks_core.lib_regions_of_interest import is_ROI_table_valid
from fractal_tasks_core.lib_regions_of_interest import load_region
from fractal_tasks_core.lib_ROI_overlaps import find_overlaps_in_ROI_indices
from fractal_tasks_core.lib_ROI_overlaps import get_overlapping_pairs_3D
from fractal_tasks_core.lib_write import prepare_label_group
from fractal_tasks_core.lib_write import write_table
from fractal_tasks_core.lib_zattrs_utils import rescale_datasets

from isit.prediction.apply_model import dynamically_apply_instance_seg_model
from isit.prediction.postprocessing import postprocess
from isit.prediction.preprocessing import getfun_match_scaling_and_normalize

logger = logging.getLogger(__name__)
__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

def segment_ROI_3D(img, model_fun, tiling_patch_size, tiling_border, peak_min_distance, model_spacing, labels_fill_holes, labels_size_threshold, labels_keep_largest, labels_sequential, image_spacing, num_post_processes):
    """
    Applies 3D RDCNet segmenation to img
    img needs to be 4D to work with the wrapper task, but RDCNet internally
    wants a 3D array.
    """
    logger.info("[segment_ROI_3D] Starting preprocessing")
    img = np.squeeze(img)
    img, _, img_shape = preprocessing_fun(img, '', False)
    img_shape = tf.convert_to_tensor(img_shape)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    logger.info("[segment_ROI_3D] Starting segmentation")
    labels, _, _ = dynamically_apply_instance_seg_model(
            model_fun,
            img,
            tf.convert_to_tensor(''), # no image output required
            img_shape,
            tiling_patch_size, tiling_border,
            peak_min_distance,
            model_spacing)
    logger.info("[segment_ROI_3D] Segmentation done, starting postprocessing")
    labels = postprocess(
            labels,
            img_shape,
            labels_fill_holes,
            labels_size_threshold,
            labels_keep_largest,
            labels_sequential,
            image_spacing,
            num_post_processes)
    logger.info("[segment_ROI_3D] Postprocessing Done")
    return labels

def init_segmentation_3D(model_path,core_count,image_spacing):
    # the following will limit the amount of CPU cores TensorFlow will use for CPU tasks.
    # there is one CPU only task at the end of the dynamically_apply_instance_seg_model() function
    # see also https://stackoverflow.com/questions/57925061/how-can-i-reduce-the-number-of-cpus-used-by-tensorlfow-keras
    tf.config.threading.set_intra_op_parallelism_threads(core_count)
    tf.config.threading.set_inter_op_parallelism_threads(core_count)

    model = tf.saved_model.load(model_path)
    model_spacing = model.signatures['get_config']()['spacing'].numpy()

    peak_min_distance = model.signatures['get_config']()['peak_min_distance'].numpy()
    model_fun = model.signatures['model_fun']

    global preprocessing_fun
    preprocessing_fun = getfun_match_scaling_and_normalize(image_spacing, model_spacing)

    return model, model_spacing, model_fun, peak_min_distance

def segment_ROI_2D(img, model, image_spacing, patch_size):
    """
    Applies 2D RDCNet segmenation to img

    """
    #img = img[0]
    if len(img.shape) != 2:
        raise ValueError('Input image must be exactly 2D')
    pred = tiled_apply_instance_seg_model(
        model, img, image_spacing,
        patch_size = patch_size,
        border = (300, 300, 0), #Hard coded values found in drogon
        gpu_postproc = False, #in the original drogon code the variable was hard (eye emoji) coded to False
        lock = None
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

def init_segmentation_2D(model_path, core_count):
    tf.config.threading.set_intra_op_parallelism_threads(core_count)
    tf.config.threading.set_inter_op_parallelism_threads(core_count)

    model = dlutils_load_model(model_path)
    return model


@validate_arguments
def rdcnet_segmentation(
    *,
    # Fractal arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: Dict[str, Any],
    # Task-specific arguments
    level: int,
    channel: Channel,
    model_path: str,
    input_ROI_table: str = "FOV_ROI_table",
    output_ROI_table: Optional[str] = None,
    output_label_name: Optional[str] = None,
    use_masks: bool = False,
    relabeling: bool = True,
    # RDCNet-related arguments
    core_count: int = 4,
    tiling_patch_size: tuple[int] = (100, 400, 400, 1),
    tiling_border: tuple[int] = (25, 50, 50, 0),
    labels_fill_holes: bool = True,
    labels_size_threshold: int = 0,
    labels_keep_largest: bool = True,
    labels_sequential: bool = True,
    # 2D RDCNet-related arguments
    patch_size: tuple[int] = [1300, 1300, 1],
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Run RDCNet segmentation on the ROIs of a single OME-Zarr image

    Args:
        input_paths: List of input paths where the image data is stored as
            OME-Zarrs. Should point to the parent folder containing one or many
            OME-Zarr files, not the actual OME-Zarr file. Example:
            `["/some/path/"]`. This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path to the OME-Zarr image in the OME-Zarr plate that is
            processed. Example: `"some_plate.zarr/B/03/0"`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        level: Pyramid level of the image to be segmented. Choose `0` to
            process at full resolution.
        channel: Primary channel for segmentation; requires either
            `wavelength_id` (e.g. `A01_C01`) or `label` (e.g. `DAPI`).
        model_path: Full path to the trained RDCNet model
        input_ROI_table: Name of the ROI table over which the task loops to
            apply RDCNet segmentation. Examples: `FOV_ROI_table` => loop over
            the field of views, `organoid_ROI_table` => loop over the organoid
            ROI table (generated by another task), `well_ROI_table` => process
            the whole well as one image.
        output_ROI_table: If provided, a ROI table with that name is created,
            which will contain the bounding boxes of the newly segmented
            labels. ROI tables should have `ROI` in their name.
        use_masks: If `True`, try to use masked loading and fall back to
            `use_masks=False` if the ROI table is not suitable. Masked
            loading is relevant when only a subset of the bounding box should
            actually be processed (e.g. running within `organoid_ROI_table`).
        output_label_name: Name of the output label image (e.g. `"organoids"`).
        relabeling: If `True`, apply relabeling so that label values are
            unique for all objects in the well.
        core_count: How many CPU cores to use
        tiling_patch_size: Size of tiling patches to process in pixels (zyx?)
        tiling_border: Overlap between patches to use (we think)
        labels_fill_holes: Whether wholes in label masks should be filled
        labels_size_threshold: RDCNET parameter, always set to True?
        labels_keep_largest: RDCNET parameter, something about overlapping
            labels?
        labels_sequential: Enforcment of sequential labeling?
        patch_size: Size of the patch, excluding border, see RDCNet documentation.
        overwrite: If `True`, overwrite the task output.
    """
    # Handle tiling_patch_size constraints better: Wait for
    # https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/564
    tiling_patch_size = tuple(tiling_patch_size)
    tiling_border = tuple(tiling_border)
    if len(tiling_border) != 4:
        raise ValueError(
            "The parameter `tiling_border` needs to contain 4 integer values"
        )
    if len(tiling_patch_size) != 4:
        raise ValueError(
            "The parameter `tiling_border` needs to contain 4 integer values"
        )

    # Set input path
    if len(input_paths) > 1:
        raise NotImplementedError
    in_path = Path(input_paths[0])
    zarrurl = (in_path.resolve() / component).as_posix()
    logger.info(f"{zarrurl=}")

    # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(zarrurl)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    actual_res_pxl_sizes_zyx = tuple(ngff_image_meta.get_pixel_sizes_zyx(level=level))
    logger.info(f"NGFF image has {num_levels=}")
    logger.info(f"NGFF image has {coarsening_xy=}")
    logger.info(
        f"NGFF image has full-res pixel sizes {full_res_pxl_sizes_zyx}"
    )
    logger.info(
        f"NGFF image has level-{level} pixel sizes "
        f"{actual_res_pxl_sizes_zyx}"
    )

    # Find channel index
    try:
        tmp_channel: OmeroChannel = get_channel_from_image_zarr(
            image_zarr_path=zarrurl,
            wavelength_id=channel.wavelength_id,
            label=channel.label,
        )
    except ChannelNotFoundError as e:
        logger.warning(
            "Channel not found, exit from the task.\n"
            f"Original error: {str(e)}"
        )
        return {}
    ind_channel = tmp_channel.index

    # Set channel label
    if output_label_name is None:
        try:
            channel_label = tmp_channel.label
            output_label_name = f"label_{channel_label}"
        except (KeyError, IndexError):
            output_label_name = f"label_{ind_channel}"

    # Load ZYX data
    data_zyx = da.from_zarr(f"{zarrurl}/{level}")[ind_channel]
    logger.info(f"{data_zyx.shape=}")

    # Read ROI table
    ROI_table_path = f"{zarrurl}/tables/{input_ROI_table}"
    ROI_table = ad.read_zarr(ROI_table_path)

    # Perform some checks on the ROI table
    valid_ROI_table = is_ROI_table_valid(
        table_path=ROI_table_path, use_masks=use_masks
    )
    if use_masks and not valid_ROI_table:
        logger.info(
            f"ROI table at {ROI_table_path} cannot be used for masked "
            "loading. Set use_masks=False."
        )
        use_masks = False
    logger.info(f"{use_masks=}")

    # Create list of indices for 3D ROIs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices, input_ROI_table)

    # If we are not planning to use masked loading, fail for overlapping ROIs
    if not use_masks:
        overlap = find_overlaps_in_ROI_indices(list_indices)
        if overlap:
            raise ValueError(
                f"ROI indices created from {input_ROI_table} table have "
                "overlaps, but we are not using masked loading."
            )

    # Load zattrs file
    zattrs_file = f"{zarrurl}/.zattrs"
    with open(zattrs_file, "r") as jsonfile:
        zattrs = json.load(jsonfile)

    # Rescale datasets (only relevant for level>0)
    if ngff_image_meta.axes_names[0] != "c":
        raise ValueError(
            "Cannot set `remove_channel_axis=True` for multiscale "
            f"metadata with axes={ngff_image_meta.axes_names}. "
            'First axis should have name "c".'
        )
    new_datasets = rescale_datasets(
        datasets=[ds.dict() for ds in ngff_image_meta.datasets],
        coarsening_xy=coarsening_xy,
        reference_level=level,
        remove_channel_axis=True,
    )

    label_attrs = {
        "image-label": {
            "version": __OME_NGFF_VERSION__,
            "source": {"image": "../../"},
        },
        "multiscales": [
            {
                "name": output_label_name,
                "version": __OME_NGFF_VERSION__,
                "axes": [
                    ax.dict()
                    for ax in ngff_image_meta.multiscale.axes
                    if ax.type != "channel"
                ],
                "datasets": new_datasets,
            }
        ],
    }

    image_group = zarr.group(zarrurl)
    label_group = prepare_label_group(
        image_group,
        output_label_name,
        overwrite=overwrite,
        label_attrs=label_attrs,
        logger=logger,
    )

    logger.info(
        f"Helper function `prepare_label_group` returned {label_group=}"
    )
    logger.info(f"Output label path: {zarrurl}/labels/{output_label_name}/0")
    store = zarr.storage.FSStore(f"{zarrurl}/labels/{output_label_name}/0")
    label_dtype = np.uint32

    # Ensure that all output shapes & chunks are 3D (for 2D data: (1, y, x))
    # https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/398
    shape = data_zyx.shape
    if len(shape) == 2:
        shape = (1, *shape)
    chunks = data_zyx.chunksize
    if len(chunks) == 2:
        chunks = (1, *chunks)
    mask_zarr = zarr.create(
        shape=shape,
        chunks=chunks,
        dtype=label_dtype,
        store=store,
        overwrite=False,
        dimension_separator="/",
    )

    logger.info(
        f"mask will have shape {data_zyx.shape} "
        f"and chunks {data_zyx.chunks}"
    )

    # Counters for relabeling
    if relabeling:
        num_labels_tot = 0

    # Iterate over ROIs
    num_ROIs = len(list_indices)

    if output_ROI_table:
        bbox_dataframe_list = []

    def _do_seg2d(model):
        # Prepare keyword arguments for segment_ROI_2D fucntion
        kwargs_segment_ROI_2D = dict(
            model = model,
            image_spacing = actual_res_pxl_sizes_zyx[1:],
            patch_size = patch_size
        )
        # Call segment_ROI through the masked-loading wrapper, which includes
        # pre/post-processing functions if needed

        label_img_2D = masked_loading_wrapper(
            image_array=img_np[0,0],
            function=segment_ROI_2D,
            kwargs=kwargs_segment_ROI_2D,
            use_masks=use_masks,
            preprocessing_kwargs=preprocessing_kwargs,
        )

        # Fillup a 3D Volume to be compatible with saving roi later
        new_label_img = np.zeros(img_np[0].shape)
        new_label_img[:,] = label_img_2D
        return new_label_img

    def _do_seg3d(model_spacing, model_fun, peak_min_distance):
        # Prepare keyword arguments for segment_ROI_3D function
        kwargs_segment_ROI_3D = dict(
            model_fun = model_fun,
            tiling_patch_size = tiling_patch_size,
            tiling_border = tiling_border,
            peak_min_distance = peak_min_distance,
            model_spacing = model_spacing,
            labels_fill_holes = labels_fill_holes,
            labels_size_threshold = labels_size_threshold,
            labels_keep_largest = labels_keep_largest,
            labels_sequential = labels_sequential,
            image_spacing = actual_res_pxl_sizes_zyx,
            num_post_processes = core_count
        )

        # Call segment_ROI through the masked-loading wrapper, which includes
        # pre/post-processing functions if needed
        new_label_img = masked_loading_wrapper(
            image_array=img_np,
            function=segment_ROI_3D,
            kwargs=kwargs_segment_ROI_3D,
            use_masks=use_masks,
            preprocessing_kwargs=preprocessing_kwargs,
        )
        return new_label_img

    do_3D = data_zyx.shape[0] > 1 and len(data_zyx.shape) == 3

    # Initializing models
    if do_3D:
        # Initialize the 3D model
        logger.info(f"3D segmentation will be executed")

        model, model_spacing, model_fun, peak_min_distance = init_segmentation_3D(model_path=model_path, core_count=core_count, image_spacing = actual_res_pxl_sizes_zyx)
        seg_fun = _do_seg3d
        seg_fun_kwarg = {"model_spacing": model_spacing, "model_fun": model_fun, "peak_min_distance": peak_min_distance}
    else:
        # Initialize the 2D model
        logger.info(f"2D segmentation will be executed")
        model = init_segmentation_2D(model_path, core_count)
        seg_fun = _do_seg2d
        seg_fun_kwarg = {"model": model}

    logger.info(f"Now starting loop over {num_ROIs} ROIs")
    for i_ROI, indices in enumerate(list_indices):
        # Define region
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (
            slice(s_z, e_z),
            slice(s_y, e_y),
            slice(s_x, e_x),
        )
        logger.info(f"Now processing ROI {i_ROI+1}/{num_ROIs}")

        # img_np = load_region(data_zyx, region, compute=True, return_as_3D=True)
        img_np = np.expand_dims(
            load_region(data_zyx, region, compute=True, return_as_3D=True),
            axis=0,
        )

        # Prepare keyword arguments for preprocessing function
        preprocessing_kwargs = {}
        if use_masks:
            preprocessing_kwargs = dict(
                region=region,
                current_label_path=f"{zarrurl}/labels/{output_label_name}/0",
                ROI_table_path=ROI_table_path,
                ROI_positional_index=i_ROI,
            )

        new_label_img = seg_fun(**seg_fun_kwarg)

        logger.info(f"Finished segmentation for ROI {i_ROI+1}/{num_ROIs}")

        # Shift labels and update relabeling counters
        if relabeling:
            logger.info(f"Starting relabeling for ROI {i_ROI+1}")
            num_labels_roi = np.max(new_label_img)
            new_label_img[new_label_img > 0] += num_labels_tot
            num_labels_tot += num_labels_roi

            # Write some logs
            logger.info(
                f"ROI {indices}, " f"{num_labels_roi=}, " f"{num_labels_tot=}"
            )

            # Check that total number of labels is under control
            if num_labels_tot > np.iinfo(label_dtype).max:
                raise Exception(
                    "ERROR in re-labeling:"
                    f"Reached {num_labels_tot} labels, "
                    f"but dtype={label_dtype}"
                )
            logger.info(f"Finished relabeling for ROI {i_ROI+1}")

        if output_ROI_table:
            logger.info(f"Starting ROI table processing for ROI {i_ROI+1}")
            bbox_df = array_to_bounding_box_table(
                new_label_img, actual_res_pxl_sizes_zyx
            )

            bbox_dataframe_list.append(bbox_df)

            overlap_list = []
            for df in bbox_dataframe_list:
                overlap_list.extend(
                    get_overlapping_pairs_3D(df, full_res_pxl_sizes_zyx)
                )
            if len(overlap_list) > 0:
                logger.warning(
                    f"{len(overlap_list)} bounding-box pairs overlap"
                )
            logger.info(f"Finished ROI table processing for ROI {i_ROI+1}")

        # Compute and store 0-th level to disk
        da.array(new_label_img).to_zarr(
            url=mask_zarr,
            region=region,
            compute=True,
        )

    logger.info(
        f"End RDCNet_segmentation task for {zarrurl}, "
        "now building pyramids."
    )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{zarrurl}/labels/{output_label_name}",
        overwrite=False,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=chunks,
        aggregation_function=np.max,
    )

    logger.info("End building pyramids")

    if output_ROI_table:
        # Concatenate all ROI dataframes
        df_well = pd.concat(bbox_dataframe_list, axis=0, ignore_index=True)
        df_well.index = df_well.index.astype(str)
        # Extract labels and drop them from df_well
        labels = pd.DataFrame(df_well["label"].astype(str))
        df_well.drop(labels=["label"], axis=1, inplace=True)
        # Convert all to float (warning: some would be int, in principle)
        bbox_dtype = np.float32
        df_well = df_well.astype(bbox_dtype)
        # Convert to anndata
        bbox_table = ad.AnnData(df_well, dtype=bbox_dtype)
        bbox_table.obs = labels

        # Write to zarr group
        image_group = zarr.group(f"{in_path}/{component}")
        logger.info(
            "Now writing bounding-box ROI table to "
            f"{in_path}/{component}/tables/{output_ROI_table}"
        )
        table_attrs = {
            "type": "ngff:region_table",
            "region": {"path": f"../labels/{output_label_name}"},
            "instance_key": "label",
        }
        write_table(
            image_group,
            output_ROI_table,
            bbox_table,
            overwrite=overwrite,
            logger=logger,
            table_attrs=table_attrs,
        )

    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task
    run_fractal_task(
           task_function=rdcnet_segmentation,
           logger_name=logger.name,
    )

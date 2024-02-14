from isit.fractal.rdcnet_segmentation import rdcnet_segmentation
from pydantic.decorator import validate_arguments

from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

from fractal_tasks_core.channels import ChannelInputModel

@validate_arguments
def rdcnet_segmentation_3d(
    *,
    # Fractal arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: Dict[str, Any],
    # Task-specific arguments
    level: int,
    channel: ChannelInputModel,
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
) -> Dict[str, Any]:
    """
    Run RDCNet segmentation on the ROIs of a single OME-Zarr image.
    Performs 3d segmenation.
    """
    return rdcnet_segmentation(
        input_paths = input_paths,
        output_path = output_path,
        component = component,
        metadata = metadata,
        level = level,
        channel = channel,
        model_path = model_path,
        input_ROI_table = input_ROI_table,
        output_ROI_table = output_ROI_table,
        output_label_name = output_label_name,
        use_masks = use_masks,
        relabeling = relabeling,
        do_3D = True,
        core_count = core_count,
        tiling_patch_size = tiling_patch_size,
        tiling_border = tiling_border,
        labels_fill_holes = labels_fill_holes,
        labels_size_threshold = labels_size_threshold,
        labels_keep_largest = labels_keep_largest,
        labels_sequential = labels_sequential
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task
    run_fractal_task(
           task_function=rdcnet_segmentation_3d,
           logger_name=logger.name,
    )

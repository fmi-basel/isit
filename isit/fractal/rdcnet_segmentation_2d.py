from isit.fractal.rdcnet_segmentation import rdcnet_segmentation
from pydantic.decorator import validate_arguments

from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

from fractal_tasks_core.channels import ChannelInputModel

@validate_arguments
def rdcnet_segmentation_2d(
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
    # 2D RDCNet-related arguments
    patch_size: tuple[int] = [1300, 1300, 1],
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Run RDCNet segmentation on the ROIs of a single OME-Zarr MIP image
    Performs 2d segmentation.
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
        do_3D = False,
        core_count = core_count,
        patch_size = patch_size,
        overwrite = overwrite
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task
    run_fractal_task(
           task_function=rdcnet_segmentation_2d,
           logger_name=logger.name,
    )

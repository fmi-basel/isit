from fractal_tasks_core.dev.task_models import ParallelTask

TASK_LIST = [
    ParallelTask(
        name="Rdcnet Segmentation 2d Task",
        executable="fractal/rdcnet_segmentation_2d.py",
        meta={"cpus_per_task": 4, "mem": 8000, "needs_gpu": True},
    ),
    ParallelTask(
        name="Rdcnet Segmentation 3d Task",
        executable="fractal/rdcnet_segmentation_3d.py",
        meta={"cpus_per_task": 40, "mem": 10000, "needs_gpu": True},
    ),
]

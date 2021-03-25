# ISIT
2D/3D Instance Segmentation with Iterative Training. Once an initial model is trained, the provided notebooks can be used to hand-pick bad predictions and correct them before retraining.

## Installation
It is recommended to create a new python environment and install visualization libraries and cuda (GPU support) with conda:

```bash
conda create -n isit python=3.7
conda activate isit
conda install -c pyviz holoviews bokeh panel param xarray datashader pytables
conda install cudatoolkit=10.1 cudnn=7

git clone https://github.com/fmi-basel/isit.git
pip install isit/
```


## Usage
Path to the data, model architecture, etc. can be defined in a config file. 3 examples are provided as a starting point:

1. 2D-mip-org-seg: organoids segmentation on overview images acquired with yokogawa
1. 2D-nuclei-seg: nuclei segmentation on organoid images acquired with yokogawa
1. 3D-nuclei_seg: 3D nuclei segmentation in on images acquired with a W1 spinning disc

The training datasets should contain a json file containing at least the pixel/voxel size. Example in `/tungstenfs/scratch/gliberal/Users/ortiraph/datasets/ISIT`


Training a model:
```bash
LUIGI_CONFIG_PATH=./training_config.cfg luigi --local-scheduler --module isit InstanceModelExportTask
```

Applying a model
```bash
LUIGI_CONFIG_PATH=./predict_config.cfg luigi --local-scheduler --module isit SegmentInstancesTask
```

Applying a 2D model to each slice of a 3D stack
```bash
LUIGI_CONFIG_PATH=./predict_config.cfg luigi --local-scheduler --module isit BatchedSegmentInstancesTask
```


## Hyperparameters selection for the model architecture
These few rules of thumb can be used to select hyperparameters for a new project (taken from https://github.com/fmi-basel/RDCNet)

- **down/up sampling factor** highest factor where instances can still be visually recognized
- **max dilation rate** `object_size << 2 * dilation_rate * downsampling_factor << training_patch_size`
- **network capacity** primarily adjust the number of groups and channels per group
- **number of iterations** 5-6 is usually a good balance between performances and computational cost

# ISIT
2D/3D Instance Segmentation Iterative Training 

## Installation
It is recommended to create a new python environment and install visualization libraries and cuda (GPU support) with conda:

```bash
conda create -n isit python=3.7
conda activate isit
conda install -c pyviz holoviews bokeh panel param xarray datashader pytables
conda install cudatoolkit=10.1 cudnn=7

git clone TODO
pip install TODO
```


## Usage
## Configuration file
## experiment config

## Training a model
```bash
LUIGI_CONFIG_PATH=./training_config.cfg luigi --local-scheduler --module isit InstanceModelExportTask
```

## Applying a model
```bash
LUIGI_CONFIG_PATH=./predict_config.cfg luigi --local-scheduler --module isit SegmentInstancesTask
```

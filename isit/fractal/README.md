# Custom Fractal task to run RDCNets

Requirements:
1. A conda environment with fractal-tasks-core 0.12.2, isit and tensorflow < 2.7 installed.

```
git clone https://github.com/fmi-basel/isit
cd isit/
git checkout dev-enrico
pip install "tensorflow<2.7"
pip install -e .
pip install fractal-tasks-core==0.12.2
```

2. The task in rdcnet_segmentation.py, its schema in rdcnet_segmentation.json and a meta schema in meta_rdcnet_segmentation.json

3. Add the task via the CLI client (to be able to provide a schema):
export F_PYTHON_PATH=/path/to/fractal/rdcnet_env_39/bin/python 
export BASE_PATH=/path/to/Users/luetjoel/isit/isit/fractal/

fractal task new --input-type zarr --output-type zarr --version 0.0.1 --meta-file $BASE_PATH/meta_rdcnet_segmentation.json --args-schema $BASE_PATH/rdcnet_segmentation.json --args-schema-version pydantic_v1 "RDCNet Segmentation" "$F_PYTHON_PATH $BASE_PATH/rdcnet_segmentation.py" jluethi:rdcnet_segmentation_0.0.1

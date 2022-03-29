# MLHarness
[![Build Status](https://dev.azure.com/yhchang/c3sr/_apis/build/status/c3sr.mlharness?branchName=master)](https://dev.azure.com/yhchang/c3sr/_build/latest?definitionId=11&branchName=master)

MLHarness is a scalable benchmarking harness system for MLCommons Inference with three distinctive features: 
- MLHarness codifies the standard benchmark process as defined by MLCommons Inference including the models, datasets, DL frameworks, and software and hardware systems; 
- MLHarness provides an easy and declarative approach for model developers to contribute their models and datasets to MLCommons Inference; and
- MLHarness includes the support of a wide range of models with varying inputs/outputs modalities so that MLHarness can scalably benchmark models across different datasets, frameworks, and hardware systems.

Please see the [MLHarness Paper](https://doi.org/10.1016/j.tbench.2021.100002) and the [MLHarness Presentation](https://www.youtube.com/watch?v=IzAhn5QUClU) for detailed descriptions and case studies that demonstrate the unique value of MLHarness.

### Tutorial
The easiest way to use MLHarness is through the [pre-built docker images](https://hub.docker.com/r/c3sr/mlharness/tags). Instructions for installing docker can be found at [docker's guiding documents](https://docs.docker.com/get-docker/).

To get started, choose a configuration from the table below that fits best to your system.

| System | ONNX Runtime v1.7.1 | MXNet v1.8.0 | PyTorch v1.8.1 | TensorFlow v1.14.0 | 
| :---: | :---: | :---: | :--: | :--: |
| CPU Only | `c3sr/mlharness:amd64-cpu-onnxruntime1.7.1-latest` | `c3sr/mlharness:amd64-cpu-mxnet1.8.0-latest` | `c3sr/mlharness:amd64-cpu-pytorch1.8.1-latest` | `c3sr/mlharness:amd64-cpu-tensorflow1.14.0-latest` |
| GPU with CUDA 10.0 | <center>—</center> | `c3sr/mlharness:amd64-gpu-mxnet1.8.0-cuda10.0-latest` | `c3sr/mlharness:amd64-gpu-pytorch1.8.1-cuda10.0-latest` | `c3sr/mlharness:amd64-gpu-tensorflow1.14.0-cuda10.0-latest` |
| GPU with CUDA 10.1 | `c3sr/mlharness:amd64-gpu-onnxruntime1.7.1-cuda10.1-latest` | `c3sr/mlharness:amd64-gpu-mxnet1.8.0-cuda10.1-latest` | `c3sr/mlharness:amd64-gpu-pytorch1.8.1-cuda10.1-latest` | `c3sr/mlharness:amd64-gpu-tensorflow1.14.0-cuda10.1-latest` |
| GPU with CUDA 10.2 | `c3sr/mlharness:amd64-gpu-onnxruntime1.7.1-cuda10.2-latest` | `c3sr/mlharness:amd64-gpu-mxnet1.8.0-cuda10.2-latest` | `c3sr/mlharness:amd64-gpu-pytorch1.8.1-cuda10.2-latest` | `c3sr/mlharness:amd64-gpu-tensorflow1.14.0-cuda10.2-latest` |
| GPU with CUDA 11.0 | `c3sr/mlharness:amd64-gpu-onnxruntime1.7.1-cuda11.0-latest` | `c3sr/mlharness:amd64-gpu-mxnet1.8.0-cuda11.0-latest` | `c3sr/mlharness:amd64-gpu-pytorch1.8.1-cuda11.0-latest` | <center>—</center> |
| GPU with CUDA 11.1 | `c3sr/mlharness:amd64-gpu-onnxruntime1.7.1-cuda11.1-latest` | `c3sr/mlharness:amd64-gpu-mxnet1.8.0-cuda11.1-latest`  | `c3sr/mlharness:amd64-gpu-pytorch1.8.1-cuda11.1-latest`  | <center>—</center> |
| GPU with CUDA 11.2 | `c3sr/mlharness:amd64-gpu-onnxruntime1.7.1-cuda11.2-latest` | `c3sr/mlharness:amd64-gpu-mxnet1.8.0-cuda11.2-latest` | `c3sr/mlharness:amd64-gpu-pytorch1.8.1-cuda11.2-latest` | <center>—</center> |

After choosing a docker image, there are other two required components, which are models and datasets, where MLHarness uses manifests to codify them. Examples of model manifests can be found at [dlmodel/models](https://github.com/c3sr/dlmodel/tree/master/models) and examples of dataset manifests can be found at [dldataset/datasets](https://github.com/c3sr/dldataset/tree/master/datasets). As not all models and datasets are public, some of the manifests only provide methods to manipulate models and data without having a download method. To address this issue, we can set the environment variable `$DATA_DIR` to the directory containing models and datasets pre-downloaded, and use this environment variable to find models and datasets we want. 

Here is an example run. Suppose we choose ONNX Runtime as our backend and we have a GPU with CUDA 11.2. Therefore, we use `c3sr/mlharness:amd64-gpu-onnxruntime1.7.1-cuda11.2-latest` as the pre-built docker image. Then, we choose to benchmark the BERT model ([manifest](https://github.com/c3sr/dlmodel/blob/master/models/language/onnxruntime/BERT.yml)) on the SQuAD v1.1 dataset ([manifest](https://github.com/c3sr/dldataset/blob/master/datasets/squad.yml)). We setup our directory as follow, where we need `dev-v1.1.json` and `vocab.txt` from the SQuAD v1.1 dataset, and we can get the manifests of models and datasets by cloning [dlmodel](https://github.com/c3sr/dlmodel) and [dldataset](https://github.com/c3sr/dldataset).

```
~/data/
├── SQuAD
│   ├── dev-v1.1.json
│   └── vocab.txt
├── dlmodel
│   └── models
│       └── language
│           └── onnxruntime
│               └── BERT.yml
└── dldataset
    └── datasets
        └── squad.yml
```

To get the help information of MLHarness, we can run the following command by setting `$GPUID` to the GPU ID you wan to use:
```
docker run --rm --gpus device=$GPUID c3sr/mlharness:amd64-gpu-onnxruntime1.7.1-cuda11.2-latest -h
```

Following the help information, we can run the following command to get a simple run:
```
docker run --rm \
  -v ~/data:/root/data \
  --env DATA_DIR=/root/data/SQuAD \
  --gpus device=$GPUID \
  --shm-size 1g --ulimit memlock=-1 --ulimit stack=67108864 --privileged=true --network host \
  c3sr/mlharness:amd64-gpu-onnxruntime1.7.1-cuda11.2-latest \
  --dataset squad --dataset_path /root/data/dldataset/datasets/squad.yml \
  --backend onnxruntime --model_path /root/data/dlmodel/models/language/onnxruntime/BERT.yml \
  --use_gpu 1 --gpu_id $GPUID \
  --accuracy --count 10 \
  --scenario Offline
```
The description follows:
- `docker run --rm`: Run MLHarness as a docker container, remove it after execution.
- `-v ~/data:/root/data`: Mount the directory we prepared.
- `--env DATA_DIR=/root/data/SQuAD`: Set environment variable to the dataset directory we downloaded.
- `--gpus device=$GPUID`: Expose GPU to docker, please replace `$GPUID` with the GPU ID you want to use.
- `--shm-size 1g --ulimit memlock=-1 --ulimit stack=67108864 --privileged=true --network host`: Configure resources in docker container.
- `c3sr/mlharness:amd64-gpu-onnxruntime1.7.1-cuda11.2-latest`: The [pre-build docker image](https://hub.docker.com/r/c3sr/mlharness/tags) we choose above.
- `--dataset squad --dataset_path /root/data/dldataset/datasets/squad.yml`: The dataset and the path to the dataset manifest file in the mounted directory.
- `--backend onnxruntime --model_path /root/data/dlmodel/models/language/onnxruntime/BERT.yml`: The backend and the path to the model manifest file in the mounted directory.
- `--use_gpu 1 --gpu_id $GPUID`: Let MLHarness know that we want to use GPU in the program. Please replace `$GPUID` with the GPU ID you want to use.
- `--accuracy --count 10`: Generate MLCommons Inference reports in accuracy mode, and only run 10 samples for simplicity.
- `--scenario Offline`: Scenario for MLCommons Inference.

After the execution, we are supposed to get `{"exact_match": 70.0, "f1": 70.0}` as the result for the first 10 samples.

For more details, please refer to the help information from the command line.

### Tracer
MLHarness also provides tracing information across software and hardware stacks. To enable this service, start [jaeger](http://jaeger.readthedocs.io/en/latest/getting_started/) by:
```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 jaegertracing/all-in-one:1.21.0
```
And choose one of `--trace_level` from `{NO_TRACE,APPLICATION_TRACE,MODEL_TRACE,FRAMEWORK_TRACE,ML_LIBRARY_TRACE,SYSTEM_LIBRARY_TRACE,HARDWARE_TRACE,FULL_TRACE}` when running experiments.

The tracing information can be found at port 16686. Note that there is a limit on the number of traces that can be recorded, hence please also limit your sample size using `--count`.

### Customization
Aside from using the manifests we already have, we can also create and contribute our manifests, by replacing the corresponding fields in the manifests. Examples of model manifests can be found at [dlmodel/models](https://github.com/c3sr/dlmodel/tree/master/models) and examples of dataset manifests can be found at [dldataset/datasets](https://github.com/c3sr/dldataset/tree/master/datasets). Note that dataset manifest and model manifest share the same python interpreter hence global python variables can be used between two manifests. Some detailed descriptions are as follow:

#### Dataset Manifest
```yaml
name: ... # The name of your dataset
init: |
  def init(count):
    # args:
    ### count: The count argument provided from the command line
    ########## can be used here to restrict the dataset size.
    # description: Reading files and global preprocessing that can not be done per input.
    # return:
    ### real_count: This function must return the real size of the dataset that is involved.
load: |
  def load(sample_list):
    # args:
    ### sample_list: A list of samples used in the current inference.
    # description: In case the dataset can not fit into 
    ############## RAM, dynamically loading samples can be a solution.
    # return: None
unload: |
  def unload(sample_list):
    # args:
    ### sample_list: A list of samples used in the current inference.
    # description: Unload samples that were loaded into RAM.
    # return: None
```

#### Model Manifest
```yaml
name: ... # name of your model
framework:
  name: ... # framework for the model
  version: ... # framework version constraint
version: ... # version for the manifest
description: ... # human read-able description
references: ... # references for the model
license: ... # license of the model
modality: general # general modality is used to fit into the MLCommons format
inputs: # human read-able descriptions
  - type: ... # such as images, sentences ...
    description: ... # descriptions
    parameters: # type parameters
      element_type: ... # element type of this input
  - type: ... # such as images, sentences ...
    description: ... # descriptions
    parameters: # type parameters
      element_type: ... # element type of this input
  ...
outputs: # human read-able descriptions
  - type: ... # such as images, sentences ...
    description: ... # descriptions
    parameters: # type parameters
      element_type: ... # element type of this output
  - type: ... # such as images, sentences ...
    description: ... # descriptions
    parameters: # type parameters
      element_type: ... # element type of this output
  ...
model: # specifies model resources
    is_archive:
        ... # if set, then the base_url is a url to an archive
        # the graph_path and weights_path then denote the
        # file names of the graph and weights within the archive
    graph_path: ... # url to the model
    graph_checksum: ... # model checksum
preprocess: |
  def preprocess(ctx, data):
    # args:
    ### ctx: Extra information from manifest, stored as dictionary.
    ### data: The loaded data provided from data manifest.
    # description: Preprocessing inputs.
    # return: A tuple of numpy arrays that can be used for inference.
postprocess: |
  def postprocess(ctx, data):
    # args:
    ### ctx: Extra information from manifest, stored as dictionary.
    ### data: The outputs from the model.
    # description: Postprocessing outputs.
    # return: A list of json byte arrays that fits into MLCommons format.
attributes: # extra network attributes
  ...
```
# MLHarness
[![Build Status](https://dev.azure.com/yhchang/c3sr/_apis/build/status/c3sr.mlharness?branchName=master)](https://dev.azure.com/yhchang/c3sr/_build/latest?definitionId=11&branchName=master)

MLHarness is a scalable benchmarking harness system for MLCommons Inference with three distinctive features: 
- MLHarness codifies the standard benchmark process as defined by MLCommons Inference including the models, datasets, DL frameworks, and software and hardware systems; 
- MLHarness provides an easy and declarative approach for model developers to contribute their models and datasets to MLCommons Inference; and
- MLHarness includes the support of a wide range of models with varying inputs/outputs modalities so that MLHarness can scalably benchmark models across different datasets, frameworks, and hardware systems.

Please see the [MLHarness Paper](https://arxiv.org/abs/2111.05231) for detailed descriptions and case studies that demonstrate the unique value of MLHarness.

### Tutorial
The easiest way to use MLHarness is through the [pre-built docker images](https://hub.docker.com/r/c3sr/mlharness). Instructions for installing docker can be found at [docker's guiding documents](https://docs.docker.com/get-docker/).

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

After choosing a docker image, there are other two required components, which are models and datasets, where MLHarness uses manifests to codify them. Examples of model manifests can be found at [dlmodel/models](https://github.com/c3sr/dlmodel/tree/master/models) and examples of dataset manifests can be found at [dldataset/datasets](https://github.com/c3sr/dldataset/tree/master/datasets). As not all models and datasets are public, some of the manifests only provide methods to manipulate models and data without having a download method. To address this issue, we can set the environment variable `$DATA_DIR` to the directory containing models and datasets we have pre-downloaded, and use this environment variable to find models and datasets we want. 

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
  --accuracy --count=10 \
  --scenario Offline
```
The description follows:
- `docker run --rm`: Run MLHarness as a docker container, remove it after execution.
- `-v ~/data:/root/data`: Mount the directory we prepared.
- `--env DATA_DIR=/root/data/SQuAD`: Set environment variable to the dataset directory we downloaded.
- `--gpus device=$GPUID`: Expose GPU to docker, please replace `$GPUID` with the GPU ID you want to use.
- `--shm-size 1g --ulimit memlock=-1 --ulimit stack=67108864 --privileged=true --network host`: Configure resources in docker container.
- `c3sr/mlharness:amd64-gpu-onnxruntime1.7.1-cuda11.2-latest`: The [pre-build docker image](https://hub.docker.com/r/c3sr/mlharness) we choose above.
- `--dataset squad --dataset_path /root/data/dldataset/datasets/squad.yml`: The dataset and the path to the dataset manifest file in the mounted directory.
- `--backend onnxruntime --model_path /root/data/dlmodel/models/language/onnxruntime/BERT.yml`: The backend and the path to the model manifest file in the mounted directory.
- `--use_gpu 1 --gpu_id $GPUID`: Let MLHarness know that we want to use GPU in the program. Please replace `$GPUID` with the GPU ID you want to use.
- `--accuracy --count=10`: Generate MLCommons Inference reports in accuracy mode, and only run 10 samples for simplicity.
- `--scenario Offline`: Scenario for MLCommons Inference.

After the execution, we are supposed to get `{"exact_match": 70.0, "f1": 70.0}` as the result for the first 10 samples.

### Customization
Aside from using the manifests we already have above, we can also create and contribute our manifests, by replacing the corresponding fields in the manifests.

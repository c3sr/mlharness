# MLHarness
[![Build Status](https://dev.azure.com/yhchang/c3sr/_apis/build/status/c3sr.mlharness?branchName=master)](https://dev.azure.com/yhchang/c3sr/_build/latest?definitionId=11&branchName=master)

MLHarness is a scalable benchmarking harness system for MLCommons Inference with three distinctive features: 
- MLHarness codifies the standard benchmark process as defined by MLCommons Inference including the models, datasets, DL frameworks, and software and hardware systems; 
- MLHarness provides an easy and declarative approach for model developers to contribute their models and datasets to MLCommons Inference; and
- MLHarness includes the support of a wide range of models with varying inputs/outputs modalities so that MLHarness can scalably benchmark models across different datasets, frameworks, and hardware systems.

Please see the [MLHarness Paper](https://arxiv.org/abs/2111.05231) for detailed descriptions and case studies that demonstrate the unique value of MLHarness.

### Tutorial
The easiest way to use MLHarness is through the [pre-built docker images](https://hub.docker.com/repository/docker/c3sr/mlharness). Instructions for installing docker can be found at [docker's guiding documents](https://docs.docker.com/get-docker/).

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


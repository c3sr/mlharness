## Result

### Close Division

|   System | Processor | # | Accelerator | # | Software | Framework | Model | Task | SingleStream (queries/s) | Offline (samples/s) |
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:| :--------:| :--------:| :--------:|
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | ResNet50-v1.5 | Image Classification | 130  | 345 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | MobileNet-v1 | Image Classification | 187  | 423 |

### Open Division
| System | Processor | # | Accelerator | # | Software | Framework | Model | Task | SingleStream (queries/s) | Offline (samples/s) |
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:| :--------:| :--------:| :--------:|
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_Alexnet | Image Classification | 193  | 472 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_ResNet18 | Image Classification | 172 | 434 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_ResNet34 | Image Classification | 147 | 396 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_ResNet50 | Image Classification | 131 | 332 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_ResNet101 | Image Classification | 95 | 268 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_ResNet152 | Image Classification | 75 | 222 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_Alexnet | Image Classification | 148 | 474 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_VGG_11 | Image Classification | 132 | 368 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_ResNet_18 | Image Classification | 98 | 437 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_ResNet_34 | Image Classification | 76 | 402 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_ResNet50 | Image Classification | 62 | 346 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_ResNet101 | Image Classification | 45 | 284 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_ResNet101 | Image Classification | 34 | 227 |

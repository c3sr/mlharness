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
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_VGG11 | Image Classification | 158 | 339 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_VGG13 | Image Classification | 148 | 291 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_VGG16 | Image Classification | 137 | 256 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_VGG19 | Image Classification | 126 | 229 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_VGG11_BN | Image Classification | 159 | 341 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_VGG13_BN | Image Classification | 149 | 281 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_VGG16_BN | Image Classification | 137 | 259 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_VGG19_BN | Image Classification | 127 | 231 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_DenseNet121 | Image Classification | 88 | 333 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_DenseNet161 | Image Classification | 59 | 225 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_DenseNet169 | Image Classification | 68 | 300 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | ONNX Runtime 1.6.0 | TorchVision_DenseNet201 | Image Classification | 53 | 274 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_Alexnet | Image Classification | 148 | 474 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_ResNet_18 | Image Classification | 98 | 437 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_ResNet_34 | Image Classification | 76 | 402 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_ResNet50 | Image Classification | 62 | 346 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_ResNet101 | Image Classification | 45 | 284 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_ResNet152 | Image Classification | 34 | 227 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_VGG11 | Image Classification | 132 | 368 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_VGG13 | Image Classification | 125 | 320 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_VGG16 | Image Classification | 118 | 291 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_VGG19 | Image Classification | 111 | 268 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_VGG11_BN | Image Classification | 116 | 358 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_VGG13_BN | Image Classification | 111 | 300 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_VGG16_BN | Image Classification | 102 | 271 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_VGG19_BN | Image Classification | 95 | 254 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_DenseNet121 | Image Classification | 37 | 337 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_DenseNet161 | Image Classification | 32 | 227 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_DenseNet169 | Image Classification | 30 | 298 |
| raider-01| Intel(R) Core(TM) i7-7820X CPU @ 3.60GHz | 1 | NVIDIA TITAN V | 1 | CUDA 10.2 | PyTorch 1.5.0 | TorchVision_DenseNet201 | Image Classification | 26 | 269 |


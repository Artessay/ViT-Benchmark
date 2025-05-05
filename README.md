# ViT-Benchmark

## Download Dataset and Checkpoints

Download ImageNet dataset on `https://image-net.org`.

Download ViT checkpoint:

```sh
wget https://download.pytorch.org/models/vit_b_16-c867db91.pth -O checkpoints/vit_b_16.pth
```

## Evaluation Result

### Train

| dataset | method | accuracy |
|:----:|:----:|:----:|
| **cifar - 10** | pt | 94.58 |
| | ft | 97.67 |
| | cft | 97.99 |
| | ncft-random  | 98.02 |
| | ncft-weight  | 97.99 |
| | ncft-shapley | 97.78 |
| **cifar - 100** | pt | 79.69 |
| | ft | 85.52 |
| | cft | 87.05 |
| | ncft-random-0.7  | 86.74 |
| | ncft-weight-0.1  | 84.72 |
| | ncft-weight-0.3  | 86.31 |
| | ncft-weight-0.5  | 86.49 |
| | ncft-weight-0.7  | 86.76 |
| | ncft-shapley-0.5 | 86.49 |
| | ncft-shapley-0.7 | 86.45 |
| **imagenet** | pt | 80.14 |
| | ft | 78.09 |
| | cft | 78.51 |
| | ncft-random | - |
| | ncft-weight | - |

### Inference


| dataset | method | accuracy |
|:----:|:----:|:----:|
| **cifar - 10** | pt | 94.58 |
| | ft | 97.67 |
| | cft | 97.99 |
| | ncft-random  | 98.02 |
| | ncft-weight  | 97.99 |
| | ncft-shapley | - |
| **cifar - 100** | pt | 79.69 |
| | ft | 85.52 |
| | cft | 87.05 |
| | ncft-random-0.7  | 86.74 |
| | ncft-weight-0.1  | 84.72 |
| | ncft-weight-0.3  | 86.31 |
| | ncft-weight-0.5  | 86.49 |
| | ncft-weight-0.7  | 86.76 |
| | ncft-shapley-0.5 | - |
| | ncft-shapley-0.7 | - |
| **imagenet** | pt | 80.14 |
| | ft | 78.09 |
| | cft | 78.51 |
| | ncft-random | - |
| | ncft-weight | - |
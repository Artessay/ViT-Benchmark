# ViT-Benchmark

## Download Dataset and Checkpoints

Download ImageNet dataset on `https://image-net.org`.

Download ViT checkpoint:

```sh
wget https://download.pytorch.org/models/vit_b_16-c867db91.pth -O checkpoints/vit_b_16.pth
```

## Evaluation Result

| dataset | method | accuracy |
|:----:|:----:|:----:|
| **cifar - 10** | pt | 94.58 |
| | ft | 97.67 |
| | cft | 97.99 |
| | ncft-random | 98.02 |
| **cifar - 100** | pt | 79.69 |
| | ft | 85.52 |
| | cft | 87.05 |
| | ncft-random | 86.74 |
| **imagenet** | pt | - |
| | ft | - |
| | cft | - |
| | ncft-random | - |
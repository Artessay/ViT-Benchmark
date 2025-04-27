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
| **cifar - 10** | pt | - |
| | ft | 97.67 |
| | cft | - |
| | ncft | - |
| **cifar - 100** | pt | - |
| | ft | 85.52 |
| | cft | - |
| | ncft | - |
| **imagenet** | pt | - |
| | ft | - |
| | cft | - |
| | ncft | - |
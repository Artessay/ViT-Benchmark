# ViT-Benchmark

## Download Dataset and Checkpoints

Download ImageNet dataset on `https://image-net.org` and place it in the folder `data/imagenet/`.

Download ViT checkpoint:

```sh
wget https://download.pytorch.org/models/vit_b_16-c867db91.pth -O checkpoints/vit_b_16.pth
```

## Get Start

Train model:

```sh
python main.py
```

Evaluate model:

```sh
python eval.py
```

## Evaluation Result

### Train

| dataset | method | accuracy |
|:----:|:----:|:----:|
| **cifar - 10** | pt | 94.58 |
| | ft | 97.67 |
| | cft | 97.99 |
| | ncft-random  | 98.02 |
| | ncft-gradient| 97.85 |
| | ncft-weight  | 97.91 |
| | ncft-shapley | 97.78 |
| **cifar - 100** | pt | 79.69 |
| | ft | 85.52 |
| | cft | 87.05 |
| | ncft-random-0.3  | 85.64 |
| | ncft-random-0.7  | 86.74 |
| | ncft-weight-0.1  | 84.72 |
| | ncft-weight-0.3  | 86.30 |
| | ncft-weight-0.5  | 86.49 |
| | ncft-weight-0.7  | 86.77 |
| | ncft-shapley-0.3 | 86.35 |
| | ncft-shapley-0.5 | 86.49 |
| | ncft-shapley-0.7 | 86.45 |
| **imagenet** | pt | 80.14 |
| | ft | 78.09 |
| | cft | 78.51 |
| | ncft-random-0.3 | 79.70 |
| | ncft-weight-0.3 | 79.61 |
| | ncft-shapley-0.3 | 79.60 |
| | ncft-random-0.7 | 79.57 |
| | ncft-weight-0.7 | 79.54 |
| | ncft-shapley-0.7 | 79.57 |

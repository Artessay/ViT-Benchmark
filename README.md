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
python main.py -m [mode] -d [dataset] -s [seed]
```

Evaluate model:

```sh
python eval.py -m [mode] -d [dataset] -s [seed]
```
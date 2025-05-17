#!/bin/bash

mkdir -p logs/eval

CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m pt -d cifar-100 >> logs/eval/pt_cifar-100.log 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m ft -d cifar-100 >> logs/eval/ft_cifar-100.log 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m cft -d cifar-100 >> logs/eval/cft_cifar-100.log 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m r-ncft -d cifar-100 >> logs/eval/r-ncft_cifar-100.log 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m g-ncft -d cifar-100 >> logs/eval/g-ncft_cifar-100.log 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m w-ncft -d cifar-100 >> logs/eval/w-ncft_cifar-100.log 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m s-ncft -d cifar-100 >> logs/eval/s-ncft_cifar-100.log 2>&1


CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m pt -d imagenet >> logs/eval/pt_imagenet.log 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m ft -d imagenet >> logs/eval/ft_imagenet.log 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m cft -d imagenet >> logs/eval/cft_imagenet.log 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m r-ncft -d imagenet >> logs/eval/r-ncft_imagenet.log 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m g-ncft -d imagenet >> logs/eval/g-ncft_imagenet.log 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m w-ncft -d imagenet >> logs/eval/w-ncft_imagenet.log 2>&1
CUDA_VISIBLE_DEVICES=0 nohup python eval.py -m s-ncft -d imagenet >> logs/eval/s-ncft_imagenet.log 2>&1


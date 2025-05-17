#!/bin/bash

mkdir -p logs/train

CUDA_VISIBLE_DEVICES=0 nohup python main.py -m s-ncft -d cifar-100 -lr 1e-5 -activate_ratio 0.1 >> logs/train/s-ncft_cifar-100_lr_1e-5_activate_ratio_0.1.log 2>&1 
CUDA_VISIBLE_DEVICES=0 nohup python main.py -m s-ncft -d cifar-100 -lr 1e-5 -activate_ratio 0.3 >> logs/train/s-ncft_cifar-100_lr_1e-5_activate_ratio_0.3.log 2>&1 
CUDA_VISIBLE_DEVICES=0 nohup python main.py -m s-ncft -d cifar-100 -lr 1e-5 -activate_ratio 0.5 >> logs/train/s-ncft_cifar-100_lr_1e-5_activate_ratio_0.5.log 2>&1 
CUDA_VISIBLE_DEVICES=0 nohup python main.py -m s-ncft -d cifar-100 -lr 1e-5 -activate_ratio 0.7 >> logs/train/s-ncft_cifar-100_lr_1e-5_activate_ratio_0.7.log 2>&1 
CUDA_VISIBLE_DEVICES=0 nohup python main.py -m s-ncft -d cifar-100 -lr 1e-5 -activate_ratio 0.9 >> logs/train/s-ncft_cifar-100_lr_1e-5_activate_ratio_0.9.log 2>&1 
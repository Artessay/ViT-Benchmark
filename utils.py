import argparse
import logging
import os
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(log_file) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file output
    file_handler = logging.FileHandler(log_file, mode="w")  # use 'w' to overwrite existing file, default is 'a' (append)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="ViT Benchmark")

    # pretrain, finetune, continue train
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="train mode",
        default="ft",
        choices=["pt", "ft", "cft", "r-ncft", "g-ncft", "w-ncft", "s-ncft"],
    )

    parser.add_argument(
        "-d", "--dataset", type=str, default="cifar-100", help="Dataset name", choices=["cifar-10", "cifar-100", "imagenet"]
    )

    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("-activate_ratio", "--activate_ratio", type=float, default=0.5, help="Activate ratio")

    args = parser.parse_args()
    return args

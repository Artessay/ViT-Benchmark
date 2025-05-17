import json

import torch
from torch.utils.tensorboard import SummaryWriter

import neural_function
from config import CONFIG
from data_loader import get_data_loader
from train_and_eval import evaluate, train_with_partial_activate
from utils import seed_everything, setup_logger
from vit_loader import create_vit_model, load_vit_model


def main(args):
    seed = args.seed
    mode = args.mode
    dataset_name = args.dataset
    seed_everything(seed)

    config = CONFIG.get(dataset_name)
    model_name = config["model_name"]
    batch_size = config["batch_size"]
    train_loader, val_loader, test_loader = get_data_loader(dataset_name, batch_size=batch_size)

    # 加载预训练的ViT模型
    num_classes = config["num_classes"]
    if mode in ["pt", "ft"]:
        vit_model = create_vit_model(model_name, num_classes=num_classes)
    elif "cft" in mode:
        weights_path = f"checkpoints/{model_name}_{dataset_name}_pt_{seed}.pth"
        vit_model = load_vit_model(weights_path, num_classes=num_classes)
    else:
        raise ValueError(f"Mode {mode} is not supported.")

    # 准备日志
    # activate_ratio = config["activate_ratio"]
    activate_ratio = args.activate_ratio
    lr = args.learning_rate
    if "ncft" in mode:
        experiment_name = f"{model_name}_{dataset_name}_{mode}_{seed}_r{activate_ratio}"
    else:
        experiment_name = f"{model_name}_{dataset_name}_{mode}_{seed}"

    experiment_name = experiment_name + f"_lr{lr}"

    save_path = f"checkpoints/{experiment_name}.pth"
    logger = setup_logger(f"logs/{experiment_name}.log")
    logger.info(f"Model: {model_name}, Dataset: {dataset_name}, Mode: {mode}")
    writer = SummaryWriter(f"runs/{experiment_name}")

    epochs, patience = config["epochs"], config["patience"]
    # lr = config["pt_lr"] if mode == "pt" else config["lr"]
    weight_decay = config["weight_decay"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == "pt":
        neural_function.activate_head(vit_model)
    elif mode == "ft" or mode == "cft":
        neural_function.activate_full(vit_model)
    elif mode == "r-ncft":
        neural_function.activate_random(vit_model, activate_ratio)
    elif mode == "g-ncft":
        neural_function.activate_based_on_gradient(vit_model, activate_ratio, val_loader, device)
    elif mode == "w-ncft":
        neural_function.activate_based_on_gradient_trace(vit_model, activate_ratio, val_loader, device)
    elif mode == "s-ncft":
        neural_function.activate_based_on_shapley_value(vit_model, activate_ratio, val_loader, device)
    else:
        raise ValueError(f"Mode {mode} is not supported.")

    train_with_partial_activate(
        vit_model, train_loader, val_loader, epochs, patience, lr, weight_decay, save_path, device, logger, writer
    )
    logger.info("Training completed.")

    # 测试预训练模型
    vit_model.load_state_dict(torch.load(save_path))
    accuracy = evaluate(vit_model, test_loader, device)
    logger.info(f"Accuracy on the test set: {100 * accuracy: .2f}%")
    writer.close()

    config["accuracy"] = round(100 * accuracy, 2)
    result_path = f"results/train_{experiment_name}.json"
    with open(result_path, "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    from utils import get_args

    args = get_args()
    main(args)

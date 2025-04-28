import json
import torch

import neural_function
from config import CONFIG
from data_loader import get_data_loader
from vit_loader import create_vit_model, load_vit_model
from train_and_eval import evaluate
from utils import seed_everything

def main(args):
    seed = args.seed
    mode = args.mode
    dataset_name = args.dataset
    seed_everything(seed)

    config = CONFIG.get(dataset_name)
    model_name = config['model_name']
    batch_size = config['batch_size']
    _, val_loader, test_loader = get_data_loader(dataset_name, batch_size=batch_size)

    # 准备日志
    activate_ratio = config['activate_ratio']
    if "ncft" in mode:
        experiment_name = f"{model_name}_{dataset_name}_{mode}_{seed}_r{activate_ratio}"
    else:
        experiment_name = f"{model_name}_{dataset_name}_{mode}_{seed}"
    save_path = f"checkpoints/{experiment_name}.pth"
    result_path = f"results/{experiment_name}.json"


    # 加载预训练的ViT模型
    num_classes = config['num_classes']
    if mode in ['pt', 'ft', 'cft']:
        vit_model = load_vit_model(save_path, num_classes=num_classes)
    elif 'ncft' in mode:
        weights_path = f"checkpoints/{model_name}_{dataset_name}_cft_{seed}.pth"
        vit_model = load_vit_model(weights_path, num_classes=num_classes)
    else:
        raise ValueError(f"Mode {mode} is not supported.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode in ['pt', 'ft', 'cft']:
        pass    # do nothing
    elif mode == 'r-ncft':
        neural_function.activate_neuron_random(vit_model, activate_ratio)
    elif mode == 'w-ncft':
        neural_function.activate_neuron_based_on_gradient_trace(vit_model, activate_ratio, val_loader, device)
    else:
        raise ValueError(f"Mode {mode} is not supported.")
    
    # 测试预训练模型
    accuracy = evaluate(vit_model, test_loader, device)

    config["accuracy"] = round(100 * accuracy, 2)
    with open(result_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Accuracy: {config["accuracy"]}")

if __name__ == "__main__":
    from utils import get_args
    args = get_args()
    main(args)
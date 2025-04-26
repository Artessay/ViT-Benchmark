import torch
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG
from data_loader import get_data_loader
from vit_loader import create_vit_model, load_vit_model
from train_and_eval import train_head, train_full, evaluate
from utils import seed_everything, setup_logger

def main(args):
    seed = args.seed
    mode = args.mode
    dataset_name = args.dataset
    seed_everything(seed)

    config = CONFIG.get(dataset_name)
    model_name = config['model_name']
    batch_size = config['batch_size']
    train_loader, val_loader, test_loader = get_data_loader(dataset_name, batch_size=batch_size)

    # 加载预训练的ViT模型
    num_classes = config['num_classes']
    if mode in ['pt', 'ft']:
        vit_model = create_vit_model(model_name, num_classes=num_classes)
    elif mode in ['cft', 'ncft']:
        weights_path = f"checkpoints/{model_name}_{dataset_name}_pt_{seed}.pth"
        vit_model = load_vit_model(weights_path, num_classes=num_classes)

    # 准备日志
    save_path = f"checkpoints/{model_name}_{dataset_name}_{mode}_{seed}.pth"
    logger = setup_logger(save_path)
    logger.info(f"Model: {model_name}, Dataset: {dataset_name}, Mode: {mode}")
    writer = SummaryWriter(f'runs/{model_name}_{dataset_name}_{mode}_{seed}')

    epochs, patience, lr = config['epochs'], config['patience'], config['lr']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练分类头
    if mode == 'pt':
        train_head(vit_model, train_loader, val_loader, epochs, patience, lr, save_path, device, logger, writer)
    else:
        train_full(vit_model, train_loader, val_loader, epochs, patience, lr, save_path, device, logger, writer)
    logger.info("Training completed.")

    # 测试预训练模型
    vit_model.load_state_dict(torch.load(save_path))
    accuracy_pretrained = evaluate(vit_model, test_loader, device)
    logger.info(f'Accuracy on the test set: {100 * accuracy_pretrained: .4f}%')
    writer.close()
    
if __name__ == "__main__":
    from utils import get_args
    args = get_args()
    main(args)
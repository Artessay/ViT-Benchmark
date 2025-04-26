import torch

from data_loader import get_data_loader
from model_loader import load_vit_model
from train_and_eval import train_full, evaluate
from utils import seed_everything

batch_size = 512
epochs = 50
patience = 5
lr = 3e-5
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    seed_everything(seed)

    dataset_name = "cifar-100"
    model_name = "vit_b_16"
    train_loader, val_loader, test_loader = get_data_loader(dataset_name, batch_size=batch_size)

    # 加载预训练的ViT模型
    weights_path = f"checkpoints/{model_name}_{dataset_name}_pt_head.pth"
    vit_model = load_vit_model(weights_path, num_classes=100)

    # 微调模型
    save_path = f"checkpoints/{model_name}_{dataset_name}_ct_full.pth"
    train_full(vit_model, train_loader, val_loader, epochs, patience, lr, save_path, device)
    print("Finetuning completed.")

    # 测试预训练模型
    vit_model.load_state_dict(torch.load(save_path))
    accuracy_pretrained = evaluate(vit_model, test_loader, device)
    print(f'Accuracy of the continue learning model on the test images: {100 * accuracy_pretrained}%')
    
if __name__ == "__main__":
    main()
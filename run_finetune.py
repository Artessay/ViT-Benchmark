import torch

from data_loader import get_data_loader
from train_and_eval import train_full, evaluate
from utils import seed_everything
from model import create_vit_model

batch_size = 512
epochs = 100
patience = 5
lr = 3e-5
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    seed_everything(seed)

    dataset_name = "cifar-100"
    model_name = "vit_b_16"
    train_loader, test_loader = get_data_loader(dataset_name, batch_size=batch_size)

    # 加载预训练的ViT模型
    vit_model = create_vit_model(model_name)

    # 微调模型
    save_path = f"checkpoints/{model_name}_{dataset_name}_ft_full.pth"
    train_full(vit_model, train_loader, test_loader, epochs, patience, lr, save_path, device)
    print("Finetuning completed.")

    # 测试预训练模型
    accuracy_pretrained = evaluate(vit_model, test_loader, device)
    print(f'Accuracy of the pretrained model on the test images: {100 * accuracy_pretrained}%')
    
if __name__ == "__main__":
    main()
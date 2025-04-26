import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_data_loader
from train_and_eval import train, evaluate
from utils import seed_everything
from model import create_vit_model

batch_size = 512
epochs = 20
lr = 3e-5
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    seed_everything(seed)
    train_loader, test_loader = get_data_loader("cifar-100", batch_size=batch_size)

    # 加载预训练的ViT模型
    model_pretrained = create_vit_model("vit_b_16")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_pretrained.parameters(), lr=lr)

    # 微调模型
    train(model_pretrained, train_loader, criterion, optimizer, device)
    print("Finetuning completed.")

    # 测试预训练模型
    accuracy_pretrained = evaluate(model_pretrained, test_loader, device)
    print(f'Accuracy of the pretrained model on the test images: {100 * accuracy_pretrained}%')
    
if __name__ == "__main__":
    main()
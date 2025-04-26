import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16, ViT_B_16_Weights

from data_loader import get_data_loader
from utils import seed_everything

batch_size = 64
epochs = 20
lr = 3e-5
seed = 42

seed_everything(seed)
train_loader, test_loader = get_data_loader("cifar-100", batch_size=batch_size)


# 加载预训练的ViT模型
model_pretrained = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
num_ftrs = model_pretrained.heads.head.in_features
model_pretrained.heads.head = nn.Linear(num_ftrs, 100)

# 微调模型
model_finetuned = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
num_ftrs = model_finetuned.heads.head.in_features
model_finetuned.heads.head = nn.Linear(num_ftrs, 100)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_finetuned.parameters(), lr=lr)

# 微调模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_finetuned.to(device)
model_finetuned.train()
for epoch in range(5):  # 训练5个epoch
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model_finetuned(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')


# 测试预训练模型
model_pretrained.to(device)
model_pretrained.eval()
correct_pretrained = 0
total_pretrained = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model_pretrained(images)
        _, predicted = torch.max(outputs.data, 1)
        total_pretrained += labels.size(0)
        correct_pretrained += (predicted == labels).sum().item()

# 测试微调后的模型
model_finetuned.eval()
correct_finetuned = 0
total_finetuned = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model_finetuned(images)
        _, predicted = torch.max(outputs.data, 1)
        total_finetuned += labels.size(0)
        correct_finetuned += (predicted == labels).sum().item()

print(f'Accuracy of the pretrained model on the 10000 test images: {100 * correct_pretrained / total_pretrained}%')
print(f'Accuracy of the finetuned model on the 10000 test images: {100 * correct_finetuned / total_finetuned}%')
    
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.models import VisionTransformer
from tqdm import tqdm

def train_one_epoch(model: VisionTransformer, train_loader: DataLoader, criterion: nn.Module, optimizer: Optimizer, device, epoch: int):
    running_loss = 0.0
    for data in tqdm(train_loader, ncols=80, desc="Train", leave=False):
        images, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

def train_head(model: VisionTransformer, train_loader: DataLoader, epochs: int, criterion: nn.Module, optimizer: Optimizer, device):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
    print("Training head completed.")

def train_full(model: VisionTransformer, train_loader: DataLoader, epochs: int, criterion: nn.Module, optimizer: Optimizer, device):
    model.to(device)
    model.train()



def evaluate(model: VisionTransformer, test_loader: DataLoader, device):
    model.to(device)
    model.eval()
    correct_pretrained = 0
    total_pretrained = 0
    with torch.no_grad():
        for data in tqdm(test_loader, ncols=80, desc="Test"):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total_pretrained += labels.size(0)
            correct_pretrained += (predicted == labels).sum().item()

    return correct_pretrained / total_pretrained
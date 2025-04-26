import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.models import VisionTransformer
from tqdm import tqdm

def train_one_epoch(model: VisionTransformer, train_loader: DataLoader, optimizer: Optimizer, device):
    model.to(device)
    model.train()

    running_loss = 0.0
    for data in tqdm(train_loader, ncols=80, desc="Train", leave=False):
        images, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

def train(model: VisionTransformer, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, epochs: int, patience: int, save_path: str, device):
    best_accuracy = 0.0
    patience_counter = 0
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f'\nEpoch {epoch + 1}, Loss: {loss:.5f}')
        val_accuracy = evaluate(model, val_loader, device)
        print(f'\nValidation Accuracy: {val_accuracy * 100:.2f}%')

        # Save the model checkpoint if validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(f"Saving model with accuracy: {best_accuracy * 100:.2f}%")
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")

def train_head(model: VisionTransformer, train_loader: DataLoader, val_loader: DataLoader, epochs: int, patience: int, lr: float, save_path: str, device):
    for name, param in model.named_parameters():
        if "heads" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    
    train(model, train_loader, val_loader, optimizer, epochs, patience, save_path, device)

def train_full(model: VisionTransformer, train_loader: DataLoader, val_loader: DataLoader, epochs: int, patience: int, lr: float, save_path: str, device):
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(model, train_loader, val_loader, optimizer, epochs, patience, save_path, device)


def evaluate(model: VisionTransformer, test_loader: DataLoader, device):
    model.to(device)
    model.eval()
    correct_pretrained = 0
    total_pretrained = 0
    with torch.no_grad():
        for data in tqdm(test_loader, ncols=80, desc="Test", leave=False):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total_pretrained += labels.size(0)
            correct_pretrained += (predicted == labels).sum().item()

    return correct_pretrained / total_pretrained
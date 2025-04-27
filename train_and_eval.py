import torch
import torch.nn.functional as F

from tqdm import tqdm
from logging import Logger
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.models import VisionTransformer
from torch.utils.tensorboard import SummaryWriter

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

def train(model: VisionTransformer, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, epochs: int, patience: int, save_path: str, device: torch.device, logger: Logger, writer: SummaryWriter):
    best_accuracy = 0.0
    patience_counter = 0
    for epoch in range(epochs):
        # Train the model
        loss = train_one_epoch(model, train_loader, optimizer, device)
        logger.info(f'Epoch {epoch + 1}, Loss: {loss:.5f}')
        writer.add_scalar('Loss/train', loss, epoch)
        # Validate the model
        val_accuracy = evaluate(model, val_loader, device)
        logger.info(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # Save the model checkpoint if validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            logger.info(f"Saving model with accuracy: {best_accuracy * 100:.2f}%")
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
    logger.info(f"Best validation accuracy: {best_accuracy * 100:.2f}%")

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

def train_with_partial_activate(
        model: VisionTransformer, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        epochs: int, 
        patience: int, 
        lr: float, 
        weight_decay: float,
        save_path: str, 
        device: torch.device, 
        logger: Logger, 
        writer: SummaryWriter,
):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    train(model, train_loader, val_loader, optimizer, epochs, patience, save_path, device, logger, writer)


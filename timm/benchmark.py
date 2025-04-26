import torch
import timm
from data_loader import get_data_loader
from config import CONFIG

def load_model(model_name, pretrained=True, num_classes=1000):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def finetune_model(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # 验证步骤
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(targets).sum().item()

        print(f'Epoch {epoch+1}/{epochs} - Val loss: {val_loss/len(val_loader):.4f} Acc: {correct/len(val_loader.dataset):.4f}')

    return model


def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy


if __name__ == '__main__':
    import csv
    with open(CONFIG['output_file'], 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset','model','pretrained_accuracy','finetuned_accuracy'])
        writer.writeheader()
        
        for dataset_cfg in CONFIG['datasets']:
            # 加载配置适配模型
            model = load_model(
                dataset_cfg['model_mapping'],
                num_classes=dataset_cfg['num_classes']
            )
            
            # 获取适配数据加载器
            train_loader, val_loader = get_data_loader(
                dataset_cfg['name'],
                input_size=dataset_cfg['input_size']
            )
            
            # 评估预训练模型
            pretrained_accuracy = evaluate_model(model, val_loader)
            
            # 微调模型
            finetuned_model = finetune_model(model, train_loader, val_loader)
            finetuned_accuracy = evaluate_model(finetuned_model, val_loader)
            
            # 记录结果
            writer.writerow({
                'dataset': dataset_cfg['name'],
                'model': dataset_cfg['model_mapping'],
                'pretrained_accuracy': f"{pretrained_accuracy:.4f}",
                'finetuned_accuracy': f"{finetuned_accuracy:.4f}",
            })